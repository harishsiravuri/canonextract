"""Adapter that runs our ContributionRecord predictions through the official
SciREX evaluator (`scirex_relation_evaluate.py`).

The official evaluator expects three input JSONL files:
  - NER:               {"doc_id": ..., "ner": [[start_tok, end_tok, label], ...]}
  - Salient clusters:  {"doc_id": ..., "clusters": {name: [[s,e], ...]}, "coref": {name: [[s,e], ...]}}
  - Relations:         {"doc_id": ..., "predicted_relations": [[[c1,c2,c3,c4], score, label], ...]}

For each predicted entity name in our ContributionRecord we:
  1. Find every (case-insensitive, multi-token) occurrence of the name in
     the gold paper's whitespace-tokenised `words` list.
  2. Add token-level [start, end+1] spans to the NER list with the
     appropriate SciREX label ('Method' / 'Task' / 'Material' / 'Metric').
  3. Group the spans by canonical name → one salient cluster per unique
     (label, normalized name).
  4. For each contribution, emit a relation tuple of cluster names
     (Material, Metric, Task, Method) with score=1, label=1.

If any of the four cluster slots in a contribution has no matched
mentions, we drop that relation (the official scorer requires all four).

The adapter is conservative: a predicted entity that doesn't appear
verbatim in the paper text is silently dropped — the SciREX evaluator
otherwise treats unknown spans as automatic precision misses.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

from paper1.loaders import GoldPaper
from paper1.schema import ContributionRecord
from paper1.voting import _norm

# Map our ContributionUnit field names to SciREX entity types
KIND_TO_LABEL = {"method": "Method", "task": "Task", "dataset": "Material", "metric": "Metric"}
USED_ENTITIES = ["Material", "Metric", "Task", "Method"]


def _norm_words(words: list[str]) -> list[str]:
    return [w.lower() for w in words]


def _find_mentions(name_tokens: list[str], words_lower: list[str]) -> list[tuple[int, int]]:
    """Return all [start, end+1] token spans where `name_tokens` matches `words_lower` exactly."""
    out: list[tuple[int, int]] = []
    n = len(name_tokens)
    if n == 0:
        return out
    for i in range(len(words_lower) - n + 1):
        if words_lower[i : i + n] == name_tokens:
            out.append((i, i + n))
    return out


def _load_gold_words(gold_papers: list[GoldPaper]) -> dict[str, list[str]]:
    """Re-load the SciREX gold JSONL to recover per-doc `words` arrays."""
    base = Path("data/raw/scirex/scirex_dataset/release_data")
    out: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        p = base / f"{split}.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                doc = json.loads(line)
                out[f"scirex:{doc['doc_id']}"] = doc.get("words", [])
    return out


def build_prediction_files(
    records: dict[str, ContributionRecord],
    gold_papers: list[GoldPaper],
    out_dir: Path,
) -> dict:
    """Write ner.jsonl, clusters.jsonl, relations.jsonl in the SciREX format.

    `records` maps paper_id (e.g. "scirex:abc...") → our ContributionRecord.
    `gold_papers` is the SciREX dev set (or the subset we evaluated on).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    words_by_doc = _load_gold_words(gold_papers)

    ner_path = out_dir / "ner.jsonl"
    clusters_path = out_dir / "clusters.jsonl"
    relations_path = out_dir / "relations.jsonl"

    with ner_path.open("w") as fner, clusters_path.open("w") as fclu, relations_path.open("w") as frel:
        for gp in gold_papers:
            doc_id_full = gp.paper_id  # "scirex:..."
            doc_id_short = doc_id_full.split(":", 1)[1]
            words = words_by_doc.get(doc_id_full, [])
            words_lower = _norm_words(words)
            rec = records.get(doc_id_full)

            ner_spans: list[list] = []  # [start, end, label]
            clusters: dict[str, list[list[int]]] = {}  # cluster_name → [[s,e], ...]

            def _register(kind: str, name: str | None) -> str | None:
                """Find mentions of `name` (kind ∈ {method,task,dataset,metric}),
                add NER spans, append to cluster, return the cluster_name."""
                if not name:
                    return None
                clean = re.sub(r"\s+", " ", name.strip())
                tokens = clean.lower().split()
                if not tokens:
                    return None
                spans = _find_mentions(tokens, words_lower)
                if not spans:
                    return None
                label = KIND_TO_LABEL[kind]
                cluster_name = f"{label}__{_norm(clean) or clean.lower()}"
                bucket = clusters.setdefault(cluster_name, [])
                for s, e in spans:
                    ner_spans.append([s, e, label])
                    if [s, e] not in bucket:
                        bucket.append([s, e])
                return cluster_name

            relations: list[list] = []  # [[c1,c2,c3,c4], score, label]
            if rec is not None:
                for c in rec.contributions:
                    method_c = _register("method", c.method.name)
                    task_c = _register("task", c.task.name)
                    # For datasets and metrics, register all of them but pick the first
                    # for the relation slot (SciREX gold relations are one-of)
                    ds_clusters = []
                    for d in c.datasets:
                        cn = _register("dataset", d.name)
                        if cn:
                            ds_clusters.append(cn)
                    mt_clusters = []
                    for m in c.metrics:
                        cn = _register("metric", m.name)
                        if cn:
                            mt_clusters.append(cn)
                    dataset_c = ds_clusters[0] if ds_clusters else None
                    metric_c = mt_clusters[0] if mt_clusters else None
                    if all([method_c, task_c, dataset_c, metric_c]):
                        # SciREX expects [Material, Metric, Task, Method] order
                        relations.append([[dataset_c, metric_c, task_c, method_c], 1, 1])

            # Emit
            fner.write(json.dumps({"doc_id": doc_id_short, "ner": ner_spans}) + "\n")
            fclu.write(json.dumps({"doc_id": doc_id_short, "clusters": clusters, "coref": clusters}) + "\n")
            frel.write(json.dumps({"doc_id": doc_id_short, "predicted_relations": relations}) + "\n")

    return {
        "ner_file": str(ner_path),
        "clusters_file": str(clusters_path),
        "relations_file": str(relations_path),
        "n_papers": len(gold_papers),
    }


def run_official_evaluator(
    pred_dir: Path,
    gold_split: str = "dev",
    base: Path = Path("data/raw/scirex"),
) -> dict:
    """Invoke `python -m scirex.evaluation_scripts.scirex_relation_evaluate ...`
    and capture the printed metrics.

    The evaluator prints multi-section output ("Salient Clustering Metrics",
    "Relation Metrics n=2", "Relation Metrics n=4"). We capture and parse
    those into a dict.
    """
    gold_file = (base / "scirex_dataset" / "release_data" / f"{gold_split}.jsonl").resolve()
    pred_dir = pred_dir.resolve()
    cmd = [
        sys.executable,
        "-m", "scirex.evaluation_scripts.scirex_relation_evaluate",
        "--gold-file", str(gold_file),
        "--ner-file", str(pred_dir / "ner.jsonl"),
        "--clusters-file", str(pred_dir / "clusters.jsonl"),
        "--relations-file", str(pred_dir / "relations.jsonl"),
    ]
    env = {"PYTHONPATH": str(base)}
    import os
    env_full = {**os.environ, **env}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base), env=env_full, timeout=900)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": " ".join(cmd)}


def parse_evaluator_output(text: str) -> dict:
    """Pull (p, r, f1) out of the evaluator's printed output."""
    out: dict[str, dict] = {}
    section = None
    cur: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Salient Clustering Metrics"):
            section = "salient_clustering"; cur = {}
        elif s.startswith("Relation Metrics n="):
            if section and cur:
                out[section] = cur
            section = s.replace(" ", "_").replace("=", ""); cur = {}
        else:
            for k in ("p", "r", "f1"):
                m = re.match(rf"^{k}\s+([0-9.eE+-]+)\s*$", s)
                if m:
                    cur[k] = float(m.group(1))
    if section and cur:
        out[section] = cur
    return out
