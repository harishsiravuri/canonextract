"""Build per-paper vocabulary indices for CanonExtract (Component 2).

Three tiers per paper:
  - local:    row/col headers + captions + neighborhoods from this paper's tex.
  - external: MetaLead-gold union + SciREX-train-gold union (skip PwC — API
              was unreachable in probe; fall back per spec).
  - fallback: paper-text n-grams (deterministic).

Output: outputs/canonextract_build/vocab/{paper_id}.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

OUT_DIR = Path("outputs/canonextract_build/vocab")
METALEAD_JSON = Path("data/metalead/annotations/metalead_annotations.json")
SCIREX_TRAIN_JSONL = Path("data/raw/scirex/scirex_dataset/release_data/train.jsonl")
SCIREX_DEV_JSONL = Path("data/raw/scirex/scirex_dataset/release_data/dev.jsonl")
METALEAD_TABLES_DIR = Path("outputs/canonextract_diag_v2/metalead_tables")
SCIREX_TABLES_DIR = Path("outputs/canonextract_diag/tables")
SCIREX_ANCHORS_DIR = Path("outputs/canonextract_diag_v2/scirex_expanded_anchors")
METALEAD_TEX_DIR = Path("data/metalead_tex")
SCIREX_TEX_DIR = Path("data/scirex_tex")
EXTERNAL_CACHE = Path("outputs/canonextract_build/external_vocab.json")

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str | None) -> str:
    if not s: return ""
    s = str(s).lower().replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _local_from_metalead(paper_key: str) -> dict:
    p = METALEAD_TABLES_DIR / f"{paper_key}.json"
    if not p.exists(): return {"tasks": [], "datasets": [], "metrics": []}
    d = json.loads(p.read_text())
    row = [normalize(r) for r in d.get("row_headers", []) if r]
    col = [normalize(c) for c in d.get("column_headers", []) if c]
    cells = [normalize(c) for c in d.get("cells", []) if c]
    caps = [normalize(c) for c in d.get("table_captions", []) if c]
    fcaps = [normalize(c) for c in d.get("figure_captions", []) if c]
    nbrs = [normalize(c) for c in d.get("table_neighborhoods", []) if c]
    all_str = row + col + cells + caps + fcaps + nbrs
    # Split multi-word into candidate phrases: full, minus parenthesized qualifier, minus unit
    phrases = set()
    for s in all_str:
        if not s: continue
        phrases.add(s)
        no_paren = re.sub(r"\(.*?\)", " ", s)
        no_paren = _WS_RE.sub(" ", no_paren).strip()
        if no_paren: phrases.add(no_paren)
        # strip trailing "% ,accuracy" etc
        trimmed = re.sub(r"\s+(?:accuracy|f1|score|value|s|w|test|dev|train|%)$", "", s).strip()
        if trimmed: phrases.add(trimmed)
    return {"tasks": [], "datasets": [], "metrics": [], "all_phrases": sorted(phrases)}


def _local_from_scirex(doc_id: str) -> dict:
    tp = SCIREX_TABLES_DIR / f"{doc_id}.json"
    if not tp.exists(): return {"tasks": [], "datasets": [], "metrics": [], "all_phrases": []}
    d = json.loads(tp.read_text())
    row, col, cells = [], [], []
    for t in d.get("tables", []):
        row.extend(t.get("row_headers", []))
        col.extend(t.get("column_headers", []))
        cells.extend(t.get("cells", []))
    caps, fcaps, nbrs = [], [], []
    ap = SCIREX_ANCHORS_DIR / f"{doc_id}.json"
    if ap.exists():
        e = json.loads(ap.read_text())
        caps = e.get("table_captions", []); fcaps = e.get("figure_captions", []); nbrs = e.get("table_neighborhoods", [])
    all_str = [normalize(s) for s in (row + col + cells + caps + fcaps + nbrs) if s]
    phrases = set()
    for s in all_str:
        if not s: continue
        phrases.add(s)
        no_paren = re.sub(r"\(.*?\)", " ", s)
        no_paren = _WS_RE.sub(" ", no_paren).strip()
        if no_paren: phrases.add(no_paren)
        trimmed = re.sub(r"\s+(?:accuracy|f1|score|value|s|w|test|dev|train|%)$", "", s).strip()
        if trimmed: phrases.add(trimmed)
    return {"tasks": [], "datasets": [], "metrics": [], "all_phrases": sorted(phrases)}


def _build_external_vocab() -> dict:
    """Union of MetaLead + SciREX train+dev gold vocab. PwC skipped (unreachable)."""
    if EXTERNAL_CACHE.exists():
        return json.loads(EXTERNAL_CACHE.read_text())

    tasks, datasets, metrics = Counter(), Counter(), Counter()
    # MetaLead
    m = json.loads(METALEAD_JSON.read_text())
    for _, entry in m.items():
        for tdm in entry.get("TDMs") or []:
            if tdm.get("Task"): tasks[normalize(tdm["Task"])] += 1
            if tdm.get("Test-Dataset"): datasets[normalize(tdm["Test-Dataset"])] += 1
            if tdm.get("Train-Dataset"): datasets[normalize(tdm["Train-Dataset"])] += 1
            if tdm.get("Metric"): metrics[normalize(tdm["Metric"])] += 1
    # SciREX train+dev (avoid test leakage — only train + dev)
    for path in (SCIREX_TRAIN_JSONL, SCIREX_DEV_JSONL):
        if not path.exists(): continue
        with path.open() as f:
            for line in f:
                doc = json.loads(line)
                for rel in (doc.get("n_ary_relations") or []):
                    if rel.get("Task"): tasks[normalize(rel["Task"])] += 1
                    if rel.get("Material"): datasets[normalize(rel["Material"])] += 1
                    if rel.get("Metric"): metrics[normalize(rel["Metric"])] += 1
    out = {
        "source": "MetaLead + SciREX (train+dev). PwC skipped (API unreachable).",
        "tasks": [k for k, _ in tasks.most_common(500)],
        "datasets": [k for k, _ in datasets.most_common(500)],
        "metrics": [k for k, _ in metrics.most_common(500)],
        "tasks_all": sorted(tasks.keys()),
        "datasets_all": sorted(datasets.keys()),
        "metrics_all": sorted(metrics.keys()),
    }
    EXTERNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EXTERNAL_CACHE.write_text(json.dumps(out, indent=2))
    return out


def _fallback_from_text(text: str) -> dict:
    """Paper 1 style: n-gram extraction from paper prose."""
    tokens = normalize(text).split()
    ngrams = Counter()
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            g = " ".join(tokens[i:i+n])
            ngrams[g] += 1
    return {"top_ngrams": [k for k, v in ngrams.most_common(200)]}


@app.command()
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build external vocab once
    ext = _build_external_vocab()
    console.print(f"external vocab: tasks={len(ext['tasks_all'])}  datasets={len(ext['datasets_all'])}  metrics={len(ext['metrics_all'])}")

    # MetaLead papers (eval + dev = 27 + 4 = 31 papers with tex)
    metalead_json = json.loads(METALEAD_JSON.read_text())
    import re as _re
    ml_papers = [k.removesuffix(".pdf") for k in metalead_json.keys()
                 if _re.match(r"^\d{4}\.\d{4,6}$", k.removesuffix(".pdf"))
                 and (METALEAD_TEX_DIR / k.removesuffix(".pdf")).exists()]
    console.print(f"MetaLead papers to vocab: {len(ml_papers)}")

    # SciREX test papers (58 with tex)
    sx_docs = []
    with Path("data/raw/scirex/scirex_dataset/release_data/test.jsonl").open() as f:
        for line in f:
            sx_docs.append(json.loads(line))
    sx_papers = [d["doc_id"] for d in sx_docs
                 if (SCIREX_TEX_DIR / d["doc_id"]).exists()]
    console.print(f"SciREX papers to vocab: {len(sx_papers)}")

    # Emit vocab per paper
    for paper_key in ml_papers:
        local = _local_from_metalead(paper_key)
        (OUT_DIR / f"metalead_{paper_key}.json").write_text(json.dumps({
            "paper_id": paper_key, "corpus": "metalead",
            "local": local, "external": ext, "fallback_note": "paper text n-grams computed at inference time",
        }, indent=2))

    for doc_id in sx_papers:
        local = _local_from_scirex(doc_id)
        (OUT_DIR / f"scirex_{doc_id}.json").write_text(json.dumps({
            "paper_id": doc_id, "corpus": "scirex",
            "local": local, "external": ext, "fallback_note": "paper text n-grams computed at inference time",
        }, indent=2))

    console.print(f"[green]wrote {len(ml_papers) + len(sx_papers)} vocab files[/green]")


if __name__ == "__main__":
    app()
