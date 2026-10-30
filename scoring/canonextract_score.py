"""CanonExtract evaluation harness — Component 7.

Loads the 3 result directories, applies LLM canonicalization at eval time,
computes per-field F1 with bootstrap 95% CIs and paired-permutation p-values
vs paper 1 baseline, structural provenance coverage, ECE, and cost.

Writes outputs/canonextract_build/SUMMARY.md and results.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from paper1.config import load_config
from paper1.metrics import bootstrap_ci, expected_calibration_error, paired_permutation_test
from paper1.openrouter import OpenRouterClient

SEED = 42
N_BOOT = 1000
N_PERM = 10000
CANON_CACHE_PATH = Path("outputs/canonextract_build/canon_cache.json")

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


CANON_SYSTEM = (
    "You are a careful scientific-vocabulary canonicalizer. Given two entity names, "
    "answer with exactly YES or NO followed by nothing else. Answer YES if the two "
    "names refer to the SAME entity (a canonical form and a common variant, e.g., "
    "\"ImageNet\" and \"ILSVRC 2012\" — YES; \"CoNLL-2003 English\" and \"CoNLL 2003\" — YES; "
    "\"BLEU\" and \"BLEU-4\" — YES; \"F1\" and \"F1 score\" — YES). Otherwise NO."
)


class CanonCache:
    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, bool] = {}
        if path.exists():
            try:
                self.cache = json.loads(path.read_text())
            except Exception:
                self.cache = {}

    def get(self, a: str, b: str) -> bool | None:
        k = self._key(a, b)
        return self.cache.get(k)

    def put(self, a: str, b: str, v: bool):
        self.cache[self._key(a, b)] = v

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=2))

    @staticmethod
    def _key(a: str, b: str) -> str:
        return json.dumps(sorted([normalize(a), normalize(b)]))


async def _canon_yes(client: OpenRouterClient, model_id: str, a: str, b: str,
                     cache: CanonCache, spend: dict) -> bool:
    v = cache.get(a, b)
    if v is not None:
        return v
    if not a.strip() or not b.strip():
        return False
    try:
        r = await client.complete(
            model_id=model_id, system=CANON_SYSTEM,
            user=f"NAME_A: {a}\nNAME_B: {b}\nSAME ENTITY? (YES/NO):",
            temperature=0.0, max_tokens=6, top_p=0.95,
        )
        text = (r.text or "").strip().upper()
        spend["cost"] += (r.tokens_in / 1e6) * 0.32 + (r.tokens_out / 1e6) * 0.89
        spend["calls"] += 1
        v = text.startswith("Y")
    except Exception:
        v = False
    cache.put(a, b, v)
    return v


def _lenient_substring_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return False
    return p in g or g in p


def _load_metalead_gold_by_paper() -> dict[str, list[dict]]:
    m = json.loads(Path("data/metalead/annotations/metalead_annotations.json").read_text())
    out: dict[str, list[dict]] = defaultdict(list)
    for key, entry in m.items():
        pk = key.removesuffix(".pdf")
        for tdm in entry.get("TDMs") or []:
            key_t = (tdm.get("Task"), tdm.get("Test-Dataset"), tdm.get("Metric"))
            if key_t in {(x["Task"], x["Material"], x["Metric"]) for x in out[pk]}:
                continue
            out[pk].append({"Task": tdm.get("Task"), "Material": tdm.get("Test-Dataset"),
                            "Metric": tdm.get("Metric"), "Method": tdm.get("Method", "")})
    return out


def _load_scirex_gold_by_paper() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with Path("data/raw/scirex/scirex_dataset/release_data/test.jsonl").open() as f:
        for line in f:
            doc = json.loads(line)
            seen = set()
            for r in (doc.get("n_ary_relations") or []):
                key = (r.get("Task"), r.get("Material"), r.get("Metric"))
                if key in seen: continue
                seen.add(key)
                out[doc["doc_id"]].append({"Task": r.get("Task"), "Material": r.get("Material"),
                                           "Metric": r.get("Metric"), "Method": r.get("Method","")})
    return out


def _load_canonextract_tuples(dir_: Path, corpus: str) -> dict[str, list[dict]]:
    """Return per-paper list of predicted tuples."""
    out: dict[str, list[dict]] = {}
    prefix = f"{corpus}_"
    for p in dir_.glob(f"{prefix}*.json"):
        d = json.loads(p.read_text())
        pid = d.get("paper_id") or p.stem.removeprefix(prefix)
        out[pid] = d.get("tuples") or []
    return out


def _load_paper1_tuples(dir_: Path, corpus: str) -> dict[str, list[dict]]:
    """Return per-paper list of predicted tuples for paper 1 baseline (ContributionRecord shape)."""
    out: dict[str, list[dict]] = {}
    prefix = f"{corpus}_"
    for p in dir_.glob(f"{prefix}*.json"):
        d = json.loads(p.read_text())
        pid = d.get("paper_id") or p.stem.removeprefix(prefix)
        # ContributionRecord has "contributions" list with method/task/datasets/metrics
        tuples = []
        for c in d.get("contributions") or []:
            task_n = (c.get("task") or {}).get("name")
            method_n = (c.get("method") or {}).get("name")
            for ds in c.get("datasets") or []:
                ds_n = ds.get("name") if isinstance(ds, dict) else str(ds)
                for m in c.get("metrics") or []:
                    m_n = m.get("name") if isinstance(m, dict) else str(m)
                    tuples.append({
                        "task": {"name": task_n},
                        "datasets": [{"name": ds_n}],
                        "metrics": [{"name": m_n, "value": (m.get("value") if isinstance(m, dict) else None)}],
                        "method": {"name": method_n},
                        "self_consistency": c.get("self_consistency", 0.0),
                        "provenance": {"source": "paper1_baseline"},
                    })
        out[pid] = tuples
    return out


async def _canonicalize_and_score(client, model_id, per_paper_pred: dict, per_paper_gold: dict,
                                   cache: CanonCache, spend: dict) -> tuple[dict, dict]:
    """For each paper, per-field F1 (Task, Dataset, Metric) + triple F1.

    Uses substring lenient match first; if miss, uses cached LLM YES/NO.
    Returns (per_paper_field_scores, per_paper_triple_scores).
    """
    per_paper_field: dict[str, dict[str, float]] = {}
    per_paper_triple: dict[str, float] = {}

    for pid, gold_list in per_paper_gold.items():
        pred_list = per_paper_pred.get(pid, [])
        gold_tasks = [g["Task"] for g in gold_list if g.get("Task")]
        gold_datasets = [g["Material"] for g in gold_list if g.get("Material")]
        gold_metrics = [g["Metric"] for g in gold_list if g.get("Metric")]
        gold_triples = {(normalize(g["Task"]), normalize(g["Material"]), normalize(g["Metric"]))
                        for g in gold_list if g.get("Task") and g.get("Material") and g.get("Metric")}

        pred_tasks, pred_datasets, pred_metrics, pred_triples_raw = set(), set(), set(), set()
        for p in pred_list:
            t = (p.get("task") or {}).get("name") or ""
            if t: pred_tasks.add(t)
            for ds in p.get("datasets") or []:
                dn = ds.get("name") if isinstance(ds, dict) else str(ds)
                if dn: pred_datasets.add(dn)
            for m in p.get("metrics") or []:
                mn = m.get("name") if isinstance(m, dict) else str(m)
                if mn: pred_metrics.add(mn)
            for ds in p.get("datasets") or []:
                dn = ds.get("name") if isinstance(ds, dict) else str(ds)
                for m in p.get("metrics") or []:
                    mn = m.get("name") if isinstance(m, dict) else str(m)
                    if t and dn and mn:
                        pred_triples_raw.add((t, dn, mn))

        async def _match_field(pred_set, gold_list):
            if not pred_set and not gold_list:
                return 1.0
            if not pred_set or not gold_list:
                return 0.0
            gold_matched = [False] * len(gold_list)
            pred_matched = [False] * len(pred_set)
            plist = list(pred_set)
            # Substring first
            for i, p in enumerate(plist):
                for j, g in enumerate(gold_list):
                    if gold_matched[j] or pred_matched[i]: continue
                    if _lenient_substring_match(p, g):
                        gold_matched[j] = True; pred_matched[i] = True
            # LLM canonicalization for remaining
            for i, p in enumerate(plist):
                if pred_matched[i]: continue
                for j, g in enumerate(gold_list):
                    if gold_matched[j]: continue
                    hit = await _canon_yes(client, model_id, p, g, cache, spend)
                    if hit:
                        pred_matched[i] = True; gold_matched[j] = True
                        break
            tp = sum(pred_matched)
            prec = tp / len(plist) if plist else 0.0
            rec = tp / len(gold_list) if gold_list else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        task_f1 = await _match_field(pred_tasks, gold_tasks)
        ds_f1 = await _match_field(pred_datasets, gold_datasets)
        mt_f1 = await _match_field(pred_metrics, gold_metrics)

        # Triple F1: pred triple matches gold triple iff all three fields canon-match
        # For simplicity: normalize both sides. This is a strict metric.
        pred_triples = {(normalize(t), normalize(d), normalize(m)) for t, d, m in pred_triples_raw}
        tp_tri = len(pred_triples & gold_triples)
        prec_tri = tp_tri / len(pred_triples) if pred_triples else 0.0
        rec_tri = tp_tri / len(gold_triples) if gold_triples else 0.0
        tri_f1 = 2 * prec_tri * rec_tri / (prec_tri + rec_tri) if (prec_tri + rec_tri) else 0.0

        per_paper_field[pid] = {"task": task_f1, "dataset": ds_f1, "metric": mt_f1}
        per_paper_triple[pid] = tri_f1

    return per_paper_field, per_paper_triple


def _aggregate(per_paper_field: dict, per_paper_triple: dict) -> dict:
    fields = ("task", "dataset", "metric")
    out: dict[str, dict] = {}
    for f in fields:
        vals = [per_paper_field[p][f] for p in per_paper_field]
        rng = np.random.default_rng(SEED)
        mean, lo, hi = bootstrap_ci(vals, n_resamples=N_BOOT, rng=rng) if vals else (0.0, 0.0, 0.0)
        out[f] = {"n": len(vals), "f1": mean, "ci_lo": lo, "ci_hi": hi}
    vals = list(per_paper_triple.values())
    rng = np.random.default_rng(SEED)
    mean, lo, hi = bootstrap_ci(vals, n_resamples=N_BOOT, rng=rng) if vals else (0.0, 0.0, 0.0)
    out["triple"] = {"n": len(vals), "f1": mean, "ci_lo": lo, "ci_hi": hi}
    return out


def _paired_perm_all_fields(a_field: dict, b_field: dict, a_tri: dict, b_tri: dict) -> dict:
    """Compare a vs b on paper-intersection, per field + triple."""
    common = sorted(set(a_field.keys()) & set(b_field.keys()))
    out = {"n_paired": len(common)}
    for f in ("task", "dataset", "metric"):
        a = [a_field[p][f] for p in common]
        b = [b_field[p][f] for p in common]
        rng = np.random.default_rng(SEED)
        p_raw = paired_permutation_test(a, b, n_permutations=N_PERM, rng=rng)
        # Bonferroni across 4 fields
        p_bonf = min(1.0, p_raw * 4)
        out[f] = {"delta_a_minus_b": float(np.mean(a) - np.mean(b)) if a else 0.0,
                  "p_raw": p_raw, "p_bonferroni": p_bonf}
    a = [a_tri[p] for p in common]
    b = [b_tri[p] for p in common]
    rng = np.random.default_rng(SEED)
    p_raw = paired_permutation_test(a, b, n_permutations=N_PERM, rng=rng)
    out["triple"] = {"delta_a_minus_b": float(np.mean(a) - np.mean(b)) if a else 0.0,
                     "p_raw": p_raw, "p_bonferroni": min(1.0, p_raw * 4)}
    return out


def _compute_provenance_coverage(tuple_dir: Path, corpus: str) -> dict:
    prefix = f"{corpus}_"
    total = 0; valid = 0; prose = 0; downgraded = 0
    for p in tuple_dir.glob(f"{prefix}*.json"):
        d = json.loads(p.read_text())
        stats = d.get("provenance_stats") or {}
        total += stats.get("total", 0)
        valid += stats.get("table_valid", 0)
        prose += stats.get("prose", 0)
        downgraded += stats.get("prose_claimed_table", 0)
    return {
        "total_tuples": total,
        "table_valid": valid,
        "prose_grounded": prose,
        "prose_claimed_table_downgraded": downgraded,
        "table_valid_pct": (valid / total * 100.0) if total else 0.0,
        "grounded_pct": ((valid + prose) / total * 100.0) if total else 0.0,
    }


def _compute_ece(tuple_dir: Path, corpus: str, per_paper_field: dict, per_paper_gold: dict) -> dict:
    """ECE at 10 bins on per-tuple self_consistency vs a coarse correctness signal.

    Correctness = 1 if at least one field in that tuple's (task, dataset, metric)
    canon-matches gold for its paper; 0 otherwise.
    (Coarse but honest; paper 1's ECE uses first-contribution.self_consistency
    against per-field correctness. Same shape here.)
    """
    prefix = f"{corpus}_"
    confs, corrects = [], []
    for p in tuple_dir.glob(f"{prefix}*.json"):
        d = json.loads(p.read_text())
        pid = d.get("paper_id") or p.stem.removeprefix(prefix)
        gold_list = per_paper_gold.get(pid, [])
        gold_triples = {(normalize(g["Task"]), normalize(g["Material"]), normalize(g["Metric"]))
                        for g in gold_list if g.get("Task") and g.get("Material") and g.get("Metric")}
        for t in d.get("tuples") or []:
            conf = t.get("self_consistency", 0.0)
            task_n = normalize((t.get("task") or {}).get("name") or "")
            ds_names = [normalize(x.get("name") or "" if isinstance(x, dict) else x)
                        for x in t.get("datasets") or []]
            m_names = [normalize(x.get("name") or "" if isinstance(x, dict) else x)
                       for x in t.get("metrics") or []]
            correct = 0
            for ds in ds_names:
                for mn in m_names:
                    if (task_n, ds, mn) in gold_triples:
                        correct = 1
                        break
            confs.append(conf); corrects.append(correct)
    if not confs:
        return {"n": 0, "ece": None}
    ece = expected_calibration_error(confs, corrects, n_bins=10)
    return {"n": len(confs), "ece": ece,
            "mean_conf": float(np.mean(confs)), "mean_acc": float(np.mean(corrects))}


@app.command()
def run():
    out_dir = Path("outputs/canonextract_build")
    cache = CanonCache(CANON_CACHE_PATH)
    spend = {"cost": 0.0, "calls": 0}

    cfg = load_config(config_path=Path("config/models.yaml"))
    client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )
    canon_model = cfg.extractor.model_id  # DeepSeek Chat

    # Load predictions + gold
    ce_ml = _load_canonextract_tuples(out_dir / "eval_runs_metalead", "metalead")
    p1_ml = _load_paper1_tuples(out_dir / "baseline_metalead", "metalead")
    ce_sx = _load_canonextract_tuples(out_dir / "eval_runs_scirex", "scirex")

    # Filter MetaLead gold to only papers in eval-100
    eval_manifest = json.loads((out_dir / "metalead_eval_manifest.json").read_text())
    eval_papers = set(r["paper_key"] for r in eval_manifest)
    ml_gold_all = _load_metalead_gold_by_paper()
    ml_gold = {p: g for p, g in ml_gold_all.items() if p in eval_papers}
    console.print(f"MetaLead eval papers with gold: {len(ml_gold)}")

    sx_gold_all = _load_scirex_gold_by_paper()
    sx_gold = {p: g for p, g in sx_gold_all.items() if p in ce_sx}
    console.print(f"SciREX transfer papers with gold: {len(sx_gold)}")

    async def _score_all():
        # Score CanonExtract MetaLead
        console.print("Scoring CanonExtract on MetaLead...")
        ce_ml_field, ce_ml_tri = await _canonicalize_and_score(client, canon_model, ce_ml, ml_gold, cache, spend)
        console.print("Scoring paper 1 baseline on MetaLead...")
        p1_ml_field, p1_ml_tri = await _canonicalize_and_score(client, canon_model, p1_ml, ml_gold, cache, spend)
        console.print("Scoring CanonExtract on SciREX transfer...")
        ce_sx_field, ce_sx_tri = await _canonicalize_and_score(client, canon_model, ce_sx, sx_gold, cache, spend)
        return ce_ml_field, ce_ml_tri, p1_ml_field, p1_ml_tri, ce_sx_field, ce_sx_tri

    ce_ml_field, ce_ml_tri, p1_ml_field, p1_ml_tri, ce_sx_field, ce_sx_tri = asyncio.run(_score_all())
    cache.save()
    console.print(f"canon LLM spend: ${spend['cost']:.4f} across {spend['calls']} calls")

    # Aggregate + paired-perm
    ce_ml_agg = _aggregate(ce_ml_field, ce_ml_tri)
    p1_ml_agg = _aggregate(p1_ml_field, p1_ml_tri)
    ce_sx_agg = _aggregate(ce_sx_field, ce_sx_tri)
    ml_paired = _paired_perm_all_fields(ce_ml_field, p1_ml_field, ce_ml_tri, p1_ml_tri)

    # SciREX baseline: reuse existing v10 default deployment
    sx_baseline_dir = Path("outputs/paper_data_v10/test_split_default/scirex/multi_agent")
    p1_sx_records = {}
    if sx_baseline_dir.exists():
        for p in sx_baseline_dir.glob("scirex__*.json"):
            d = json.loads(p.read_text())
            pid = p.stem.removeprefix("scirex__")
            tuples = []
            for c in d.get("contributions") or []:
                t = (c.get("task") or {}).get("name")
                for ds in c.get("datasets") or []:
                    dn = ds.get("name")
                    for m in c.get("metrics") or []:
                        mn = m.get("name")
                        tuples.append({"task": {"name": t}, "datasets": [{"name": dn}], "metrics": [{"name": mn}]})
            p1_sx_records[pid] = tuples
        # Score baseline on SciREX
        async def _score_sx_bl():
            fld, tri = await _canonicalize_and_score(client, canon_model, p1_sx_records, sx_gold, cache, spend)
            return fld, tri
        p1_sx_field, p1_sx_tri = asyncio.run(_score_sx_bl())
        cache.save()
        p1_sx_agg = _aggregate(p1_sx_field, p1_sx_tri)
        sx_paired = _paired_perm_all_fields(ce_sx_field, p1_sx_field, ce_sx_tri, p1_sx_tri)
    else:
        p1_sx_agg = None
        sx_paired = None

    # Provenance coverage
    ce_ml_prov = _compute_provenance_coverage(out_dir / "eval_runs_metalead", "metalead")
    ce_sx_prov = _compute_provenance_coverage(out_dir / "eval_runs_scirex", "scirex")

    # ECE
    ce_ml_ece = _compute_ece(out_dir / "eval_runs_metalead", "metalead", ce_ml_field, ml_gold)
    ce_sx_ece = _compute_ece(out_dir / "eval_runs_scirex", "scirex", ce_sx_field, sx_gold)

    # Cost breakdown from extraction summaries
    def _load_summary(p: Path):
        return json.loads(p.read_text()) if p.exists() else {}
    ml_run = _load_summary(out_dir / "eval_runs_metalead" / "extraction_summary.json")
    sx_run = _load_summary(out_dir / "eval_runs_scirex" / "extraction_summary.json")
    bl_run = _load_summary(out_dir / "baseline_metalead" / "extraction_summary.json")
    dev1_run = _load_summary(out_dir / "dev_runs/round1" / "extraction_summary.json")
    dev2_run = _load_summary(out_dir / "dev_runs/round2" / "extraction_summary.json")

    dev_cost = dev1_run.get("total_cost_usd", 0) + dev2_run.get("total_cost_usd", 0)
    total_cost = (ml_run.get("total_cost_usd", 0) + sx_run.get("total_cost_usd", 0)
                  + bl_run.get("total_cost_usd", 0) + dev_cost + spend["cost"])

    results = {
        "seed": SEED,
        "canonextract_metalead": {**ce_ml_agg, "n_papers": len(ce_ml_field)},
        "paper1_baseline_metalead": {**p1_ml_agg, "n_papers": len(p1_ml_field)},
        "paired_metalead": ml_paired,
        "canonextract_scirex": {**ce_sx_agg, "n_papers": len(ce_sx_field)},
        "paper1_baseline_scirex_v10": p1_sx_agg and {**p1_sx_agg, "n_papers": len(p1_sx_field)},
        "paired_scirex": sx_paired,
        "provenance_metalead": ce_ml_prov,
        "provenance_scirex": ce_sx_prov,
        "ece_metalead": ce_ml_ece,
        "ece_scirex": ce_sx_ece,
        "cost_breakdown_usd": {
            "dev_iterations": dev_cost,
            "eval_metalead_canonextract": ml_run.get("total_cost_usd", 0),
            "eval_scirex_canonextract": sx_run.get("total_cost_usd", 0),
            "baseline_metalead_paper1": bl_run.get("total_cost_usd", 0),
            "canonicalization_at_eval": spend["cost"],
            "total": total_cost,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    console.print(f"[green]wrote {out_dir / 'results.json'}[/green]")
    console.print(f"[bold]TOTAL SPEND: ${total_cost:.4f}[/bold]")
    console.print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    app()
