import os
"""Score cost-matched SC baseline against gold using h1_reanalysis functions.

Reuses norm / lenient / emitted_triples / per_paper_metrics / paired_boot /
evaluate exactly. Does NOT redefine matching rules.

Produces:
  1. Three-system table (CE, paper 1 baseline, SC) on paired subsets.
  2. Paired-bootstrap 95% CI for three deltas:
       - CE vs paper 1 baseline (reproduces h1_reanalysis_full.json)
       - CE vs SC (new; primary)
       - SC vs paper 1 baseline (new; secondary)
  3. Decision rule outcome for CE-vs-SC.
  4. If |delta_F1(CE-SC)| < 0.05: mechanism analysis.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

# Import the frozen scoring functions
sys.path.insert(0, str(Path(__file__).parent))
from h1_reanalysis import (  # noqa: E402
    norm, lenient, emitted_triples, per_paper_metrics, paired_boot,
    load_gold, load_predictions, evaluate,
)


ROOT = Path(os.environ.get("CANONEXTRACT_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD_PATH = ROOT / "data/metalead/annotations/metalead_annotations.json"
CE_GLOB = str(ROOT / "outputs/canonextract_build/eval_runs_metalead_cycle5/metalead_*.json")
BL_GLOB = str(ROOT / "outputs/canonextract_build/baseline_metalead/metalead_*.json")
SC_GLOB = str(ROOT / "outputs/canonextract_build/metalead_sc_baseline/metalead_*.json")


def main():
    gold = load_gold(str(GOLD_PATH))
    ce = load_predictions(CE_GLOB, key_pref="tuples")
    bl = {}
    # baseline_metalead is not present here — pull from ICMLA repo
    bl_glob_icmla = str(ROOT / "baselines" / "single_call_metalead" / "metalead_*.json")
    bl = load_predictions(bl_glob_icmla, key_pref="contributions")
    sc = load_predictions(SC_GLOB, key_pref="contributions")
    print(f"CE={len(ce)}  BL={len(bl)}  SC={len(sc)}  gold={len(gold)}")

    # ------------- three pairwise comparisons -------------
    ce_vs_bl = evaluate(ce, bl, gold, "CE", "BL")
    ce_vs_sc = evaluate(ce, sc, gold, "CE", "SC")
    sc_vs_bl = evaluate(sc, bl, gold, "SC", "BL")

    # Also compute three-system table on the intersection of all three
    common_all = sorted(set(ce) & set(bl) & set(sc))
    print(f"\n== triple-common set (n={len(common_all)}) ==")
    def _triple_row(preds, name):
        rows = []
        for pid in common_all:
            triples = emitted_triples(preds.get(pid, []))
            g = gold.get(pid, {'tasks':set(),'datasets':set(),'metrics':set(),'triples':set()})
            p, r, f, tp, ne, gc, gn = per_paper_metrics(triples, g)
            rows.append((pid, p, r, f))
        p_m = float(np.mean([x[1] for x in rows]))
        r_m = float(np.mean([x[2] for x in rows]))
        f_m = float(np.mean([x[3] for x in rows]))
        return {"n": len(rows), "prec_mean": p_m, "rec_mean": r_m, "f1_mean": f_m,
                "per_paper": [{"pid": x[0], "prec": x[1], "rec": x[2], "f1": x[3]} for x in rows]}
    triple_ce = _triple_row(ce, "CE")
    triple_bl = _triple_row(bl, "BL")
    triple_sc = _triple_row(sc, "SC")

    # ------------- CE vs SC on the 27 paired set (largest common; no baseline) -------------
    ce_vs_sc_wide = evaluate(ce, sc, gold, "CE", "SC")  # 27 paired (CE and SC both cover all 27)

    # ------------- Decision rule (CE vs SC on the largest common-paper set = ce_vs_sc_wide) -------------
    f1_row = ce_vs_sc_wide.get("f1", {})
    delta_f1 = f1_row.get("delta", float("nan"))
    ci_lo = f1_row.get("ci_lo", float("nan"))
    ci_hi = f1_row.get("ci_hi", float("nan"))
    ci_excludes_zero = f1_row.get("ci_excludes_zero", False)

    h1_matched_budget = (abs(delta_f1) >= 0.05) and (delta_f1 > 0) and ci_excludes_zero
    sc_within_5pp = (abs(delta_f1) < 0.05) or (not ci_excludes_zero)
    sc_beats_ce = (delta_f1 <= -0.05) and ci_excludes_zero

    if h1_matched_budget:
        decision = "H1_MATCHED_BUDGET_CONFIRMED"
    elif sc_beats_ce:
        decision = "SC_BEATS_CE_H1_FAILS"
    else:
        decision = "SC_WITHIN_5PP_TRIGGER_MECHANISM"

    print(f"\n== decision rule ==")
    print(f"delta_F1(CE-SC) = {delta_f1:+.4f}  CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  excludes_zero={ci_excludes_zero}")
    print(f"decision: {decision}")

    # ------------- Mechanism analysis if triggered -------------
    mech = None
    if decision == "SC_WITHIN_5PP_TRIGGER_MECHANISM":
        mech = _mechanism_analysis(ce, sc, gold, common_all if len(common_all)==0 else sorted(set(ce)&set(sc)))

    # ------------- write out -------------
    out = {
        "seed": 42,
        "run": "h1_cost_matched_sc",
        "ce_vs_bl_paired22": ce_vs_bl,
        "ce_vs_sc_paired27": ce_vs_sc_wide,
        "sc_vs_bl_paired22": sc_vs_bl,
        "triple_common_set": {
            "n": triple_ce["n"],
            "paper_ids": common_all,
            "CE":  {"prec_mean": triple_ce["prec_mean"], "rec_mean": triple_ce["rec_mean"], "f1_mean": triple_ce["f1_mean"]},
            "BL":  {"prec_mean": triple_bl["prec_mean"], "rec_mean": triple_bl["rec_mean"], "f1_mean": triple_bl["f1_mean"]},
            "SC":  {"prec_mean": triple_sc["prec_mean"], "rec_mean": triple_sc["rec_mean"], "f1_mean": triple_sc["f1_mean"]},
            "per_paper_CE": triple_ce["per_paper"],
            "per_paper_BL": triple_bl["per_paper"],
            "per_paper_SC": triple_sc["per_paper"],
        },
        "decision_rule": {
            "delta_F1_CE_minus_SC": delta_f1,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "ci_excludes_zero": ci_excludes_zero,
            "outcome": decision,
        },
        "mechanism_analysis": mech,
    }
    out_p = ROOT / "outputs/canonextract_build/h1_cost_matched_sc.json"
    out_p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_p}")


def _mechanism_analysis(ce, sc, gold, papers):
    """For each gold triple CE covered but SC did NOT (CE-exclusive), classify how
    CanonExtract found it via provenance bucket (table vs prose)."""
    # Load cycle-6 buckets for CE tuples
    cycle6_bucket_lookup = {}
    for f in glob.glob(str(ROOT/"outputs/canonextract_build/eval_runs_metalead_cycle6/metalead_*.json")):
        d = json.load(open(f))
        pid = d.get("paper_id") or Path(f).stem.removeprefix("metalead_")
        for i, t in enumerate(d.get("tuples") or []):
            cycle6_bucket_lookup[(pid, i)] = (t.get("provenance") or {}).get("cycle6_bucket", "unknown")

    # For each paper, compute the set of gold triples SC didn't cover
    from h1_reanalysis import triple_ok, lenient as _len
    ce_exclusive_gold: list[dict] = []
    sc_exclusive_gold: list[dict] = []
    for pid in papers:
        g = gold.get(pid, {})
        if not g or not g.get("triples"): continue
        ce_pred = emitted_triples(ce.get(pid, []))
        sc_pred = emitted_triples(sc.get(pid, []))
        for gt in g["triples"]:
            ce_hit = any(_len(pt[0],[gt[0]]) and _len(pt[1],[gt[1]]) and _len(pt[2],[gt[2]]) for pt in ce_pred)
            sc_hit = any(_len(pt[0],[gt[0]]) and _len(pt[1],[gt[1]]) and _len(pt[2],[gt[2]]) for pt in sc_pred)
            if ce_hit and not sc_hit:
                # Identify which CE tuple covered gt (first match)
                cover_idx = None
                for i, pt in enumerate(ce.get(pid, [])):
                    task = (pt.get("task") or {}).get("name") or ""
                    ds_list = pt.get("datasets") or []
                    m_list = pt.get("metrics") or []
                    matched = False
                    for ds in ds_list:
                        dsn = ds.get("name") if isinstance(ds, dict) else str(ds)
                        for m in m_list:
                            mn = m.get("name") if isinstance(m, dict) else str(m)
                            if _len(task,[gt[0]]) and _len(dsn,[gt[1]]) and _len(mn,[gt[2]]):
                                cover_idx = i; matched = True; break
                        if matched: break
                    if matched: break
                bucket = cycle6_bucket_lookup.get((pid, cover_idx), "unknown") if cover_idx is not None else "unknown"
                ce_exclusive_gold.append({"paper_id": pid, "gold_triple": list(gt),
                                            "ce_tuple_idx": cover_idx, "ce_bucket": bucket})
            elif sc_hit and not ce_hit:
                sc_exclusive_gold.append({"paper_id": pid, "gold_triple": list(gt)})

    # Classify CE-exclusive by provenance
    from collections import Counter
    cls = Counter()
    for e in ce_exclusive_gold:
        b = e["ce_bucket"]
        if b in ("strict_table_valid", "partial_table_valid",
                 "verifier_downgrade_partial"):  # cycle 6 bucket names
            cls["table"] += 1
        elif b in ("prose", "prose_claimed_table"):
            cls["prose"] += 1
        else:
            cls["ambiguous"] += 1
    total_ce_excl = sum(cls.values())
    frac_table = (cls.get("table", 0) / total_ce_excl) if total_ce_excl else None
    frac_prose = (cls.get("prose", 0) / total_ce_excl) if total_ce_excl else None

    return {
        "n_ce_exclusive_gold": total_ce_excl,
        "ce_exclusive_by_bucket_class": dict(cls),
        "frac_table_only": frac_table,
        "frac_prose_only": frac_prose,
        "ce_exclusive_gold": ce_exclusive_gold,
        "n_sc_exclusive_gold": len(sc_exclusive_gold),
        "sc_exclusive_gold": sc_exclusive_gold,
    }


if __name__ == "__main__":
    main()
