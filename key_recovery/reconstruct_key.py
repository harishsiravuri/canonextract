#!/usr/bin/env python3
"""Reconstruct the attribution key for the whole-record blind packet by
content-join against the frozen per-system record outputs.

Usage:
    python reconstruct_key.py BLIND_CSV CE_DIR BASELINE_DIR OUT_JSON

No LLM calls. Pure deterministic content comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def norm_text(s) -> str:
    if s is None:
        return ""
    return _WS_RE.sub(" ", str(s).strip().lower())


def norm_value(s):
    """Return (float_value_or_None, raw_string_normalized)."""
    if s is None or s == "":
        return (None, "")
    raw = str(s).strip()
    try:
        return (float(raw), raw.lower())
    except ValueError:
        return (None, raw.lower())


# --------------------------------------------------------------------------
# Record extraction from each system's frozen outputs
# --------------------------------------------------------------------------

def _flatten_record(method_name, task_name, datasets, metrics):
    """Produce one (method, task, dataset, metric_name, metric_value) record
    for every (dataset, metric) combination inside a tuple/contribution.
    When metrics is empty, emit one record per dataset with metric/value
    left as None, so dataset-only contributions (prose records with no
    explicit cell) remain joinable."""
    out = []
    for d in datasets or []:
        d_name = d.get("name") if isinstance(d, dict) else d
        if not metrics:
            out.append({
                "method": method_name,
                "task": task_name,
                "dataset": d_name,
                "metric": None,
                "value": None,
                "unit": None,
            })
            continue
        for m in metrics or []:
            if not isinstance(m, dict):
                continue
            out.append({
                "method": method_name,
                "task": task_name,
                "dataset": d_name,
                "metric": m.get("name"),
                "value": m.get("value"),
                "unit": m.get("unit"),
            })
    return out


def load_ce_records(ce_dir: Path, paper_id: str):
    """Load CanonExtract records for one paper. CE output shape: top-level
    key `tuples`, each tuple has method.name, task.name, datasets[].name,
    metrics[].{name,value,unit}. Datasets and metrics are typically length 1
    in the cycle5/6/merged outputs, but we flatten defensively."""
    path = ce_dir / f"metalead_{paper_id}.json"
    if not path.exists():
        return []
    j = json.loads(path.read_text())
    out = []
    for t in j.get("tuples", []):
        m_name = (t.get("method") or {}).get("name")
        t_name = (t.get("task") or {}).get("name")
        out.extend(_flatten_record(m_name, t_name, t.get("datasets"), t.get("metrics")))
    return out


def load_baseline_records(baseline_dir: Path, paper_id: str):
    """Load baseline records for one paper. Baseline output shape: top-level
    key `contributions`, each contribution has method.name, task.name,
    datasets[].name, metrics[].{name,value,unit}. Datasets and metrics are
    multi-valued per contribution, so flatten across the cross product."""
    path = baseline_dir / f"metalead_{paper_id}.json"
    if not path.exists():
        return []
    j = json.loads(path.read_text())
    out = []
    for c in j.get("contributions", []):
        m_name = (c.get("method") or {}).get("name")
        t_name = (c.get("task") or {}).get("name")
        out.extend(_flatten_record(m_name, t_name, c.get("datasets"), c.get("metrics")))
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _key_strict(rec):
    v_float, v_raw = norm_value(rec.get("value"))
    return (
        norm_text(rec.get("method")),
        norm_text(rec.get("task")),
        norm_text(rec.get("dataset")),
        norm_text(rec.get("metric")),
        v_float if v_float is not None else v_raw,
    )


def match_strict(blind_rec, system_records):
    """v1 strict exact match on (method, task, dataset, metric, value)
    after whitespace+case normalization. Value compared as float when parseable."""
    bk = _key_strict(blind_rec)
    for r in system_records:
        if _key_strict(r) == bk:
            return True
    return False


def _value_close(a, b, tol=0.01):
    """Within 1% relative tolerance, or absolute 0.001 for small values."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol or abs(a - b) <= 1e-3


def match_lenient(blind_rec, system_records):
    """v2 lenient: method exact, task exact, dataset substring either way,
    metric substring either way, value within 1% tolerance."""
    bm = norm_text(blind_rec.get("method"))
    bt = norm_text(blind_rec.get("task"))
    bd = norm_text(blind_rec.get("dataset"))
    bmet = norm_text(blind_rec.get("metric"))
    bv_f, bv_raw = norm_value(blind_rec.get("value"))

    for r in system_records:
        rm = norm_text(r.get("method"))
        rt = norm_text(r.get("task"))
        rd = norm_text(r.get("dataset"))
        rmet = norm_text(r.get("metric"))
        rv_f, rv_raw = norm_value(r.get("value"))
        if rm != bm:
            continue
        if rt != bt:
            continue
        if not (bd and rd and (bd in rd or rd in bd)):
            continue
        if not (bmet and rmet and (bmet in rmet or rmet in bmet)):
            continue
        if bv_f is not None and rv_f is not None:
            if not _value_close(bv_f, rv_f):
                continue
        else:
            if bv_raw != rv_raw:
                continue
        return True
    return False


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def classify(ce_hit, base_hit, used_lenient):
    if ce_hit and base_hit:
        klass = "shared_record_both_systems"
    elif ce_hit and not base_hit:
        klass = "clean_single_system_match"
    elif base_hit and not ce_hit:
        klass = "clean_single_system_match"
    else:
        klass = "unmatched_neither_system"
    if used_lenient and (ce_hit or base_hit):
        klass = klass + "__via_lenient"
    return klass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blind_csv")
    ap.add_argument("ce_dir")
    ap.add_argument("baseline_dir")
    ap.add_argument("out_json")
    args = ap.parse_args()

    ce_dir = Path(args.ce_dir)
    baseline_dir = Path(args.baseline_dir)

    # Cache per-paper record lists so we read each JSON once.
    ce_cache: dict[str, list] = {}
    base_cache: dict[str, list] = {}

    def ce_for(pid):
        if pid not in ce_cache:
            ce_cache[pid] = load_ce_records(ce_dir, pid)
        return ce_cache[pid]

    def base_for(pid):
        if pid not in base_cache:
            base_cache[pid] = load_baseline_records(baseline_dir, pid)
        return base_cache[pid]

    out = []
    with open(args.blind_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["paper_id"].strip()
            blind = {
                "method": row.get("method"),
                "task": row.get("task"),
                "dataset": row.get("dataset"),
                "metric": row.get("metric"),
                "value": row.get("value"),
            }
            ce_recs = ce_for(pid)
            base_recs = base_for(pid)

            # v1 strict
            ce_hit = match_strict(blind, ce_recs)
            base_hit = match_strict(blind, base_recs)
            used_lenient = False

            # v2 lenient fallback (only if strict found nothing in that system)
            ce_hit_v2 = False
            base_hit_v2 = False
            if not ce_hit:
                ce_hit_v2 = match_lenient(blind, ce_recs)
            if not base_hit:
                base_hit_v2 = match_lenient(blind, base_recs)
            if ce_hit_v2 or base_hit_v2:
                used_lenient = True
            ce_any = ce_hit or ce_hit_v2
            base_any = base_hit or base_hit_v2

            matched = []
            if ce_any:
                matched.append("CE")
            if base_any:
                matched.append("baseline")

            out.append({
                "row_id": row["row_id"],
                "paper_id": pid,
                "matched_systems": matched,
                "exact_match_v1": {
                    "CE": ce_hit,
                    "baseline": base_hit,
                },
                "lenient_match_v2": {
                    "CE": ce_hit_v2,
                    "baseline": base_hit_v2,
                },
                "ambiguity_class": classify(ce_any, base_any, used_lenient),
            })

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Short stdout summary
    from collections import Counter
    cnt = Counter(r["ambiguity_class"] for r in out)
    sys_cnt = Counter(tuple(r["matched_systems"]) for r in out)
    print("ambiguity_class counts:")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    print("matched_systems counts:")
    for k, v in sorted(sys_cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k or '()'}: {v}")


if __name__ == "__main__":
    main()
