#!/usr/bin/env python3
"""Chapter 1 yield endpoint — pre-registered estimator.

Deterministic. Seed 42. No network. No LLM calls.

Inputs (hard-coded, all absolute):
  - Blind labels CSV with per-field YES/NO verdicts.
  - Reconstructed attribution key (content-join reconstruction of the
    lost whole_record_packet_key.json).
  - Per-paper CanonExtract tuples under eval_runs_metalead_cycle5/.
  - Per-paper baseline contributions under baseline_metalead/.

Output:
  - yield_results.json (next to this script).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHD_ROOT = os.environ.get(
    "CANONEXTRACT_ROOT", str(Path(__file__).resolve().parents[1])
)
CSV_PATH = f"{PHD_ROOT}/outputs/whole_record_packet_blind.csv"
KEY_PATH = f"{PHD_ROOT}/paper_btsd/key_recovery/reconstructed_key.json"
CE_DIR = (
    f"{PHD_ROOT}/contribution_extraction_paper1_icmla/canonextract_build/"
    "outputs/eval_runs_metalead_cycle5"
)
BASE_DIR = (
    f"{PHD_ROOT}/contribution_extraction_paper1_icmla/canonextract_build/"
    "outputs/baseline_metalead"
)
OUT_PATH = f"{PHD_ROOT}/paper_btsd/key_recovery/yield_results.json"

SEED = 42
N_BOOT = 1000

FIELDS = [
    "field_method_correct",
    "field_task_correct",
    "field_dataset_correct",
    "field_metric_correct",
    "field_value_correct",
    "field_conditions_correct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def is_correct(row: dict) -> bool:
    """All six per-field verdicts are YES or N/A."""
    for f in FIELDS:
        v = (row.get(f) or "").strip().upper()
        if v not in {"YES", "N/A"}:
            return False
    return True


def ce_emitted(paper_id: str) -> int:
    """CanonExtract emitted records per paper = len(tuples)."""
    with open(f"{CE_DIR}/metalead_{paper_id}.json") as fh:
        d = json.load(fh)
    return len(d.get("tuples", []) or [])


def baseline_emitted(paper_id: str) -> int:
    """Baseline emitted records per paper = flattened contributions.

    For each contribution, cross datasets x metrics; when a contribution
    has datasets but no metrics, emit one record per dataset. This
    mirrors the join flattening in reconstruct_key.py.
    """
    with open(f"{BASE_DIR}/metalead_{paper_id}.json") as fh:
        d = json.load(fh)
    n = 0
    for c in d.get("contributions", []) or []:
        datasets = c.get("datasets") or []
        metrics = c.get("metrics") or []
        if datasets and metrics:
            n += len(datasets) * len(metrics)
        elif datasets:
            n += len(datasets)
        else:
            # No datasets at all: one record for the contribution.
            n += 1
    return n


def paper_bootstrap_mean(values: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paper-clustered bootstrap: resample papers with replacement,
    compute mean of the per-paper value, return (mean_point, lo, hi)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = sum(values) / n
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return (point, lo, hi)


def paper_bootstrap_ratio(
    numers: list[float], denoms: list[float], n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Paper-clustered bootstrap for mean(numer)/mean(denom)."""
    rng = random.Random(seed)
    n = len(numers)
    assert n == len(denoms)
    mn = sum(numers) / n
    md = sum(denoms) / n
    point = mn / md if md else float("nan")
    ratios: list[float] = []
    for _ in range(n_boot):
        sn = 0.0
        sd = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sn += numers[idx]
            sd += denoms[idx]
        if sd:
            ratios.append(sn / sd)
    if not ratios:
        return (point, float("nan"), float("nan"))
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios)) - 1]
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_key(path: str) -> dict[str, dict]:
    with open(path) as fh:
        return {r["row_id"]: r for r in json.load(fh)}


def stratum_of(matched: list[str]) -> str:
    has_ce = "CE" in matched
    has_bl = "baseline" in matched
    if has_ce and has_bl:
        return "shared"
    if has_ce:
        return "ce_only"
    if has_bl:
        return "base_only"
    return "neither"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = load_csv(CSV_PATH)
    key = load_key(KEY_PATH)

    # Stratum tallies: (correct, total) per stratum.
    strata = {s: [0, 0] for s in ("ce_only", "base_only", "shared", "neither")}
    for r in rows:
        row_id = r["row_id"]
        k = key.get(row_id)
        if k is None:
            raise RuntimeError(f"no key entry for {row_id}")
        s = stratum_of(k["matched_systems"])
        strata[s][1] += 1
        if is_correct(r):
            strata[s][0] += 1

    # Per-system pooled accuracy under two policies.
    ce_only_k, ce_only_n = strata["ce_only"]
    bl_only_k, bl_only_n = strata["base_only"]
    sh_k, sh_n = strata["shared"]

    # Count-both: shared rows go into BOTH systems.
    ce_cb_k = ce_only_k + sh_k
    ce_cb_n = ce_only_n + sh_n
    bl_cb_k = bl_only_k + sh_k
    bl_cb_n = bl_only_n + sh_n

    # Exclude-shared: shared rows removed from both.
    ce_ex_k, ce_ex_n = ce_only_k, ce_only_n
    bl_ex_k, bl_ex_n = bl_only_k, bl_only_n

    def acc(k_, n_):
        return (k_ / n_) if n_ else float("nan")

    pooled = {
        "count_both": {
            "canonextract": {
                "correct": ce_cb_k, "total": ce_cb_n,
                "accuracy": acc(ce_cb_k, ce_cb_n),
                "wilson95": wilson_ci(ce_cb_k, ce_cb_n),
            },
            "baseline": {
                "correct": bl_cb_k, "total": bl_cb_n,
                "accuracy": acc(bl_cb_k, bl_cb_n),
                "wilson95": wilson_ci(bl_cb_k, bl_cb_n),
            },
        },
        "exclude_shared": {
            "canonextract": {
                "correct": ce_ex_k, "total": ce_ex_n,
                "accuracy": acc(ce_ex_k, ce_ex_n),
                "wilson95": wilson_ci(ce_ex_k, ce_ex_n),
            },
            "baseline": {
                "correct": bl_ex_k, "total": bl_ex_n,
                "accuracy": acc(bl_ex_k, bl_ex_n),
                "wilson95": wilson_ci(bl_ex_k, bl_ex_n),
            },
        },
    }

    per_stratum = {
        "ce_only": {
            "correct": ce_only_k, "total": ce_only_n,
            "accuracy": acc(ce_only_k, ce_only_n),
            "wilson95": wilson_ci(ce_only_k, ce_only_n),
        },
        "base_only": {
            "correct": bl_only_k, "total": bl_only_n,
            "accuracy": acc(bl_only_k, bl_only_n),
            "wilson95": wilson_ci(bl_only_k, bl_only_n),
        },
        "shared": {
            "correct": sh_k, "total": sh_n,
            "accuracy": acc(sh_k, sh_n),
            "wilson95": wilson_ci(sh_k, sh_n),
        },
        "neither": {
            "correct": strata["neither"][0], "total": strata["neither"][1],
        },
    }

    # Paired paper list = intersection of the two system directories.
    ce_papers = {
        f[len("metalead_"):-len(".json")]
        for f in os.listdir(CE_DIR)
        if f.startswith("metalead_") and f.endswith(".json")
    }
    bl_papers = {
        f[len("metalead_"):-len(".json")]
        for f in os.listdir(BASE_DIR)
        if f.startswith("metalead_") and f.endswith(".json")
    }
    paired = sorted(ce_papers & bl_papers)

    emissions = {
        p: {"ce": ce_emitted(p), "baseline": baseline_emitted(p)}
        for p in paired
    }

    # Yields per paper under both policies.
    acc_cb_ce = pooled["count_both"]["canonextract"]["accuracy"]
    acc_cb_bl = pooled["count_both"]["baseline"]["accuracy"]
    acc_ex_ce = pooled["exclude_shared"]["canonextract"]["accuracy"]
    acc_ex_bl = pooled["exclude_shared"]["baseline"]["accuracy"]

    yields_cb_ce = [emissions[p]["ce"] * acc_cb_ce for p in paired]
    yields_cb_bl = [emissions[p]["baseline"] * acc_cb_bl for p in paired]
    yields_ex_ce = [emissions[p]["ce"] * acc_ex_ce for p in paired]
    yields_ex_bl = [emissions[p]["baseline"] * acc_ex_bl for p in paired]

    delta_cb = [a - b for a, b in zip(yields_cb_ce, yields_cb_bl)]
    delta_ex = [a - b for a, b in zip(yields_ex_ce, yields_ex_bl)]

    # Mean records/paper per system (two policies — policy only changes
    # accuracy, not emission counts, but record the figures for
    # completeness; "mean records per paper" is a pure-emission figure).
    records_ce = [emissions[p]["ce"] for p in paired]
    records_bl = [emissions[p]["baseline"] for p in paired]

    mean_records_ce = paper_bootstrap_mean(records_ce, N_BOOT, SEED)
    mean_records_bl = paper_bootstrap_mean(records_bl, N_BOOT, SEED)

    mean_yield_cb_ce = paper_bootstrap_mean(yields_cb_ce, N_BOOT, SEED)
    mean_yield_cb_bl = paper_bootstrap_mean(yields_cb_bl, N_BOOT, SEED)
    mean_yield_ex_ce = paper_bootstrap_mean(yields_ex_ce, N_BOOT, SEED)
    mean_yield_ex_bl = paper_bootstrap_mean(yields_ex_bl, N_BOOT, SEED)

    mean_delta_cb = paper_bootstrap_mean(delta_cb, N_BOOT, SEED)
    mean_delta_ex = paper_bootstrap_mean(delta_ex, N_BOOT, SEED)

    yield_ratio_cb = paper_bootstrap_ratio(yields_cb_ce, yields_cb_bl, N_BOOT, SEED)

    out = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "paired_papers": paired,
        "n_paired_papers": len(paired),
        "attribution_counts": {
            "ce_only": ce_only_n,
            "base_only": bl_only_n,
            "shared": sh_n,
            "neither": strata["neither"][1],
            "total": sum(v[1] for v in strata.values()),
        },
        "per_stratum_accuracies": per_stratum,
        "pooled_accuracies": pooled,
        "per_paper_emissions": emissions,
        "per_paper_yields": {
            "count_both": {
                p: {"ce": yc, "baseline": yb, "delta": yc - yb}
                for p, yc, yb in zip(paired, yields_cb_ce, yields_cb_bl)
            },
            "exclude_shared": {
                p: {"ce": yc, "baseline": yb, "delta": yc - yb}
                for p, yc, yb in zip(paired, yields_ex_ce, yields_ex_bl)
            },
        },
        "mean_records_per_paper": {
            "canonextract": {
                "mean": mean_records_ce[0],
                "ci95": [mean_records_ce[1], mean_records_ce[2]],
            },
            "baseline": {
                "mean": mean_records_bl[0],
                "ci95": [mean_records_bl[1], mean_records_bl[2]],
            },
        },
        "mean_yield_per_paper": {
            "count_both": {
                "canonextract": {
                    "mean": mean_yield_cb_ce[0],
                    "ci95": [mean_yield_cb_ce[1], mean_yield_cb_ce[2]],
                },
                "baseline": {
                    "mean": mean_yield_cb_bl[0],
                    "ci95": [mean_yield_cb_bl[1], mean_yield_cb_bl[2]],
                },
            },
            "exclude_shared": {
                "canonextract": {
                    "mean": mean_yield_ex_ce[0],
                    "ci95": [mean_yield_ex_ce[1], mean_yield_ex_ce[2]],
                },
                "baseline": {
                    "mean": mean_yield_ex_bl[0],
                    "ci95": [mean_yield_ex_bl[1], mean_yield_ex_bl[2]],
                },
            },
        },
        "mean_paired_delta": {
            "count_both": {
                "mean": mean_delta_cb[0],
                "ci95": [mean_delta_cb[1], mean_delta_cb[2]],
                "ci_excludes_zero": (mean_delta_cb[1] > 0) or (mean_delta_cb[2] < 0),
            },
            "exclude_shared": {
                "mean": mean_delta_ex[0],
                "ci95": [mean_delta_ex[1], mean_delta_ex[2]],
                "ci_excludes_zero": (mean_delta_ex[1] > 0) or (mean_delta_ex[2] < 0),
            },
        },
        "yield_ratio_count_both": {
            "mean": yield_ratio_cb[0],
            "ci95": [yield_ratio_cb[1], yield_ratio_cb[2]],
        },
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
