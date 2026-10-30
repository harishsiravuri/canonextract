"""Bootstrap CIs, paired permutation tests, and ECE — pure numpy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = rng or np.random.default_rng(0)
    n = arr.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), lo, hi


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided paired permutation test on the difference of means.

    Returns p-value. a[i] and b[i] must be paired observations (same paper).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        return 1.0
    rng = rng or np.random.default_rng(0)
    diff = a_arr - b_arr
    obs = diff.mean()
    n = diff.size
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        m = (diff * signs).mean()
        if abs(m) >= abs(obs):
            count += 1
    return (count + 1) / (n_permutations + 1)


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Standard ECE with equal-width bins on [0, 1]."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    if conf.size == 0 or conf.shape != correct.shape:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_diagram_data(
    confidences: Sequence[float],
    correctness: Sequence[float],
    n_bins: int = 10,
) -> list[dict]:
    """Per-bin data for a reliability diagram. Returns list of bin records."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
            }
        )
    return out


def percentiles(values: Sequence[float], qs: Sequence[float] = (0.5, 0.95, 0.99)) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q * 100)}"] = float(np.quantile(arr, q))
    out["mean"] = float(arr.mean())
    return out
