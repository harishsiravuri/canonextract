"""Platt-style temperature scaling for binary correctness probabilities.

We treat the model's `self_consistency` ∈ [0, 1] as a probability of correctness.
Temperature scaling fits a single scalar T so that the calibrated probability is
the sigmoid of (logit(p) / T). T is fit by minimising NLL on a held-out split.

Returns: (best_T, ece_uncalibrated, ece_calibrated, calibrated_probs).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from paper1.metrics.stats import expected_calibration_error


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(probs: Sequence[float], correct: Sequence[float]) -> float:
    """Find T > 0 minimising NLL on these (probs, correct) pairs."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.size == 0:
        return 1.0
    z = _logit(p)
    # Coarse-then-fine 1-D grid search; T in [0.05, 20]
    Ts = np.concatenate([np.linspace(0.05, 1.0, 40), np.linspace(1.0, 20.0, 80)[1:]])
    best_t, best_nll = 1.0, float("inf")
    for t in Ts:
        loss = _nll(_sigmoid(z / t), y)
        if loss < best_nll:
            best_nll, best_t = loss, float(t)
    return best_t


def apply_temperature(probs: Sequence[float], T: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(probs, dtype=float)) / max(T, 1e-6))


def calibrate_and_evaluate(
    probs: Sequence[float],
    correct: Sequence[float],
    fit_frac: float = 0.2,
    n_bins: int = 10,
    seed: int = 0,
) -> dict:
    """Fit T on a held-out fit_frac, evaluate ECE on the rest."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.size < 10:
        return {"n": int(p.size), "T": 1.0, "ece_uncalibrated": 0.0, "ece_calibrated": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(p.size)
    n_fit = max(2, int(round(p.size * fit_frac)))
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]
    T = fit_temperature(p[fit_idx], y[fit_idx])
    p_eval = p[eval_idx]
    y_eval = y[eval_idx]
    p_cal = apply_temperature(p_eval, T)
    ece_un = expected_calibration_error(p_eval.tolist(), y_eval.tolist(), n_bins=n_bins)
    ece_cal = expected_calibration_error(p_cal.tolist(), y_eval.tolist(), n_bins=n_bins)
    return {
        "n": int(p.size),
        "n_fit": int(n_fit),
        "n_eval": int(eval_idx.size),
        "T": float(T),
        "ece_uncalibrated": float(ece_un),
        "ece_calibrated": float(ece_cal),
        "mean_conf_uncal": float(p_eval.mean()) if p_eval.size else 0.0,
        "mean_conf_cal": float(p_cal.mean()) if p_cal.size else 0.0,
        "mean_acc": float(y_eval.mean()) if y_eval.size else 0.0,
    }
