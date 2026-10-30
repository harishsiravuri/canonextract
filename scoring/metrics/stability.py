"""Stability metrics — Jaccard similarity between two ContributionRecords.

A ContributionRecord may have multiple contributions; for the pilot we
flatten the per-record fields into sets and compute set-Jaccard per field.
"""

from __future__ import annotations

from typing import Any

from paper1.schema import ContributionRecord
from paper1.voting import _norm


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _method_names(record: ContributionRecord) -> set[str]:
    out: set[str] = set()
    for c in record.contributions:
        n = _norm(c.method.name)
        if n:
            out.add(n)
    return out


def _task_names(record: ContributionRecord) -> set[str]:
    out: set[str] = set()
    for c in record.contributions:
        n = _norm(c.task.name)
        if n:
            out.add(n)
    return out


def _dataset_names(record: ContributionRecord) -> set[str]:
    out: set[str] = set()
    for c in record.contributions:
        for ds in c.datasets:
            n = _norm(ds.name)
            if n:
                out.add(n)
    return out


def _metric_names(record: ContributionRecord) -> set[str]:
    out: set[str] = set()
    for c in record.contributions:
        for m in c.metrics:
            n = _norm(m.name)
            if n:
                out.add(n)
    return out


def _claim_strengths(record: ContributionRecord) -> set[str]:
    out: set[str] = set()
    for c in record.contributions:
        if c.claim_strength is not None:
            out.add(c.claim_strength)
    return out


def jaccard_stability(
    record_a: ContributionRecord, record_b: ContributionRecord
) -> dict[str, Any]:
    """Compute Jaccard similarity per field plus an overall mean.

    Returns a dict with per-field scores and an "overall" mean.
    """

    method_j = _jaccard(_method_names(record_a), _method_names(record_b))
    task_j = _jaccard(_task_names(record_a), _task_names(record_b))
    datasets_j = _jaccard(_dataset_names(record_a), _dataset_names(record_b))
    metrics_j = _jaccard(_metric_names(record_a), _metric_names(record_b))
    claim_j = _jaccard(_claim_strengths(record_a), _claim_strengths(record_b))

    per_field = {
        "method.name": method_j,
        "task.name": task_j,
        "datasets": datasets_j,
        "metrics": metrics_j,
        "claim_strength": claim_j,
    }
    overall = sum(per_field.values()) / len(per_field)

    return {
        "per_field": per_field,
        "overall": overall,
    }
