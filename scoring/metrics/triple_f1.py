"""(Task, Dataset, Metric) triple F1 against TDMSci / NLP-TDMS gold.

Triples are canonicalized on both predicted and gold sides before comparison
(see paper1.canonicalize). This collapses surface-form variation
("avg accuracy" vs "accuracy", "BLEU-4" vs "BLEU", "SQuAD v1.1" vs "SQuAD")
that was driving v1 triple F1 near zero.
"""

from __future__ import annotations

from collections.abc import Iterable

from paper1.canonicalize import (
    canonicalize_dataset_full,
    canonicalize_metric_name,
    canonicalize_task_name,
)
from paper1.voting import _norm


def _canon_triple(t: str | None, d: str | None, m: str | None) -> tuple[str, str, str] | None:
    nt = canonicalize_task_name(t)
    nd = canonicalize_dataset_full(d)
    nm = canonicalize_metric_name(m)
    if not (nt and nd and nm):
        return None
    return (_norm(nt) or "", _norm(nd) or "", _norm(nm) or "")


def _norm_triples(
    triples: Iterable[tuple[str | None, str | None, str | None]],
) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for t, d, m in triples:
        ct = _canon_triple(t, d, m)
        if ct and all(ct):
            out.add(ct)
    return out


def triple_f1(
    pred_triples: Iterable[tuple[str | None, str | None, str | None]],
    gold_triples: Iterable[tuple[str | None, str | None, str | None]],
) -> dict[str, float]:
    p = _norm_triples(pred_triples)
    g = _norm_triples(gold_triples)
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
