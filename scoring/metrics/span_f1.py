"""Set-level F1 between predicted and gold entity sets.

The original SciREX joint model reports F1 on flat per-document entity sets
(after coreference resolution maps mention spans to canonical names). We
reuse that protocol here — both predicted and gold are sets of normalized
strings per paper, F1 is set-wise precision/recall.

`strict` mode: exact normalized-string match.
`lenient` mode: substring-overlap match (either direction). This rewards the
LLM for producing "BERT-large" when gold is "bert" and vice versa.
"""

from __future__ import annotations

from collections.abc import Iterable

from paper1.canonicalize import (
    canonicalize_dataset_full,
    canonicalize_method_name,
    canonicalize_metric_name,
    canonicalize_task_name,
)
from paper1.voting import _norm


def _canon(x: str | None, kind: str) -> str | None:
    if not x:
        return None
    if kind == "method":
        return canonicalize_method_name(x)
    if kind == "task":
        return canonicalize_task_name(x)
    if kind == "dataset":
        return canonicalize_dataset_full(x)
    if kind == "metric":
        return canonicalize_metric_name(x)
    return x


def _norm_set(items: Iterable[str | None], kind: str | None = None) -> set[str]:
    out: set[str] = set()
    for x in items:
        n = _norm(_canon(x, kind) if kind else x)
        if n:
            out.add(n)
    return out


def set_f1(
    pred: Iterable[str | None],
    gold: Iterable[str | None],
    lenient: bool = False,
    kind: str | None = None,
) -> dict[str, float]:
    p = _norm_set(pred, kind=kind)
    g = _norm_set(gold, kind=kind)
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if lenient:
        # lenient: count predicted as TP if any gold substring-overlaps
        tp = sum(1 for x in p if any(x in y or y in x for y in g))
        fp = max(0, len(p) - tp)
        fn = sum(1 for y in g if not any(y in x or x in y for x in p))
    else:
        tp = len(p & g)
        fp = len(p - g)
        fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
