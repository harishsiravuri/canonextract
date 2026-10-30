"""Phase Y3 — measures the critic's precision on its UNSUPPORTED verdicts.

Approach (using only saved records, no re-extraction):

For each SciREX paper, compare the FULL multi-agent record against the
NO-CRITIC ablation record on the same paper. A field where:
  * NoCritic has a non-null name (the extractor proposed it),
  * Full has a null name AND `critic_verdict.<field>` == "UNSUPPORTED",
counts as a *critic-suppressed extraction*.

We then check that suppressed extraction against SciREX gold:
  * Suppression is **correct** (truly wrong) if the no-critic name does
    NOT match any gold entity of that type (lenient substring match).
  * Suppression is **a false suppression** if the no-critic name DOES
    match a gold entity of that type — the extractor was right and the
    critic killed a correct claim.

Also count the *missed-suppressions* — fields where Full retained a name
that does NOT match gold (the critic should have suppressed it but
didn't). These are needed for the recall side of the verdict.

Reports:
  critic_precision = correct_suppressions / total_suppressions
  critic_recall    = correct_suppressions / (correct_suppressions + missed_suppressions)
"""

from __future__ import annotations

from collections.abc import Iterable

from paper1.canonicalize import (
    canonicalize_dataset_full,
    canonicalize_method_name,
    canonicalize_metric_name,
    canonicalize_task_name,
)
from paper1.loaders import GoldPaper
from paper1.schema import ContributionRecord
from paper1.voting import _norm

_CANON = {
    "method": canonicalize_method_name,
    "task": canonicalize_task_name,
    "dataset": canonicalize_dataset_full,
    "metric": canonicalize_metric_name,
}


def _norm_canon(name: str | None, kind: str) -> str | None:
    if not name:
        return None
    fn = _CANON.get(kind)
    if fn is None:
        return _norm(name)
    return _norm(fn(name))


def _matches_gold(name_norm: str | None, gold_set: set[str]) -> bool:
    if not name_norm or not gold_set:
        return False
    for g in gold_set:
        if name_norm in g or g in name_norm:
            return True
    return False


def _record_singletons(rec: ContributionRecord) -> dict[str, str | None]:
    """Reduce a record to a single name per (method.name, task.name).
    For datasets/metrics, return any non-null name (since they're lists)."""
    if not rec.contributions:
        return {"method.name": None, "task.name": None, "datasets": None, "metrics": None}
    c = rec.contributions[0]
    return {
        "method.name": _norm_canon(c.method.name, "method"),
        "task.name": _norm_canon(c.task.name, "task"),
        "datasets": next(
            (n for n in (_norm_canon(d.name, "dataset") for d in c.datasets) if n), None
        ),
        "metrics": next(
            (n for n in (_norm_canon(m.name, "metric") for m in c.metrics) if n), None
        ),
    }


def _record_critic_verdict(rec: ContributionRecord) -> dict[str, str | None]:
    if not rec.contributions:
        return {"method.name": None, "task.name": None, "datasets": None, "metrics": None}
    cv = rec.contributions[0].critic_verdict
    return {
        "method.name": cv.method,
        "task.name": cv.task,
        "datasets": cv.datasets,
        "metrics": cv.metrics,
    }


def critic_validation(
    full_record: ContributionRecord,
    no_critic_record: ContributionRecord,
    gold: GoldPaper,
) -> dict:
    """Return per-paper (correct_suppressions, false_suppressions, missed_suppressions, retained_correct)."""
    full = _record_singletons(full_record)
    no_critic = _record_singletons(no_critic_record)
    verdict = _record_critic_verdict(full_record)
    gold_sets = {
        "method.name": gold.gold.methods,
        "task.name": gold.gold.tasks,
        "datasets": gold.gold.datasets,
        "metrics": gold.gold.metrics,
    }
    by_field: dict[str, dict] = {}
    for field, gold_set in gold_sets.items():
        nc_name = no_critic[field]
        full_name = full[field]
        v = verdict[field]

        suppressed = (full_name is None) and (nc_name is not None)
        critic_unsupported = v == "UNSUPPORTED"

        nc_in_gold = _matches_gold(nc_name, gold_set)
        full_in_gold = _matches_gold(full_name, gold_set)

        # critic-suppressed = full dropped the field AND critic said UNSUPPORTED
        # (we use the OR of the two heuristics so we count both explicit and implicit drops)
        is_critic_suppression = suppressed and critic_unsupported
        is_implicit_suppression = suppressed and not critic_unsupported  # consolidator dropped without explicit verdict

        # Outcome buckets:
        correct_suppression = is_critic_suppression and not nc_in_gold
        false_suppression = is_critic_suppression and nc_in_gold
        # missed-suppression: full has a name not in gold (extractor wrong, critic didn't fix it)
        missed_suppression = (full_name is not None) and not full_in_gold
        retained_correct = (full_name is not None) and full_in_gold

        by_field[field] = {
            "no_critic_name": nc_name,
            "full_name": full_name,
            "critic_verdict": v,
            "is_critic_suppression": bool(is_critic_suppression),
            "is_implicit_suppression": bool(is_implicit_suppression),
            "correct_suppression": bool(correct_suppression),
            "false_suppression": bool(false_suppression),
            "missed_suppression": bool(missed_suppression),
            "retained_correct": bool(retained_correct),
        }
    return by_field


def aggregate(per_paper: Iterable[dict]) -> dict:
    fields = ("method.name", "task.name", "datasets", "metrics")
    totals = {f: {"correct_supp": 0, "false_supp": 0, "missed_supp": 0,
                  "retained_correct": 0, "implicit_supp": 0} for f in fields}
    overall = {"correct_supp": 0, "false_supp": 0, "missed_supp": 0,
               "retained_correct": 0, "implicit_supp": 0, "n_papers": 0}
    for paper in per_paper:
        overall["n_papers"] += 1
        for f, d in paper.items():
            t = totals[f]
            t["correct_supp"] += int(d["correct_suppression"])
            t["false_supp"] += int(d["false_suppression"])
            t["missed_supp"] += int(d["missed_suppression"])
            t["retained_correct"] += int(d["retained_correct"])
            t["implicit_supp"] += int(d["is_implicit_suppression"])
            overall["correct_supp"] += int(d["correct_suppression"])
            overall["false_supp"] += int(d["false_suppression"])
            overall["missed_supp"] += int(d["missed_suppression"])
            overall["retained_correct"] += int(d["retained_correct"])
            overall["implicit_supp"] += int(d["is_implicit_suppression"])

    def _pr(t: dict) -> dict:
        n_supp = t["correct_supp"] + t["false_supp"]
        wrong_total = t["correct_supp"] + t["missed_supp"]
        return {
            **t,
            "n_critic_suppressions": n_supp,
            "critic_precision": (t["correct_supp"] / n_supp) if n_supp else None,
            "critic_recall": (t["correct_supp"] / wrong_total) if wrong_total else None,
        }

    return {
        "by_field": {f: _pr(totals[f]) for f in fields},
        "overall": _pr(overall),
    }
