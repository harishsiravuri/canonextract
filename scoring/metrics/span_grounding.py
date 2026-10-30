"""Span-grounding accuracy metric (Phase Y2 — headline contribution).

A predicted entity grounds correctly if:
  (a) its evidence_span (start, end) overlaps a gold span of the same entity type,
  AND
  (b) the entity's normalized name appears as a substring of the gold span's text
      (or vice versa).

Per-paper:
  precision = (# claims that ground correctly) / (# claims with non-null spans)
  recall    = (# gold entities that have a correctly-grounded claim) / (# gold entities of that type)
  f1        = 2pr / (p+r)

Aggregated across papers as macro-F1 (mean over papers).
"""

from __future__ import annotations

from collections.abc import Iterable

from paper1.loaders import GoldPaper
from paper1.schema import ContributionRecord
from paper1.voting import _norm


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _claims_from_record(rec: ContributionRecord) -> list[tuple[str, str, tuple[int, int] | None]]:
    """Yield (label, name_norm, span) for every entity with a non-null name in rec."""
    out: list[tuple[str, str, tuple[int, int] | None]] = []
    for c in rec.contributions:
        for label, ent in (("Method", c.method), ("Task", c.task)):
            n = _norm(ent.name)
            if not n:
                continue
            span = None
            if ent.evidence_span:
                span = (ent.evidence_span.start, ent.evidence_span.end)
            out.append((label, n, span))
        for d in c.datasets:
            n = _norm(d.name)
            if not n:
                continue
            span = None
            if d.evidence_span:
                span = (d.evidence_span.start, d.evidence_span.end)
            out.append(("Dataset", n, span))
        for m in c.metrics:
            n = _norm(m.name)
            if not n:
                continue
            span = None
            if m.evidence_span:
                span = (m.evidence_span.start, m.evidence_span.end)
            out.append(("Metric", n, span))
    return out


def _resolve_span_from_name(name: str, paper_text: str) -> tuple[int, int] | None:
    """Find the first occurrence of `name` (case-insensitive) in paper_text;
    return its character offsets. Falls back to whitespace-collapsed match.
    """
    if not name or not paper_text:
        return None
    needle = name.strip().lower()
    hay = paper_text.lower()
    idx = hay.find(needle)
    if idx >= 0:
        return (idx, idx + len(needle))
    # try to collapse to single tokens — split words and look for first
    return None


def per_paper_grounding_resolved(
    rec: ContributionRecord, gold: GoldPaper
) -> dict:
    """Same as per_paper_grounding, but ignores the LLM's evidence_span and
    instead resolves each predicted entity name to a char span via string
    match in the paper text. Reports the LLM's "post-processed" grounding
    capability — what the system can do given a deterministic name-to-span
    resolver downstream of the agents.
    """
    claims = _claims_from_record(rec)
    gold_spans_by_label: dict[str, list[tuple[int, int, str, str]]] = {}
    for s, e, lab, surf in gold.gold.gold_spans:
        gold_spans_by_label.setdefault(lab, []).append((s, e, lab, surf))

    resolved: list[tuple[str, str, tuple[int, int]]] = []
    for lab, name_norm, _orig_span in claims:
        sp = _resolve_span_from_name(name_norm, gold.full_text)
        if sp is not None:
            resolved.append((lab, name_norm, sp))

    correct_claims = 0
    matched_unique: set[tuple[str, str]] = set()
    for lab, name_norm, span in resolved:
        for gs, ge, _glab, gsurf in gold_spans_by_label.get(lab, []):
            if not _overlap(span, (gs, ge)):
                continue
            gn = _norm(gsurf) or ""
            if name_norm in gn or gn in name_norm:
                correct_claims += 1
                if gn:
                    matched_unique.add((lab, gn))
                break

    gold_unique: set[tuple[str, str]] = set()
    for _s, _e, lab, surf in gold.gold.gold_spans:
        gn = _norm(surf)
        if gn:
            gold_unique.add((lab, gn))

    precision = correct_claims / len(resolved) if resolved else 0.0
    recall = len(matched_unique) / len(gold_unique) if gold_unique else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_claims_resolved": len(resolved),
        "n_claims_total": len(claims),
        "n_correct": correct_claims,
        "n_gold_unique": len(gold_unique),
        "n_gold_matched": len(matched_unique),
    }


def per_paper_grounding(rec: ContributionRecord, gold: GoldPaper) -> dict:
    """Compute per-paper precision/recall/f1 of span grounding.

    Recall denominator is the unique set of (label, normalized_surface) gold
    entities — SciREX may have 259 mention-level spans per paper, but most are
    repeated mentions of the same ~10–20 entity types. The pipeline produces
    one canonical entity per type per contribution, so deduping gold to types
    aligns the comparison.
    """
    claims = _claims_from_record(rec)
    claims_with_span = [(lab, n, sp) for lab, n, sp in claims if sp is not None]
    gold_spans_by_label: dict[str, list[tuple[int, int, str, str]]] = {}
    for s, e, lab, surf in gold.gold.gold_spans:
        gold_spans_by_label.setdefault(lab, []).append((s, e, lab, surf))

    # PRECISION: of claimed spans, how many ground correctly?
    correct_claims = 0
    for lab, name_norm, span in claims_with_span:
        candidates = gold_spans_by_label.get(lab, [])
        for gs, ge, _glab, gsurf in candidates:
            if not _overlap(span, (gs, ge)):
                continue
            gnorm = _norm(gsurf) or ""
            if name_norm in gnorm or gnorm in name_norm:
                correct_claims += 1
                break
    precision = correct_claims / len(claims_with_span) if claims_with_span else 0.0

    # RECALL: collapse gold mentions to unique (label, normalized_surface) entities.
    gold_unique: set[tuple[str, str]] = set()
    for s, e, lab, surf in gold.gold.gold_spans:
        gn = _norm(surf)
        if gn:
            gold_unique.add((lab, gn))

    matched_unique: set[tuple[str, str]] = set()
    for lab, name_norm, span in claims_with_span:
        for gs, ge, _glab, gsurf in gold_spans_by_label.get(lab, []):
            if not _overlap(span, (gs, ge)):
                continue
            gn = _norm(gsurf) or ""
            if not gn:
                continue
            if name_norm in gn or gn in name_norm:
                matched_unique.add((lab, gn))
    recall = len(matched_unique) / len(gold_unique) if gold_unique else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_claims_with_span": len(claims_with_span),
        "n_correct": correct_claims,
        "n_gold_unique": len(gold_unique),
        "n_gold_matched": len(matched_unique),
    }


def aggregate_grounding(per_paper_results: Iterable[dict]) -> dict:
    rows = list(per_paper_results)
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    n = len(rows)
    return {
        "precision": sum(r["precision"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "n": n,
        "per_paper": rows,
    }
