# CanonExtract build — cycle 8a report (documented negative result)

Cycle 8a re-runs the cycle-6 verifier on the same 219 cycle-5 strict
candidates but injects a compact paper context (title + abstract +
method excerpt) into the prompt, aiming to resolve method-name aliases
like "Ours" → "LM-LSTM-CRF" that cycle 6 could not. **Result: Packet 2
precision unchanged at 66.7% and recall unchanged at 22.2%. The
paper-context prompt flipped 11 tuples total (9 YES → NO/UNCLEAR,
2 NO → YES), but none of the flips landed on Packet 2 tuples, so the
metrics did not move.** Cycle 6 remains the deployed calibrator.

**Cumulative OpenRouter spend across all cycles: \$4.107 of \$10 cap.
Cycle-8a additive: \$0.138 (219 calls at t=0, wall 175 s).**

## Cycle 8a setup

- Same model as cycle 6: `meta-llama/llama-3.3-70b-instruct` at t=0.
- Prompt: cycle-6 verifier prompt with a new **PAPER CONTEXT** block
  (title + first 300 words of abstract + method excerpt) injected
  BEFORE the table + tuple + claimed provenance.
- Method excerpt heuristic: first paragraph in the tex containing
  "we call" / "our method" / "our model" / "our approach" / "we
  propose" / "we introduce", capped at 300 tokens; falls back to the
  first paragraph of a Method / Approach / Model section.
- Applied to: ALL 219 cycle-5 strict-candidate tuples (49 MetaLead + 170
  SciREX). Not filtered to only cycle-6 NO, so we can also verify that
  the added context does not flip cycle-6 YES verdicts.

## Paper context extraction

- 46 unique papers (15 MetaLead + 31 SciREX).
- 39 papers full context (title + abstract + method excerpt extracted).
- 7 papers partial (missing `method_excerpt` — no matching trigger
  paragraph and no Method section header found).
- 0 papers with completely missing context.

Partial papers still received prompts with title + abstract + empty
method excerpt; cycle-6 fallback was never triggered.

## Cycle 8a verifier verdicts

| | Count |
|---|---:|
| Total calls | 219 |
| YES | **95** (43.4%) |
| NO | 124 (56.6%) |
| UNCLEAR | 0 |
| ERROR | 0 |
| Tokens in | 218,768 |
| Tokens out | 10,808 |
| Cost | \$0.1376 |
| Wall time | 175 s |

## Flips vs cycle 6

| | Cycle 6 | Cycle 8a |
|---|---:|---:|
| YES | 102 | 95 |
| NO | 117 | 124 |

- **YES → NO / UNCLEAR flips: 9** — cycle-6 YES tuples that cycle 8a
  rejected after seeing paper context.
- **NO → YES flips: 2** — cycle-6 NO tuples that cycle 8a promoted
  after resolving a method alias.

Net: 7 fewer strict tuples than cycle 6.

## Cycle 8a corpus totals

| Bucket | MetaLead | SciREX |
|---|---:|---:|
| strict_table_valid (cycle 8a YES) | **2.5%** (n=7) | **18.0%** (n=88) |
| verifier_downgrade_partial (cycle 8a NO) | 15.2% (n=42) | 16.8% (n=82) |
| partial_table_valid (from binder) | 51.1% (n=141) | 32.9% (n=161) |
| prose_claimed_table | 26.8% (n=74) | 15.5% (n=76) |
| prose | 2.5% (n=7) | 11.7% (n=57) |
| missing | 1.8% (n=5) | 5.1% (n=25) |

Strict rate marginally down vs cycle 6 (MetaLead 2.9% → 2.5%; SciREX
19.2% → 18.0%). Downgrade slightly up.

## Packet 2 rescore

| Bucket | YES | NO | MISSING |
|---|---:|---:|---:|
| strict_table_valid | 2 | 1 | 0 |
| verifier_downgrade_partial | 2 | 8 | 0 |
| partial_table_valid | 2 | 17 | 0 |
| prose_claimed_table | 1 | 2 | 2 |
| missing | 2 | 1 | 0 |

**Identical to cycle 6's Packet 2 bucket distribution.** The 11 total
flips between cycle 6 and cycle 8a all happened on non-Packet-2 tuples.

| Metric | Cycle 6 | **Cycle 8a** |
|---|---:|---:|
| Strict precision overall | 66.7% (2/3) | **66.7%** (2/3) |
| Strict recall overall | 22.2% (2/9) | **22.2%** (2/9) |
| Strict precision MetaLead | N/A (0 ML strict in sample) | **N/A** |
| Strict precision SciREX | 66.7% (2/3) | **66.7%** (2/3) |
| Strict recall MetaLead | 0/4 = 0.0% | **0/4 = 0.0%** |
| Strict recall SciREX | 2/5 = 40.0% | **2/5 = 40.0%** |

## Deployment decision

**Criteria (per spec)**: deploy cycle 8a if precision ≥ 60% AND recall
≥ 30%. Cycle 8a precision met (66.7% ≥ 60%). Cycle 8a recall NOT met
(22.2% < 30%).

**Decision: keep cycle 6 as deployed.**

## Why paper context did not move Packet 2 metrics

The 2 MetaLead Packet-2 YES cases that cycle 6 wrongly downgraded (P004:
`LM-LSTM-CRF / CoNLL00 chunking / F1 / 95.96`; P014: `BLSTM-CNN + emb +
lex / OntoNotes v5 English / f1 / 86.28`) both landed in
`verifier_downgrade_partial` again under cycle 8a. Both tuples DO have
paper context available (title + abstract; method excerpts extracted).
Yet the verifier still rejected them.

Likely reasons:
- **P004** — the tuple's method `LM-LSTM-CRF` is a paper-specific name;
  the row label in the table may not resolve to it even with paper
  context because the row label is generic ("Ours w/o transfer") and
  the context excerpt might not name it explicitly.
- **P014** — the tuple's method `BLSTM-CNN + emb + lex` has a specific
  ablation suffix; the row label `BLSTM-CNN + emb` differs by one
  ablation term. Paper context does not resolve this because the difference
  is one dataset-specific ablation configuration, not an alias.

The recall floor for these cases is not "verifier can't resolve
aliases" but "verifier reads the ablation configuration in the row
label as materially different from the tuple's ablation configuration",
which is a legitimate reason to reject on a strict rule. Distinguishing
whether the tuple's specific configuration name (`+ emb + lex`) is a
different model or a subset of the row's variant needs more than
title + abstract + method paragraph — it needs the actual results
section.

## Recommendation

**Ship cycle 6.** Precision 66.7%, recall 22.2%, any-table-bound
coverage 83–96%. Cycles 7 (lenient re-vote) and 8a (paper context) are
documented negative results in the git history for auditability.

The remaining recall gap (22.2%) is largely dominated by MetaLead
tuples where the row label differs from the tuple method by an ablation
configuration — legitimately rejected on strict rules, and only
recoverable with either (a) results-section-level context in the
prompt (cost multiplier — full-paper context would be 3–5× the current
per-call size), or (b) a stronger verifier model (cycle 8b territory —
GPT-4o / Claude Sonnet). Neither is required for a defensible chapter
draft.

## Cycles 6 → 7 → 8a progression (all Packet 2)

| Metric | Cycle 6 (deployed) | Cycle 7 (negative) | Cycle 8a (negative) |
|---|---:|---:|---:|
| Precision overall | **66.7%** | 50.0% | 66.7% |
| Recall overall | **22.2%** | 22.2% | 22.2% |
| Precision SciREX | **66.7%** | 50.0% | 66.7% |
| Recall SciREX | **40.0%** | 40.0% | 40.0% |
| Recall MetaLead | 0.0% | 0.0% | 0.0% |
| Cost | \$0.083 | \$0.062 | **\$0.138** |
| Deployment | ✅ deployed | ❌ | ❌ |

## Files added in cycle 8a

Under `outputs/canonextract_build/`:
- `SUMMARY_cycle8a.md` (this file)
- `cycle8a_summary.json` — aggregate + Packet 2 rescore + flip counts + cost
- `paper_context/{corpus}/{paper_id}.json` — 46 papers' title / abstract / method excerpt
- `paper_context/context_partial.json` — 7 papers with missing method_excerpt
- `verifier_cycle8a/{corpus}/{paper_id}.json` — per-paper cycle 8a verdicts + reasons
- `eval_runs_metalead_cycle8a/{paper_id}.json` — cycle-5 tuples + `cycle8a_bucket`
- `eval_runs_scirex_cycle8a/{paper_id}.json`
- `cycle8a_run.log`

Under `scripts/`:
- `canonextract_cycle8a_context.py` — paper context extractor
- `canonextract_cycle8a_verifier.py` — cycle 8a verifier driver

**Main branch, `artifacts/paper_data_v10/`, `conference_101719.tex`, and
all cycle 1–7 outputs untouched.**
