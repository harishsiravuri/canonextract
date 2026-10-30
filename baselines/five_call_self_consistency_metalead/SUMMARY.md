# Cost-matched self-consistency baseline on MetaLead — SUMMARY

**Post-lock addition to the H1 whole-record reanalysis.** Produced 2026-07-28 for the endpoint completion of `REPORT_h1_whole_record.md`.

## Method
- Model: `deepseek/deepseek-chat` (same as CanonExtract's and paper 1's Extractor)
- Prompt: paper 1's `config/prompts/extractor.md`, unchanged
- Samples per paper: 5 at temperatures {0.0, 0.3, 0.5, 0.7, 0.7}
- Voting: per contribution key `(task_norm, first_dataset_norm, first_metric_norm)`, keep contributions with fractional agreement ≥ 3/5, tie-break to t=0.0
- Paper text cap: 45,000 chars (same as original paper 1 baseline)

## Run
Two-phase run: Phase 1 hit the initial \$0.50 cap at 20 papers with 5/5 samples each. User approved additional spend for the remaining 7. Phase 2 (resume) processed them all.

| Phase | Papers | Cost | Tokens in | Tokens out | Wall |
|---|---:|---:|---:|---:|---:|
| Run 1 | 20 of 27 (5/5) | \$0.5131 | 1,265,260 | 121,581 | 1162 s |
| Run 2 (resume) | 7 of 7 (5/5) | \$0.1883 | 452,050 | 49,070 | 492 s |
| **Total** | **27 of 27** | **\$0.7014** | **1,717,310** | **170,651** | **1654 s** |

Cumulative OpenRouter spend cycles 1..SC: **~\$4.81 of \$10 cap**.

## Files
- `metalead_{paper_id}.json` — per-paper voted contributions (schema mirrors paper 1 baseline) with `_raw_samples` (5 per paper) and `_meta` (tokens, cost, wall, temperatures, seed)
- `extraction_summary.json` — resume-phase 7-paper summary
- `extraction_summary_run1.json` — initial 20-paper summary
- `../sc_baseline_run.log`, `../sc_baseline_run2.log` — run logs

## Scoring output
See `../h1_cost_matched_sc.json` and the "Cost-matched self-consistency baseline (post-lock addition)" section in `../REPORT_h1_whole_record.md`.

## Result headlines (per paper-clustered bootstrap on n=27 CE-vs-SC paired)
- Whole-record F1: **CE 0.388 vs SC 0.312** (Δ +7.67 pp, 95% CI [−6.78, +21.03] — includes 0)
- Recall: CE 0.379 vs SC 0.271 (Δ +10.85 pp, CI [−3.20, +24.89])
- Precision: CE 0.451 vs SC 0.457 (Δ −0.56 pp, CI [−17.28, +16.63])
- Decision rule: **H1's matched-budget claim NOT satisfied at 95 % CI**; mechanism analysis triggered.
- Mechanism: 80 % (40/50) of CE-exclusive gold coverage traces to table-anchored provenance; 18 % (9/50) to prose.
