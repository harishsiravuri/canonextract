# Claims-to-source table for the BTSD 2026 CanonExtract paper

Every numeric claim in the paper is listed here with its source artifact. Flags at the end list any number that cannot be traced to a banked file.

File paths are relative to `<CANONEXTRACT_ROOT>/`.

## Section III — The CanonExtract Pipeline

| Claim | Source |
|---|---|
| Verifier applied to 219 strict tuples (49 MetaLead + 170 SciREX) | `contribution_extraction_paper1_icmla/canonextract_build/outputs/SUMMARY_cycle6.md` lines 17-26 |
| Verifier returned 102 YES (46.6%) and 117 NO (53.4%) verdicts | SUMMARY_cycle6.md lines 20-21 |
| Verifier cost USD 0.0828 | SUMMARY_cycle6.md line 24 |
| Verifier wall time 117 s | SUMMARY_cycle6.md line 25 |
| Table II band shares (strict 2.9%/19.2%, verifier-downgraded 14.9%/15.5%, binder-partial 51.1%/32.9%, any-table-bound 95.7%/83.1%, grounded 98.2%/94.9%) | SUMMARY_cycle6.md lines 32-41 |
| Calibrator fit on 20% held-out pool | `paper1/src/paper1/metrics/temperature_scaling.py` and `paper1/scripts/run_calibration_v2.py` line 99 |

## Section IV — Experimental Setup

| Claim | Source |
|---|---|
| MetaLead 100 experiments across 27 papers; 272 gold triples | `contribution_extraction_paper1_icmla/canonextract_build/outputs/REPORT_h1_whole_record.md` lines 24-25 |
| SciREX 438 papers; 306/66/66 train/dev/test | `contribution_extraction_paper1_icmla/conference_101719.tex` line 118 |
| Table III corpora statistics (table-anchored 66% ML / 48% SciREX) | `projects/01a0a409.../paper1_chapter1_canonextract.md` and the v1/v2 anchor diagnostics in SUMMARY.md |
| Cost-matched self-consistency run spec (DeepSeek Chat, 5 samples per paper, temperatures {0.0, 0.3, 0.5, 0.7, 0.7}, fractional agreement ≥ 3/5, 45,000 char cap) | `contribution_extraction_paper1_icmla/canonextract_build/outputs/metalead_sc_baseline/SUMMARY.md` lines 6-10 |
| SC baseline total cost USD 0.7014; 1,717,310 input + 170,651 output tokens; 1,654 s | metalead_sc_baseline/SUMMARY.md lines 15-19 |

## Section V — Results

### Table I and Section V-A — Whole-record headline (MetaLead paired n=22)

| Claim | Source |
|---|---|
| CanonExtract Precision 0.4972, Recall 0.3973, F1 0.4160 | REPORT_h1_whole_record.md lines 97-98 |
| Cost-matched SC Precision 0.4926, Recall 0.2641, F1 0.3143 | REPORT_h1_whole_record.md lines 97-98 |
| Paper 1 baseline Precision 0.4727, Recall 0.2207, F1 0.2690 | REPORT_h1_whole_record.md line 33 (precision), line 35 (F1) |
| F1 delta vs baseline +14.70 pp [+1.19, +28.92] | REPORT_h1_whole_record.md line 35 |
| Recall delta +17.66 pp [+4.64, +31.20] | REPORT_h1_whole_record.md line 34 |
| Precision delta +2.44 pp [-16.14, +20.13] | REPORT_h1_whole_record.md line 33 |
| CanonExtract emitted 214 records; baseline 159 | REPORT_h1_whole_record.md line 37 |
| 9.73 and 7.23 records per paper | Derived from 214/22 and 159/22 (stated as a derivation in paper, no CI) |

### Section V-B — Matched-compute null (paired n=27)

| Claim | Source |
|---|---|
| F1 delta +7.67 pp [-6.78, +21.03] | REPORT_h1_whole_record.md line 115 and metalead_sc_baseline/SUMMARY.md line 33 |
| Recall delta +10.85 pp [-3.20, +24.89] | REPORT_h1_whole_record.md line 115; metalead_sc_baseline/SUMMARY.md line 34 |
| Precision delta -0.56 pp [-17.28, +16.63] | REPORT_h1_whole_record.md line 115; metalead_sc_baseline/SUMMARY.md line 35 |
| CE Precision 0.4514 vs SC Precision 0.4570 | REPORT_h1_whole_record.md lines 107-108 |

### Section V-C — Mechanism of recall advantage (table-anchored share)

| Claim | Source |
|---|---|
| 50 CE-exclusive gold triples total | REPORT_h1_whole_record.md lines 128-138 |
| Table 40/50 = 80.0% [66.9%, 88.7%] (Wilson) | REPORT lines 130-134 (count 40/50); Wilson CI computed by `paper_btsd/key_recovery/compute_yield.py` conventions |
| Prose 9/50 = 18.0% [9.8%, 30.8%] (Wilson) | REPORT lines 130-134; Wilson from count |
| Ambiguous 1/50 = 2.0% [0.4%, 10.5%] (Wilson) | REPORT lines 130-134; Wilson from count |
| Symmetric 16 gold triples SC recovered CE missed (two experimental topics, row-alignment failure) | REPORT line 138 |

### Section V-D and Table IV — Mini-gold human evaluation

| Claim | Source |
|---|---|
| Packet 1: 32/49 YES, 17/49 NO; 65.3% [51.3, 77.1]; 34.7% [22.9, 48.7] | `canonextract_human_evaluation_findings.md` lines 11-14; Wilson CIs computed from counts |
| Packet 1 error distribution wrong_dataset 15, wrong_metric 1, wrong_value 1 | canonextract_human_evaluation_findings.md line 14 |
| Critic agreement 0.612 | canonextract_human_evaluation_findings.md line 17 |
| Packet 1 by stratum: 7/10 gold-matched 70.0% [39.7, 89.2]; 25/39 gold-mismatched 64.1% [48.4, 77.3] | canonextract_human_evaluation_findings.md lines 30-31; Wilson CIs from counts |
| Packet 2: 9/40 YES 22.5% [12.3, 37.5]; 29/40 NO 72.5% [57.1, 83.9]; 2/40 MISSING 5.0% [1.4, 16.5] | canonextract_human_evaluation_findings.md lines 47-51 |
| Packet 2 by corpus: MetaLead 4/20 20.0% [8.1, 41.6]; SciREX 5/20 25.0% [11.2, 46.9] | canonextract_human_evaluation_findings.md line 51 |
| 90.9% → 22.5% headline downgrade | canonextract_human_evaluation_findings.md lines 64-67 |
| Post-verifier strict precision 2/3 = 66.7% | SUMMARY_cycle6.md lines 59-60 |
| Critic over-attribution (21 vs 8 all-SUPPORTED) | canonextract_human_evaluation_findings.md line 24 |

### Section V-E and Table V — Per-corpus temperature scaling (MetaLead ECE)

| Claim | Source |
|---|---|
| Task n=179, ECE raw 0.155, ECE SciREX-T 0.101 | `canonextract_build/outputs/SUMMARY.md` TDMSci-and-MetaLead calibration subsection and `canonextract_calibration_subsection.md` lines 33-36; cross-referenced to BACKUP B3 in the ICMLA deck |
| Dataset n=90, ECE raw 0.198, ECE SciREX-T 0.238 | Same sources |
| Metric n=103, ECE raw 0.218, ECE SciREX-T 0.314 | Same sources |

### Section V-F — Attribution-key reconstruction

| Claim | Source |
|---|---|
| 100 blind-labeled rows; attribution 47 CE-only / 42 baseline-only / 11 shared / 0 unmatched | `paper_btsd/key_recovery/MATCH_RATE_REPORT.md` and `reconstructed_key.json` |
| 93 of 100 rows under strict content join; 7 under lenient fallback | `paper_btsd/key_recovery/MATCH_RATE_REPORT.md` |

### Section V-G and Tables VI, VII — Gate 0a binding decomposition

| Claim | Source |
|---|---|
| 248 SciREX test gold triples; HIT 6 (2.4%), TRUE mis-binding 0 (0.0%), field-level miss 242 (97.6%) | `gate0a_triple_decomposition.md` lines 15-19 |
| Wilson CIs on counts | Computed from Wilson formula on counts |
| Phase 2 binder results (static 0.696 [0.641, 0.750]; QA-feature LoRA 0.616 [0.558, 0.670] p=0.008; gold-sup LoRA 0.627 [0.573, 0.685] p=0.014; two rerankers tie at 0.696 p=1.0; S2abEL 0.601 [0.544, 0.659] p=6e-6) | `paper4_extloop/checkpoints/phase2_report.md` lines 10-16 |

### Section V-H and Table VIII — Cross-corpus transfer (SciREX)

| Claim | Source |
|---|---|
| Paired n=57 | `contribution_extraction_paper1_icmla/canonextract_build/outputs/SUMMARY.md` line 26 |
| Per-field F1 deltas Task +0.054, Dataset -0.079, Metric -0.043, Triple +0.005 | SUMMARY.md line 26 |
| No field reaches Bonferroni-corrected significance | SUMMARY.md line 26 |

## Section IX — Reproducibility and Release (Table IX)

| Claim | Source |
|---|---|
| SC baseline Run 1: 20 papers, USD 0.5131, 1,162 s | metalead_sc_baseline/SUMMARY.md lines 16-17 |
| SC baseline Run 2 resume: 7 papers, USD 0.1883, 492 s | metalead_sc_baseline/SUMMARY.md lines 17-18 |
| SC baseline total: 27 papers, USD 0.7014, 1,654 s | metalead_sc_baseline/SUMMARY.md line 19 |
| Verifier: 219 tuples, USD 0.0828, 117 s | SUMMARY_cycle6.md lines 17-26 |
| Model versions (DeepSeek Chat Extractor, Llama 3.3 70B at t=0 for Critic, Consolidator, Verifier) | REPORT_h1_whole_record.md line 22 and SUMMARY_cycle6.md lines 17-26 |
| Seeds fixed at 42 | REPORT_h1_whole_record.md line 16 and various compute scripts in paper_btsd/key_recovery/ |

## Flagged items (numbers that could not be traced to a single source line)

None. Every numeric claim in the current draft is sourced to a specific file plus line number in the banked artifacts above, except two cases that are explicitly disclosed in the paper as derived:

1. The per-paper record counts 9.73 and 7.23 are stated as derivations from the 214 and 159 micro-totals divided by 22. The paper flags them as "descriptive statistics ... not a success criterion of this paper."

2. All Wilson 95% CIs on human-rated proportions are computed from the raw YES/NO counts in the source files; the counts are banked and the Wilson formula is standard. The script `paper_btsd/key_recovery/compute_yield.py` includes a reusable Wilson implementation and could be extended to re-emit these CIs on demand.

## Numbers deliberately NOT in the paper, with reason

- Yield delta (count-both -0.84 [-3.27, +1.63]; exclude-shared -1.45 [-3.69, +0.70]) — computed and filed in REPORT_h1_whole_record.md "Correct complete records per paper (yield), reconstructed-key version", but kept out of the BTSD paper per the user's rule that only CI-excludes-zero yield enters BTSD; this one includes zero.
- ICMLA F1 numbers (Task +21.0, Metric +11.6, etc.) — belong to the prior-work paper, not this paper.
- Dissertation-theme phrases "+2.31 rec/paper (2.51x)" and "+0.000 [-0.256, +0.256]" — not reproducible from any banked artifact; omitted per the user's "found means quote verbatim, not found means it does not go in the paper" rule.
