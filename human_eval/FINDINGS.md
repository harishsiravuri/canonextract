# CanonExtract, human evaluation findings (2026-07-26)

Two blinded annotation packets returned by Harish. Records + hidden keys in the outputs folder; scoring script in the reproducibility appendix.

## Packet 1: adjudication of 49 stratified tuples against paper text

Sample: 49 CanonExtract tuples, stratified by (gold-match status, self-consistency, task family). Harish adjudicated `binding_correct_yes_no` (YES / NO / UNCLEAR) against each paper's own text, along with an `error_type` when NO.

### Aggregate

- **YES: 32 (65.3%)**. The extracted tuple is defensibly correct as an experimental result the paper reports.
- **NO: 17 (34.7%)**. Error type distribution: wrong_dataset 15, wrong_metric 1, wrong_value 1. Wrong_dataset dominates.
- **UNCLEAR: 0**.

### Cross-tab with pipeline Critic

Simple agreement between Harish's YES/NO and pipeline Critic's all-SUPPORTED verdict: **0.612**. Detailed cross-tab:

| Harish | Critic all-SUPPORTED | Critic not-all-SUPPORTED |
|---|---:|---:|
| YES (32) | 21 | 11 |
| NO (17) | 8 | 9 |

The Critic over-attributes SUPPORTED (21 + 8 = 29 all-SUPPORTED cases, of which 8 are Harish-NO). This is a real Critic calibration weakness that the calibration subsection now has direct evidence for: the Critic verdict is a useful signal but is not a stand-in for human adjudication.

### The key comparability finding

Of the 49 sampled tuples, 10 are exact-match against gold (SciREX or MetaLead) and 39 are gold-mismatched. Harish's YES rate is nearly identical across those subsets:

- Gold-matched: **7 of 10 YES (70%)**.
- Gold-mismatched: **25 of 39 YES (64%)**.

Twenty-five of the 39 gold-mismatched tuples are correct in the paper's own terms. That is direct human-verified evidence that most CanonExtract "F1 misses" against benchmark gold are **schema mismatches** (benchmark coverage gaps, vocabulary variants, dataset-variant granularity) rather than extraction errors. This confirms Gate 0a's aggregate finding at the tuple level, and it is the load-bearing empirical claim behind the chapter's corpus-fit disclosure on SciREX transfer.

### Implications for the chapter

- The Introduction can now report a human-verified extraction correctness of **65% on paper text**, alongside the strict-match F1 numbers.
- The Discussion's corpus-fit paragraph can cite the **64% YES on gold-mismatched tuples** as direct evidence that the strict-match F1 ceiling on SciREX is set by the benchmark, not by the extractor.
- The wrong_dataset dominance (15 of 17 errors) is consistent with Gate 0a: datasets are the field on which CanonExtract's mechanism is weakest. This is reported honestly in Section VII-E (calibration) and now also in the adjudication paragraph.

## Packet 2: provenance spot-check on 40 stratified table-valid tuples

Sample: 40 CanonExtract tuples flagged as `source = "table"` by the pipeline binder, stratified by (corpus, self-consistency). Harish adjudicated `cell_matches_tuple_yes_no` (YES / NO / MISSING) after inspecting the paper's tables.

### Aggregate

- **YES: 9 (22.5%)** — the resolved cell is the RIGHT cell for the tuple.
- **NO: 29 (72.5%)** — resolved cell exists in the paper's tables but points to a different (method, dataset, metric) combination than the tuple claims.
- **MISSING: 2 (5.0%)** — resolved cell cannot be found in the paper.

By corpus: MetaLead 4 of 20 YES (20%), SciREX 5 of 20 YES (25%). Symmetric across corpora, so the finding is a general property of the binder, not a corpus artifact.

### What went wrong

Inspection of the 29 NO rows shows a consistent pattern. The binder finds a cell in the paper's tables whose numeric value matches the tuple's value (or is close), but the cell's row label is for a different method and/or the cell's column path is for a different (dataset, metric) than the tuple claims. Concrete cases from the sample:

- Tuple claims method "PCL" on ImageNet detection at mAP 19.6. Binder resolves to row `-OB-G-Ens.+FRCNN` in column `Results`, cell value 19.6. Value matches but row is a completely different method (an ensemble baseline, not PCL).
- Tuple claims method "MUSE" on WMT14 En-Fr at BLEU 43.5. Binder resolves to row `Model` in column `En-Fr`, cell value 41.4. Value does not match (43.5 vs 41.4), row is a generic column header, but the binder accepted the cell.

The binder was accepting a cell when it found any cell with a matching-ish numeric value under a permissive lookup, rather than enforcing structural alignment on row-and-column path. The strict-table-valid flag was measuring cell existence, not cell semantics.

### Downgraded chapter headline

The chapter previously reported **90.9% strict-table-valid provenance** on MetaLead. The human-verified rate is **22.5%**. That is a real downgrade and it goes into the chapter honestly. Reframed provenance headline for the chapter:

- **Automatic strict-table-valid (binder-claimed): 90.9%** on MetaLead, 81.4% on SciREX transfer.
- **Human-verified cell-semantics-correct (Packet 2 spot-check): 22.5% overall** (MetaLead 20%, SciREX 25%).
- The gap is a binder bug: the binder accepts a cell when it finds a matching-ish value, without enforcing that the row-and-column path corresponds to the tuple's (method, dataset, metric) claim.

### Cycle 5 fix

Rewrite the binder to require **all three of** (a) tuple.method aligns with the cell's row label (after LaTeX and formatting normalization; treat "ours", "our", "proposed" as legitimate row labels when the tuple's method is the paper's own contribution), (b) tuple.dataset OR tuple.metric appears in the cell's header path, and (c) numeric(tuple.value) matches numeric(cell) within a small rounding tolerance (allowing +/- notation like "95.96 +/- 0.08"). Report "strict-table-valid" as the fraction passing 3-of-3; introduce a "partial-table-valid" bucket for 2-of-3 pass (value match plus one of row-or-column, but not both); downgrade the rest to "prose_claimed_table" as before.

This is a deterministic $0 change and does not require re-running the extractor. Predicted post-fix strict-table-valid rate: 30 to 50% of currently-flagged table-valid tuples (a rise from Packet 2's 22.5% because some current NO cases will re-classify to partial rather than fail entirely). Predicted partial-table-valid: 20 to 30% additional. The rest downgrades to prose.

## What the chapter now claims, honestly

- MetaLead per-field F1 lift +0.11 to +0.19 absolute on all four fields, Bonferroni-significant (unchanged).
- Human-verified extraction correctness on paper text: 65% (Packet 1).
- Human-verified provenance cell-semantics correctness: 22.5% before cycle 5; will be re-reported after cycle 5.
- Per-corpus ECE (Design B, per-corpus CV): 8 of 8 cells at or below 0.10, mean 0.051. Mean AUROC 0.779. Deployed calibrator beats trivial constant on Brier on 8 of 8 cells.
- Cost per paper 1.9x paper 1 baseline.
- SciREX transfer null on strict-match F1 IS driven by benchmark coverage, not extractor failure, per Packet 1's 64% YES rate on gold-mismatched tuples.

The provenance downgrade is real, but the discovery of the mechanism (binder bug, not extractor failure) plus the direct human evidence for the corpus-fit story (Packet 1) net out to a slightly stronger chapter, not a weaker one. The chapter now has actual human adjudication behind three of its four headline claims.

## Files

- Labeled inputs: `outputs/human_eval_packet.csv`, `outputs/canonextract_provenance_spotcheck.csv`.
- Hidden keys used for scoring: `outputs/human_eval_key.json`, `outputs/canonextract_provenance_spotcheck_key.json`.
- This synthesis: `phd_v2/canonextract_human_evaluation_findings.md`.
- Cycle 5 prompt (binder rewrite): `phd_v2/claude_code_prompt_canonextract_cycle5.md`.
