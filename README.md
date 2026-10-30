# CanonExtract

Code and data for *Table-Aware Extraction of Scientific Results* by Harish
Varma Siravuri and Hamed Alhoori (Northern Illinois University), to appear
at the 6th International Workshop on Big Data and AI-Assisted Tools (BTSD),
IEEE BigData 2026.

CanonExtract is an extension of the prose-only multi-agent extractor from
our prior ICMLA 2026 paper. It adds a table parser, a per-paper vocabulary
index, source-reference assignment on every emitted result record, and a
per-tuple verifier that reviews candidate strict-category records.

## What is in this repository

```
pipeline/            Table parser, vocabulary builder, binder, verifier,
                     the Extractor/Critic/Consolidator prompts, and the
                     orchestration script.
scoring/             The scorer (TDM-triple F1), bootstrap and permutation
                     tests, and the compute-matched self-consistency
                     analysis script.
baselines/           Per-paper outputs from the two baselines on MetaLead:
                     single_call_metalead/ and
                     five_call_self_consistency_metalead/.
records/             Per-paper outputs from the deployed CanonExtract
                     configuration on MetaLead and SciREX, plus the
                     six-category provenance band assignments.
human_eval/          Both human-rated packets described in the paper:
                     packet1_per_record/ (49 records, per-record
                     correctness) and packet2_cell_assignment/ (40
                     records, cell-assignment correctness). FINDINGS.md
                     summarises both packets.
calibration/         Per-field expected calibration error on MetaLead
                     and the SciREX-fit temperature-scaling parameter.
key_recovery/        Scripts and artifacts for reconstructing the Packet 1
                     attribution key via deterministic content join.
CLAIMS_TO_SOURCE.md  Every number in the paper traced to its source file.
```

## Reproducing the paper numbers

The deployed configuration uses DeepSeek Chat as the Extractor at
temperatures {0, 0.3, 0.7} and Llama 3.3 70B at temperature 0 as the
Critic, Consolidator, and per-tuple verifier. Both are hosted on
OpenRouter. All sampling, bootstrap, and random-split seeds are fixed
at 42.

Per-paper record outputs for both corpora are released in `records/`.
Baseline per-paper outputs are released in `baselines/`. These are the
inputs the scoring code reads; no network calls are needed to reproduce
the reported numbers once the per-system record outputs are present.

To re-score with the released scorer:

```bash
cd scoring
python canonextract_score.py \
  --predictions ../records/metalead_cycle8a \
  --baseline ../baselines/single_call_metalead \
  --seed 42
```

The scoring code is deterministic given the seed. The paired delta is
the per-paper mean of (system A F1) minus (system B F1), with 1000-
resample paper-clustered bootstrap 95 % confidence intervals.

## Model and corpus notes

- **Corpora.** MetaLead 2026 is the primary extraction evaluation corpus
  (100-experiment subsample spanning 27 papers). SciREX test (66 papers)
  is used as a transfer check and for calibration fitting.
- **Model versions.** DeepSeek Chat as the Extractor; Llama 3.3 70B at
  temperature 0 as the Critic, Consolidator, and per-tuple verifier.
- **Hosting.** OpenRouter for both DeepSeek Chat and Llama 3.3 70B.
- **Paper-text cap.** 45,000 characters per paper for both baselines and
  the extended system.

## Dependencies

The pipeline scripts have minimal external dependencies: `requests`
(OpenRouter calls), `pydantic` (schema validation), `beautifulsoup4` and
`lxml` (table parsing from arXiv LaTeX source). The scoring code uses
`numpy` for the bootstrap. The temperature-scaling calibrator in
`scoring/metrics/` depends on `scipy` for `optimize.minimize`.

The modules under `pipeline/` and `scoring/` originally lived inside a
larger research codebase (`paper1/`). The versions in this repository
have had their `paper1.*` imports flattened to local modules where
possible; a few scripts still reference `paper1.openrouter_client` and
similar paths and will not run as-is without the original client.
These scripts are included for traceability and inspection rather than
turnkey execution.

## Citation

```
@inproceedings{siravuri2026btsd,
  author    = {Siravuri, Harish Varma and Alhoori, Hamed},
  title     = {Table-Aware Extraction of Scientific Results},
  booktitle = {6th International Workshop on Big Data and AI-Assisted Tools (BTSD)
               at IEEE BigData 2026},
  year      = {2026}
}
```

## Licenses

Code is released under Apache License 2.0 (see `LICENSE`). Data,
including the human-rated packets and per-paper record outputs, is
released under CC BY 4.0 (see `LICENSE-data`).

## Contact

Harish Varma Siravuri — hsiravuri@niu.edu
Hamed Alhoori — alhoori@niu.edu
