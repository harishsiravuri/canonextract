# Attribution key recovery for the whole-record blind packet

The original `whole_record_packet_key.json` was not committed to the
repository and is not recoverable from history. The 100 blind-labeled
rows do carry full record content, so attribution has been rebuilt by
deterministic content join against the frozen per-system record
outputs. No new LLM calls were issued. No row attribution is guessed.

## a. CanonExtract output directory chosen

The CanonExtract side of the join uses
`contribution_extraction_paper1_icmla/canonextract_build/outputs/eval_runs_metalead_cycle5/`.
This directory holds the cycle-5 provenance pass, which is the
provenance layer the cycle-6 deployed configuration reads through.
The three candidate directories (`eval_runs_metalead_cycle5`,
`eval_runs_metalead_cycle6`, `eval_runs_metalead_merged`) were
compared on paper `1812.09471`, which is the example named in the
task brief. All three were observed to carry the same `tuples` list
for that paper (14 tuples, identical method, task, dataset, metric
and value fields). Cycle-5 was selected because it is the layer the
blind packet's record shapes most closely mirror (for example the
`intent detection and slot filling` task phrasing used in blind row
`W001` appears verbatim in the cycle-5 output for `1812.09471`). The
baseline side uses `baseline_metalead/`, as named in the task brief.

## b. Join method

Each system's per-paper JSON was flattened into one record per
(method, task, dataset, metric, value) combination:

- CanonExtract: iterate `tuples[]`; for each tuple, cross `datasets[]`
  with `metrics[]`.
- Baseline: iterate `contributions[]`; for each contribution, cross
  `datasets[]` with `metrics[]`.
- When a tuple or contribution carries datasets but no metrics (which
  occurs for prose records such as the UniLM GLUE and CoQA
  contributions in `baseline_metalead/metalead_1905.03197.json`), one
  record per dataset is still emitted, with the metric and value held
  as `None`. This preserves joinability for the four blind rows whose
  metric and value cells are themselves empty (`W034`, `W050`, `W058`,
  `W075`).

Two match passes were applied in order:

- v1 strict exact match on the 5-tuple
  `(method, task, dataset, metric, value)` after whitespace and case
  normalization. Values parseable as floats are compared as floats;
  otherwise the lowercased raw string is used.
- v2 lenient fallback, applied only when a system failed v1 on a
  given row. v2 requires exact `method` and `task`, substring match
  either direction on `dataset`, substring match either direction on
  `metric`, and numeric value within 1 percent relative tolerance.
  v2 is used specifically to absorb cosmetic differences such as
  `PubMed` vs `PubMed Dataset`, `SemEval 2010` vs `SemEval 2010 WSI
  dataset`, and `F1` vs `Paired F1`, which show up only for paper
  `1811.09242`.

Paper identifiers were used only to scope the per-paper file; the
`paper_id` is required to match exactly, which it does in all 100
rows.

## c. Attribution histogram

Under the strict v1 join (counts out of 100):

- CanonExtract only: 47
- Baseline only: 46
- Both systems (shared record): 7
- Neither system: 0

With the lenient v2 fallback applied only to v1-unmatched systems:

- CanonExtract only: 47
- Baseline only: 42
- Both systems (shared record): 11
- Neither system: 0

The lenient pass does not change which rows are attributed; it only
promotes four rows from `baseline_only` to `shared_record` by
accepting that CanonExtract's `Paired F1` on `PubMed Dataset` is the
same cell as Baseline's `F1` on `PubMed` for paper `1811.09242`.

## d. Ambiguity analysis

Of the 100 rows, 89 are clean single-system matches, 11 are shared
records produced by both systems, and 0 are unmatched. The shared
set breaks down as follows:

- 7 rows match both systems exactly under v1 strict. Examples:
  `W011` (`1703.06345`, hierarchical RNN, PTB, F1, 91.26) and
  `W099` (`1511.08308`, BLSTM-CNN, CoNLL-2003 NER, F1, 91.62). These
  are the uncomplicated case: both systems emitted the same cell
  with the same labels.
- 4 rows match both systems only under v2 lenient. All four are from
  paper `1811.09242` (AutoSense). CanonExtract emits dataset names
  that carry a `Dataset` or `WSI dataset` suffix and metric names
  prefixed `Paired`; the baseline emits the shorter forms. The
  numeric values are identical and the (method, task) pair is the
  same, so these are treated as legitimately shared.

The 11 shared rows are listed in `ambiguous_rows.csv`. No row in the
packet is "truly ambiguous" in the sense of matching two inconsistent
records; every ambiguity here is a shared-emission ambiguity, not a
labelling ambiguity.

## e. Readiness for yield computation

Yield is not computed in this run. The following observations bear
on whether yield computation can proceed:

- All 100 blind rows are attributable to at least one system under
  the strict v1 join. There are no unmatched or orphan rows that
  would force a judgment call.
- 11 of 100 rows are shared records that cannot be attributed to a
  single system without a tie-breaking rule. 7 of those are exact
  shared matches and 4 are lenient-only shared matches.
- The ratio of lenient-only shared matches to total rows is 4 of
  100. All 4 come from a single paper, which is a concentration
  worth noting rather than a cross-corpus pattern.

A reasonable threshold for proceeding is that fewer than roughly 15
of 100 rows require a shared-record policy, and that no row is left
unattributed. Both conditions hold in this reconstruction. The
decision Harish needs to make before yield is computed is the
shared-record policy itself: whether shared rows (a) count once for
each system, (b) are excluded from the per-system yield
denominators, or (c) are split by a documented tie-breaker. The
reconstruction does not pick that policy; it only exposes the 11
shared rows for the decision to be made over.

## Artifacts

- `reconstruct_key.py`: the deterministic join script.
- `reconstructed_key.json`: per-row attribution, 100 records.
- `ambiguous_rows.csv`: the 11 non-clean rows with full packet columns
  plus `matched_systems` and `ambiguity_class`.
- `MATCH_RATE_REPORT.md`: this document.
