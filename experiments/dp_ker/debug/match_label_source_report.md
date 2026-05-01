# Match Label Source Report

This report audits whether the local raw files contain a true resume-JD matching label. It does not use `valid_resume_and_jd` as a match label.

## Raw File Scope

- Local JSON files: 276
- Filename category counts:
  - match: 94
  - empty_additional_info: 40
  - invalid_resume: 55
  - invalid_job_description: 57
  - invalid_gibberish_resume: 15
  - invalid_gibberish_job_description: 15

## Field Findings

The JSON structure contains `input`, `output`, and `details`. The relevant scoring fields are under `output.scores`, especially `output.scores.aggregated_scores.macro_scores` and `output.scores.aggregated_scores.micro_scores`. The field `output.valid_resume_and_jd` exists, but it is a validity/evaluability flag, not a match label.

No explicit JSON field named `label`, `matched`, `mismatched`, `category`, `final_score`, or `requirement_score` was found as a top-level matching target in the local JSON files. The dataset card and README describe matched/mismatched/invalid/missing-info groups, and the local file names expose `match`, `invalid_*`, and `empty_additional_info` categories. The local subset does not contain non-invalid `mismatch_*.json` files.

## Label Decision

Priority 1 cannot be used for the local subset because there is no explicit paired `matched`/`mismatched` field and no non-invalid mismatched filename group in the available files.

Priority 2 is used: score-threshold pseudo labels derived from aggregated scores. For `match_*.json` files only, `raw_score = mean(aggregated macro score, aggregated micro score)`. The binary matching label is:

- `raw_score >= 7.0` -> `label=1`
- `raw_score < 7.0` -> `label=0`

This is a threshold-based pseudo label, not an explicit human-provided matched/mismatched label.

Invalid files and missing-additional-info files are excluded from the matching experiments.

## Fixed Label Counts

- label_source: `score_threshold`
- matched / positive pseudo labels: 23
- mismatched / negative pseudo labels: 71
- included matching rows: 94
- excluded invalid rows: 142
- excluded missing-info rows: 40

## Stop Condition

The local data supports only threshold-based pseudo matching labels. It does not support a strict claim that the fixed labels are original ground-truth matched/mismatched annotations. If the paper requires explicit matched/mismatched labels, the full dataset or additional metadata must be restored before final experiments.
