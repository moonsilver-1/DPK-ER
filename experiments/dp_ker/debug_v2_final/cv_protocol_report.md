# CV Protocol Report

- Total rows: 815
- Positive labels: 150
- Negative labels: 665

## Protocols

- Row-level StratifiedKFold: 5 folds, seed=3407.
- Hold-out split: 60/20/20 design, final test uses 20% untouched data.

## Group Split Notes

- Group column `source_file` has no repeated groups (815 unique groups for 815 rows). GroupKFold would be equivalent to row-level splitting, so it was skipped.
