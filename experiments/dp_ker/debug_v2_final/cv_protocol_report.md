# CV Protocol Report

- Total rows: 815
- Positive labels: 150
- Negative labels: 665

## Protocols

- Row-level StratifiedKFold: 5 folds, seed=3407.
- Repeated hold-out: seeds=[3407, 42, 2026, 2027, 2028].
- Each hold-out run uses 80% training pool and 20% untouched test set.
- Model selection is performed inside the training pool; test sets are not used for model selection.

## Group Split Notes

- Group column `source_file` has no repeated groups (815 unique groups for 815 rows). GroupKFold would be equivalent to row-level splitting, so it was skipped.
