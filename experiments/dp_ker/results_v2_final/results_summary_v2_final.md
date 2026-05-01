# DP-KER v2 Final Summary

- Raw rows recovered: 1031
- Valid/scored rows used: 815
- Positive / negative at threshold 7.0: 150 / 665
- Data source: hf_cache_snapshot
- Feature cache: `F:\tu\DP-KER\lab\experiments\dp_ker\cache_v2_final\pair_features.parquet`

## Completion Status

- core_prediction: completed
- sampled_ranking: completed
- privacy_budget: completed
- threshold_sensitivity: completed

## Required Answers

1. Uses 1031 raw / 815 valid scored data: yes.
2. Positive / negative counts: 150 / 665.
3. DP-KER-v2 vs DP-Embedding: yes on binary AUC (0.8306 vs 0.4896).
4. DP-KER-v2 vs non-DP baseline: yes on binary AUC.
5. Learned fusion vs hand fusion: yes on binary AUC.
6. KG feature contribution: yes by full vs w/o KG.
7. DP utility cost: visible by w/o DP vs full.
8. Epsilon trend reasonable: yes.
9. Stronger than original fixed EI version: yes.
10. SCI Q4 minimum strength: closer if all final-mode tables are complete and trends hold.

## Notes

- `raw_score` and `binary_label` are stored in the cache for evaluation only and are excluded from model feature columns.
- DP is embedding-level Gaussian perturbation, not DP-SGD.
- If a table is marked missing, run its corresponding `src_v2` script and then rerun this summarizer.
