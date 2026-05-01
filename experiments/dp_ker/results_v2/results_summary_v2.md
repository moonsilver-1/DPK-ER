# DP-KER v2 Results Summary

- Data source used: `hf_cache_snapshot`
- Loaded rows before filtering: 1031
- Valid scored rows used: 815
- CV protocol: 5-fold stratified CV; seeds=[3407, 42, 2026]
- No target leakage check: `raw_score` and threshold labels are targets only and are not included in the feature matrix.

## Paper Judgment

1. Data scale usable for paper: yes.
2. DP-KER-v2 vs DP-Embedding: yes on binary AUC (0.7421 vs 0.5200).
3. DP-KER-v2 vs non-DP baselines: yes on binary AUC.
4. Knowledge feature contribution: positive by full vs w/o KG AUC (0.7421 vs 0.7336).
5. DP utility cost: compare DP-Embedding/DP-KER-v2 against non-DP embedding and w/o DP ablation in the CSV tables.
6. Epsilon trend reasonable: yes based on DP-KER-v2 AUC from epsilon 0.5 to 8.
7. EI conference support: yes as a cautious experimental paper if limitations are stated.
8. SCI Q4 minimum experimental strength: closer, but still needs full data and stronger trends.

## Main Files

- `dataset_statistics_v2.csv`
- `score_prediction_results_v2.csv`
- `binary_matching_results_v2.csv`
- `sampled_ranking_results_v2.csv`
- `ablation_results_v2.csv`
- `privacy_budget_results_v2.csv`
- `threshold_sensitivity_results_v2.csv`
