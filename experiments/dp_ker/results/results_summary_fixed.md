# DP-KER Fixed Experimental Results Summary

This summary uses the repaired label protocol. The old `valid_resume_and_jd`-based pairwise/ranking/ablation/privacy results must not be used as paper main results.

## Label Fix

`valid_resume_and_jd` is no longer used as a matching label. The local JSON files do not contain an explicit ground-truth `matched`/`mismatched` field, and the available local subset has no non-invalid `mismatch_*.json` files. Therefore, the fixed labels use a score-threshold pseudo-label protocol based on `output.scores.aggregated_scores`:

- `raw_score = mean(macro aggregated score, micro aggregated score)`
- `raw_score >= 7.0` -> positive match (`label=1`)
- `raw_score < 7.0` -> negative match (`label=0`)
- invalid and missing-additional-info samples are excluded

See `experiments/dp_ker/debug/match_label_source_report.md` for the full audit.

## Dataset Statistics Fixed

| Metric | Value |
|---|---:|
| raw_files | 276 |
| profiles_rows | 276 |
| candidates_rows | 276 |
| labels_rows | 94 |
| matched_label_1 | 23 |
| mismatched_label_0 | 71 |
| excluded_invalid_rows | 142 |
| excluded_missing_info_rows | 40 |
| label_source | score_threshold |

## Fixed Label Score Distribution

| label | count | raw_score_min | raw_score_max | raw_score_mean | raw_score_median |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 71.0000 | 1.0000 | 6.9500 | 5.1072 | 5.4700 |
| 1.0000 | 23.0000 | 7.0050 | 9.0000 | 7.7822 | 7.6500 |

## Pairwise Matching Results Fixed

| Method | AUC_mean | Accuracy_mean | F1_mean | Precision_mean | Recall_mean | Spearman_mean | Pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF pair similarity | 0.6509 | 0.6809 | 0.4444 | 0.3871 | 0.5217 | 0.4538 | 0.4978 |
| BM25 pair similarity | 0.7275 | 0.7553 | 0.2069 | 0.5000 | 0.1304 | 0.3885 | 0.4119 |
| tfidf-svd pair similarity | 0.6497 | 0.6277 | 0.4262 | 0.3421 | 0.5652 | 0.4281 | 0.4677 |
| KG-Enhanced pair score | 0.5211 | 0.5319 | 0.3333 | 0.2558 | 0.4783 | 0.2117 | 0.2826 |
| DP-Embedding pair similarity | 0.5499 | 0.4752 | 0.3754 | 0.2647 | 0.6522 | 0.0706 | 0.0874 |
| DP-KER full pair score | 0.5593 | 0.4716 | 0.3748 | 0.2657 | 0.6522 | 0.1258 | 0.1170 |

### Pairwise Audit

After fixing labels and correcting score direction where necessary, AUC values are no longer systematically below 0.5. `DP-KER full pair score` has AUC 0.5593 versus `DP-Embedding pair similarity` AUC 0.5499. Thus DP-KER is slightly higher on AUC in this fixed pairwise run, but not clearly better across all metrics: Accuracy, F1, Precision, and Recall remain very close or mixed. This should be reported cautiously.

Pairwise consistency is intentionally removed because there is no method-dependent generated explanation in this protocol.

## Score Direction Check

| Method | AUC_score_label | AUC_negative_score_label | used_direction |
| --- | --- | --- | --- |
| TF-IDF pair similarity | 0.6509 | 0.3491 | score |
| BM25 pair similarity | 0.7275 | 0.2725 | score |
| tfidf-svd pair similarity | 0.6497 | 0.3503 | score |
| KG-Enhanced pair score | 0.5211 | 0.4789 | score |
| DP-Embedding pair similarity seed=3407 | 0.3564 | 0.6436 | negative_score |
| DP-KER full pair score seed=3407 | 0.4391 | 0.5609 | negative_score |
| DP-Embedding pair similarity seed=42 | 0.4942 | 0.5058 | negative_score |
| DP-KER full pair score seed=42 | 0.5487 | 0.4513 | score |
| DP-Embedding pair similarity seed=2026 | 0.5003 | 0.4997 | score |
| DP-KER full pair score seed=2026 | 0.5683 | 0.4317 | score |

The direction issue was real for some DP seed-specific scores. The fixed pairwise pipeline now uses the better direction when `AUC(-score,label)` exceeds `AUC(score,label)`.

## Sampled Ranking Results Fixed

| Method | NegativeType | Recall@5_mean | NDCG@5_mean | MRR_mean | Hit@1_mean | Consistency_mean |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF+Cosine | random | 0.2464 | 0.1644 | 0.2122 | 0.0870 | 0.3696 |
| BM25 | random | 0.2754 | 0.1460 | 0.1652 | 0.0290 | 0.3603 |
| Embedding | random | 0.3188 | 0.1982 | 0.2271 | 0.1014 | 0.3734 |
| KG-Enhanced | random | 0.2319 | 0.1244 | 0.1543 | 0.0435 | 0.4147 |
| DP-Embedding | random | 0.3188 | 0.2033 | 0.2251 | 0.1014 | 0.3466 |
| DP-KER | random | 0.3478 | 0.1702 | 0.1680 | 0.0145 | 0.3591 |
| TF-IDF+Cosine | hard | 0.0870 | 0.0405 | 0.0763 | 0.0000 | 0.3759 |
| BM25 | hard | 0.4783 | 0.4214 | 0.4511 | 0.3478 | 0.3507 |
| Embedding | hard | 0.1304 | 0.0660 | 0.0938 | 0.0000 | 0.3755 |
| KG-Enhanced | hard | 0.1739 | 0.1057 | 0.1425 | 0.0435 | 0.4142 |
| DP-Embedding | hard | 0.2754 | 0.1840 | 0.2224 | 0.0870 | 0.3561 |
| DP-KER | hard | 0.3768 | 0.2185 | 0.2181 | 0.0725 | 0.3729 |

### Sampled Ranking Audit

Under random negatives, DP-KER improves Recall@5 over DP-Embedding but has lower NDCG@5, MRR, and Hit@1. Under hard negatives, DP-KER improves Recall@5 and NDCG@5 over DP-Embedding, while MRR and Hit@1 remain comparable or weaker. KG-related methods improve consistency over plain embedding in both random and hard settings; the KG-Enhanced consistency gain over Embedding is 0.0413 for random negatives and 0.0387 for hard negatives.

## Ablation Results Fixed

| Method | AUC_mean | Accuracy_mean | F1_mean | Precision_mean | Recall_mean | Spearman_mean | Pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.5593 | 0.4716 | 0.3748 | 0.2657 | 0.6522 | 0.1258 | 0.1170 |
| w/o_dp | 0.6069 | 0.6170 | 0.3333 | 0.2903 | 0.3913 | 0.3667 | 0.4555 |
| w/o_knowledge | 0.5499 | 0.4752 | 0.3754 | 0.2647 | 0.6522 | 0.0706 | 0.0874 |
| w/o_ranking_fusion | 0.5227 | 0.4929 | 0.3497 | 0.2580 | 0.5652 | -0.0071 | -0.0068 |
| retrieval_only | 0.6509 | 0.6809 | 0.4444 | 0.3871 | 0.5217 | 0.4538 | 0.4978 |

### Ablation Audit

The full DP-KER score is slightly stronger than `w/o_knowledge` on AUC (0.5593 vs 0.5499), but weaker than `w/o_dp` and `retrieval_only`. This supports only a limited claim: KG/fusion can modestly improve the noisy DP representation on AUC, but the privacy mechanism still carries a utility cost.

## Privacy Budget Results Fixed

| epsilon | sigma | AUC_mean | Accuracy_mean | F1_mean | Precision_mean | Recall_mean | Spearman_mean | Pearson_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 9.6896 | 0.5597 | 0.4681 | 0.3729 | 0.2637 | 0.6522 | 0.1254 | 0.1185 |
| 1.0000 | 4.8448 | 0.5593 | 0.4681 | 0.3729 | 0.2637 | 0.6522 | 0.1254 | 0.1178 |
| 2.0000 | 2.4224 | 0.5593 | 0.4716 | 0.3748 | 0.2657 | 0.6522 | 0.1258 | 0.1170 |
| 4.0000 | 1.2112 | 0.5575 | 0.4681 | 0.3726 | 0.2632 | 0.6522 | 0.1243 | 0.1168 |
| 8.0000 | 0.6056 | 0.5528 | 0.4574 | 0.3677 | 0.2590 | 0.6522 | 0.1255 | 0.1165 |

### Privacy Audit

The sigma schedule is correct: epsilon increases while sigma decreases. However, the utility metrics do not improve monotonically with larger epsilon. AUC is nearly flat from epsilon 0.5 to 2.0 and then slightly decreases at larger epsilon values. This means the fixed experiment verifies the DP parameter relation, but it does not show a clean privacy-utility trend.

## Final Answers

1. Fixed label status: not an explicit original matched/mismatched label; it is a score-threshold pseudo matching label from aggregated scores.
2. Fixed label counts: 23 positive and 71 negative samples.
3. Invalid/missing-info handling: 142 invalid rows and 40 missing-info rows were excluded from matching experiments.
4. DP-KER vs DP-Embedding: DP-KER is slightly higher on fixed pairwise AUC, but not consistently better across all pairwise metrics. In sampled ranking, it is stronger on some ranking metrics but not all.
5. KG and consistency: consistency should not be compared in pairwise results. In sampled ranking, KG-Enhanced improves method-dependent top-k consistency over Embedding.
6. Epsilon trend: sigma follows the expected privacy relation, but utility does not show a clean monotonic improvement as epsilon increases.

## Which Tables Can Be Used

- Keep: `dataset_statistics_fixed.csv`, `pairwise_results_fixed.csv`, `sampled_ranking_results_fixed.csv`, `ablation_results_fixed.csv`, and `privacy_budget_results_fixed.csv`, with the pseudo-label caveat.
- Do not use as paper main results: old `pairwise_results.csv`, `sampled_ranking_results.csv`, `ablation_results.csv`, `privacy_budget_results.csv`, and deprecated full-retrieval results.
- Rerun needed before strong claims: restore the full 1,031-sample dataset or obtain explicit matched/mismatched metadata.
