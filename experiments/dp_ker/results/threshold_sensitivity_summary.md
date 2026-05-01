# Threshold Sensitivity Analysis

This experiment rebuilds score-threshold pseudo labels at thresholds 6.5, 7.0, and 7.5 without modifying existing fixed result CSV files.

## Label Counts

| threshold | positive_count | negative_count | total_count |
| --- | --- | --- | --- |
| 6.5000 | 36.0000 | 58.0000 | 94.0000 |
| 7.0000 | 23.0000 | 71.0000 | 94.0000 |
| 7.5000 | 15.0000 | 79.0000 | 94.0000 |

## Pairwise Results

| threshold | Method | AUC_mean | AUC_std | Accuracy_mean | F1_mean | Precision_mean | Recall_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5000 | TF-IDF pair similarity | 0.6193 | 0.0000 | 0.6064 | 0.4478 | 0.4839 | 0.4167 |
| 6.5000 | BM25 pair similarity | 0.6509 | 0.0000 | 0.6170 | 0.1429 | 0.5000 | 0.0833 |
| 6.5000 | DP-Embedding pair similarity | 0.5570 | 0.0492 | 0.5355 | 0.5156 | 0.4279 | 0.6574 |
| 6.5000 | DP-KER full pair score | 0.5694 | 0.0434 | 0.5035 | 0.4966 | 0.4119 | 0.6389 |
| 7.0000 | TF-IDF pair similarity | 0.6509 | 0.0000 | 0.6809 | 0.4444 | 0.3871 | 0.5217 |
| 7.0000 | BM25 pair similarity | 0.7275 | 0.0000 | 0.7553 | 0.2069 | 0.5000 | 0.1304 |
| 7.0000 | DP-Embedding pair similarity | 0.5499 | 0.0663 | 0.4752 | 0.3754 | 0.2647 | 0.6522 |
| 7.0000 | DP-KER full pair score | 0.5593 | 0.0081 | 0.4716 | 0.3748 | 0.2657 | 0.6522 |
| 7.5000 | TF-IDF pair similarity | 0.6515 | 0.0000 | 0.6809 | 0.3478 | 0.2581 | 0.5333 |
| 7.5000 | BM25 pair similarity | 0.6667 | 0.0000 | 0.8404 | 0.2857 | 0.5000 | 0.2000 |
| 7.5000 | DP-Embedding pair similarity | 0.5598 | 0.0221 | 0.4681 | 0.2994 | 0.1902 | 0.7111 |
| 7.5000 | DP-KER full pair score | 0.5451 | 0.0386 | 0.5071 | 0.2443 | 0.1636 | 0.4889 |

## Sampled Ranking Results

| threshold | Method | NegativeType | Recall@5_mean | Recall@5_std | NDCG@5_mean | NDCG@5_std | MRR_mean | Hit@1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5000 | TF-IDF+Cosine | random | 0.2407 | 0.0346 | 0.1591 | 0.0088 | 0.2075 | 0.0741 |
| 6.5000 | BM25 | random | 0.2315 | 0.0131 | 0.1377 | 0.0159 | 0.1765 | 0.0463 |
| 6.5000 | DP-Embedding | random | 0.2963 | 0.0693 | 0.1975 | 0.0405 | 0.2322 | 0.1019 |
| 6.5000 | DP-KER | random | 0.3333 | 0.0600 | 0.1943 | 0.0208 | 0.2081 | 0.0648 |
| 6.5000 | TF-IDF+Cosine | hard | 0.1111 | 0.0000 | 0.0767 | 0.0000 | 0.1159 | 0.0278 |
| 6.5000 | BM25 | hard | 0.6111 | 0.0000 | 0.4759 | 0.0000 | 0.4713 | 0.3333 |
| 6.5000 | DP-Embedding | hard | 0.2778 | 0.0600 | 0.1696 | 0.0415 | 0.2037 | 0.0648 |
| 6.5000 | DP-KER | hard | 0.3704 | 0.1164 | 0.2101 | 0.0807 | 0.2106 | 0.0648 |
| 7.0000 | TF-IDF+Cosine | random | 0.2464 | 0.0410 | 0.1644 | 0.0138 | 0.2122 | 0.0870 |
| 7.0000 | BM25 | random | 0.2754 | 0.0542 | 0.1460 | 0.0179 | 0.1652 | 0.0290 |
| 7.0000 | DP-Embedding | random | 0.3188 | 0.1247 | 0.2033 | 0.0912 | 0.2251 | 0.1014 |
| 7.0000 | DP-KER | random | 0.3478 | 0.1280 | 0.1702 | 0.0657 | 0.1680 | 0.0145 |
| 7.0000 | TF-IDF+Cosine | hard | 0.0870 | 0.0000 | 0.0405 | 0.0000 | 0.0763 | 0.0000 |
| 7.0000 | BM25 | hard | 0.4783 | 0.0000 | 0.4214 | 0.0000 | 0.4511 | 0.3478 |
| 7.0000 | DP-Embedding | hard | 0.2754 | 0.0893 | 0.1840 | 0.0982 | 0.2224 | 0.0870 |
| 7.0000 | DP-KER | hard | 0.3768 | 0.2255 | 0.2185 | 0.1636 | 0.2181 | 0.0725 |
| 7.5000 | TF-IDF+Cosine | random | 0.2889 | 0.0314 | 0.1902 | 0.0312 | 0.2278 | 0.1111 |
| 7.5000 | BM25 | random | 0.3333 | 0.0544 | 0.1911 | 0.0321 | 0.2014 | 0.0667 |
| 7.5000 | DP-Embedding | random | 0.2667 | 0.0544 | 0.1737 | 0.0463 | 0.2153 | 0.0667 |
| 7.5000 | DP-KER | random | 0.3556 | 0.1133 | 0.2203 | 0.0775 | 0.2340 | 0.0889 |
| 7.5000 | TF-IDF+Cosine | hard | 0.0667 | 0.0000 | 0.0333 | 0.0000 | 0.0742 | 0.0000 |
| 7.5000 | BM25 | hard | 0.6000 | 0.0000 | 0.3977 | 0.0000 | 0.3650 | 0.2000 |
| 7.5000 | DP-Embedding | hard | 0.3111 | 0.0831 | 0.2010 | 0.0798 | 0.2322 | 0.0889 |
| 7.5000 | DP-KER | hard | 0.4889 | 0.2200 | 0.3278 | 0.1737 | 0.3231 | 0.1778 |

## DP-KER vs DP-Embedding Deltas

| threshold | pairwise_auc_delta | random_recall_delta | random_ndcg_delta | hard_recall_delta | hard_ndcg_delta |
| --- | --- | --- | --- | --- | --- |
| 6.5000 | 0.0125 | 0.0370 | -0.0032 | 0.0926 | 0.0405 |
| 7.0000 | 0.0094 | 0.0290 | -0.0330 | 0.1014 | 0.0345 |
| 7.5000 | -0.0146 | 0.0889 | 0.0466 | 0.1778 | 0.1268 |

## Stability Audit

- Pairwise AUC: not stable; DP-KER is compared against DP-Embedding at each threshold.
- Random sampled ranking Recall@5/NDCG@5: not stable.
- Hard sampled ranking Recall@5/NDCG@5: stable.

The result should be interpreted as sensitivity analysis for pseudo labels, not as validation against explicit matched/mismatched ground truth. If space is limited, this is better used as a short robustness sentence or appendix-style table rather than a main claim table.
