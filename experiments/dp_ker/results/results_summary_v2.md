# DP-KER Results Summary V2

This summary uses the repaired evaluation protocol. The old full-corpus Top-K retrieval result is deprecated because the dataset is pairwise resume-JD matching data, not a fully annotated recommendation corpus.

## Protocol Changes

- Deprecated old full-library ranking output: `deprecated_full_retrieval_results.csv`.
- New main result: `pairwise_results.csv`.
- New auxiliary ranking result: `sampled_ranking_results.csv`.
- Ablation and privacy-budget outputs were regenerated under the pairwise protocol.
- No CSV values were edited by hand; all reported numbers come from scripts.

## Dataset Scope

The available local raw directory contains 276 JSON samples. The dataset card says 1,031 samples, but the local Hugging Face cache also contains metadata for only 276 JSON files. The converter processes all local JSON files, so the 276-row converted dataset is not caused by filtering in `convert_hf_dataset.py`.

The full download attempt was interrupted by the user. Therefore, the current experiments must be described as using the available local subset.

See `dataset_debug_report.md` for the detailed data-range check.

## Table I. Dataset Statistics

| Metric | Value |
|---|---:|
| raw_files | 276 |
| profiles_rows | 276 |
| candidates_rows | 276 |
| labels_rows | 276 |
| knowledge_rows | 276 |
| valid_true | 131 |
| valid_false | 145 |
| empty_additional_info | 40 |

## Table II. Pairwise Matching Results

| Method | AUC | Accuracy | F1 | Precision | Recall | Spearman | Pearson | Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF pair similarity | 0.2305 | 0.1232 | 0.0000 | 0.0000 | 0.0000 | -0.4002 | -0.6859 | 0.4587 |
| BM25 pair similarity | 0.2649 | 0.5217 | 0.0000 | 0.0000 | 0.0000 | -0.3274 | -0.1507 | 0.4587 |
| TF-IDF/SVD embedding pair similarity | 0.2201 | 0.1341 | 0.0245 | 0.0263 | 0.0229 | -0.4266 | -0.6581 | 0.4587 |
| KG-Enhanced pair score | 0.2047 | 0.1304 | 0.0083 | 0.0090 | 0.0076 | -0.4555 | -0.5881 | 0.4587 |
| DP-Embedding pair similarity | 0.4594 +- 0.0385 | 0.4601 +- 0.0430 | 0.3964 +- 0.0617 | 0.4284 +- 0.0362 | 0.3868 +- 0.1170 | -0.0596 +- 0.0731 | -0.0481 +- 0.0622 | 0.4587 |
| DP-KER full pair score | 0.1674 +- 0.0106 | 0.2005 +- 0.0133 | 0.0669 +- 0.0183 | 0.0743 +- 0.0172 | 0.0611 +- 0.0187 | -0.5290 +- 0.0223 | -0.5193 +- 0.0289 | 0.4587 |

### Audit

The new pairwise protocol is more appropriate for the dataset, but the current DP-KER scoring formula is still not competitive. DP-Embedding is much stronger than DP-KER on AUC, Accuracy, F1, Precision, and Recall. This means the paper cannot claim that the current DP-KER implementation improves pairwise matching accuracy.

The correlation values are negative for most non-DP methods, which suggests the current heuristic scores are not aligned with the aggregated score labels. This is an important remaining modeling issue.

## Table III. Sampled Ranking Results

| Method | Negatives | Recall@5 | NDCG@5 | MRR | Hit@1 | Consistency |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF+Cosine | random | 0.2850 +- 0.0036 | 0.2023 +- 0.0033 | 0.2438 +- 0.0033 | 0.1221 +- 0.0062 | 0.3737 +- 0.0034 |
| BM25 | random | 0.1934 +- 0.0036 | 0.1447 +- 0.0025 | 0.2028 +- 0.0057 | 0.0992 +- 0.0062 | 0.3717 +- 0.0026 |
| Embedding | random | 0.2290 +- 0.0062 | 0.1678 +- 0.0067 | 0.2173 +- 0.0064 | 0.1069 +- 0.0108 | 0.3699 +- 0.0032 |
| KG-Enhanced | random | 0.3588 +- 0.0062 | 0.2419 +- 0.0009 | 0.2622 +- 0.0019 | 0.1272 +- 0.0036 | 0.4123 +- 0.0041 |
| DP-Embedding | random | 0.2112 +- 0.0180 | 0.1213 +- 0.0166 | 0.1603 +- 0.0150 | 0.0356 +- 0.0130 | 0.3344 +- 0.0029 |
| DP-KER | random | 0.2977 +- 0.0108 | 0.1992 +- 0.0028 | 0.2287 +- 0.0024 | 0.0967 +- 0.0036 | 0.3919 +- 0.0041 |
| TF-IDF+Cosine | hard | 0.1221 | 0.0915 | 0.1292 | 0.0611 | 0.4573 |
| BM25 | hard | 0.1527 | 0.0950 | 0.1342 | 0.0458 | 0.4480 |
| Embedding | hard | 0.1298 | 0.0931 | 0.1282 | 0.0534 | 0.4570 |
| KG-Enhanced | hard | 0.2290 | 0.1572 | 0.1896 | 0.0840 | 0.4856 |
| DP-Embedding | hard | 0.2316 +- 0.0307 | 0.1334 +- 0.0213 | 0.1678 +- 0.0184 | 0.0483 +- 0.0190 | 0.3588 +- 0.0009 |
| DP-KER | hard | 0.2087 +- 0.0259 | 0.1241 +- 0.0113 | 0.1583 +- 0.0047 | 0.0458 +- 0.0062 | 0.4671 +- 0.0030 |

### Audit

The sampled ranking protocol is more reasonable than full-library Top-K retrieval. Under random negatives, DP-KER improves over DP-Embedding on Recall@5, NDCG@5, MRR, Hit@1, and Consistency. Under hard negatives, DP-KER improves Consistency substantially but is weaker than DP-Embedding on ranking metrics. This supports a cautious claim: knowledge enhancement helps explanation consistency and can improve sampled ranking under easier negative sampling, but it does not robustly dominate DP-Embedding under hard negatives.

## Table IV. Ablation Study

| Setting | AUC | Accuracy | F1 | Precision | Recall | Consistency |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.1674 +- 0.0106 | 0.2005 +- 0.0133 | 0.0669 +- 0.0183 | 0.0743 +- 0.0172 | 0.0611 +- 0.0187 | 0.4587 |
| w/o_dp | 0.2151 | 0.1377 | 0.0246 | 0.0265 | 0.0229 | 0.4587 |
| w/o_knowledge | 0.4594 +- 0.0385 | 0.4601 +- 0.0430 | 0.3964 +- 0.0617 | 0.4284 +- 0.0362 | 0.3868 +- 0.1170 | 0.4587 |
| w/o_ranking_fusion | 0.1490 +- 0.0085 | 0.2053 +- 0.0181 | 0.2385 +- 0.0447 | 0.2176 +- 0.0340 | 0.2646 +- 0.0599 | 0.4587 |
| retrieval_only | 0.2305 | 0.1232 | 0.0000 | 0.0000 | 0.0000 | 0.4587 |

### Audit

The pairwise ablation does not support the current full scoring formula. The `w/o_knowledge` variant, which is effectively DP-Embedding, is much stronger on core classification metrics. The current pairwise consistency implementation is pair-level and does not vary by method, so it should not be used to claim ablation-level explanation improvements. For consistency claims, Table III is more informative.

## Table V. Privacy Budget Analysis

| epsilon | sigma | AUC | Accuracy | F1 | Precision | Recall | Consistency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 9.6896 | 0.1671 +- 0.0101 | 0.2053 +- 0.0133 | 0.0671 +- 0.0219 | 0.0748 +- 0.0206 | 0.0611 +- 0.0225 | 0.4587 |
| 1.0 | 4.8448 | 0.1674 +- 0.0103 | 0.2041 +- 0.0120 | 0.0697 +- 0.0215 | 0.0774 +- 0.0204 | 0.0636 +- 0.0219 | 0.4587 |
| 2.0 | 2.4224 | 0.1674 +- 0.0106 | 0.2005 +- 0.0133 | 0.0669 +- 0.0183 | 0.0743 +- 0.0172 | 0.0611 +- 0.0187 | 0.4587 |
| 4.0 | 1.2112 | 0.1672 +- 0.0108 | 0.1969 +- 0.0163 | 0.0695 +- 0.0122 | 0.0769 +- 0.0107 | 0.0636 +- 0.0130 | 0.4587 |
| 8.0 | 0.6056 | 0.1655 +- 0.0102 | 0.1860 +- 0.0197 | 0.0636 +- 0.0085 | 0.0699 +- 0.0067 | 0.0585 +- 0.0095 | 0.4587 |

### Audit

The DP implementation now clearly satisfies the expected sigma relation: epsilon increases from 0.5 to 8.0 while sigma decreases from 9.6896 to 0.6056. However, the matching metrics do not improve monotonically with larger epsilon. This suggests that the current DP-KER score is dominated by the weak KG/fusion formulation rather than showing a clean privacy-utility curve.

## What Was Fixed

- The old full-library Top-K retrieval output is no longer used as the main result.
- `main_results.csv` was renamed to `deprecated_full_retrieval_results.csv`.
- A pairwise matching protocol was added for the resume-JD pair dataset.
- A sampled ranking protocol with random and hard negatives was added.
- DP noise now supports L2 clipping, Gaussian noise, normalize option, deterministic seed, and multi-seed reporting.
- The sigma schedule is correct: larger epsilon gives smaller sigma.
- Dataset scope is documented in `dataset_debug_report.md`.

## What Still Remains

- The full 1,031-example dataset was not restored; current results use the 276-sample local subset.
- DP-KER does not beat DP-Embedding in pairwise matching.
- DP-KER only beats DP-Embedding on sampled ranking with random negatives, not hard negatives.
- Pairwise correlation metrics are weak or negative, meaning current heuristic scores are not aligned with continuous label scores.
- The current full DP-KER weight formula should be redesigned or tuned before making strong claims.

## Paper Guidance

Use Table II as the new main pairwise matching result, but state clearly that the current DP-KER implementation does not improve pairwise classification performance. Use Table III to show that sampled ranking is a better protocol than full-library retrieval and that knowledge helps consistency. Table IV should be presented as diagnostic rather than as a strong success table. Table V can be used to show the DP mechanism is implemented correctly, while noting that the current utility trend is not clean.

The current experiments are more credible than the original full-retrieval setup, but they support only cautious claims. A 4-5 page EI paper can still be written as a framework and pilot evaluation paper, but not as a strong empirical superiority paper.

