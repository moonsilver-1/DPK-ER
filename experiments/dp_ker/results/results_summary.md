# DP-KER Experimental Results Audit

This document audits the already generated CSV files only. No experiment was rerun, and no new data were invented.

## Executive Summary

- The current local raw dataset contains 276 JSON files, not 1,031. This is a major scope mismatch versus the dataset card / README claim and must be clarified before writing final paper claims.
- `DP-KER` does **not** outperform `DP-Embedding` on ranking metrics in the current results. Recall@5 is tied, and NDCG@5 is worse.
- `DP-KER` does improve `Consistency` versus `w/o_knowledge`, so the knowledge component is visible in the current ablation.
- Adding DP reduces Recall@5 / NDCG@5 / MRR relative to `w/o_dp`, which is directionally reasonable. However, the privacy-budget trend is weak and not monotonic.
- The current evidence is enough for a pilot-style experiments section, but not strong enough to support a confident superiority claim for `DP-KER`.

## Table 1. Dataset Statistics

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
| match_files | 94 |
| invalid_resume_files | 55 |
| invalid_job_description_files | 57 |
| invalid_gibberish_resume_files | 15 |
| invalid_gibberish_job_description_files | 15 |
| empty_additional_info_files | 40 |

### Audit note

The local conversion pipeline is internally consistent: all four JSONL files have 276 rows, and the label split sums to 276. The issue is external scope: the repository content currently reflects only 276 raw files, while the dataset description says 1,031 samples. If the full dataset is not present locally, the paper should not generalize these numbers as if they covered the full corpus.

### English analysis paragraph

The dataset statistics indicate that the current experimental corpus is internally coherent after conversion: each derived JSONL file contains the same number of rows, and the label split is well-defined. However, the visible raw dataset size is much smaller than the dataset card claims, so the experiments should be described as running on the currently available local subset unless the missing samples are recovered. This difference matters because it affects how strongly the results can support the final paper claims.

## Table 2. Main Results

| Method | Recall@5 | NDCG@5 | MRR | Consistency | Valid Queries |
|---|---:|---:|---:|---:|---:|
| TF-IDF+Cosine | 0.0763 | 0.0520 | 0.0630 | 0.5259 | 131 |
| BM25 | 0.0687 | 0.0463 | 0.0513 | 0.5304 | 131 |
| SBERT+Cosine | 0.0763 | 0.0497 | 0.0557 | 0.5256 | 131 |
| RAG-based Rec. | 0.0687 | 0.0464 | 0.0566 | 0.5456 | 131 |
| DP-Embedding | 0.0229 | 0.0154 | 0.0269 | 0.3277 | 131 |
| DP-KER | 0.0229 | 0.0126 | 0.0263 | 0.4962 | 131 |

### Relative change for DP-KER

| Comparison | Recall@5 relative change | NDCG@5 relative change |
|---|---:|---:|
| vs TF-IDF+Cosine | -70.00% | -75.78% |
| vs BM25 | -66.67% | -72.81% |
| vs SBERT+Cosine | -70.00% | -74.66% |
| vs DP-Embedding | 0.00% | -18.29% |

### Audit note

`DP-KER` is not better than `DP-Embedding` here. Recall@5 is exactly tied, while NDCG@5 is lower. The current table therefore does not support a strong statement that the knowledge enhancement improves ranking quality under privacy noise. The only clearly positive signal in the current main table is that `DP-KER` has much higher `Consistency` than `DP-Embedding`, but even that does not exceed the strongest non-DP baseline.

### English analysis paragraph

The main results show a clear utility drop after adding differential privacy, which is expected in principle. What is not supported by the current numbers is a claim that DP-KER improves ranking performance over a plain DP embedding baseline. In fact, the ranking metrics are either tied or worse. If the paper keeps the current data, the discussion should frame DP-KER as a privacy-preserving framework with a visible utility trade-off, rather than as a method that improves all ranking metrics.

## Table 3. Ablation Study

| Setting | Recall@5 | NDCG@5 | MRR | Consistency | Valid Queries |
|---|---:|---:|---:|---:|---:|
| full | 0.0229 | 0.0126 | 0.0263 | 0.4962 | 131 |
| w/o_dp | 0.0687 | 0.0474 | 0.0573 | 0.5478 | 131 |
| w/o_knowledge | 0.0229 | 0.0154 | 0.0269 | 0.3277 | 131 |
| w/o_ranking_fusion | 0.0229 | 0.0154 | 0.0268 | 0.4107 | 131 |
| retrieval_only | 0.0763 | 0.0520 | 0.0630 | 0.5259 | 131 |

### Relative change of full vs ablations

| Comparison | Consistency relative change |
|---|---:|
| full vs w/o_knowledge | +51.42% |
| full vs w/o_ranking_fusion | +20.84% |
| full vs w/o_dp | -9.42% |

### Audit note

The ablation table supports the contribution of the knowledge component and the fusion design, because `full` clearly improves `Consistency` over `w/o_knowledge` and `w/o_ranking_fusion`. However, `w/o_dp` is stronger than `full` on every ranking metric and on `Consistency`, so the current numbers do not show DP as a performance-improving module. The paper should therefore present DP as the privacy mechanism that introduces a cost, not as an accuracy enhancer.

### English analysis paragraph

The ablation study provides partial support for the framework design. The knowledge and fusion components are visible in the consistency score, which suggests that the model is using the auxiliary structure in a meaningful way. At the same time, the no-DP variant remains stronger on ranking and consistency, so the final text should be careful not to overclaim the benefit of the privacy module itself. A balanced interpretation is that knowledge helps recover structure after privacy perturbation, but the privacy perturbation still reduces raw retrieval quality.

## Table 4. Privacy Budget Analysis

| epsilon | Recall@5 | NDCG@5 | MRR | Consistency | Valid Queries |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.0229 | 0.0129 | 0.0266 | 0.4965 | 131 |
| 1.0 | 0.0229 | 0.0129 | 0.0266 | 0.4968 | 131 |
| 2.0 | 0.0229 | 0.0126 | 0.0263 | 0.4962 | 131 |
| 4.0 | 0.0229 | 0.0116 | 0.0247 | 0.4979 | 131 |
| 8.0 | 0.0153 | 0.0076 | 0.0232 | 0.5048 | 131 |

### Audit note

The privacy-budget trend is only weakly aligned with intuition. Recall@5 is flat from epsilon 0.5 to 4.0 and only drops at 8.0. NDCG@5 and MRR drift downward overall, but not monotonically. `Consistency` even rises slightly at larger epsilon values. This does not invalidate the table, but it means the current numbers are not strong evidence of a clean monotonic privacy-utility curve.

### English analysis paragraph

The privacy-budget experiment suggests that stronger privacy noise generally hurts ranking quality, but the effect is not smooth enough to support a strong monotonic claim. The metrics remain mostly flat across intermediate epsilon values, and consistency does not move in a clean direction. In the paper, this table should be presented as a rough trade-off study rather than a precise privacy calibration result.

## What can go into the paper directly

- `dataset_statistics.csv` can be used directly, but only with a scope note that the local corpus contains 276 files.
- `main_results.csv` can be used directly as a pilot result table, but not as evidence that DP-KER beats DP-Embedding.
- `ablation_results.csv` can be used directly to argue that knowledge and fusion matter.
- `privacy_budget_results.csv` can be used directly with a caveat that the epsilon trend is weak.

## What needs rerun or repair

- The dataset scope mismatch should be resolved before a final paper submission.
- If the intended claim is that DP-KER beats DP-Embedding, the current main results do not support it, so the model or scoring design needs to be rerun or redesigned.
- If the intended claim is a monotonic privacy-budget curve, the current experiment needs a more stable evaluation setup.

## Can this support a 4-5 page EI paper?

Yes, but only as a concise methods-and-pilot-results paper with cautious claims. It is not yet strong enough for a confident superiority story. The current evidence is enough to motivate the framework, explain the pipeline, and show that knowledge helps after privacy perturbation, but it is too weak to claim broad empirical dominance.

## Suggested Experiments Section Structure

1. Dataset and preprocessing
2. Experimental setup and baselines
3. Main results
4. Ablation analysis
5. Privacy budget analysis
6. Limitations and scope note

For the narrative, keep the tone measured: emphasize the privacy-utility trade-off, explain that knowledge improves consistency under perturbation, and explicitly note that the privacy module reduces raw ranking quality. Avoid claiming that DP-KER outperforms the non-private or plain DP baseline unless the numbers are rerun and actually support that statement.

