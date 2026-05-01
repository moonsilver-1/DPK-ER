# Debug Summary

## Answers

1. Is the label reversed?

No. The label is not numerically reversed; `label=1` means `valid_resume_and_jd=True`. The problem is semantic: this is a validity/evaluability label, not a complete match-vs-nonmatch relevance label.

2. Is the score direction reversed?

For many heuristic methods, yes relative to the current validity label. `AUC(-score, label)` is much higher than `AUC(score, label)` for TF-IDF, BM25, embedding, KG, and DP-KER. This means higher similarity often corresponds to invalid/negative records under the current label definition.

3. Why is AUC below 0.5?

Because the binary label is `valid_resume_and_jd`, while the scores measure textual/semantic similarity or requirement overlap. Invalid samples can still have high resume-JD lexical overlap, including cases where invalid resumes/JDs are malformed, duplicated, or too similar. The score is therefore not aligned with the validity label.

4. Should pairwise consistency be deleted?

Yes for pairwise method comparison. The current pairwise consistency depends only on the gold pair content and not on method-specific predictions or generated explanations. It is identical for all methods and should be removed from pairwise result tables. Consistency can remain in sampled ranking only if computed from each method's top-ranked candidates.

5. Which tables can be kept and which must be rerun?

- Keep dataset statistics and dataset debug report as scope documentation.
- Keep sampled ranking as a diagnostic/auxiliary table, because its consistency is method-dependent through top-ranked candidates.
- Rerun pairwise matching after redefining the label target, removing pairwise consistency, and choosing a score aligned with the target.
- Rerun ablation and privacy budget after the pairwise target/metric definition is fixed.

## Files Generated

- `label_debug_report.md`
- `score_direction_report.csv`
- `label_score_distribution.csv`
- `sanity_check_pairs.md`
- `debug_summary.md`