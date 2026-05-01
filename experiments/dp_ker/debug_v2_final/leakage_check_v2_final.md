# Leakage Check v2 Final

- Model feature columns: tfidf_cosine, bm25_score, embedding_cosine, dp_embedding_cosine, kg_skill_coverage, skill_overlap_count, missing_skill_ratio, requirement_coverage, resume_length, jd_length
- Excluded target columns: raw_score, binary_label.
- `raw_score` is used only as regression target and for constructing `binary_label`.
- `binary_label` is used only as classification target and for stratified splits.
- Feature cache includes target columns for evaluation bookkeeping, but training code selects only FEATURE_COLUMNS.
