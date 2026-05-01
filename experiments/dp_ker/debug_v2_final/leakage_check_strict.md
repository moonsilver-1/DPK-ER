# Strict Leakage Check

## Training Feature Columns

- tfidf_cosine
- bm25_score
- embedding_cosine
- dp_embedding_cosine
- kg_skill_coverage
- skill_overlap_count
- missing_skill_ratio
- requirement_coverage
- resume_length
- jd_length

## Explicitly Excluded / Checked Columns

- raw_score: present
- binary_label: present
- label: not present
- score: not present
- aggregated_score: not present
- valid_resume_and_jd: not present
- profile_id: not present
- resume_id: not present
- candidate_id: not present
- job_id: not present
- jd_id: not present
- source_file: present
- file_name: not present

## Final Judgment

- No obvious label leakage detected from the configured feature columns.
- raw_score and binary_label are not included in FEATURE_COLUMNS.
- ID/file columns are not included in FEATURE_COLUMNS.
