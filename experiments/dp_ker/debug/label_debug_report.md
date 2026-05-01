# Label Debug Report

Total label rows: 276
Positive label rows: 131
Negative label rows: 145

## Field Sample

```json
{"sample_id": "empty_additional_info_0", "source_file": "empty_additional_info_0.json", "filename_category": "empty_additional_info", "sample_index": 0, "label": true, "valid_resume_and_jd": true, "category_label_from_filename": false, "macro_score": 9.0, "micro_score": 9.0, "requirements_total": 3, "requirements_met": 3, "requirements_met_rate": 1.0, "justification_count": 4, "macro_scores": [{"criteria": "leadership", "score": 9}, {"criteria": "technical expertise", "score": 9}], "micro_scores": [{"criteria": "project management", "score": 9}, {"criteria": "installation and commissioning", "score": 9}], "requirements": [{"criteria": "Proficiency in conducting and managing FAT/SAT", "meets": true}, {"criteria": "Bachelor's degree in Electrical Engineering", "meets": true}, {"criteria": "10+ years of experience in project management and electrical design", "meets": true}], "aggregated_scores": {"macro_scores": 9.0, "micro_scores": 9.0}}
```
```json
{"sample_id": "empty_additional_info_1", "source_file": "empty_additional_info_1.json", "filename_category": "empty_additional_info", "sample_index": 1, "label": true, "valid_resume_and_jd": true, "category_label_from_filename": false, "macro_score": 7.0, "micro_score": 6.0, "requirements_total": 3, "requirements_met": 2, "requirements_met_rate": 0.6666666666666666, "justification_count": 4, "macro_scores": [{"criteria": "technical expertise", "score": 7}], "micro_scores": [{"criteria": "electrical design evaluation", "score": 6}, {"criteria": "project management", "score": 8}, {"criteria": "contractor management", "score": 5}], "requirements": [{"criteria": "Proficiency in conducting and managing FAT/SAT", "meets": false}, {"criteria": "Experience in managing and coordinating with multiple stakeholders", "meets": true}, {"criteria": "Bachelor's degree in Electrical Engineering", "meets": true}], "aggregated_scores": {"macro_scores": 7.0, "micro_scores": 6.0}}
```
```json
{"sample_id": "empty_additional_info_10", "source_file": "empty_additional_info_10.json", "filename_category": "empty_additional_info", "sample_index": 10, "label": true, "valid_resume_and_jd": true, "category_label_from_filename": false, "macro_score": 9.0, "micro_score": 8.09, "requirements_total": 2, "requirements_met": 2, "requirements_met_rate": 1.0, "justification_count": 4, "macro_scores": [{"criteria": "experience", "score": 9}], "micro_scores": [{"criteria": "project management", "score": 9}, {"criteria": "installation, testing & commissioning", "score": 8}, {"criteria": "vendor management", "score": 8}], "requirements": [{"criteria": "Proficiency in electrical design and project execution", "meets": true}, {"criteria": "10+ years of industry experience in project management", "meets": true}], "aggregated_scores": {"macro_scores": 9.0, "micro_scores": 8.09}}
```
```json
{"sample_id": "empty_additional_info_11", "source_file": "empty_additional_info_11.json", "filename_category": "empty_additional_info", "sample_index": 11, "label": true, "valid_resume_and_jd": true, "category_label_from_filename": false, "macro_score": 5.96, "micro_score": 7.28, "requirements_total": 3, "requirements_met": 2, "requirements_met_rate": 0.6666666666666666, "justification_count": 4, "macro_scores": [{"criteria": "technical expertise", "score": 6}, {"criteria": "team collaboration", "score": 5}], "micro_scores": [{"criteria": "matlab", "score": 8}, {"criteria": "arduino", "score": 0}], "requirements": [{"criteria": "Proficiency in MATLAB and basic programming (Python/C++)", "meets": true}, {"criteria": "Bachelor's degree in Electrical Engineering or a related field", "meets": true}, {"criteria": "Relevant internship experience in automation or control systems", "meets": false}], "aggregated_scores": {"macro_scores": 5.96, "micro_scores": 7.28}}
```
```json
{"sample_id": "empty_additional_info_110", "source_file": "empty_additional_info_110.json", "filename_category": "empty_additional_info", "sample_index": 110, "label": true, "valid_resume_and_jd": true, "category_label_from_filename": false, "macro_score": 3.42, "micro_score": 5.0, "requirements_total": 4, "requirements_met": 1, "requirements_met_rate": 0.25, "justification_count": 4, "macro_scores": [{"criteria": "leadership", "score": 4}, {"criteria": "international business experience", "score": 3}, {"criteria": "strategic planning", "score": 3}], "micro_scores": [{"criteria": "client acquisition", "score": 5}], "requirements": [{"criteria": "Proven experience in market entry and compliance with international standards", "meets": false}, {"criteria": "Proficiency in strategic planning and client management", "meets": false}, {"criteria": "Bachelor's degree in Business, Marketing, or relevant field", "meets": true}, {"criteria": "5+ years of experience in international business development", "meets": false}], "aggregated_scores": {"macro_scores": 3.42, "micro_scores": 5.0}}
```

## Label by Filename Category

| filename_category | label | count |
|---|---:|---:|
| empty_additional_info | 0 | 1 |
| empty_additional_info | 1 | 39 |
| invalid_gibberish_job_description | 0 | 15 |
| invalid_gibberish_resume | 0 | 15 |
| invalid_job_description | 0 | 57 |
| invalid_resume | 0 | 55 |
| match | 0 | 2 |
| match | 1 | 92 |

## Interpretation

The converted `label` field is copied from `valid_resume_and_jd`. It indicates whether the resume/JD pair is valid for evaluation, not whether the pair is a semantic match. This is confirmed by filenames such as `empty_additional_info_*`, which are label=1 but not filename-category `match`, and by invalid filename categories that are label=0.

## Positive Samples

- sample_id=empty_additional_info_0, source_file=empty_additional_info_0.json, category=empty_additional_info, label=True, valid=True, macro=9.0, micro=9.0, requirements_met_rate=1.0
- sample_id=empty_additional_info_1, source_file=empty_additional_info_1.json, category=empty_additional_info, label=True, valid=True, macro=7.0, micro=6.0, requirements_met_rate=0.6666666666666666
- sample_id=empty_additional_info_10, source_file=empty_additional_info_10.json, category=empty_additional_info, label=True, valid=True, macro=9.0, micro=8.09, requirements_met_rate=1.0
- sample_id=empty_additional_info_11, source_file=empty_additional_info_11.json, category=empty_additional_info, label=True, valid=True, macro=5.96, micro=7.28, requirements_met_rate=0.6666666666666666
- sample_id=empty_additional_info_110, source_file=empty_additional_info_110.json, category=empty_additional_info, label=True, valid=True, macro=3.42, micro=5.0, requirements_met_rate=0.25
- sample_id=empty_additional_info_111, source_file=empty_additional_info_111.json, category=empty_additional_info, label=True, valid=True, macro=4.0, micro=6.0, requirements_met_rate=0.0
- sample_id=empty_additional_info_112, source_file=empty_additional_info_112.json, category=empty_additional_info, label=True, valid=True, macro=5.41, micro=5.0, requirements_met_rate=0.0
- sample_id=empty_additional_info_114, source_file=empty_additional_info_114.json, category=empty_additional_info, label=True, valid=True, macro=5.2, micro=3.54, requirements_met_rate=0.5
- sample_id=empty_additional_info_115, source_file=empty_additional_info_115.json, category=empty_additional_info, label=True, valid=True, macro=8.0, micro=8.53, requirements_met_rate=1.0
- sample_id=empty_additional_info_116, source_file=empty_additional_info_116.json, category=empty_additional_info, label=True, valid=True, macro=6.27, micro=6.680000000000001, requirements_met_rate=1.0

## Negative Samples

- sample_id=empty_additional_info_113, source_file=empty_additional_info_113.json, category=empty_additional_info, label=False, valid=False, macro=3.0, micro=2.0, requirements_met_rate=0.0
- sample_id=invalid_gibberish_job_description_match_20, source_file=invalid_gibberish_job_description_match_20.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_278, source_file=invalid_gibberish_job_description_match_278.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_289, source_file=invalid_gibberish_job_description_match_289.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_292, source_file=invalid_gibberish_job_description_match_292.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_398, source_file=invalid_gibberish_job_description_match_398.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_40, source_file=invalid_gibberish_job_description_match_40.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_459, source_file=invalid_gibberish_job_description_match_459.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_497, source_file=invalid_gibberish_job_description_match_497.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None
- sample_id=invalid_gibberish_job_description_match_522, source_file=invalid_gibberish_job_description_match_522.json, category=invalid_gibberish_job_description, label=False, valid=False, macro=None, micro=None, requirements_met_rate=None