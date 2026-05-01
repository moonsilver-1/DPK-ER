# Full Dataset Loading Report

```json
{
  "dataset": "netsol/resume-score-details",
  "source_used": "hf_cache_snapshot",
  "load_error": "datasets.load_dataset failed, recovered JSON files from HF cache. Error: DatasetGenerationError('An error occurred while generating the dataset')",
  "splits": {
    "hf_cache": 1031
  },
  "fields": {
    "hf_cache": [
      "input",
      "output",
      "details"
    ]
  },
  "total_loaded_rows": 1031,
  "included_valid_scored_rows": 815,
  "positive_count_threshold_7": 150,
  "negative_count_threshold_7": 665,
  "filename_categories": {
    "empty_additional_info": 40,
    "invalid_gibberish_job_description": 15,
    "invalid_gibberish_resume": 15,
    "invalid_job_description": 57,
    "invalid_resume": 55,
    "match": 648,
    "mismatch": 201
  },
  "note": "If source_used is local_raw, datasets.load_dataset did not complete in this environment. The v2 experiment is then limited to the available local subset."
}
```
