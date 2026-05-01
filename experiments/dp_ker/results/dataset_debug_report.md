# Dataset Debug Report

This report records the current local data state used by the repaired experiments.

## Local Files

- Raw data directory: `data/raw/resume-score-details`
- JSON files present locally: 276
- Hugging Face download metadata files for JSON objects: 276
- Non-JSON files present: `.gitattributes`, `README.md`, `dataset-card.yaml`

## Dataset Card Claim

The local `dataset-card.yaml` reports:

- `num_examples: 1031`
- 648 matched samples
- 201 mismatched samples
- 142 invalid samples
- 40 samples missing additional info

## Conversion Scope

`convert_hf_dataset.py` iterates over `data/raw/resume-score-details/*.json` and converts every JSON file present in that directory. The converter is not filtering to a subset by split, filename, or label. Therefore, the current 276-row converted dataset is caused by the local raw directory containing only 276 JSON files, not by the conversion script dropping records.

## Download Completeness

The local Hugging Face cache metadata also lists 276 JSON metadata files, matching the 276 raw JSON files. A follow-up attempt was made to restore the full dataset with `uv run huggingface-cli download netsol/resume-score-details --repo-type dataset --local-dir data/raw/resume-score-details`, using a workspace-local `UV_CACHE_DIR`. That download attempt was interrupted by the user before completion, so the full 1,031-example corpus was not restored in this run.

## Current Experimental Scope

The repaired experiments are based on the available local subset of 276 JSON samples. The current result tables must not be described as full-dataset results unless the remaining raw files are restored and the experiments are rerun.

