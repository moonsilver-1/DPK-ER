from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data_loader import (
    EXPERIMENT_ROOT,
    RAW_DATA_DIR,
    build_candidate_text,
    build_knowledge_text,
    filename_metadata,
    load_raw_samples,
    safe_list,
    safe_text,
)


DEBUG_V2_DIR = EXPERIMENT_ROOT / "debug_v2"
RESULTS_V2_DIR = EXPERIMENT_ROOT / "results_v2"
HF_DATASET_NAME = "netsol/resume-score-details"


def _raw_score(payload: dict[str, Any]) -> float | None:
    aggregated = payload.get("output", {}).get("scores", {}).get("aggregated_scores", {})
    macro = aggregated.get("macro_scores")
    micro = aggregated.get("micro_scores")
    values = [float(value) for value in (macro, micro) if value is not None]
    return float(np.mean(values)) if values else None


def _record_from_payload(payload: dict[str, Any], sample_id: str, source: str, split: str = "local") -> dict[str, Any]:
    input_block = payload.get("input", {})
    output_block = payload.get("output", {})
    scores = output_block.get("scores", {})
    raw_score = _raw_score(payload)
    additional_info = safe_text(input_block.get("additional_info"))
    valid = bool(output_block.get("valid_resume_and_jd"))
    source_stem = Path(source).stem
    meta = filename_metadata(source) if source.endswith(".json") else {"filename_category": "hf", "sample_index": None}
    include = valid and bool(additional_info.strip()) and raw_score is not None
    return {
        "sample_id": sample_id,
        "split": split,
        "source_file": source,
        "filename_category": meta["filename_category"],
        "include_in_matching": include,
        "raw_score": raw_score,
        "label": int(raw_score >= 7.0) if include and raw_score is not None else None,
        "candidate": {
            "candidate_text": build_candidate_text(payload),
            "skills": payload.get("details", {}).get("skills", []),
            "source_file": source,
        },
        "knowledge": {
            "job_description": safe_text(input_block.get("job_description")),
            "additional_info": additional_info,
            "minimum_requirements": safe_list(input_block.get("minimum_requirements")),
            "macro_dict": input_block.get("macro_dict", {}) or {},
            "micro_dict": input_block.get("micro_dict", {}) or {},
            "knowledge_text": build_knowledge_text(payload),
            "requirements": scores.get("requirements", []),
            "justification": output_block.get("justification", []),
        },
        "payload": payload,
    }


def _load_hf_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import DatasetDict, load_dataset

    ds = load_dataset(HF_DATASET_NAME)
    split_names = list(ds.keys()) if isinstance(ds, DatasetDict) else ["train"]
    records: list[dict[str, Any]] = []
    fields: dict[str, list[str]] = {}
    split_sizes: dict[str, int] = {}
    for split in split_names:
        split_ds = ds[split] if isinstance(ds, DatasetDict) else ds
        split_sizes[split] = len(split_ds)
        fields[split] = list(split_ds.column_names)
        for idx, row in enumerate(split_ds):
            payload = dict(row)
            records.append(_record_from_payload(payload, sample_id=f"{split}_{idx}", source=f"hf:{split}:{idx}", split=split))
    return records, {"source": "huggingface", "splits": split_sizes, "fields": fields}


def _load_local_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_raw_samples()
    records = [
        _record_from_payload(sample["payload"], sample_id=sample["source_stem"], source=sample["source_file"], split="local")
        for sample in samples
    ]
    return records, {
        "source": "local_raw",
        "raw_dir": str(RAW_DATA_DIR),
        "splits": {"local": len(records)},
        "fields": {"local": ["input", "output", "details"]},
    }


def _load_hf_cache_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--netsol--resume-score-details" / "snapshots"
    snapshots = [path for path in cache_root.glob("*") if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No Hugging Face snapshot cache found under {cache_root}")
    json_files: list[Path] = []
    for snapshot in sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True):
        files = sorted(snapshot.glob("*.json"))
        if len(files) > len(json_files):
            json_files = files
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {cache_root}")
    records = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(_record_from_payload(payload, sample_id=path.stem, source=path.name, split="hf_cache"))
    return records, {
        "source": "hf_cache_snapshot",
        "raw_dir": str(cache_root),
        "splits": {"hf_cache": len(records)},
        "fields": {"hf_cache": ["input", "output", "details"]},
    }


def load_dataset_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    DEBUG_V2_DIR.mkdir(parents=True, exist_ok=True)
    try:
        records, meta = _load_hf_rows()
        meta["load_error"] = ""
    except Exception as exc:
        hf_error = repr(exc)
        try:
            records, meta = _load_hf_cache_rows()
            meta["load_error"] = f"datasets.load_dataset failed, recovered JSON files from HF cache. Error: {hf_error}"
        except Exception as cache_exc:
            records, meta = _load_local_rows()
            meta["load_error"] = f"datasets.load_dataset failed: {hf_error}; HF cache fallback failed: {cache_exc!r}"
    write_loading_report(records, meta)
    return records, meta


def write_loading_report(records: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    categories = Counter(record.get("filename_category", "unknown") for record in records)
    included = [record for record in records if record.get("include_in_matching")]
    positives = sum(1 for record in included if record.get("label") == 1)
    negatives = sum(1 for record in included if record.get("label") == 0)
    report = {
        "dataset": HF_DATASET_NAME,
        "source_used": meta.get("source"),
        "load_error": meta.get("load_error", ""),
        "splits": meta.get("splits", {}),
        "fields": meta.get("fields", {}),
        "total_loaded_rows": len(records),
        "included_valid_scored_rows": len(included),
        "positive_count_threshold_7": positives,
        "negative_count_threshold_7": negatives,
        "filename_categories": dict(categories),
        "note": (
            "If source_used is local_raw, datasets.load_dataset did not complete in this environment. "
            "The v2 experiment is then limited to the available local subset."
        ),
    }
    (DEBUG_V2_DIR / "full_dataset_loading_report.md").write_text(
        "# Full Dataset Loading Report\n\n```json\n"
        + json.dumps(report, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )


def filtered_records(records: list[dict[str, Any]], threshold: float = 7.0) -> list[dict[str, Any]]:
    output = []
    for record in records:
        if not record.get("include_in_matching"):
            continue
        raw_score = float(record["raw_score"])
        updated = dict(record)
        updated["label"] = int(raw_score >= threshold)
        output.append(updated)
    return output
