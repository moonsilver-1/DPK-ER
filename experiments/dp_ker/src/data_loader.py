from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


SEED = 3407


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("无法在当前脚本路径向上找到 pyproject.toml，无法识别项目根目录。")


PROJECT_ROOT = find_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "resume-score-details"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "dp_ker"
EXPERIMENT_DATA_DIR = EXPERIMENT_ROOT / "data"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
PROFILES_PATH = EXPERIMENT_DATA_DIR / "profiles.jsonl"
CANDIDATES_PATH = EXPERIMENT_DATA_DIR / "candidates.jsonl"
LABELS_PATH = EXPERIMENT_DATA_DIR / "labels.jsonl"
KNOWLEDGE_ITEMS_PATH = EXPERIMENT_DATA_DIR / "knowledge_items.jsonl"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_statistics.csv"
DATASET_STATS_FIXED_PATH = RESULTS_DIR / "dataset_statistics_fixed.csv"


def ensure_output_dirs() -> None:
    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(safe_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def filename_metadata(name: str) -> dict[str, Any]:
    stem = Path(name).stem
    parts = stem.split("_")
    category = parts[0]
    if stem.startswith("invalid_gibberish_resume"):
        category = "invalid_gibberish_resume"
    elif stem.startswith("invalid_gibberish_job_description"):
        category = "invalid_gibberish_job_description"
    elif stem.startswith("invalid_job_description"):
        category = "invalid_job_description"
    elif stem.startswith("invalid_resume"):
        category = "invalid_resume"
    elif stem.startswith("empty_additional_info"):
        category = "empty_additional_info"
    elif stem.startswith("match"):
        category = "match"
    match = re.search(r"(\d+)$", stem)
    sample_index = int(match.group(1)) if match else None
    return {
        "source_stem": stem,
        "filename_category": category,
        "sample_index": sample_index,
    }


def ensure_expected_keys(payload: dict[str, Any], source_file: str) -> None:
    expected_top = {"input", "output", "details"}
    actual_top = set(payload.keys())
    if actual_top != expected_top:
        raise ValueError(
            f"{source_file} 的顶层字段与预期不一致。预期={sorted(expected_top)}，实际={sorted(actual_top)}"
        )
    expected_input = {
        "job_description",
        "macro_dict",
        "micro_dict",
        "additional_info",
        "minimum_requirements",
        "resume",
    }
    expected_output = {
        "justification",
        "scores",
        "personal_info",
        "valid_resume_and_jd",
    }
    expected_details = {
        "name",
        "number",
        "skills",
        "email_id",
        "location",
        "projects",
        "education",
        "achievements",
        "publications",
        "certifications",
        "additional_urls",
        "executive_summary",
        "employment_history",
    }
    blocks = {
        "input": expected_input,
        "output": expected_output,
        "details": expected_details,
    }
    for block_name, expected in blocks.items():
        block = payload.get(block_name, {})
        if not isinstance(block, dict):
            raise ValueError(f"{source_file} 的 {block_name} 不是对象，实际类型={type(block).__name__}")
        actual = set(block.keys())
        if actual != expected:
            raise ValueError(
                f"{source_file} 的 {block_name} 字段与预期不一致。预期={sorted(expected)}，实际={sorted(actual)}"
            )


def build_candidate_text(payload: dict[str, Any]) -> str:
    details = payload["details"]
    personal_info = payload["output"].get("personal_info", {})
    sections = [
        safe_text(payload["input"].get("resume")),
        safe_text(details.get("executive_summary")),
        safe_text(details.get("skills")),
        safe_text(details.get("education")),
        safe_text(details.get("projects")),
        safe_text(details.get("employment_history")),
        safe_text(details.get("achievements")),
        safe_text(details.get("certifications")),
        safe_text(details.get("publications")),
        safe_text(details.get("additional_urls")),
        safe_text(personal_info),
    ]
    return "\n\n".join(section for section in sections if section)


def build_knowledge_text(payload: dict[str, Any]) -> str:
    job_description = safe_text(payload["input"].get("job_description"))
    additional_info = safe_text(payload["input"].get("additional_info"))
    minimum_requirements = safe_text(payload["input"].get("minimum_requirements"))
    macro_dict = safe_text(payload["input"].get("macro_dict"))
    micro_dict = safe_text(payload["input"].get("micro_dict"))
    sections = [job_description, additional_info, minimum_requirements, macro_dict, micro_dict]
    return "\n\n".join(section for section in sections if section)


def load_raw_samples() -> list[dict[str, Any]]:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"未找到原始数据目录：{RAW_DATA_DIR}\n请先下载 netsol/resume-score-details 到 data/raw/resume-score-details。"
        )
    samples: list[dict[str, Any]] = []
    for json_file in sorted(RAW_DATA_DIR.glob("*.json")):
        payload = read_json(json_file)
        ensure_expected_keys(payload, json_file.name)
        meta = filename_metadata(json_file.name)
        samples.append(
            {
                "source_file": json_file.name,
                "source_path": str(json_file),
                **meta,
                "payload": payload,
            }
        )
    return samples


def build_converted_rows(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = sample["payload"]
    input_block = payload["input"]
    output_block = payload["output"]
    details = payload["details"]
    scores = output_block.get("scores", {})
    macro_scores = scores.get("macro_scores", [])
    micro_scores = scores.get("micro_scores", [])
    requirements = scores.get("requirements", [])
    aggregated = scores.get("aggregated_scores", {})
    personal_info = output_block.get("personal_info", {})

    sample_id = sample["source_stem"]
    candidate_text = build_candidate_text(payload)
    knowledge_text = build_knowledge_text(payload)
    requirements_met = [item for item in safe_list(requirements) if isinstance(item, dict) and item.get("meets")]
    total_requirements = len(requirements)
    macro_score_value = aggregated.get("macro_scores")
    micro_score_value = aggregated.get("micro_scores")

    profile_row = {
        "sample_id": sample_id,
        "source_file": sample["source_file"],
        "filename_category": sample["filename_category"],
        "sample_index": sample["sample_index"],
        "candidate_name": personal_info.get("name") or details.get("name"),
        "email": personal_info.get("email") or details.get("email_id"),
        "phone": personal_info.get("phone") or details.get("number"),
        "location": details.get("location"),
        "current_position": personal_info.get("current_position"),
        "current_company": personal_info.get("current_company"),
        "employment_start_date_current_company": personal_info.get("employment_start_date_current_company"),
        "first_company_start_date": personal_info.get("first_company_start_date"),
        "urls": personal_info.get("urls", []),
        "education": details.get("education", []),
        "skills": details.get("skills", []),
        "achievements": details.get("achievements", []),
        "certifications": details.get("certifications", []),
        "additional_urls": details.get("additional_urls", []),
        "executive_summary": details.get("executive_summary"),
    }
    candidate_row = {
        "sample_id": sample_id,
        "source_file": sample["source_file"],
        "filename_category": sample["filename_category"],
        "sample_index": sample["sample_index"],
        "resume_text": safe_text(input_block.get("resume")),
        "candidate_text": candidate_text,
        "employment_history": details.get("employment_history", []),
        "projects": details.get("projects", []),
        "education": details.get("education", []),
        "achievements": details.get("achievements", []),
        "publications": details.get("publications", []),
        "certifications": details.get("certifications", []),
        "skills": details.get("skills", []),
        "executive_summary": details.get("executive_summary"),
        "additional_urls": details.get("additional_urls", []),
    }
    raw_score = None
    if macro_score_value is not None and micro_score_value is not None:
        raw_score = (float(macro_score_value) + float(micro_score_value)) / 2.0
    elif macro_score_value is not None:
        raw_score = float(macro_score_value)
    elif micro_score_value is not None:
        raw_score = float(micro_score_value)

    matching_label = None
    label_source = None
    include_in_matching = sample["filename_category"] == "match" and raw_score is not None
    if include_in_matching:
        matching_label = int(raw_score >= 7.0)
        label_source = "score_threshold"

    label_row = {
        "sample_id": sample_id,
        "profile_id": sample_id,
        "candidate_id": sample_id,
        "source_file": sample["source_file"],
        "filename_category": sample["filename_category"],
        "sample_index": sample["sample_index"],
        "label": matching_label,
        "label_source": label_source,
        "raw_score": raw_score,
        "normalized_score": (raw_score / 10.0) if raw_score is not None else None,
        "include_in_matching": include_in_matching,
        "valid_resume_and_jd": bool(output_block.get("valid_resume_and_jd")),
        "category_label_from_filename": sample["filename_category"] == "match",
        "macro_score": macro_score_value,
        "micro_score": micro_score_value,
        "requirements_total": total_requirements,
        "requirements_met": len(requirements_met),
        "requirements_met_rate": (len(requirements_met) / total_requirements) if total_requirements else None,
        "justification_count": len(safe_list(output_block.get("justification"))),
        "macro_scores": macro_scores,
        "micro_scores": micro_scores,
        "requirements": requirements,
        "aggregated_scores": aggregated,
    }
    knowledge_row = {
        "sample_id": sample_id,
        "source_file": sample["source_file"],
        "filename_category": sample["filename_category"],
        "sample_index": sample["sample_index"],
        "job_description": safe_text(input_block.get("job_description")),
        "additional_info": safe_text(input_block.get("additional_info")),
        "minimum_requirements": safe_list(input_block.get("minimum_requirements")),
        "macro_dict": input_block.get("macro_dict", {}),
        "micro_dict": input_block.get("micro_dict", {}),
        "knowledge_text": knowledge_text,
    }
    return profile_row, candidate_row, label_row, knowledge_row


def load_converted_dataset() -> list[dict[str, Any]]:
    profiles = {row["sample_id"]: row for row in read_jsonl(PROFILES_PATH)}
    candidates = {row["sample_id"]: row for row in read_jsonl(CANDIDATES_PATH)}
    labels = {
        row["sample_id"]: row
        for row in read_jsonl(LABELS_PATH)
        if row.get("label") is not None and row.get("include_in_matching", True)
    }
    knowledge_items = {row["sample_id"]: row for row in read_jsonl(KNOWLEDGE_ITEMS_PATH)}
    sample_ids = sorted(set(profiles) & set(candidates) & set(labels) & set(knowledge_items))
    return [
        {
            "sample_id": sample_id,
            "profile": profiles[sample_id],
            "candidate": candidates[sample_id],
            "label": labels[sample_id],
            "knowledge": knowledge_items[sample_id],
        }
        for sample_id in sample_ids
    ]
