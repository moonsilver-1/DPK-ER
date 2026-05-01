from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from data_loader import (
    DATASET_STATS_FIXED_PATH,
    DATASET_STATS_PATH,
    CANDIDATES_PATH,
    EXPERIMENT_DATA_DIR,
    KNOWLEDGE_ITEMS_PATH,
    LABELS_PATH,
    PROFILES_PATH,
    build_converted_rows,
    ensure_output_dirs,
    load_raw_samples,
    write_jsonl,
)


def main() -> None:
    ensure_output_dirs()
    try:
        samples = load_raw_samples()
    except Exception as exc:
        print("数据转换失败。")
        print(str(exc))
        print("请检查 data/raw/resume-score-details 中的 JSON 字段是否与预期一致。")
        raise SystemExit(1) from exc

    profile_rows = []
    candidate_rows = []
    label_rows = []
    all_label_rows = []
    knowledge_rows = []
    category_counter = Counter()
    valid_counter = Counter()
    additional_info_empty = 0

    for sample in samples:
        profile_row, candidate_row, label_row, knowledge_row = build_converted_rows(sample)
        profile_rows.append(profile_row)
        candidate_rows.append(candidate_row)
        all_label_rows.append(label_row)
        if label_row.get("include_in_matching") and label_row.get("label") is not None:
            label_rows.append(label_row)
        knowledge_rows.append(knowledge_row)
        category_counter[label_row["filename_category"]] += 1
        valid_counter[bool(label_row["valid_resume_and_jd"])] += 1
        if not knowledge_row["additional_info"].strip():
            additional_info_empty += 1

    write_jsonl(profile_rows, PROFILES_PATH)
    write_jsonl(candidate_rows, CANDIDATES_PATH)
    write_jsonl(label_rows, LABELS_PATH)
    write_jsonl(knowledge_rows, KNOWLEDGE_ITEMS_PATH)

    stats_rows = [
        ("raw_files", str(len(samples))),
        ("profiles_rows", str(len(profile_rows))),
        ("candidates_rows", str(len(candidate_rows))),
        ("labels_rows", str(len(label_rows))),
        ("knowledge_rows", str(len(knowledge_rows))),
        ("label_source", "score_threshold"),
        ("score_threshold_raw", "7.0"),
        ("matched_label_1", str(sum(1 for row in label_rows if int(row["label"]) == 1))),
        ("mismatched_label_0", str(sum(1 for row in label_rows if int(row["label"]) == 0))),
        ("excluded_rows", str(len(all_label_rows) - len(label_rows))),
        ("excluded_invalid_rows", str(sum(1 for row in all_label_rows if str(row["filename_category"]).startswith("invalid")))),
        ("excluded_missing_info_rows", str(sum(1 for row in all_label_rows if row["filename_category"] == "empty_additional_info"))),
        ("all_raw_label_rows_before_matching_filter", str(len(all_label_rows))),
        ("valid_true", str(valid_counter[True])),
        ("valid_false", str(valid_counter[False])),
        ("empty_additional_info", str(additional_info_empty)),
        ("match_files", str(category_counter["match"])),
        ("invalid_resume_files", str(category_counter["invalid_resume"])),
        ("invalid_job_description_files", str(category_counter["invalid_job_description"])),
        ("invalid_gibberish_resume_files", str(category_counter["invalid_gibberish_resume"])),
        ("invalid_gibberish_job_description_files", str(category_counter["invalid_gibberish_job_description"])),
        ("empty_additional_info_files", str(category_counter["empty_additional_info"])),
    ]
    DATASET_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_STATS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(stats_rows)
    with DATASET_STATS_FIXED_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(stats_rows)

    print(f"已写入 {PROFILES_PATH}")
    print(f"已写入 {CANDIDATES_PATH}")
    print(f"已写入 {LABELS_PATH}")
    print(f"已写入 {KNOWLEDGE_ITEMS_PATH}")
    print(f"已写入 {DATASET_STATS_PATH}")


if __name__ == "__main__":
    main()
