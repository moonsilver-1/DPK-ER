from __future__ import annotations

import csv

from baselines import evaluate_methods
from data_loader import RESULTS_DIR, load_converted_dataset
from metrics import format_results_row, ranking_metrics


def main() -> None:
    records = load_converted_dataset()
    if not records:
        raise FileNotFoundError(
            "未找到转换后的 JSONL 数据。请先运行 convert_hf_dataset.py 生成 experiments/dp_ker/data 下的文件。"
        )
    table = evaluate_methods(records)
    rows = []
    for row in table:
        score_matrix = row.pop("score_matrix")
        metrics = ranking_metrics(records, score_matrix, top_k=5)
        rows.append(format_results_row(row["method"], metrics))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "main_results.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {output_path}")


if __name__ == "__main__":
    main()
