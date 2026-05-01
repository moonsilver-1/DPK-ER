from __future__ import annotations

from typing import Any

import numpy as np

from kg_score import explanation_consistency


def _rank_of_true(scores_row: np.ndarray, true_index: int) -> int:
    order = np.argsort(-np.asarray(scores_row, dtype=float), kind="mergesort")
    return int(np.where(order == true_index)[0][0] + 1)


def ranking_metrics(
    records: list[dict[str, Any]],
    score_matrix: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    score_matrix = np.asarray(score_matrix, dtype=float)
    if score_matrix.ndim != 2:
        raise ValueError("score_matrix 必须是二维矩阵")
    if score_matrix.shape[0] != len(records):
        raise ValueError("score_matrix 的行数必须与 records 数量一致")

    valid_indices = [i for i, record in enumerate(records) if bool(record["label"]["label"])]
    recall_hits = []
    ndcg_values = []
    mrr_values = []
    consistency_values = []

    for i, record in enumerate(records):
        top_pred = int(np.argmax(score_matrix[i]))
        predicted = records[top_pred]
        consistency_values.append(
            explanation_consistency(
                candidate_text=record["candidate"]["candidate_text"],
                knowledge_text=predicted["knowledge"]["knowledge_text"],
                minimum_requirements=predicted["knowledge"].get("minimum_requirements", []),
                additional_info=predicted["knowledge"].get("additional_info", ""),
                macro_dict=predicted["knowledge"].get("macro_dict", {}),
                micro_dict=predicted["knowledge"].get("micro_dict", {}),
            )
        )

        if i not in valid_indices:
            continue
        rank = _rank_of_true(score_matrix[i], i)
        recall_hits.append(1.0 if rank <= top_k else 0.0)
        ndcg_values.append(1.0 / np.log2(rank + 1) if rank <= top_k else 0.0)
        mrr_values.append(1.0 / rank)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        f"Recall@{top_k}": _mean(recall_hits),
        f"NDCG@{top_k}": _mean(ndcg_values),
        "MRR": _mean(mrr_values),
        "Consistency": _mean(consistency_values),
        "ValidQueries": float(len(valid_indices)),
    }


def format_results_row(method: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {"Method": method, **metrics}
