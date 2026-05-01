from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np

from baselines import BM25Okapi, build_candidate_texts, build_text_corpus
from data_loader import (
    CANDIDATES_PATH,
    EXPERIMENT_ROOT,
    KNOWLEDGE_ITEMS_PATH,
    LABELS_PATH,
    PROFILES_PATH,
    RESULTS_DIR,
    read_jsonl,
)
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from kg_score import tokenize
from run_pairwise_matching import (
    EPSILON,
    binary_metrics,
    bm25_pair_scores,
    dp_embedding_pair_scores,
    embedding_pair_scores,
    kg_pair_scores,
    labels_and_scores,
    minmax,
    orient_scores,
    tfidf_pair_scores,
)
from run_sampled_ranking import (
    NEGATIVES_PER_QUERY,
    TOP_K,
    build_pools,
    build_similarity_matrix,
    direction_multiplier,
    evaluate_pools,
    kg_scores_for_pool,
    rowwise_cosine,
)


THRESHOLDS = [6.5, 7.0, 7.5]
METHODS_PAIRWISE = ["TF-IDF pair similarity", "BM25 pair similarity", "DP-Embedding pair similarity", "DP-KER full pair score"]
METHODS_RANKING = ["TF-IDF+Cosine", "BM25", "DP-Embedding", "DP-KER"]
PAIRWISE_OUTPUT = RESULTS_DIR / "threshold_sensitivity_pairwise.csv"
RANKING_OUTPUT = RESULTS_DIR / "threshold_sensitivity_ranking.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "threshold_sensitivity_summary.md"
DEBUG_DIRECTION_OUTPUT = EXPERIMENT_ROOT / "debug" / "threshold_sensitivity_score_direction.csv"


def load_base_rows() -> list[dict[str, Any]]:
    profiles = {row["sample_id"]: row for row in read_jsonl(PROFILES_PATH)}
    candidates = {row["sample_id"]: row for row in read_jsonl(CANDIDATES_PATH)}
    labels = {row["sample_id"]: row for row in read_jsonl(LABELS_PATH) if row.get("raw_score") is not None}
    knowledge = {row["sample_id"]: row for row in read_jsonl(KNOWLEDGE_ITEMS_PATH)}
    sample_ids = sorted(set(profiles) & set(candidates) & set(labels) & set(knowledge))
    records = []
    for sample_id in sample_ids:
        label = labels[sample_id]
        if label.get("filename_category") != "match":
            continue
        records.append(
            {
                "sample_id": sample_id,
                "profile": profiles[sample_id],
                "candidate": candidates[sample_id],
                "label": label,
                "knowledge": knowledge[sample_id],
            }
        )
    return records


def records_for_threshold(base_records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    records = []
    for record in base_records:
        raw_score = float(record["label"]["raw_score"])
        new_label = dict(record["label"])
        new_label["label"] = int(raw_score >= threshold)
        new_label["label_source"] = f"score_threshold_{threshold}"
        new_label["threshold"] = threshold
        new_label["normalized_score"] = raw_score / 10.0
        records.append({**record, "label": new_label})
    return records


def aggregate_metrics(
    threshold: float,
    method: str,
    metric_rows: list[dict[str, float]],
    positive_count: int,
    negative_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "threshold": threshold,
        "Method": method,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "total_count": positive_count + negative_count,
    }
    if extra:
        row.update(extra)
    for metric in ["AUC", "Accuracy", "F1", "Precision", "Recall"]:
        values = np.asarray([metric_row[metric] for metric_row in metric_rows], dtype=float)
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(np.nanstd(values))
    return row


def aggregate_ranking(
    threshold: float,
    method: str,
    negative_type: str,
    metric_rows: list[dict[str, float]],
    positive_count: int,
    negative_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "threshold": threshold,
        "Method": method,
        "NegativeType": negative_type,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "total_count": positive_count + negative_count,
    }
    if extra:
        row.update(extra)
    for metric in ["Recall@5", "NDCG@5", "MRR", "Hit@1"]:
        values = np.asarray([metric_row[metric] for metric_row in metric_rows], dtype=float)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def write_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pairwise_threshold(records: list[dict[str, Any]], threshold: float, direction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_true, continuous = labels_and_scores(records)
    positive_count = int(np.sum(y_true))
    negative_count = int(len(y_true) - positive_count)
    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)

    tfidf_scores = tfidf_pair_scores(candidate_texts, job_texts)
    bm25_scores = bm25_pair_scores(candidate_texts, job_texts)
    embedding_scores, _backend, candidate_vectors, job_vectors = embedding_pair_scores(candidate_texts, job_texts)
    kg_scores = kg_pair_scores(records)

    threshold_direction_rows: list[dict[str, Any]] = []
    tfidf_scores = orient_scores(f"threshold={threshold} TF-IDF pair similarity", y_true, tfidf_scores, threshold_direction_rows)
    bm25_scores = orient_scores(f"threshold={threshold} BM25 pair similarity", y_true, bm25_scores, threshold_direction_rows)
    kg_scores = orient_scores(f"threshold={threshold} KG-Enhanced pair score", y_true, kg_scores, threshold_direction_rows)
    direction_rows.extend(threshold_direction_rows)

    rows = [
        aggregate_metrics(
            threshold,
            "TF-IDF pair similarity",
            [binary_metrics(y_true, tfidf_scores, continuous)],
            positive_count,
            negative_count,
        ),
        aggregate_metrics(
            threshold,
            "BM25 pair similarity",
            [binary_metrics(y_true, bm25_scores, continuous)],
            positive_count,
            negative_count,
        ),
    ]

    dp_rows = []
    dp_ker_rows = []
    for seed in DEFAULT_DP_SEEDS:
        seed_direction_rows: list[dict[str, Any]] = []
        dp_scores = dp_embedding_pair_scores(candidate_vectors, job_vectors, epsilon=EPSILON, seed=seed)
        dp_ker_scores = 0.45 * minmax(dp_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
        dp_scores = orient_scores(f"threshold={threshold} DP-Embedding seed={seed}", y_true, dp_scores, seed_direction_rows)
        dp_ker_scores = orient_scores(f"threshold={threshold} DP-KER seed={seed}", y_true, dp_ker_scores, seed_direction_rows)
        direction_rows.extend(seed_direction_rows)
        dp_rows.append(binary_metrics(y_true, dp_scores, continuous))
        dp_ker_rows.append(binary_metrics(y_true, dp_ker_scores, continuous))

    dp_extra = {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)}
    rows.append(aggregate_metrics(threshold, "DP-Embedding pair similarity", dp_rows, positive_count, negative_count, dp_extra))
    rows.append(aggregate_metrics(threshold, "DP-KER full pair score", dp_ker_rows, positive_count, negative_count, dp_extra))
    return rows


def run_ranking_threshold(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    y_true = np.asarray([int(record["label"]["label"]) for record in records], dtype=int)
    positive_count = int(np.sum(y_true))
    negative_count = int(len(y_true) - positive_count)
    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    hard_similarity = build_similarity_matrix(candidate_texts, job_texts)
    embedding_scores, _backend, embedding_candidate_vectors, embedding_job_vectors = embedding_pair_scores(candidate_texts, job_texts)
    bm25_global = BM25Okapi([tokenize(text) for text in job_texts])
    bm25_diag = np.asarray([bm25_global.get_scores(tokenize(text))[index] for index, text in enumerate(candidate_texts)], dtype=float)
    kg_diag = np.asarray([kg_scores_for_pool(records, index, [index])[0] for index in range(len(records))], dtype=float)
    base_multipliers = {
        "TF-IDF+Cosine": direction_multiplier(y_true, np.diag(hard_similarity)),
        "BM25": direction_multiplier(y_true, bm25_diag),
    }

    rows = []
    for negative_type in ["random", "hard"]:
        for method in METHODS_RANKING:
            metric_rows = []
            seeds = DEFAULT_DP_SEEDS if negative_type == "random" or method.startswith("DP") else [DEFAULT_DP_SEEDS[0]]
            for seed in seeds:
                pools = build_pools(records, negative_type=negative_type, seed=seed, hard_similarity=hard_similarity)
                score_multiplier = base_multipliers.get(method, 1.0)
                dp_candidate_vectors = None
                dp_job_vectors = None
                if method.startswith("DP"):
                    dp_candidate_vectors = private_embeddings(embedding_candidate_vectors, epsilon=EPSILON, seed=seed, normalize=True)
                    dp_job_vectors = private_embeddings(embedding_job_vectors, epsilon=EPSILON, seed=seed + 1, normalize=True)
                    dp_diag = rowwise_cosine(dp_candidate_vectors, dp_job_vectors)
                    if method == "DP-KER":
                        full_diag = 0.45 * minmax(dp_diag) + 0.45 * minmax(kg_diag) + 0.10 * minmax(bm25_diag)
                        score_multiplier = direction_multiplier(y_true, full_diag)
                    else:
                        score_multiplier = direction_multiplier(y_true, dp_diag)
                metric_rows.append(
                    evaluate_pools(
                        records=records,
                        pools=pools,
                        method=method,
                        candidate_texts=candidate_texts,
                        job_texts=job_texts,
                        embedding_candidate_vectors=embedding_candidate_vectors,
                        embedding_job_vectors=embedding_job_vectors,
                        dp_candidate_vectors=dp_candidate_vectors,
                        dp_job_vectors=dp_job_vectors,
                        score_multiplier=score_multiplier,
                    )
                )
            extra = {}
            if method.startswith("DP"):
                extra.update({"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON)})
            extra["seeds"] = ";".join(str(seed) for seed in seeds)
            rows.append(aggregate_ranking(threshold, method, negative_type, metric_rows, positive_count, negative_count, extra))
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_summary(pairwise_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]) -> None:
    counts = []
    for threshold in THRESHOLDS:
        row = next(row for row in pairwise_rows if row["threshold"] == threshold)
        counts.append(
            {
                "threshold": threshold,
                "positive_count": row["positive_count"],
                "negative_count": row["negative_count"],
                "total_count": row["total_count"],
            }
        )

    pair_by = {(float(row["threshold"]), row["Method"]): row for row in pairwise_rows}
    rank_by = {(float(row["threshold"]), row["Method"], row["NegativeType"]): row for row in ranking_rows}
    stability_lines = []
    for threshold in THRESHOLDS:
        dp = pair_by[(threshold, "DP-Embedding pair similarity")]
        dpker = pair_by[(threshold, "DP-KER full pair score")]
        auc_delta = float(dpker["AUC_mean"]) - float(dp["AUC_mean"])
        random_recall_delta = float(rank_by[(threshold, "DP-KER", "random")]["Recall@5_mean"]) - float(
            rank_by[(threshold, "DP-Embedding", "random")]["Recall@5_mean"]
        )
        random_ndcg_delta = float(rank_by[(threshold, "DP-KER", "random")]["NDCG@5_mean"]) - float(
            rank_by[(threshold, "DP-Embedding", "random")]["NDCG@5_mean"]
        )
        hard_recall_delta = float(rank_by[(threshold, "DP-KER", "hard")]["Recall@5_mean"]) - float(
            rank_by[(threshold, "DP-Embedding", "hard")]["Recall@5_mean"]
        )
        hard_ndcg_delta = float(rank_by[(threshold, "DP-KER", "hard")]["NDCG@5_mean"]) - float(
            rank_by[(threshold, "DP-Embedding", "hard")]["NDCG@5_mean"]
        )
        stability_lines.append(
            {
                "threshold": threshold,
                "pairwise_auc_delta": auc_delta,
                "random_recall_delta": random_recall_delta,
                "random_ndcg_delta": random_ndcg_delta,
                "hard_recall_delta": hard_recall_delta,
                "hard_ndcg_delta": hard_ndcg_delta,
            }
        )

    pairwise_stable = all(float(row["pairwise_auc_delta"]) > 0 for row in stability_lines)
    random_stable = all(float(row["random_recall_delta"]) > 0 and float(row["random_ndcg_delta"]) > 0 for row in stability_lines)
    hard_stable = all(float(row["hard_recall_delta"]) > 0 and float(row["hard_ndcg_delta"]) > 0 for row in stability_lines)

    summary = f"""# Threshold Sensitivity Analysis

This experiment rebuilds score-threshold pseudo labels at thresholds 6.5, 7.0, and 7.5 without modifying existing fixed result CSV files.

## Label Counts

{markdown_table(counts, ["threshold", "positive_count", "negative_count", "total_count"])}

## Pairwise Results

{markdown_table(pairwise_rows, ["threshold", "Method", "AUC_mean", "AUC_std", "Accuracy_mean", "F1_mean", "Precision_mean", "Recall_mean"])}

## Sampled Ranking Results

{markdown_table(ranking_rows, ["threshold", "Method", "NegativeType", "Recall@5_mean", "Recall@5_std", "NDCG@5_mean", "NDCG@5_std", "MRR_mean", "Hit@1_mean"])}

## DP-KER vs DP-Embedding Deltas

{markdown_table(stability_lines, ["threshold", "pairwise_auc_delta", "random_recall_delta", "random_ndcg_delta", "hard_recall_delta", "hard_ndcg_delta"])}

## Stability Audit

- Pairwise AUC: {'stable' if pairwise_stable else 'not stable'}; DP-KER is compared against DP-Embedding at each threshold.
- Random sampled ranking Recall@5/NDCG@5: {'stable' if random_stable else 'not stable'}.
- Hard sampled ranking Recall@5/NDCG@5: {'stable' if hard_stable else 'not stable'}.

The result should be interpreted as sensitivity analysis for pseudo labels, not as validation against explicit matched/mismatched ground truth. If space is limited, this is better used as a short robustness sentence or appendix-style table rather than a main claim table.
"""
    SUMMARY_OUTPUT.write_text(summary, encoding="utf-8")


def main() -> None:
    base_records = load_base_rows()
    if not base_records:
        raise FileNotFoundError("No score-threshold source labels found. Run convert_hf_dataset.py first.")

    pairwise_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        records = records_for_threshold(base_records, threshold)
        pairwise_rows.extend(run_pairwise_threshold(records, threshold, direction_rows))
        ranking_rows.extend(run_ranking_threshold(records, threshold))

    write_csv(PAIRWISE_OUTPUT, pairwise_rows)
    write_csv(RANKING_OUTPUT, ranking_rows)
    write_csv(DEBUG_DIRECTION_OUTPUT, direction_rows)
    write_summary(pairwise_rows, ranking_rows)
    print(f"wrote {PAIRWISE_OUTPUT}")
    print(f"wrote {RANKING_OUTPUT}")
    print(f"wrote {SUMMARY_OUTPUT}")
    print(f"wrote {DEBUG_DIRECTION_OUTPUT}")


if __name__ == "__main__":
    main()
