from __future__ import annotations

import csv
from typing import Any

import numpy as np

from baselines import build_candidate_texts, build_text_corpus
from data_loader import RESULTS_DIR, load_converted_dataset
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from run_pairwise_matching import (
    binary_metrics,
    bm25_pair_scores,
    embedding_pair_scores,
    kg_pair_scores,
    labels_and_scores,
    minmax,
    orient_scores,
)


EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]


def aggregate(epsilon: float, metric_rows: list[dict[str, float]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epsilon": epsilon,
        "sigma": gaussian_sigma(epsilon),
        "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS),
    }
    for metric in ["AUC", "Accuracy", "F1", "Precision", "Recall", "Spearman", "Pearson"]:
        values = np.asarray([metric_row[metric] for metric_row in metric_rows], dtype=float)
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(np.nanstd(values))
    return row


def main() -> None:
    records = load_converted_dataset()
    if not records:
        raise FileNotFoundError("Converted JSONL files are missing. Run convert_hf_dataset.py first.")

    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    y_true, continuous = labels_and_scores(records)
    direction_rows: list[dict[str, Any]] = []
    bm25_scores = bm25_pair_scores(candidate_texts, job_texts)
    _embedding_scores, _backend, candidate_vectors, job_vectors = embedding_pair_scores(candidate_texts, job_texts)
    kg_scores = kg_pair_scores(records)
    bm25_scores = orient_scores("privacy bm25", y_true, bm25_scores, direction_rows)
    kg_scores = orient_scores("privacy kg", y_true, kg_scores, direction_rows)

    rows = []
    for epsilon in EPSILONS:
        metric_rows = []
        for seed in DEFAULT_DP_SEEDS:
            private_candidates = private_embeddings(candidate_vectors, epsilon=epsilon, seed=seed, normalize=True)
            private_jobs = private_embeddings(job_vectors, epsilon=epsilon, seed=seed + 1, normalize=True)
            dp_scores = np.sum(private_candidates * private_jobs, axis=1)
            dp_ker_scores = 0.45 * minmax(dp_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
            dp_ker_scores = orient_scores(f"privacy epsilon={epsilon} seed={seed}", y_true, dp_ker_scores, direction_rows)
            metric_rows.append(binary_metrics(y_true, dp_ker_scores, continuous))
        rows.append(aggregate(epsilon, metric_rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "privacy_budget_results_fixed.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
