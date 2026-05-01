from __future__ import annotations

import csv
from typing import Any

import numpy as np

from data_loader import RESULTS_DIR, load_converted_dataset
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from run_pairwise_matching import (
    EPSILON,
    aggregate_rows,
    binary_metrics,
    bm25_pair_scores,
    embedding_pair_scores,
    kg_pair_scores,
    labels_and_scores,
    minmax,
    orient_scores,
    tfidf_pair_scores,
)
from baselines import build_candidate_texts, build_text_corpus


def main() -> None:
    records = load_converted_dataset()
    if not records:
        raise FileNotFoundError("Converted JSONL files are missing. Run convert_hf_dataset.py first.")

    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    y_true, continuous = labels_and_scores(records)
    direction_rows: list[dict[str, Any]] = []

    tfidf_scores = tfidf_pair_scores(candidate_texts, job_texts)
    bm25_scores = bm25_pair_scores(candidate_texts, job_texts)
    embedding_scores, _backend, candidate_vectors, job_vectors = embedding_pair_scores(candidate_texts, job_texts)
    kg_scores = kg_pair_scores(records)
    tfidf_scores = orient_scores("ablation retrieval_only", y_true, tfidf_scores, direction_rows)
    bm25_scores = orient_scores("ablation bm25", y_true, bm25_scores, direction_rows)
    embedding_scores = orient_scores("ablation embedding", y_true, embedding_scores, direction_rows)
    kg_scores = orient_scores("ablation kg", y_true, kg_scores, direction_rows)

    rows: list[dict[str, Any]] = []
    full_metrics = []
    no_kg_metrics = []
    no_fusion_metrics = []
    for seed in DEFAULT_DP_SEEDS:
        private_candidates = private_embeddings(candidate_vectors, epsilon=EPSILON, seed=seed, normalize=True)
        private_jobs = private_embeddings(job_vectors, epsilon=EPSILON, seed=seed + 1, normalize=True)
        dp_scores = np.sum(private_candidates * private_jobs, axis=1)
        full_scores = 0.45 * minmax(dp_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
        no_fusion_scores = np.maximum(minmax(dp_scores), minmax(kg_scores))
        full_scores = orient_scores(f"ablation full seed={seed}", y_true, full_scores, direction_rows)
        dp_scores = orient_scores(f"ablation w/o_knowledge seed={seed}", y_true, dp_scores, direction_rows)
        no_fusion_scores = orient_scores(f"ablation w/o_ranking_fusion seed={seed}", y_true, no_fusion_scores, direction_rows)
        full_metrics.append(binary_metrics(y_true, full_scores, continuous))
        no_kg_metrics.append(binary_metrics(y_true, dp_scores, continuous))
        no_fusion_metrics.append(binary_metrics(y_true, no_fusion_scores, continuous))

    no_dp_scores = 0.45 * minmax(embedding_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
    retrieval_only_scores = tfidf_scores
    no_dp_scores = orient_scores("ablation w/o_dp", y_true, no_dp_scores, direction_rows)

    rows.append(
        aggregate_rows(
            "full",
            full_metrics,
            {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)},
        )
    )
    rows.append(aggregate_rows("w/o_dp", [binary_metrics(y_true, no_dp_scores, continuous)]))
    rows.append(
        aggregate_rows(
            "w/o_knowledge",
            no_kg_metrics,
            {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)},
        )
    )
    rows.append(
        aggregate_rows(
            "w/o_ranking_fusion",
            no_fusion_metrics,
            {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)},
        )
    )
    rows.append(aggregate_rows("retrieval_only", [binary_metrics(y_true, retrieval_only_scores, continuous)]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ablation_results_fixed.csv"
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
