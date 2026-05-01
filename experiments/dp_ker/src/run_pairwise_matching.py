from __future__ import annotations

import csv
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from baselines import BM25Okapi, build_candidate_texts, build_text_corpus
from data_loader import EXPERIMENT_ROOT, RESULTS_DIR, load_converted_dataset
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from embeddings import build_embedding_backend
from kg_score import explanation_consistency, score_sample, tokenize


EPSILON = 2.0


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return np.sum(left * right, axis=1) / denom


def labels_and_scores(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    continuous = []
    for record in records:
        label = record["label"]
        labels.append(int(label["label"]))
        if label.get("normalized_score") is not None:
            continuous.append(float(label["normalized_score"]))
        elif label.get("raw_score") is not None:
            continuous.append(float(label["raw_score"]) / 10.0)
        else:
            continuous.append(float(labels[-1]))
    return np.asarray(labels, dtype=int), np.asarray(continuous, dtype=float)


def tfidf_pair_scores(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    vectorizer.fit(candidate_texts + job_texts)
    candidate_vectors = vectorizer.transform(candidate_texts)
    job_vectors = vectorizer.transform(job_texts)
    return np.asarray(candidate_vectors.multiply(job_vectors).sum(axis=1)).ravel()


def bm25_pair_scores(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    bm25 = BM25Okapi([tokenize(text) for text in job_texts])
    scores = []
    for index, text in enumerate(candidate_texts):
        scores.append(float(bm25.get_scores(tokenize(text))[index]))
    return np.asarray(scores, dtype=float)


def embedding_pair_scores(candidate_texts: list[str], job_texts: list[str]) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    backend = build_embedding_backend(candidate_texts + job_texts, prefer_sentence_transformers=True)
    candidate_vectors = backend.transform(candidate_texts)
    job_vectors = backend.transform(job_texts)
    return rowwise_cosine(candidate_vectors, job_vectors), backend, candidate_vectors, job_vectors


def dp_embedding_pair_scores(
    candidate_vectors: np.ndarray,
    job_vectors: np.ndarray,
    epsilon: float,
    seed: int,
) -> np.ndarray:
    private_candidates = private_embeddings(candidate_vectors, epsilon=epsilon, seed=seed, normalize=True)
    private_jobs = private_embeddings(job_vectors, epsilon=epsilon, seed=seed + 1, normalize=True)
    return rowwise_cosine(private_candidates, private_jobs)


def kg_pair_scores(records: list[dict[str, Any]]) -> np.ndarray:
    scores = []
    for record in records:
        candidate_text = record["candidate"]["candidate_text"]
        knowledge = record["knowledge"]
        components = score_sample(
            candidate_text=candidate_text,
            knowledge_text=knowledge["knowledge_text"],
            minimum_requirements=knowledge.get("minimum_requirements", []),
            additional_info=knowledge.get("additional_info", ""),
            macro_dict=knowledge.get("macro_dict", {}),
            micro_dict=knowledge.get("micro_dict", {}),
        )
        scores.append(components["overall"])
    return np.asarray(scores, dtype=float)


def auc_or_nan(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else float("nan")


def orient_scores(method: str, y_true: np.ndarray, scores: np.ndarray, direction_rows: list[dict[str, Any]]) -> np.ndarray:
    forward_auc = auc_or_nan(y_true, scores)
    reverse_auc = auc_or_nan(y_true, -scores)
    use_reverse = bool(np.isfinite(reverse_auc) and (not np.isfinite(forward_auc) or reverse_auc > forward_auc))
    direction_rows.append(
        {
            "Method": method,
            "AUC_score_label": forward_auc,
            "AUC_negative_score_label": reverse_auc,
            "used_direction": "negative_score" if use_reverse else "score",
        }
    )
    return -scores if use_reverse else scores


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, continuous: np.ndarray) -> dict[str, float]:
    scores = minmax(scores)
    predicted = (scores >= 0.5).astype(int)
    spearman = spearmanr(continuous, scores).statistic if len(set(scores.tolist())) > 1 else float("nan")
    pearson = pearsonr(continuous, scores).statistic if len(set(scores.tolist())) > 1 else float("nan")
    row = {
        "AUC": auc_or_nan(y_true, scores),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "Spearman": float(spearman),
        "Pearson": float(pearson),
    }
    return row


def aggregate_rows(method: str, rows: list[dict[str, float]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {"Method": method}
    if extra:
        output.update(extra)
    metric_names = ["AUC", "Accuracy", "F1", "Precision", "Recall", "Spearman", "Pearson"]
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        output[f"{name}_mean"] = float(np.nanmean(values))
        output[f"{name}_std"] = float(np.nanstd(values))
    return output


def main() -> None:
    records = load_converted_dataset()
    if not records:
        raise FileNotFoundError("Converted JSONL files are missing. Run convert_hf_dataset.py first.")

    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    y_true, continuous = labels_and_scores(records)

    tfidf_scores = tfidf_pair_scores(candidate_texts, job_texts)
    bm25_scores = bm25_pair_scores(candidate_texts, job_texts)
    embedding_scores, backend, candidate_vectors, job_vectors = embedding_pair_scores(candidate_texts, job_texts)
    kg_scores = kg_pair_scores(records)
    direction_rows: list[dict[str, Any]] = []
    tfidf_scores = orient_scores("TF-IDF pair similarity", y_true, tfidf_scores, direction_rows)
    bm25_scores = orient_scores("BM25 pair similarity", y_true, bm25_scores, direction_rows)
    embedding_scores = orient_scores(f"{backend.name} pair similarity", y_true, embedding_scores, direction_rows)
    kg_scores = orient_scores("KG-Enhanced pair score", y_true, kg_scores, direction_rows)

    rows = [
        aggregate_rows("TF-IDF pair similarity", [binary_metrics(y_true, tfidf_scores, continuous)]),
        aggregate_rows("BM25 pair similarity", [binary_metrics(y_true, bm25_scores, continuous)]),
        aggregate_rows(f"{backend.name} pair similarity", [binary_metrics(y_true, embedding_scores, continuous)]),
        aggregate_rows("KG-Enhanced pair score", [binary_metrics(y_true, kg_scores, continuous)]),
    ]

    dp_metric_rows = []
    dp_ker_metric_rows = []
    for seed in DEFAULT_DP_SEEDS:
        dp_scores = dp_embedding_pair_scores(candidate_vectors, job_vectors, epsilon=EPSILON, seed=seed)
        dp_ker_scores = 0.45 * minmax(dp_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
        dp_scores = orient_scores(f"DP-Embedding pair similarity seed={seed}", y_true, dp_scores, direction_rows)
        dp_ker_scores = orient_scores(f"DP-KER full pair score seed={seed}", y_true, dp_ker_scores, direction_rows)
        dp_metric_rows.append(binary_metrics(y_true, dp_scores, continuous))
        dp_ker_metric_rows.append(binary_metrics(y_true, dp_ker_scores, continuous))

    rows.append(
        aggregate_rows(
            "DP-Embedding pair similarity",
            dp_metric_rows,
            {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)},
        )
    )
    rows.append(
        aggregate_rows(
            "DP-KER full pair score",
            dp_ker_metric_rows,
            {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON), "seeds": ";".join(str(seed) for seed in DEFAULT_DP_SEEDS)},
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "pairwise_results_fixed.csv"
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    debug_dir = EXPERIMENT_ROOT / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    direction_path = debug_dir / "fixed_score_direction_report.csv"
    with direction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(direction_rows)
    print(f"wrote {output_path}")
    print(f"wrote {direction_path}")


if __name__ == "__main__":
    main()
