from __future__ import annotations

import csv
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from baselines import BM25Okapi, build_candidate_texts, build_text_corpus
from data_loader import RESULTS_DIR, load_converted_dataset
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from embeddings import build_embedding_backend
from kg_score import explanation_consistency, score_sample, tokenize


EPSILON = 2.0
NEGATIVES_PER_QUERY = 19
TOP_K = 5


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def build_similarity_matrix(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    vectorizer.fit(candidate_texts + job_texts)
    candidate_vectors = vectorizer.transform(candidate_texts)
    job_vectors = vectorizer.transform(job_texts)
    return np.asarray(cosine_similarity(candidate_vectors, job_vectors), dtype=float)


def build_pools(
    records: list[dict[str, Any]],
    negative_type: str,
    seed: int,
    hard_similarity: np.ndarray,
) -> list[tuple[int, list[int]]]:
    rng = np.random.default_rng(seed)
    positive_indices = [index for index, record in enumerate(records) if bool(record["label"]["label"])]
    negative_indices = [index for index, record in enumerate(records) if not bool(record["label"]["label"])]
    pools: list[tuple[int, list[int]]] = []
    for index in positive_indices:
        candidates = [candidate for candidate in negative_indices if candidate != index]
        if len(candidates) < NEGATIVES_PER_QUERY:
            candidates = [candidate for candidate in range(len(records)) if candidate != index]
        if negative_type == "random":
            negatives = rng.choice(candidates, size=NEGATIVES_PER_QUERY, replace=False).tolist()
        elif negative_type == "hard":
            ranked = sorted(candidates, key=lambda candidate: hard_similarity[index, candidate], reverse=True)
            negatives = ranked[:NEGATIVES_PER_QUERY]
        else:
            raise ValueError(f"unknown negative_type: {negative_type}")
        pools.append((index, [index, *negatives]))
    return pools


def kg_scores_for_pool(records: list[dict[str, Any]], query_index: int, pool: list[int]) -> np.ndarray:
    query_text = records[query_index]["candidate"]["candidate_text"]
    scores = []
    for candidate_index in pool:
        knowledge = records[candidate_index]["knowledge"]
        components = score_sample(
            candidate_text=query_text,
            knowledge_text=knowledge["knowledge_text"],
            minimum_requirements=knowledge.get("minimum_requirements", []),
            additional_info=knowledge.get("additional_info", ""),
            macro_dict=knowledge.get("macro_dict", {}),
            micro_dict=knowledge.get("micro_dict", {}),
        )
        scores.append(components["overall"])
    return np.asarray(scores, dtype=float)


def score_pool(
    method: str,
    records: list[dict[str, Any]],
    query_index: int,
    pool: list[int],
    candidate_texts: list[str],
    job_texts: list[str],
    embedding_candidate_vectors: np.ndarray,
    embedding_job_vectors: np.ndarray,
    dp_candidate_vectors: np.ndarray | None = None,
    dp_job_vectors: np.ndarray | None = None,
) -> np.ndarray:
    query_text = candidate_texts[query_index]
    pool_job_texts = [job_texts[index] for index in pool]
    if method == "TF-IDF+Cosine":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        vectorizer.fit([query_text, *pool_job_texts])
        query_vector = vectorizer.transform([query_text])
        job_vectors = vectorizer.transform(pool_job_texts)
        return np.asarray(cosine_similarity(query_vector, job_vectors)).ravel()
    if method == "BM25":
        bm25 = BM25Okapi([tokenize(text) for text in pool_job_texts])
        return np.asarray(bm25.get_scores(tokenize(query_text)), dtype=float)
    if method == "Embedding":
        query_vector = embedding_candidate_vectors[query_index].reshape(1, -1)
        job_vectors = embedding_job_vectors[pool]
        return np.asarray(cosine_similarity(query_vector, job_vectors)).ravel()
    if method == "DP-Embedding":
        if dp_candidate_vectors is None or dp_job_vectors is None:
            raise ValueError("DP vectors are required for DP-Embedding")
        query_vector = dp_candidate_vectors[query_index].reshape(1, -1)
        job_vectors = dp_job_vectors[pool]
        return np.asarray(cosine_similarity(query_vector, job_vectors)).ravel()
    if method == "KG-Enhanced":
        return kg_scores_for_pool(records, query_index, pool)
    if method == "DP-KER":
        if dp_candidate_vectors is None or dp_job_vectors is None:
            raise ValueError("DP vectors are required for DP-KER")
        query_vector = dp_candidate_vectors[query_index].reshape(1, -1)
        job_vectors = dp_job_vectors[pool]
        dp_scores = np.asarray(cosine_similarity(query_vector, job_vectors)).ravel()
        kg_scores = kg_scores_for_pool(records, query_index, pool)
        bm25 = BM25Okapi([tokenize(text) for text in pool_job_texts])
        bm25_scores = np.asarray(bm25.get_scores(tokenize(query_text)), dtype=float)
        return 0.45 * minmax(dp_scores) + 0.45 * minmax(kg_scores) + 0.10 * minmax(bm25_scores)
    raise ValueError(f"unknown method: {method}")


def evaluate_pools(
    records: list[dict[str, Any]],
    pools: list[tuple[int, list[int]]],
    method: str,
    candidate_texts: list[str],
    job_texts: list[str],
    embedding_candidate_vectors: np.ndarray,
    embedding_job_vectors: np.ndarray,
    dp_candidate_vectors: np.ndarray | None = None,
    dp_job_vectors: np.ndarray | None = None,
    score_multiplier: float = 1.0,
) -> dict[str, float]:
    recall_values = []
    ndcg_values = []
    mrr_values = []
    hit_values = []
    consistency_values = []

    for query_index, pool in pools:
        scores = score_pool(
            method=method,
            records=records,
            query_index=query_index,
            pool=pool,
            candidate_texts=candidate_texts,
            job_texts=job_texts,
            embedding_candidate_vectors=embedding_candidate_vectors,
            embedding_job_vectors=embedding_job_vectors,
            dp_candidate_vectors=dp_candidate_vectors,
            dp_job_vectors=dp_job_vectors,
        )
        scores = scores * score_multiplier
        order = np.argsort(-scores, kind="mergesort")
        rank = int(np.where(order == 0)[0][0] + 1)
        top_index = pool[int(order[0])]
        knowledge = records[top_index]["knowledge"]
        consistency_values.append(
            explanation_consistency(
                candidate_text=records[query_index]["candidate"]["candidate_text"],
                knowledge_text=knowledge["knowledge_text"],
                minimum_requirements=knowledge.get("minimum_requirements", []),
                additional_info=knowledge.get("additional_info", ""),
                macro_dict=knowledge.get("macro_dict", {}),
                micro_dict=knowledge.get("micro_dict", {}),
            )
        )
        recall_values.append(1.0 if rank <= TOP_K else 0.0)
        ndcg_values.append(1.0 / np.log2(rank + 1) if rank <= TOP_K else 0.0)
        mrr_values.append(1.0 / rank)
        hit_values.append(1.0 if rank == 1 else 0.0)

    return {
        "Recall@5": float(np.mean(recall_values)),
        "NDCG@5": float(np.mean(ndcg_values)),
        "MRR": float(np.mean(mrr_values)),
        "Hit@1": float(np.mean(hit_values)),
        "Consistency": float(np.mean(consistency_values)),
    }


def aggregate(method: str, negative_type: str, metric_rows: list[dict[str, float]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"Method": method, "NegativeType": negative_type}
    if extra:
        row.update(extra)
    for metric in ["Recall@5", "NDCG@5", "MRR", "Hit@1", "Consistency"]:
        values = np.asarray([metric_row[metric] for metric_row in metric_rows], dtype=float)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return np.sum(left * right, axis=1) / denom


def auc_or_nan(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else float("nan")


def direction_multiplier(y_true: np.ndarray, scores: np.ndarray) -> float:
    forward = auc_or_nan(y_true, scores)
    reverse = auc_or_nan(y_true, -scores)
    return -1.0 if np.isfinite(reverse) and reverse > forward else 1.0


def main() -> None:
    records = load_converted_dataset()
    if not records:
        raise FileNotFoundError("Converted JSONL files are missing. Run convert_hf_dataset.py first.")

    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    y_true = np.asarray([int(record["label"]["label"]) for record in records], dtype=int)
    hard_similarity = build_similarity_matrix(candidate_texts, job_texts)
    backend = build_embedding_backend(candidate_texts + job_texts, prefer_sentence_transformers=True)
    embedding_candidate_vectors = backend.transform(candidate_texts)
    embedding_job_vectors = backend.transform(job_texts)
    bm25_global = BM25Okapi([tokenize(text) for text in job_texts])
    bm25_diag = np.asarray([bm25_global.get_scores(tokenize(text))[index] for index, text in enumerate(candidate_texts)], dtype=float)
    kg_diag = np.asarray([kg_scores_for_pool(records, index, [index])[0] for index in range(len(records))], dtype=float)
    base_multipliers = {
        "TF-IDF+Cosine": direction_multiplier(y_true, np.diag(hard_similarity)),
        "BM25": direction_multiplier(y_true, bm25_diag),
        "Embedding": direction_multiplier(y_true, rowwise_cosine(embedding_candidate_vectors, embedding_job_vectors)),
        "KG-Enhanced": direction_multiplier(y_true, kg_diag),
    }

    methods = ["TF-IDF+Cosine", "BM25", "Embedding", "KG-Enhanced", "DP-Embedding", "DP-KER"]
    rows = []
    for negative_type in ["random", "hard"]:
        for method in methods:
            metric_rows = []
            seeds = DEFAULT_DP_SEEDS if negative_type == "random" or method.startswith("DP") else [DEFAULT_DP_SEEDS[0]]
            for seed in seeds:
                pools = build_pools(records, negative_type=negative_type, seed=seed, hard_similarity=hard_similarity)
                dp_candidate_vectors = None
                dp_job_vectors = None
                score_multiplier = base_multipliers.get(method, 1.0)
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
            extra = {"epsilon": EPSILON, "sigma": gaussian_sigma(EPSILON)} if method.startswith("DP") else {}
            if negative_type == "random" or method.startswith("DP"):
                extra["seeds"] = ";".join(str(seed) for seed in seeds)
            rows.append(aggregate(method, negative_type, metric_rows, extra=extra))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "sampled_ranking_results_fixed.csv"
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
