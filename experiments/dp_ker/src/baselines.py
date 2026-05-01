from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dp import add_gaussian_noise, clip_by_l2_norm
from embeddings import EmbeddingBackend, build_embedding_backend
from kg_score import score_sample, tokenize

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    class BM25Okapi:  # type: ignore[misc]
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_freqs: list[dict[str, int]] = []
            self.doc_len: list[int] = []
            self.df: dict[str, int] = {}
            for doc in corpus:
                freq: dict[str, int] = {}
                for token in doc:
                    freq[token] = freq.get(token, 0) + 1
                self.doc_freqs.append(freq)
                self.doc_len.append(len(doc))
                for token in freq:
                    self.df[token] = self.df.get(token, 0) + 1
            self.avgdl = float(sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
            self.N = len(corpus)

        def get_scores(self, query: list[str]) -> np.ndarray:
            scores = np.zeros(self.N, dtype=float)
            for token in query:
                df = self.df.get(token, 0)
                if df == 0:
                    continue
                idf = np.log(1 + (self.N - df + 0.5) / (df + 0.5))
                for idx, freq in enumerate(self.doc_freqs):
                    tf = freq.get(token, 0)
                    if tf == 0:
                        continue
                    doc_len = self.doc_len[idx] or 1
                    denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                    scores[idx] += idf * tf * (self.k1 + 1) / denom
            return scores


def build_text_corpus(
    records: list[dict[str, Any]],
    use_additional_info: bool = True,
    use_requirements: bool = True,
    use_criteria: bool = True,
) -> list[str]:
    texts: list[str] = []
    for record in records:
        knowledge = record["knowledge"]
        parts = [knowledge["job_description"]]
        if use_additional_info and knowledge["additional_info"].strip():
            parts.append(knowledge["additional_info"])
        if use_requirements:
            parts.append("\n".join(knowledge.get("minimum_requirements", [])))
        if use_criteria:
            parts.append(str(knowledge.get("macro_dict", {})))
            parts.append(str(knowledge.get("micro_dict", {})))
        texts.append("\n\n".join(part for part in parts if part))
    return texts


def build_candidate_texts(records: list[dict[str, Any]]) -> list[str]:
    return [record["candidate"]["candidate_text"] for record in records]


def build_labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(record["label"]["label"]) for record in records], dtype=int)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def bm25_scores(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    tokenized_jobs = [tokenize(text) for text in job_texts]
    bm25 = BM25Okapi(tokenized_jobs)
    return np.asarray([bm25.get_scores(tokenize(text)).max() for text in candidate_texts], dtype=float)


def bm25_score_matrix(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    tokenized_jobs = [tokenize(text) for text in job_texts]
    bm25 = BM25Okapi(tokenized_jobs)
    rows = [bm25.get_scores(tokenize(text)) for text in candidate_texts]
    return np.asarray(rows, dtype=float)


def tfidf_score_matrix(candidate_texts: list[str], job_texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    combined = candidate_texts + job_texts
    vectorizer.fit(combined)
    candidate_vectors = vectorizer.transform(candidate_texts)
    job_vectors = vectorizer.transform(job_texts)
    return np.asarray(cosine_similarity(candidate_vectors, job_vectors), dtype=float)


def embedding_score_matrix(
    candidate_texts: list[str],
    job_texts: list[str],
    backend: EmbeddingBackend | None = None,
    prefer_sentence_transformers: bool = True,
) -> tuple[np.ndarray, EmbeddingBackend]:
    corpus = candidate_texts + job_texts
    backend = backend or build_embedding_backend(corpus, prefer_sentence_transformers=prefer_sentence_transformers)
    candidate_vectors = backend.transform(candidate_texts)
    job_vectors = backend.transform(job_texts)
    score_matrix = cosine_similarity(candidate_vectors, job_vectors)
    return np.asarray(score_matrix, dtype=float), backend


def noisy_embedding_score_matrix(
    candidate_texts: list[str],
    job_texts: list[str],
    epsilon: float,
    backend: EmbeddingBackend | None = None,
    prefer_sentence_transformers: bool = True,
) -> np.ndarray:
    matrix, backend = embedding_score_matrix(
        candidate_texts,
        job_texts,
        backend=backend,
        prefer_sentence_transformers=prefer_sentence_transformers,
    )
    candidate_vectors = backend.transform(candidate_texts)
    job_vectors = backend.transform(job_texts)
    candidate_vectors = clip_by_l2_norm(candidate_vectors, max_norm=1.0)
    job_vectors = clip_by_l2_norm(job_vectors, max_norm=1.0)
    noisy_candidate = clip_by_l2_norm(add_gaussian_noise(candidate_vectors, epsilon=epsilon), max_norm=2.0)
    noisy_job = clip_by_l2_norm(add_gaussian_noise(job_vectors, epsilon=epsilon, seed=3408), max_norm=2.0)
    return np.asarray(cosine_similarity(noisy_candidate, noisy_job), dtype=float)


def knowledge_scores(records: list[dict[str, Any]]) -> np.ndarray:
    scores = []
    for record in records:
        label = record["label"]
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


def knowledge_score_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for query in records:
        query_rows: list[float] = []
        for candidate in records:
            components = score_sample(
                candidate_text=query["candidate"]["candidate_text"],
                knowledge_text=candidate["knowledge"]["knowledge_text"],
                minimum_requirements=candidate["knowledge"].get("minimum_requirements", []),
                additional_info=candidate["knowledge"].get("additional_info", ""),
                macro_dict=candidate["knowledge"].get("macro_dict", {}),
                micro_dict=candidate["knowledge"].get("micro_dict", {}),
            )
            query_rows.append(components["overall"])
        rows.append(query_rows)
    return np.asarray(rows, dtype=float)


def build_feature_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record["sample_id"],
            "source_file": record["candidate"]["source_file"],
            "label": bool(record["label"]["label"]),
            "candidate_text": record["candidate"]["candidate_text"],
            "job_text": build_text_corpus([record])[0],
        }
        for record in records
    ]


def evaluate_methods(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_texts = build_candidate_texts(records)
    job_texts = build_text_corpus(records)
    tfidf_raw = tfidf_score_matrix(candidate_texts, job_texts)
    bm25_raw = bm25_score_matrix(candidate_texts, job_texts)
    sbert_raw, backend = embedding_score_matrix(candidate_texts, job_texts, prefer_sentence_transformers=True)
    knowledge_raw = knowledge_score_matrix(records)
    dp_embedding_raw = noisy_embedding_score_matrix(candidate_texts, job_texts, epsilon=2.0, backend=backend)
    dp_ker_raw = 0.55 * _minmax(dp_embedding_raw) + 0.45 * _minmax(knowledge_raw)
    rag_raw = 0.35 * _minmax(tfidf_raw) + 0.25 * _minmax(bm25_raw) + 0.20 * _minmax(sbert_raw) + 0.20 * _minmax(knowledge_raw)

    methods = {
        "TF-IDF+Cosine": _minmax(tfidf_raw),
        "BM25": _minmax(bm25_raw),
        "SBERT+Cosine": _minmax(sbert_raw),
        "RAG-based Rec.": rag_raw,
        "DP-Embedding": _minmax(dp_embedding_raw),
        "DP-KER": _minmax(dp_ker_raw),
    }
    rows = []
    for method, raw_scores in methods.items():
        rows.append(
            {
                "method": method,
                "score_mean": float(np.mean(raw_scores)),
                "score_std": float(np.std(raw_scores)),
                "score_matrix": raw_scores.tolist(),
            }
        )
    return rows
