from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from v2_common import CACHE_DIR, DEBUG_DIR, EPSILONS, RESULTS_DIR, SEEDS_FINAL, ensure_dirs, matrix_path, write_csv

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import BM25Okapi  # noqa: E402
from dp import gaussian_sigma, private_embeddings  # noqa: E402
from embeddings import TfidfSvdBackend  # noqa: E402
from kg_score import tokenize  # noqa: E402
from load_hf_dataset import filtered_records, load_dataset_records  # noqa: E402


def text_of(record: dict, kind: str) -> str:
    if kind == "resume":
        return record["candidate"]["candidate_text"]
    knowledge = record["knowledge"]
    parts = [
        knowledge.get("job_description", ""),
        knowledge.get("additional_info", ""),
        "\n".join(knowledge.get("minimum_requirements", [])),
        str(knowledge.get("macro_dict", {})),
        str(knowledge.get("micro_dict", {})),
    ]
    return "\n\n".join(part for part in parts if str(part).strip())


def token_matrix_features(records: list[dict]) -> dict[str, np.ndarray]:
    resume_tokens = [set(tokenize(text_of(record, "resume"))) for record in records]
    criteria_tokens = []
    requirement_tokens = []
    for record in records:
        knowledge = record["knowledge"]
        terms = set()
        for key in list((knowledge.get("macro_dict") or {}).keys()) + list((knowledge.get("micro_dict") or {}).keys()):
            terms.update(tokenize(str(key)))
        criteria_tokens.append(terms)
        req_terms = set()
        for req in knowledge.get("minimum_requirements", []):
            req_terms.update(tokenize(str(req)))
        requirement_tokens.append(req_terms)

    n = len(records)
    skill_cov = np.zeros((n, n), dtype=np.float32)
    overlap_count = np.zeros((n, n), dtype=np.float32)
    missing_ratio = np.zeros((n, n), dtype=np.float32)
    req_cov = np.zeros((n, n), dtype=np.float32)
    for i, rtokens in enumerate(resume_tokens):
        for j in range(n):
            ctokens = criteria_tokens[j]
            qtokens = requirement_tokens[j]
            overlap = len(rtokens & ctokens)
            overlap_count[i, j] = float(overlap)
            skill_cov[i, j] = float(overlap / len(ctokens)) if ctokens else 0.0
            missing_ratio[i, j] = float(1.0 - skill_cov[i, j]) if ctokens else 0.0
            req_cov[i, j] = float(len(rtokens & qtokens) / len(qtokens)) if qtokens else 0.0
    return {
        "kg_skill_coverage_matrix": skill_cov,
        "skill_overlap_count_matrix": overlap_count,
        "missing_skill_ratio_matrix": missing_ratio,
        "requirement_coverage_matrix": req_cov,
    }


def main() -> None:
    ensure_dirs()
    all_records, load_meta = load_dataset_records()
    records = filtered_records(all_records, threshold=7.0)
    if not records:
        raise RuntimeError("No valid scored records available.")

    resume_texts = [text_of(record, "resume") for record in records]
    jd_texts = [text_of(record, "jd") for record in records]
    raw_score = np.asarray([float(record["raw_score"]) for record in records], dtype=float)
    binary_label = (raw_score >= 7.0).astype(int)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_features=12000)
    vectorizer.fit(resume_texts + jd_texts)
    resume_tfidf = vectorizer.transform(resume_texts)
    jd_tfidf = vectorizer.transform(jd_texts)
    tfidf_matrix = np.asarray(cosine_similarity(resume_tfidf, jd_tfidf), dtype=np.float32)

    bm25 = BM25Okapi([tokenize(text) for text in jd_texts])
    bm25_matrix = np.asarray([bm25.get_scores(tokenize(text)) for text in resume_texts], dtype=np.float32)

    embedding_backend = TfidfSvdBackend(max_features=12000, max_components=96)
    embedding_backend.fit(resume_texts + jd_texts)
    resume_embeddings = embedding_backend.transform(resume_texts)
    jd_embeddings = embedding_backend.transform(jd_texts)
    embedding_matrix = np.asarray(cosine_similarity(resume_embeddings, jd_embeddings), dtype=np.float32)
    np.save(CACHE_DIR / "text_embeddings.npy", np.stack([resume_embeddings, jd_embeddings], axis=0))

    kg_mats = token_matrix_features(records)
    for name, matrix in kg_mats.items():
        np.save(matrix_path(name), matrix)

    dp_diag_columns: dict[str, np.ndarray] = {}
    for epsilon in EPSILONS:
        for seed in SEEDS_FINAL:
            dp_resume = private_embeddings(resume_embeddings, epsilon=epsilon, seed=seed, normalize=True)
            dp_jd = private_embeddings(jd_embeddings, epsilon=epsilon, seed=seed + 1, normalize=True)
            np.save(CACHE_DIR / f"dp_embeddings_epsilon_{str(epsilon).replace('.', '_')}_seed_{seed}.npy", np.stack([dp_resume, dp_jd], axis=0))
            dp_matrix = np.asarray(cosine_similarity(dp_resume, dp_jd), dtype=np.float32)
            name = f"dp_embedding_cosine_epsilon_{str(epsilon).replace('.', '_')}_seed_{seed}"
            np.save(matrix_path(name), dp_matrix)
            dp_diag_columns[name] = np.diag(dp_matrix)

    np.save(matrix_path("tfidf_cosine_matrix"), tfidf_matrix)
    np.save(matrix_path("bm25_score_matrix"), bm25_matrix)
    np.save(matrix_path("embedding_cosine_matrix"), embedding_matrix)

    rows = []
    for idx, record in enumerate(records):
        row = {
            "sample_id": record["sample_id"],
            "source_file": record["source_file"],
            "filename_category": record.get("filename_category"),
            "tfidf_cosine": float(tfidf_matrix[idx, idx]),
            "bm25_score": float(bm25_matrix[idx, idx]),
            "embedding_cosine": float(embedding_matrix[idx, idx]),
            "dp_embedding_cosine": float(dp_diag_columns["dp_embedding_cosine_epsilon_2_0_seed_3407"][idx]),
            "kg_skill_coverage": float(kg_mats["kg_skill_coverage_matrix"][idx, idx]),
            "skill_overlap_count": float(kg_mats["skill_overlap_count_matrix"][idx, idx]),
            "missing_skill_ratio": float(kg_mats["missing_skill_ratio_matrix"][idx, idx]),
            "requirement_coverage": float(kg_mats["requirement_coverage_matrix"][idx, idx]),
            "resume_length": len(tokenize(resume_texts[idx])),
            "jd_length": len(tokenize(jd_texts[idx])),
            "raw_score": float(raw_score[idx]),
            "binary_label": int(binary_label[idx]),
        }
        for name, values in dp_diag_columns.items():
            row[name] = float(values[idx])
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = CACHE_DIR / "pair_features.parquet"
    csv_path = CACHE_DIR / "pair_features.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        feature_path = parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        feature_path = csv_path

    stats = Counter(row["filename_category"] for row in rows)
    metadata = {
        "loaded_rows": len(all_records),
        "valid_scored_rows": len(records),
        "positive_count": int(binary_label.sum()),
        "negative_count": int(len(binary_label) - binary_label.sum()),
        "threshold": 7.0,
        "feature_path": str(feature_path),
        "cache_dir": str(CACHE_DIR),
        "source_used": load_meta.get("source"),
        "load_error": load_meta.get("load_error", ""),
        "category_counts": dict(stats),
        "epsilon_values": EPSILONS,
        "seeds": SEEDS_FINAL,
        "dp_note": "Embedding-level DP perturbation: L2 clipping plus Gaussian mechanism noise; not DP-SGD.",
    }
    (CACHE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(
        RESULTS_DIR / "dataset_statistics_v2_final.csv",
        [
            {"metric": "loaded_rows", "value": len(all_records)},
            {"metric": "valid_scored_rows", "value": len(records)},
            {"metric": "positive_count_threshold_7", "value": int(binary_label.sum())},
            {"metric": "negative_count_threshold_7", "value": int(len(binary_label) - binary_label.sum())},
            {"metric": "source_used", "value": load_meta.get("source")},
            {"metric": "feature_cache", "value": str(feature_path)},
        ],
    )
    write_csv(
        DEBUG_DIR / "label_distribution_v2_final.csv",
        [
            {"metric": "total", "value": len(records)},
            {"metric": "positive_count", "value": int(binary_label.sum())},
            {"metric": "negative_count", "value": int(len(binary_label) - binary_label.sum())},
            {"metric": "raw_score_min", "value": float(raw_score.min())},
            {"metric": "raw_score_max", "value": float(raw_score.max())},
            {"metric": "raw_score_mean", "value": float(raw_score.mean())},
        ],
    )
    print(f"wrote feature cache to {feature_path}")
    print(f"wrote metadata to {CACHE_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
