from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, train_test_split

from baselines import BM25Okapi
from data_loader import SEED
from dp import DEFAULT_DP_SEEDS, gaussian_sigma, private_embeddings
from embeddings import TfidfSvdBackend
from kg_score import score_sample, tokenize
from load_hf_dataset import DEBUG_V2_DIR, RESULTS_V2_DIR, filtered_records, load_dataset_records


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FEATURE_NAMES = [
    "tfidf_cosine",
    "bm25_score",
    "embedding_cosine",
    "dp_embedding_cosine",
    "kg_skill_coverage",
    "skill_overlap_count",
    "missing_skill_ratio",
    "requirement_coverage",
    "title_or_category_match",
]
SEEDS = DEFAULT_DP_SEEDS
EPSILON = 2.0
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
THRESHOLDS = [6.5, 7.0, 7.5]
NEGATIVES_PER_QUERY = 19
TOP_K = 5


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.zeros_like(values)
    return (values - low) / (high - low)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else float("nan")


def orient_by_train(y_train: np.ndarray, train_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    forward = safe_auc(y_train, train_scores)
    reverse = safe_auc(y_train, -train_scores)
    if np.isfinite(reverse) and reverse > forward:
        return -train_scores, -test_scores, "negative_score"
    return train_scores, test_scores, "score"


def regression_metrics(y_score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = minmax(pred)
    return {
        "MAE": float(mean_absolute_error(y_score, pred)),
        "RMSE": float(math.sqrt(np.mean((y_score - pred) ** 2))),
        "Spearman": float(spearmanr(y_score, pred).statistic) if len(set(pred.tolist())) > 1 else float("nan"),
        "Pearson": float(pearsonr(y_score, pred).statistic) if len(set(pred.tolist())) > 1 else float("nan"),
    }


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = minmax(scores)
    pred = (scores >= 0.5).astype(int)
    return {
        "AUC": safe_auc(y_true, scores),
        "PR-AUC": float(average_precision_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else float("nan"),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def aggregate(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        out = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values))
        out["runs"] = len(group_rows)
        output.append(out)
    return output


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


def candidate_text(record: dict[str, Any]) -> str:
    return record["candidate"]["candidate_text"]


def job_text(record: dict[str, Any]) -> str:
    knowledge = record["knowledge"]
    parts = [
        knowledge.get("job_description", ""),
        knowledge.get("additional_info", ""),
        "\n".join(knowledge.get("minimum_requirements", [])),
        str(knowledge.get("macro_dict", {})),
        str(knowledge.get("micro_dict", {})),
    ]
    return "\n\n".join(part for part in parts if str(part).strip())


@dataclass
class FeatureBuilder:
    tfidf: TfidfVectorizer
    embedding: TfidfSvdBackend

    @classmethod
    def fit(cls, train_records: list[dict[str, Any]]) -> "FeatureBuilder":
        texts = [candidate_text(record) for record in train_records] + [job_text(record) for record in train_records]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_features=4000)
        tfidf.fit(texts)
        embedding = TfidfSvdBackend(max_features=4000, max_components=48)
        embedding.fit(texts)
        return cls(tfidf=tfidf, embedding=embedding)

    def transform_pairs(self, candidate_texts: list[str], job_records: list[dict[str, Any]], epsilon: float, seed: int) -> np.ndarray:
        job_texts = [job_text(record) for record in job_records]
        cand_tfidf = self.tfidf.transform(candidate_texts)
        job_tfidf = self.tfidf.transform(job_texts)
        tfidf_cos = np.asarray(cand_tfidf.multiply(job_tfidf).sum(axis=1)).ravel()

        cand_emb = self.embedding.transform(candidate_texts)
        job_emb = self.embedding.transform(job_texts)
        embedding_cos = rowwise_cosine(cand_emb, job_emb)
        # Embedding-level DP perturbation only: L2 clipping plus Gaussian mechanism noise.
        # This does not claim DP-SGD or private model training.
        dp_cand = private_embeddings(cand_emb, epsilon=epsilon, seed=seed, normalize=True)
        dp_job = private_embeddings(job_emb, epsilon=epsilon, seed=seed + 1, normalize=True)
        dp_embedding_cos = rowwise_cosine(dp_cand, dp_job)

        feature_rows = []
        for cand_text, record in zip(candidate_texts, job_records):
            knowledge = record["knowledge"]
            bm25 = BM25Okapi([tokenize(job_text(record))])
            bm25_score = float(bm25.get_scores(tokenize(cand_text))[0])
            components = score_sample(
                candidate_text=cand_text,
                knowledge_text=knowledge.get("knowledge_text", job_text(record)),
                minimum_requirements=knowledge.get("minimum_requirements", []),
                additional_info=knowledge.get("additional_info", ""),
                macro_dict=knowledge.get("macro_dict", {}),
                micro_dict=knowledge.get("micro_dict", {}),
            )
            candidate_tokens = set(tokenize(cand_text))
            skill_terms = set()
            for key in list((knowledge.get("macro_dict") or {}).keys()) + list((knowledge.get("micro_dict") or {}).keys()):
                skill_terms.update(tokenize(str(key)))
            overlap = len(candidate_tokens & skill_terms)
            req_cov = components["requirement_coverage"]
            title_match = 0.0
            feature_rows.append(
                [
                    0.0,  # filled below
                    bm25_score,
                    0.0,
                    0.0,
                    components["criteria_coverage"],
                    float(overlap),
                    float(1.0 - req_cov),
                    req_cov,
                    title_match,
                ]
            )
        features = np.asarray(feature_rows, dtype=float)
        features[:, 0] = tfidf_cos
        features[:, 2] = embedding_cos
        features[:, 3] = dp_embedding_cos
        return features


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return np.sum(left * right, axis=1) / denom


def labels(records: list[dict[str, Any]], threshold: float = 7.0) -> np.ndarray:
    return np.asarray([int(float(record["raw_score"]) >= threshold) for record in records], dtype=int)


def continuous(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(record["raw_score"]) / 10.0 for record in records], dtype=float)


def method_scores_from_features(features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "TF-IDF": features[:, 0],
        "BM25": features[:, 1],
        "Embedding": features[:, 2],
        "KG-only": features[:, 4],
        "DP-Embedding": features[:, 3],
        "DP-KER-v1": 0.45 * minmax(features[:, 3]) + 0.45 * minmax(features[:, 4]) + 0.10 * minmax(features[:, 1]),
    }


def train_best_classifier(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[str, Any, list[dict[str, Any]]]:
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=30, max_depth=5, class_weight="balanced", random_state=seed, n_jobs=1)),
        ("HistGradientBoostingClassifier", HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=7, random_state=seed)),
    ]
    importance_rows = []
    if len(set(y_train.tolist())) == 2 and min(np.bincount(y_train)) >= 2 and len(y_train) >= 20:
        tr_idx, va_idx = train_test_split(
            np.arange(len(y_train)),
            test_size=0.25,
            stratify=y_train,
            random_state=seed,
        )
    else:
        tr_idx = va_idx = np.arange(len(y_train))

    best_name = models[0][0]
    best_auc = -np.inf
    best_model = models[0][1]
    for name, model in models:
        try:
            model.fit(x_train[tr_idx], y_train[tr_idx])
            scores = model.predict_proba(x_train[va_idx])[:, 1]
        except Exception:
            continue
        auc = safe_auc(y_train[va_idx], scores)
        auc_value = auc if np.isfinite(auc) else 0.0
        if auc_value > best_auc:
            best_auc = auc_value
            best_name = name
            best_model = model
    best_model.fit(x_train, y_train)
    if hasattr(best_model, "feature_importances_"):
        values = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        values = np.abs(best_model.coef_[0])
    else:
        values = np.zeros(len(FEATURE_NAMES), dtype=float)
    for feature, value in zip(FEATURE_NAMES, values):
        importance_rows.append({"model": best_name, "feature": feature, "importance": float(value)})
    return best_name, best_model, importance_rows


def choose_folds(y: np.ndarray) -> tuple[int, str]:
    min_class = int(np.min(np.bincount(y))) if len(set(y.tolist())) == 2 else 0
    if min_class >= 5:
        return 5, "5-fold stratified CV"
    if min_class >= 3:
        return 3, "auto-downgraded to 3-fold because a class has fewer than 5 samples"
    return 2, "auto-downgraded to 2-fold because a class has fewer than 3 samples"


def evaluate_cv(records: list[dict[str, Any]], threshold: float = 7.0, epsilon: float = EPSILON) -> dict[str, Any]:
    y = labels(records, threshold)
    y_cont = continuous(records)
    folds, cv_note = choose_folds(y)
    score_rows = []
    binary_rows = []
    ablation_rows = []
    importance_rows = []
    fold_records = []

    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            train_records = [records[index] for index in train_idx]
            test_records = [records[index] for index in test_idx]
            builder = FeatureBuilder.fit(train_records)
            x_train = builder.transform_pairs([candidate_text(record) for record in train_records], train_records, epsilon=epsilon, seed=seed)
            x_test = builder.transform_pairs([candidate_text(record) for record in test_records], test_records, epsilon=epsilon, seed=seed)
            y_train = y[train_idx]
            y_test = y[test_idx]
            cont_test = y_cont[test_idx]
            train_scores = method_scores_from_features(x_train)
            test_scores = method_scores_from_features(x_test)
            fold_records.append({"seed": seed, "fold": fold, "train_size": len(train_idx), "test_size": len(test_idx)})

            for method in ["TF-IDF", "BM25", "Embedding", "KG-only", "DP-Embedding", "DP-KER-v1"]:
                oriented_train, oriented_test, _direction = orient_by_train(y_train, train_scores[method], test_scores[method])
                score_rows.append({"Method": method, **regression_metrics(cont_test, oriented_test)})
                binary_rows.append({"Method": method, **binary_metrics(y_test, oriented_test)})

            best_name, model, model_importance = train_best_classifier(x_train, y_train, seed)
            v2_scores = model.predict_proba(x_test)[:, 1]
            score_rows.append({"Method": "DP-KER-v2", "selected_model": best_name, **regression_metrics(cont_test, v2_scores)})
            binary_rows.append({"Method": "DP-KER-v2", "selected_model": best_name, **binary_metrics(y_test, v2_scores)})
            for row in model_importance:
                importance_rows.append({"seed": seed, "fold": fold, **row})

            feature_sets = {
                "full": list(range(len(FEATURE_NAMES))),
                "w/o DP": [i for i, name in enumerate(FEATURE_NAMES) if name != "dp_embedding_cosine"],
                "w/o KG features": [i for i, name in enumerate(FEATURE_NAMES) if name not in {"kg_skill_coverage", "skill_overlap_count", "missing_skill_ratio", "requirement_coverage", "title_or_category_match"}],
            }
            for setting, cols in feature_sets.items():
                _name, ab_model, _imp = train_best_classifier(x_train[:, cols], y_train, seed)
                ablation_rows.append({"Setting": setting, **binary_metrics(y_test, ab_model.predict_proba(x_test[:, cols])[:, 1])})
            ablation_rows.append({"Setting": "w/o learned fusion", **binary_metrics(y_test, test_scores["DP-KER-v1"])})
            ablation_rows.append({"Setting": "embedding only", **binary_metrics(y_test, test_scores["Embedding"])})
            ablation_rows.append({"Setting": "knowledge only", **binary_metrics(y_test, test_scores["KG-only"])})

    return {
        "score_rows": score_rows,
        "binary_rows": binary_rows,
        "ablation_rows": ablation_rows,
        "importance_rows": importance_rows,
        "fold_records": fold_records,
        "cv_note": cv_note,
        "folds": folds,
    }


def build_pools(records: list[dict[str, Any]], y: np.ndarray, negative_type: str, seed: int, builder: FeatureBuilder) -> list[tuple[int, list[int]]]:
    rng = np.random.default_rng(seed)
    positives = [idx for idx, value in enumerate(y) if value == 1]
    negatives = [idx for idx, value in enumerate(y) if value == 0]
    all_candidates = [candidate_text(record) for record in records]
    sim_features = builder.transform_pairs(all_candidates, records, epsilon=EPSILON, seed=seed)
    hard_score = sim_features[:, 0] + sim_features[:, 2]
    pools = []
    for idx in positives:
        candidates = [neg for neg in negatives if neg != idx]
        if len(candidates) < NEGATIVES_PER_QUERY:
            candidates = [j for j in range(len(records)) if j != idx]
        if negative_type == "random":
            chosen = rng.choice(candidates, size=min(NEGATIVES_PER_QUERY, len(candidates)), replace=False).tolist()
        else:
            chosen = sorted(candidates, key=lambda neg: hard_score[neg], reverse=True)[:NEGATIVES_PER_QUERY]
        pools.append((idx, [idx, *chosen]))
    return pools


def evaluate_ranking(records: list[dict[str, Any]], threshold: float = 7.0, epsilon: float = EPSILON) -> list[dict[str, Any]]:
    y = labels(records, threshold)
    rows = []
    folds, _note = choose_folds(y)
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
            train_records = [records[index] for index in train_idx]
            test_records = [records[index] for index in test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            builder = FeatureBuilder.fit(train_records)
            x_train = builder.transform_pairs([candidate_text(record) for record in train_records], train_records, epsilon=epsilon, seed=seed)
            _best, model, _imp = train_best_classifier(x_train, y_train, seed)
            for negative_type in ["random", "hard"]:
                pools = build_pools(test_records, y_test, negative_type, seed, builder)
                method_metric_values: dict[str, list[dict[str, float]]] = {
                    method: [] for method in ["TF-IDF", "BM25", "Embedding", "KG-only", "DP-Embedding", "DP-KER-v1", "DP-KER-v2"]
                }
                for query_idx, pool in pools:
                    cand_texts = [candidate_text(test_records[query_idx])] * len(pool)
                    job_records = [test_records[index] for index in pool]
                    feats = builder.transform_pairs(cand_texts, job_records, epsilon=epsilon, seed=seed)
                    scores = method_scores_from_features(feats)
                    scores["DP-KER-v2"] = model.predict_proba(feats)[:, 1]
                    for method, values in scores.items():
                        order = np.argsort(-values, kind="mergesort")
                        rank = int(np.where(order == 0)[0][0] + 1)
                        method_metric_values[method].append(
                            {
                                "Recall@5": 1.0 if rank <= TOP_K else 0.0,
                                "NDCG@5": 1.0 / np.log2(rank + 1) if rank <= TOP_K else 0.0,
                                "MRR": 1.0 / rank,
                                "Hit@1": 1.0 if rank == 1 else 0.0,
                            }
                        )
                for method, metric_values in method_metric_values.items():
                    if metric_values:
                        rows.append(
                            {
                                "Method": method,
                                "NegativeType": negative_type,
                                "Recall@5": float(np.mean([row["Recall@5"] for row in metric_values])),
                                "NDCG@5": float(np.mean([row["NDCG@5"] for row in metric_values])),
                                "MRR": float(np.mean([row["MRR"] for row in metric_values])),
                                "Hit@1": float(np.mean([row["Hit@1"] for row in metric_values])),
                            }
                        )
    return rows


def privacy_budget(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for epsilon in EPSILONS:
        cv = evaluate_cv(records, threshold=7.0, epsilon=epsilon)
        for row in cv["binary_rows"]:
            if row["Method"] == "DP-KER-v2":
                row = {"epsilon": epsilon, "sigma": gaussian_sigma(epsilon), **row}
                rows.append(row)
    return rows


def threshold_sensitivity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for threshold in THRESHOLDS:
        cv = evaluate_cv(records, threshold=threshold, epsilon=EPSILON)
        for row in cv["binary_rows"]:
            if row["Method"] in {"DP-Embedding", "DP-KER-v2"}:
                rows.append({"threshold": threshold, **row})
    return rows


def diagnostics(records: list[dict[str, Any]], importance_rows: list[dict[str, Any]]) -> None:
    DEBUG_V2_DIR.mkdir(parents=True, exist_ok=True)
    raw = continuous(records)
    y = labels(records)
    label_rows = [
        {"metric": "total", "value": len(records)},
        {"metric": "positive_count", "value": int(np.sum(y))},
        {"metric": "negative_count", "value": int(len(y) - np.sum(y))},
        {"metric": "raw_score_min", "value": float(np.min(raw) * 10.0)},
        {"metric": "raw_score_max", "value": float(np.max(raw) * 10.0)},
        {"metric": "raw_score_mean", "value": float(np.mean(raw) * 10.0)},
    ]
    write_csv(DEBUG_V2_DIR / "label_distribution_v2.csv", label_rows)

    builder = FeatureBuilder.fit(records)
    features = builder.transform_pairs([candidate_text(record) for record in records], records, epsilon=EPSILON, seed=SEED)
    corr_rows = []
    for idx, name in enumerate(FEATURE_NAMES):
        values = features[:, idx]
        corr_rows.append(
            {
                "feature": name,
                "spearman_raw_score": float(spearmanr(raw, values).statistic) if len(set(values.tolist())) > 1 else float("nan"),
                "pearson_raw_score": float(pearsonr(raw, values).statistic) if len(set(values.tolist())) > 1 else float("nan"),
            }
        )
    write_csv(DEBUG_V2_DIR / "feature_correlation_v2.csv", corr_rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in importance_rows:
        grouped[row["feature"]].append(float(row["importance"]))
    imp_rows = [{"feature": key, "importance_mean": float(np.mean(values)), "importance_std": float(np.std(values))} for key, values in grouped.items()]
    imp_rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    write_csv(DEBUG_V2_DIR / "top_feature_importance_v2.csv", imp_rows)


def write_summary(meta: dict[str, Any], records: list[dict[str, Any]], cv_note: str, outputs: dict[str, list[dict[str, Any]]]) -> None:
    binary = outputs["binary"]
    ranking = outputs["ranking"]
    ablation = outputs["ablation"]
    privacy = outputs["privacy"]
    by_method = {row["Method"]: row for row in binary}
    dpker = by_method.get("DP-KER-v2", {})
    dpemb = by_method.get("DP-Embedding", {})
    tfidf = by_method.get("TF-IDF", {})
    emb = by_method.get("Embedding", {})
    dpker_auc = float(dpker.get("AUC_mean", float("nan")))
    dpemb_auc = float(dpemb.get("AUC_mean", float("nan")))
    best_non_dp = max(float(tfidf.get("AUC_mean", 0.0)), float(emb.get("AUC_mean", 0.0)))
    ab_by = {row["Setting"]: row for row in ablation}
    full_auc = float(ab_by.get("full", {}).get("AUC_mean", float("nan")))
    no_kg_auc = float(ab_by.get("w/o KG features", {}).get("AUC_mean", float("nan")))
    eps_auc = [float(row["AUC_mean"]) for row in privacy]
    eps_reasonable = eps_auc[-1] >= eps_auc[0] if eps_auc else False
    source = meta.get("source")
    data_ok = len(records) >= 500
    sci_near = data_ok and dpker_auc > dpemb_auc and full_auc >= no_kg_auc
    lines = [
        "# DP-KER v2 Results Summary",
        "",
        f"- Data source used: `{source}`",
        f"- Loaded rows before filtering: {sum(meta.get('splits', {}).values()) if isinstance(meta.get('splits'), dict) else 'unknown'}",
        f"- Valid scored rows used: {len(records)}",
        f"- CV protocol: {cv_note}; seeds={SEEDS}",
        f"- No target leakage check: `raw_score` and threshold labels are targets only and are not included in the feature matrix.",
        "",
        "## Paper Judgment",
        "",
        f"1. Data scale usable for paper: {'yes' if data_ok else 'no; this run is still limited by available data size'}.",
        f"2. DP-KER-v2 vs DP-Embedding: {'yes' if dpker_auc > dpemb_auc else 'no'} on binary AUC ({dpker_auc:.4f} vs {dpemb_auc:.4f}).",
        f"3. DP-KER-v2 vs non-DP baselines: {'yes' if dpker_auc > best_non_dp else 'no'} on binary AUC.",
        f"4. Knowledge feature contribution: {'positive' if full_auc >= no_kg_auc else 'not supported'} by full vs w/o KG AUC ({full_auc:.4f} vs {no_kg_auc:.4f}).",
        "5. DP utility cost: compare DP-Embedding/DP-KER-v2 against non-DP embedding and w/o DP ablation in the CSV tables.",
        f"6. Epsilon trend reasonable: {'yes' if eps_reasonable else 'not clearly'} based on DP-KER-v2 AUC from epsilon 0.5 to 8.",
        "7. EI conference support: yes as a cautious experimental paper if limitations are stated.",
        f"8. SCI Q4 minimum experimental strength: {'closer, but still needs full data and stronger trends' if sci_near else 'not yet; data size or empirical dominance remains insufficient'}.",
        "",
        "## Main Files",
        "",
        "- `dataset_statistics_v2.csv`",
        "- `score_prediction_results_v2.csv`",
        "- `binary_matching_results_v2.csv`",
        "- `sampled_ranking_results_v2.csv`",
        "- `ablation_results_v2.csv`",
        "- `privacy_budget_results_v2.csv`",
        "- `threshold_sensitivity_results_v2.csv`",
    ]
    (RESULTS_V2_DIR / "results_summary_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_V2_DIR.mkdir(parents=True, exist_ok=True)
    all_records, meta = load_dataset_records()
    records = filtered_records(all_records, threshold=7.0)
    if len(records) < 10:
        raise RuntimeError("Not enough valid scored records for v2 experiments.")

    y = labels(records)
    stats_rows = [
        {"metric": "source_used", "value": meta.get("source")},
        {"metric": "loaded_rows", "value": len(all_records)},
        {"metric": "valid_scored_rows", "value": len(records)},
        {"metric": "positive_count_threshold_7", "value": int(np.sum(y))},
        {"metric": "negative_count_threshold_7", "value": int(len(y) - np.sum(y))},
        {"metric": "fold_seeds", "value": ";".join(str(seed) for seed in SEEDS)},
    ]
    write_csv(RESULTS_V2_DIR / "dataset_statistics_v2.csv", stats_rows)

    cv = evaluate_cv(records, threshold=7.0, epsilon=EPSILON)
    score_results = aggregate(cv["score_rows"], ["Method"], ["MAE", "RMSE", "Spearman", "Pearson"])
    binary_results = aggregate(cv["binary_rows"], ["Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    ablation_results = aggregate(cv["ablation_rows"], ["Setting"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    ranking_results = aggregate(evaluate_ranking(records), ["Method", "NegativeType"], ["Recall@5", "NDCG@5", "MRR", "Hit@1"])
    privacy_results = aggregate(privacy_budget(records), ["epsilon", "sigma", "Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    threshold_results = aggregate(threshold_sensitivity(records), ["threshold", "Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])

    write_csv(RESULTS_V2_DIR / "score_prediction_results_v2.csv", score_results)
    write_csv(RESULTS_V2_DIR / "binary_matching_results_v2.csv", binary_results)
    write_csv(RESULTS_V2_DIR / "sampled_ranking_results_v2.csv", ranking_results)
    write_csv(RESULTS_V2_DIR / "ablation_results_v2.csv", ablation_results)
    write_csv(RESULTS_V2_DIR / "privacy_budget_results_v2.csv", privacy_results)
    write_csv(RESULTS_V2_DIR / "threshold_sensitivity_results_v2.csv", threshold_results)
    diagnostics(records, cv["importance_rows"])
    write_summary(
        meta,
        records,
        cv["cv_note"],
        {"binary": binary_results, "ranking": ranking_results, "ablation": ablation_results, "privacy": privacy_results},
    )
    print(f"wrote v2 outputs to {RESULTS_V2_DIR}")
    print(f"wrote v2 debug outputs to {DEBUG_V2_DIR}")


if __name__ == "__main__":
    main()
