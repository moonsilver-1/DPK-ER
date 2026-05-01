from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.model_selection import StratifiedKFold, train_test_split


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import EXPERIMENT_ROOT, PROJECT_ROOT, SEED  # noqa: E402
from dp import DEFAULT_DP_SEEDS, gaussian_sigma  # noqa: E402


CACHE_DIR = EXPERIMENT_ROOT / "cache_v2_final"
RESULTS_DIR = EXPERIMENT_ROOT / "results_v2_final"
DEBUG_DIR = EXPERIMENT_ROOT / "debug_v2_final"

FEATURE_COLUMNS = [
    "tfidf_cosine",
    "bm25_score",
    "embedding_cosine",
    "dp_embedding_cosine",
    "kg_skill_coverage",
    "skill_overlap_count",
    "missing_skill_ratio",
    "requirement_coverage",
    "resume_length",
    "jd_length",
]
KG_COLUMNS = ["kg_skill_coverage", "skill_overlap_count", "missing_skill_ratio", "requirement_coverage"]
NON_DP_COLUMNS = [col for col in FEATURE_COLUMNS if col != "dp_embedding_cosine"]
NO_KG_COLUMNS = [col for col in FEATURE_COLUMNS if col not in KG_COLUMNS]
SEEDS_FAST = [3407]
SEEDS_FINAL = DEFAULT_DP_SEEDS
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
THRESHOLDS = [6.5, 7.0, 7.5]
DEFAULT_EPSILON = 2.0
NEGATIVES_PER_QUERY = 19
TOP_K = 5


def parse_mode() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "final"], default="fast")
    return parser.parse_args()


def mode_config(mode: str) -> dict[str, Any]:
    if mode == "final":
        return {"folds": 5, "seeds": SEEDS_FINAL, "mode": mode}
    return {"folds": 3, "seeds": SEEDS_FAST, "mode": mode}


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dirs()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_feature_table() -> pd.DataFrame:
    parquet_path = CACHE_DIR / "pair_features.parquet"
    csv_path = CACHE_DIR / "pair_features.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Feature cache missing. Run prepare_feature_cache_v2.py first.")


def load_metadata() -> dict[str, Any]:
    return json.loads((CACHE_DIR / "metadata.json").read_text(encoding="utf-8"))


def matrix_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npy"


def load_matrix(name: str) -> np.ndarray:
    return np.load(matrix_path(name))


def dp_matrix_name(epsilon: float, seed: int) -> str:
    eps_text = str(epsilon).replace(".", "_")
    return f"dp_embedding_cosine_epsilon_{eps_text}_seed_{seed}"


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else float("nan")


def orient_by_train(y_train: np.ndarray, train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    forward = safe_auc(y_train, train_scores)
    reverse = safe_auc(y_train, -train_scores)
    return -test_scores if np.isfinite(reverse) and reverse > forward else test_scores


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(math.sqrt(np.mean((y_true - pred) ** 2))),
        "Spearman": float(spearmanr(y_true, pred).statistic) if len(set(pred.tolist())) > 1 else float("nan"),
        "Pearson": float(pearsonr(y_true, pred).statistic) if len(set(pred.tolist())) > 1 else float("nan"),
    }


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = minmax(np.asarray(scores, dtype=float))
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


def stratified_splits(y: np.ndarray, mode: str):
    config = mode_config(mode)
    folds = int(config["folds"])
    min_class = int(np.min(np.bincount(y)))
    if min_class < folds:
        folds = max(2, min(3, min_class))
    for seed in config["seeds"]:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            yield int(seed), fold, train_idx, test_idx


def base_scores(df: pd.DataFrame, epsilon: float = DEFAULT_EPSILON, seed: int = SEED) -> dict[str, np.ndarray]:
    dp_col = f"dp_embedding_cosine_epsilon_{str(epsilon).replace('.', '_')}_seed_{seed}"
    if dp_col not in df.columns:
        dp_col = "dp_embedding_cosine"
    return {
        "TF-IDF": df["tfidf_cosine"].to_numpy(float),
        "BM25": df["bm25_score"].to_numpy(float),
        "Embedding": df["embedding_cosine"].to_numpy(float),
        "KG-only": df["kg_skill_coverage"].to_numpy(float),
        "DP-Embedding": df[dp_col].to_numpy(float),
        "DP-KER-v1": (
            0.45 * minmax(df[dp_col].to_numpy(float))
            + 0.45 * minmax(df["kg_skill_coverage"].to_numpy(float))
            + 0.10 * minmax(df["bm25_score"].to_numpy(float))
        ),
    }


def train_classifier(x_train: np.ndarray, y_train: np.ndarray, seed: int, model_name: str = "auto"):
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=15, random_state=seed),
    }
    if model_name != "auto":
        model = candidates[model_name]
        model.fit(x_train, y_train)
        return model_name, model
    tr_idx, va_idx = train_test_split(np.arange(len(y_train)), test_size=0.25, stratify=y_train, random_state=seed)
    best_name = "LogisticRegression"
    best_auc = -np.inf
    best_model = candidates[best_name]
    for name, model in candidates.items():
        try:
            model.fit(x_train[tr_idx], y_train[tr_idx])
            auc = safe_auc(y_train[va_idx], model.predict_proba(x_train[va_idx])[:, 1])
        except Exception:
            continue
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
    best_model.fit(x_train, y_train)
    return best_name, best_model


def train_regressor(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    models = [
        ("Ridge", Ridge(alpha=1.0)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=1)),
        ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, random_state=seed)),
    ]
    tr_idx, va_idx = train_test_split(np.arange(len(y_train)), test_size=0.25, random_state=seed)
    best_name = "Ridge"
    best_rmse = np.inf
    best_model = models[0][1]
    for name, model in models:
        try:
            model.fit(x_train[tr_idx], y_train[tr_idx])
            pred = model.predict(x_train[va_idx])
            rmse = math.sqrt(np.mean((y_train[va_idx] - pred) ** 2))
        except Exception:
            continue
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model
    best_model.fit(x_train, y_train)
    return best_name, best_model


def feature_importance(model: Any) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(FEATURE_COLUMNS))
    return [{"feature": feature, "importance": float(value)} for feature, value in zip(FEATURE_COLUMNS, values)]
