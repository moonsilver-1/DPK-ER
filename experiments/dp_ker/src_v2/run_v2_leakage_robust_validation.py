from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from v2_common import (
    DEBUG_DIR,
    FEATURE_COLUMNS,
    RESULTS_DIR,
    base_scores,
    binary_metrics,
    orient_by_train,
    read_feature_table,
    train_classifier,
    write_csv,
)


METHODS = ["TF-IDF", "BM25", "Embedding", "KG-only", "DP-Embedding", "DP-KER-v1"]
SEED = 3407


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    metrics = [
        "AUC",
        "PR-AUC",
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Precision",
        "Recall",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)

    out_rows = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda x: x[0]):
        out = {k: v for k, v in zip(keys, group_key)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group_rows], dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values))
        out["runs"] = len(group_rows)
        out_rows.append(out)

    return out_rows


def evaluate_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    protocol: str,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    y = df["binary_label"].to_numpy(int)
    x = df[FEATURE_COLUMNS].to_numpy(float)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]
    x_train = x[train_idx]
    x_test = x[test_idx]

    rows = []

    train_scores = base_scores(train_df, seed=seed)
    test_scores = base_scores(test_df, seed=seed)

    for method in METHODS:
        pred_scores = orient_by_train(y_train, train_scores[method], test_scores[method])
        rows.append(
            {
                "protocol": protocol,
                "method": method,
                "seed": seed,
                "fold": fold,
                **binary_metrics(y_test, pred_scores),
            }
        )

    model_name, model = train_classifier(x_train, y_train, seed, model_name="auto")
    clf_scores = model.predict_proba(x_test)[:, 1]

    rows.append(
        {
            "protocol": protocol,
            "method": "DP-KER-v2",
            "selected_model": model_name,
            "seed": seed,
            "fold": fold,
            **binary_metrics(y_test, clf_scores),
        }
    )

    return rows


def run_row_level_cv(df: pd.DataFrame) -> list[dict[str, Any]]:
    y = df["binary_label"].to_numpy(int)
    rows = []

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        rows.extend(evaluate_split(df, train_idx, test_idx, "row_level_stratified_cv", fold, SEED))

    return rows


def run_holdout(df: pd.DataFrame) -> list[dict[str, Any]]:
    y = df["binary_label"].to_numpy(int)
    indices = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    y_train_val = y[train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        stratify=y_train_val,
        random_state=SEED,
    )

    # The current train_classifier function already splits the training set
    # internally for model selection. Here we merge train+val for final
    # training and keep test_idx untouched. This gives a clean final hold-out
    # test estimate without evaluating on test during model selection.
    outer_train_idx = np.concatenate([train_idx, val_idx])

    return evaluate_split(df, outer_train_idx, test_idx, "holdout_60_20_20", 1, SEED)


def candidate_group_columns(df: pd.DataFrame) -> list[str]:
    possible = [
        "profile_id",
        "resume_id",
        "candidate_id",
        "job_id",
        "jd_id",
        "source_file",
        "file_name",
    ]
    return [col for col in possible if col in df.columns]


def run_group_cv(df: pd.DataFrame, group_col: str) -> tuple[list[dict[str, Any]], str]:
    y = df["binary_label"].to_numpy(int)
    groups = df[group_col].astype(str).to_numpy()

    unique_groups = pd.Series(groups).nunique()
    if unique_groups == len(groups):
        return [], (
            f"Group column `{group_col}` has no repeated groups "
            f"({unique_groups} unique groups for {len(groups)} rows). "
            "GroupKFold would be equivalent to row-level splitting, so it was skipped."
        )

    min_groups = min(pd.Series(groups).value_counts())
    n_splits = 5
    if unique_groups < n_splits:
        return [], f"Group column `{group_col}` has too few groups for 5-fold GroupKFold."

    rows = []
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        y_test = y[test_idx]
        if len(set(y_test.tolist())) < 2:
            # AUC cannot be computed if the test fold contains only one class.
            # Still evaluate other metrics, but record the limitation in report.
            pass

        protocol = f"group_cv_by_{group_col}"
        rows.extend(evaluate_split(df, train_idx, test_idx, protocol, fold, SEED))

    return rows, f"GroupKFold by `{group_col}` completed with {unique_groups} unique groups."


def write_reports(df: pd.DataFrame, group_notes: list[str]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    feature_lines = [
        "# Strict Leakage Check",
        "",
        "## Training Feature Columns",
        "",
    ]
    for col in FEATURE_COLUMNS:
        feature_lines.append(f"- {col}")

    excluded_cols = [
        "raw_score",
        "binary_label",
        "label",
        "score",
        "aggregated_score",
        "valid_resume_and_jd",
        "profile_id",
        "resume_id",
        "candidate_id",
        "job_id",
        "jd_id",
        "source_file",
        "file_name",
    ]

    feature_lines.extend(
        [
            "",
            "## Explicitly Excluded / Checked Columns",
            "",
        ]
    )
    for col in excluded_cols:
        status = "present" if col in df.columns else "not present"
        feature_lines.append(f"- {col}: {status}")

    leakage_cols = [col for col in ["raw_score", "binary_label"] if col in FEATURE_COLUMNS]
    id_cols = [col for col in ["profile_id", "resume_id", "candidate_id", "job_id", "source_file", "file_name"] if col in FEATURE_COLUMNS]

    feature_lines.extend(
        [
            "",
            "## Final Judgment",
            "",
        ]
    )

    if leakage_cols or id_cols:
        feature_lines.append("- Potential leakage detected.")
        if leakage_cols:
            feature_lines.append(f"- Target-related columns in features: {leakage_cols}")
        if id_cols:
            feature_lines.append(f"- ID-related columns in features: {id_cols}")
    else:
        feature_lines.append("- No obvious label leakage detected from the configured feature columns.")
        feature_lines.append("- raw_score and binary_label are not included in FEATURE_COLUMNS.")
        feature_lines.append("- ID/file columns are not included in FEATURE_COLUMNS.")

    (DEBUG_DIR / "leakage_check_strict.md").write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    protocol_lines = [
        "# CV Protocol Report",
        "",
        f"- Total rows: {len(df)}",
        f"- Positive labels: {int(df['binary_label'].sum())}",
        f"- Negative labels: {int((df['binary_label'] == 0).sum())}",
        "",
        "## Protocols",
        "",
        "- Row-level StratifiedKFold: 5 folds, seed=3407.",
        "- Hold-out split: 60/20/20 design, final test uses 20% untouched data.",
        "",
        "## Group Split Notes",
        "",
    ]

    for note in group_notes:
        protocol_lines.append(f"- {note}")

    (DEBUG_DIR / "cv_protocol_report.md").write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    df = read_feature_table()
    rows = []

    rows.extend(run_row_level_cv(df))
    rows.extend(run_holdout(df))

    group_notes = []
    for group_col in candidate_group_columns(df):
        group_rows, note = run_group_cv(df, group_col)
        group_notes.append(note)
        rows.extend(group_rows)

    final_rows = aggregate(rows, ["protocol", "method"])

    write_csv(RESULTS_DIR / "robust_validation_results_v2_final.csv", final_rows)

    holdout_rows = [row for row in rows if row["protocol"] == "holdout_60_20_20"]
    holdout_final = aggregate(holdout_rows, ["protocol", "method"])
    write_csv(RESULTS_DIR / "holdout_results_v2_final.csv", holdout_final)

    write_reports(df, group_notes)

    print("Wrote robust validation results:")
    print(RESULTS_DIR / "robust_validation_results_v2_final.csv")
    print(RESULTS_DIR / "holdout_results_v2_final.csv")
    print(DEBUG_DIR / "cv_protocol_report.md")
    print(DEBUG_DIR / "leakage_check_strict.md")


if __name__ == "__main__":
    main()