from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from v2_common import (
    DEBUG_DIR,
    FEATURE_COLUMNS,
    KG_COLUMNS,
    NON_DP_COLUMNS,
    NO_KG_COLUMNS,
    RESULTS_DIR,
    aggregate,
    base_scores,
    binary_metrics,
    feature_importance,
    mode_config,
    orient_by_train,
    parse_mode,
    read_feature_table,
    regression_metrics,
    stratified_splits,
    train_classifier,
    train_regressor,
    write_csv,
)


def feature_correlation(df):
    rows = []
    raw = df["raw_score"].to_numpy(float)
    label = df["binary_label"].to_numpy(int)
    for feature in FEATURE_COLUMNS:
        values = df[feature].to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "spearman_raw_score": float(spearmanr(raw, values).statistic) if len(set(values.tolist())) > 1 else float("nan"),
                "pearson_raw_score": float(pearsonr(raw, values).statistic) if len(set(values.tolist())) > 1 else float("nan"),
                "spearman_binary_label": float(spearmanr(label, values).statistic) if len(set(values.tolist())) > 1 else float("nan"),
            }
        )
    return rows


def main() -> None:
    args = parse_mode()
    config = mode_config(args.mode)
    df = read_feature_table()
    y = df["binary_label"].to_numpy(int)
    raw = df["raw_score"].to_numpy(float)
    x_all = df[FEATURE_COLUMNS].to_numpy(float)

    score_rows = []
    binary_rows = []
    ablation_rows = []
    importance_rows = []

    for seed, fold, train_idx, test_idx in stratified_splits(y, args.mode):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        raw_train = raw[train_idx]
        raw_test = raw[test_idx]
        x_train = x_all[train_idx]
        x_test = x_all[test_idx]

        train_scores = base_scores(train_df, seed=seed)
        test_scores = base_scores(test_df, seed=seed)
        for method in ["TF-IDF", "BM25", "Embedding", "KG-only", "DP-Embedding", "DP-KER-v1"]:
            pred = orient_by_train(y_train, train_scores[method], test_scores[method])
            score_rows.append({"mode": args.mode, "Method": method, "seed": seed, "fold": fold, **regression_metrics(raw_test, pred)})
            binary_rows.append({"mode": args.mode, "Method": method, "seed": seed, "fold": fold, **binary_metrics(y_test, pred)})

        reg_name, reg_model = train_regressor(x_train, raw_train, seed)
        reg_pred = reg_model.predict(x_test)
        score_rows.append({"mode": args.mode, "Method": f"DP-KER-v2-{reg_name}", "seed": seed, "fold": fold, **regression_metrics(raw_test, reg_pred)})

        clf_name, clf_model = train_classifier(x_train, y_train, seed, model_name="auto")
        clf_scores = clf_model.predict_proba(x_test)[:, 1]
        binary_rows.append({"mode": args.mode, "Method": f"DP-KER-v2-{clf_name}", "seed": seed, "fold": fold, **binary_metrics(y_test, clf_scores)})
        binary_rows.append({"mode": args.mode, "Method": "DP-KER-v2", "seed": seed, "fold": fold, **binary_metrics(y_test, clf_scores)})
        for row in feature_importance(clf_model):
            importance_rows.append({"mode": args.mode, "seed": seed, "fold": fold, "model": clf_name, **row})

        ablation_specs = {
            "full": FEATURE_COLUMNS,
            "w/o DP": NON_DP_COLUMNS,
            "w/o KG features": NO_KG_COLUMNS,
            "w/o learned fusion": [],
            "embedding only": ["embedding_cosine"],
            "knowledge only": KG_COLUMNS,
        }
        for setting, columns in ablation_specs.items():
            if setting == "w/o learned fusion":
                ablation_rows.append({"mode": args.mode, "Setting": setting, "seed": seed, "fold": fold, **binary_metrics(y_test, test_scores["DP-KER-v1"])})
                continue
            col_idx = [FEATURE_COLUMNS.index(col) for col in columns]
            _name, model = train_classifier(x_train[:, col_idx], y_train, seed, model_name="LogisticRegression")
            scores = model.predict_proba(x_test[:, col_idx])[:, 1]
            ablation_rows.append({"mode": args.mode, "Setting": setting, "seed": seed, "fold": fold, **binary_metrics(y_test, scores)})

    score_final = aggregate(score_rows, ["mode", "Method"], ["MAE", "RMSE", "Spearman", "Pearson"])
    binary_final = aggregate(binary_rows, ["mode", "Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    ablation_final = aggregate(ablation_rows, ["mode", "Setting"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    write_csv(RESULTS_DIR / "score_prediction_results_v2_final.csv", score_final)
    write_csv(RESULTS_DIR / "binary_matching_results_v2_final.csv", binary_final)
    write_csv(RESULTS_DIR / "ablation_results_v2_final.csv", ablation_final)
    write_csv(DEBUG_DIR / "feature_correlation_v2_final.csv", feature_correlation(df))

    grouped = defaultdict(list)
    for row in importance_rows:
        grouped[row["feature"]].append(float(row["importance"]))
    imp_rows = [
        {"feature": feature, "importance_mean": float(np.mean(values)), "importance_std": float(np.std(values))}
        for feature, values in grouped.items()
    ]
    imp_rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    write_csv(DEBUG_DIR / "top_feature_importance_v2_final.csv", imp_rows)

    leakage = [
        "# Leakage Check v2 Final",
        "",
        "- Model feature columns: " + ", ".join(FEATURE_COLUMNS),
        "- Excluded target columns: raw_score, binary_label.",
        "- `raw_score` is used only as regression target and for constructing `binary_label`.",
        "- `binary_label` is used only as classification target and for stratified splits.",
        "- Feature cache includes target columns for evaluation bookkeeping, but training code selects only FEATURE_COLUMNS.",
    ]
    (DEBUG_DIR / "leakage_check_v2_final.md").write_text("\n".join(leakage) + "\n", encoding="utf-8")
    print(f"wrote core prediction outputs in {RESULTS_DIR} using mode={args.mode}, folds={config['folds']}, seeds={config['seeds']}")


if __name__ == "__main__":
    main()
