from __future__ import annotations

import numpy as np

from v2_common import (
    EPSILONS,
    FEATURE_COLUMNS,
    RESULTS_DIR,
    aggregate,
    binary_metrics,
    dp_matrix_name,
    gaussian_sigma,
    parse_mode,
    read_feature_table,
    stratified_splits,
    train_classifier,
    write_csv,
)


def main() -> None:
    args = parse_mode()
    df = read_feature_table()
    y = df["binary_label"].to_numpy(int)
    rows = []
    for epsilon in EPSILONS:
        for seed, fold, train_idx, test_idx in stratified_splits(y, args.mode):
            dp_col = dp_matrix_name(epsilon, seed)
            local_df = df.copy()
            local_df["dp_embedding_cosine"] = local_df[dp_col]
            x_train = local_df.iloc[train_idx][FEATURE_COLUMNS].to_numpy(float)
            x_test = local_df.iloc[test_idx][FEATURE_COLUMNS].to_numpy(float)
            y_train = y[train_idx]
            y_test = y[test_idx]
            _name, model = train_classifier(x_train, y_train, seed, model_name="auto")
            rows.append(
                {
                    "mode": args.mode,
                    "epsilon": epsilon,
                    "sigma": gaussian_sigma(epsilon),
                    "Method": "DP-KER-v2",
                    "seed": seed,
                    "fold": fold,
                    **binary_metrics(y_test, model.predict_proba(x_test)[:, 1]),
                }
            )
    final = aggregate(rows, ["mode", "epsilon", "sigma", "Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    write_csv(RESULTS_DIR / "privacy_budget_results_v2_final.csv", final)
    print(f"wrote privacy budget output in {RESULTS_DIR} using mode={args.mode}")


if __name__ == "__main__":
    main()
