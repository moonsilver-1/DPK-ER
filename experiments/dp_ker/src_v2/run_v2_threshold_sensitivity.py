from __future__ import annotations

import numpy as np

from v2_common import (
    DEFAULT_EPSILON,
    FEATURE_COLUMNS,
    RESULTS_DIR,
    THRESHOLDS,
    aggregate,
    binary_metrics,
    dp_matrix_name,
    parse_mode,
    read_feature_table,
    stratified_splits,
    train_classifier,
    write_csv,
)
from run_v2_sampled_ranking import build_pools, load_matrix, pool_features, rank_metrics


def main() -> None:
    args = parse_mode()
    df = read_feature_table()
    matrices = {
        "tfidf": load_matrix("tfidf_cosine_matrix"),
        "bm25": load_matrix("bm25_score_matrix"),
        "embedding": load_matrix("embedding_cosine_matrix"),
        "kg_skill_coverage": load_matrix("kg_skill_coverage_matrix"),
        "skill_overlap_count": load_matrix("skill_overlap_count_matrix"),
        "missing_skill_ratio": load_matrix("missing_skill_ratio_matrix"),
        "requirement_coverage": load_matrix("requirement_coverage_matrix"),
    }
    rows = []
    for threshold in THRESHOLDS:
        y = (df["raw_score"].to_numpy(float) >= threshold).astype(int)
        for seed, fold, train_idx, test_idx in stratified_splits(y, args.mode):
            dp_name = dp_matrix_name(DEFAULT_EPSILON, seed)
            if dp_name not in matrices:
                matrices[dp_name] = load_matrix(dp_name)
            x_train = df.iloc[train_idx][FEATURE_COLUMNS].to_numpy(float)
            x_test = df.iloc[test_idx][FEATURE_COLUMNS].to_numpy(float)
            x_train[:, FEATURE_COLUMNS.index("dp_embedding_cosine")] = df.iloc[train_idx][dp_name].to_numpy(float)
            x_test[:, FEATURE_COLUMNS.index("dp_embedding_cosine")] = df.iloc[test_idx][dp_name].to_numpy(float)
            y_train = y[train_idx]
            y_test = y[test_idx]
            _name, model = train_classifier(x_train, y_train, seed, model_name="auto")
            rows.append(
                {
                    "mode": args.mode,
                    "Task": "binary",
                    "threshold": threshold,
                    "Method": "DP-KER-v2",
                    "NegativeType": "",
                    "seed": seed,
                    "fold": fold,
                    **binary_metrics(y_test, model.predict_proba(x_test)[:, 1]),
                }
            )
            pools = build_pools(df.assign(binary_label=y), matrices, test_idx, "hard", seed)
            values = []
            for query_idx, pool in pools:
                feats = pool_features(df, matrices, query_idx, pool, seed)
                scores = model.predict_proba(feats)[:, 1]
                order = np.argsort(-scores, kind="mergesort")
                rank = int(np.where(order == 0)[0][0] + 1)
                values.append(rank_metrics(rank))
            if values:
                rows.append(
                    {
                        "mode": args.mode,
                        "Task": "hard_ranking",
                        "threshold": threshold,
                        "Method": "DP-KER-v2",
                        "NegativeType": "hard",
                        "seed": seed,
                        "fold": fold,
                        "AUC": float("nan"),
                        "PR-AUC": float("nan"),
                        "Accuracy": float("nan"),
                        "Balanced Accuracy": float("nan"),
                        "F1": float("nan"),
                        "Precision": float("nan"),
                        "Recall": float("nan"),
                        "Recall@5": float(np.mean([row["Recall@5"] for row in values])),
                        "NDCG@5": float(np.mean([row["NDCG@5"] for row in values])),
                        "MRR": float(np.mean([row["MRR"] for row in values])),
                        "Hit@1": float(np.mean([row["Hit@1"] for row in values])),
                    }
                )
    binary_rows = [row for row in rows if row["Task"] == "binary"]
    rank_rows = [row for row in rows if row["Task"] == "hard_ranking"]
    final = aggregate(binary_rows, ["mode", "Task", "threshold", "Method"], ["AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "F1", "Precision", "Recall"])
    final.extend(aggregate(rank_rows, ["mode", "Task", "threshold", "Method", "NegativeType"], ["Recall@5", "NDCG@5", "MRR", "Hit@1"]))
    write_csv(RESULTS_DIR / "threshold_sensitivity_results_v2_final.csv", final)
    print(f"wrote threshold sensitivity output in {RESULTS_DIR} using mode={args.mode}")


if __name__ == "__main__":
    main()
