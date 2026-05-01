from __future__ import annotations

import numpy as np

from v2_common import (
    DEFAULT_EPSILON,
    FEATURE_COLUMNS,
    NEGATIVES_PER_QUERY,
    RESULTS_DIR,
    SEED,
    TOP_K,
    aggregate,
    dp_matrix_name,
    load_matrix,
    mode_config,
    parse_mode,
    read_feature_table,
    stratified_splits,
    train_classifier,
    write_csv,
)


def pool_features(df, matrices: dict[str, np.ndarray], query_idx: int, pool: list[int], seed: int, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    dp_name = dp_matrix_name(epsilon, seed)
    cols = {
        "tfidf_cosine": matrices["tfidf"][query_idx, pool],
        "bm25_score": matrices["bm25"][query_idx, pool],
        "embedding_cosine": matrices["embedding"][query_idx, pool],
        "dp_embedding_cosine": matrices[dp_name][query_idx, pool],
        "kg_skill_coverage": matrices["kg_skill_coverage"][query_idx, pool],
        "skill_overlap_count": matrices["skill_overlap_count"][query_idx, pool],
        "missing_skill_ratio": matrices["missing_skill_ratio"][query_idx, pool],
        "requirement_coverage": matrices["requirement_coverage"][query_idx, pool],
        "resume_length": np.full(len(pool), float(df.iloc[query_idx]["resume_length"])),
        "jd_length": df.iloc[pool]["jd_length"].to_numpy(float),
    }
    return np.column_stack([cols[name] for name in FEATURE_COLUMNS])


def build_pools(df, matrices: dict[str, np.ndarray], indices: np.ndarray, negative_type: str, seed: int) -> list[tuple[int, list[int]]]:
    rng = np.random.default_rng(seed)
    labels = df["binary_label"].to_numpy(int)
    positives = [int(idx) for idx in indices if labels[int(idx)] == 1]
    negatives = [int(idx) for idx in indices if labels[int(idx)] == 0]
    pools = []
    hard_score = matrices["embedding"] + 0.05 * matrices["bm25"]
    for idx in positives:
        candidates = [neg for neg in negatives if neg != idx]
        if len(candidates) < NEGATIVES_PER_QUERY:
            candidates = [int(item) for item in indices if int(item) != idx]
        if negative_type == "random":
            selected = rng.choice(candidates, size=min(NEGATIVES_PER_QUERY, len(candidates)), replace=False).tolist()
        else:
            selected = sorted(candidates, key=lambda neg: hard_score[idx, neg], reverse=True)[:NEGATIVES_PER_QUERY]
        pools.append((idx, [idx, *selected]))
    return pools


def rank_metrics(rank: int) -> dict[str, float]:
    return {
        "Recall@5": 1.0 if rank <= TOP_K else 0.0,
        "NDCG@5": 1.0 / np.log2(rank + 1) if rank <= TOP_K else 0.0,
        "MRR": 1.0 / rank,
        "Hit@1": 1.0 if rank == 1 else 0.0,
    }


def main() -> None:
    args = parse_mode()
    config = mode_config(args.mode)
    df = read_feature_table()
    y = df["binary_label"].to_numpy(int)
    matrices = {
        "tfidf": load_matrix("tfidf_cosine_matrix"),
        "bm25": load_matrix("bm25_score_matrix"),
        "embedding": load_matrix("embedding_cosine_matrix"),
        "kg_skill_coverage": load_matrix("kg_skill_coverage_matrix"),
        "skill_overlap_count": load_matrix("skill_overlap_count_matrix"),
        "missing_skill_ratio": load_matrix("missing_skill_ratio_matrix"),
        "requirement_coverage": load_matrix("requirement_coverage_matrix"),
    }
    for seed in config["seeds"]:
        matrices[dp_matrix_name(DEFAULT_EPSILON, seed)] = load_matrix(dp_matrix_name(DEFAULT_EPSILON, seed))

    rows = []
    for seed, fold, train_idx, test_idx in stratified_splits(y, args.mode):
        x_train = df.iloc[train_idx][FEATURE_COLUMNS].to_numpy(float)
        y_train = y[train_idx]
        _name, model = train_classifier(x_train, y_train, seed, model_name="auto")
        for negative_type in ["random", "hard"]:
            pools = build_pools(df, matrices, test_idx, negative_type, seed)
            method_values = {method: [] for method in ["TF-IDF", "BM25", "Embedding", "KG-only", "DP-Embedding", "DP-KER-v1", "DP-KER-v2"]}
            for query_idx, pool in pools:
                feats = pool_features(df, matrices, query_idx, pool, seed)
                scores = {
                    "TF-IDF": feats[:, 0],
                    "BM25": feats[:, 1],
                    "Embedding": feats[:, 2],
                    "KG-only": feats[:, 4],
                    "DP-Embedding": feats[:, 3],
                    "DP-KER-v1": 0.45 * feats[:, 3] + 0.45 * feats[:, 4] + 0.10 * feats[:, 1],
                    "DP-KER-v2": model.predict_proba(feats)[:, 1],
                }
                for method, values in scores.items():
                    order = np.argsort(-values, kind="mergesort")
                    rank = int(np.where(order == 0)[0][0] + 1)
                    method_values[method].append(rank_metrics(rank))
            for method, values in method_values.items():
                if not values:
                    continue
                rows.append(
                    {
                        "mode": args.mode,
                        "Method": method,
                        "NegativeType": negative_type,
                        "seed": seed,
                        "fold": fold,
                        "Recall@5": float(np.mean([row["Recall@5"] for row in values])),
                        "NDCG@5": float(np.mean([row["NDCG@5"] for row in values])),
                        "MRR": float(np.mean([row["MRR"] for row in values])),
                        "Hit@1": float(np.mean([row["Hit@1"] for row in values])),
                    }
                )
    final = aggregate(rows, ["mode", "Method", "NegativeType"], ["Recall@5", "NDCG@5", "MRR", "Hit@1"])
    write_csv(RESULTS_DIR / "sampled_ranking_results_v2_final.csv", final)
    print(f"wrote sampled ranking output in {RESULTS_DIR} using mode={args.mode}")


if __name__ == "__main__":
    main()
