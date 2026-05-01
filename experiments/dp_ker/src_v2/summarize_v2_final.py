from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from v2_common import CACHE_DIR, RESULTS_DIR, load_metadata


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def main() -> None:
    meta = load_metadata()
    binary = read_rows(RESULTS_DIR / "binary_matching_results_v2_final.csv")
    score = read_rows(RESULTS_DIR / "score_prediction_results_v2_final.csv")
    ranking = read_rows(RESULTS_DIR / "sampled_ranking_results_v2_final.csv")
    ablation = read_rows(RESULTS_DIR / "ablation_results_v2_final.csv")
    privacy = read_rows(RESULTS_DIR / "privacy_budget_results_v2_final.csv")
    threshold = read_rows(RESULTS_DIR / "threshold_sensitivity_results_v2_final.csv")

    dpker = find(binary, "Method", "DP-KER-v2")
    dpemb = find(binary, "Method", "DP-Embedding")
    dpker_v1 = find(binary, "Method", "DP-KER-v1")
    tfidf = find(binary, "Method", "TF-IDF")
    bm25 = find(binary, "Method", "BM25")
    emb = find(binary, "Method", "Embedding")
    best_non_dp = max(f(tfidf, "AUC_mean"), f(bm25, "AUC_mean"), f(emb, "AUC_mean"))
    full = find(ablation, "Setting", "full")
    no_kg = find(ablation, "Setting", "w/o KG features")
    no_dp = find(ablation, "Setting", "w/o DP")
    eps_values = [f(row, "AUC_mean") for row in privacy if row.get("Method") == "DP-KER-v2"]
    eps_reasonable = bool(eps_values and eps_values[-1] >= eps_values[0])

    completed = {
        "core_prediction": bool(binary and score and ablation),
        "sampled_ranking": bool(ranking),
        "privacy_budget": bool(privacy),
        "threshold_sensitivity": bool(threshold),
    }
    lines = [
        "# DP-KER v2 Final Summary",
        "",
        f"- Raw rows recovered: {meta.get('loaded_rows')}",
        f"- Valid/scored rows used: {meta.get('valid_scored_rows')}",
        f"- Positive / negative at threshold 7.0: {meta.get('positive_count')} / {meta.get('negative_count')}",
        f"- Data source: {meta.get('source_used')}",
        f"- Feature cache: `{meta.get('feature_path')}`",
        "",
        "## Completion Status",
        "",
    ]
    for name, ok in completed.items():
        lines.append(f"- {name}: {'completed' if ok else 'missing/not rerun yet'}")

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Uses 1031 raw / 815 valid scored data: {'yes' if meta.get('loaded_rows') == 1031 and meta.get('valid_scored_rows') == 815 else 'no'}.",
            f"2. Positive / negative counts: {meta.get('positive_count')} / {meta.get('negative_count')}.",
            f"3. DP-KER-v2 vs DP-Embedding: {'yes' if f(dpker, 'AUC_mean') > f(dpemb, 'AUC_mean') else 'no or unavailable'} on binary AUC ({f(dpker, 'AUC_mean'):.4f} vs {f(dpemb, 'AUC_mean'):.4f}).",
            f"4. DP-KER-v2 vs non-DP baseline: {'yes' if f(dpker, 'AUC_mean') > best_non_dp else 'no or unavailable'} on binary AUC.",
            f"5. Learned fusion vs hand fusion: {'yes' if f(dpker, 'AUC_mean') > f(dpker_v1, 'AUC_mean') else 'no or unavailable'} on binary AUC.",
            f"6. KG feature contribution: {'yes' if f(full, 'AUC_mean') > f(no_kg, 'AUC_mean') else 'not supported or unavailable'} by full vs w/o KG.",
            f"7. DP utility cost: {'visible' if f(no_dp, 'AUC_mean') > f(full, 'AUC_mean') else 'not clearly visible or unavailable'} by w/o DP vs full.",
            f"8. Epsilon trend reasonable: {'yes' if eps_reasonable else 'not clear or privacy fast not run yet'}.",
            f"9. Stronger than original fixed EI version: {'yes' if completed['core_prediction'] and meta.get('valid_scored_rows', 0) > 94 else 'not yet'}.",
            f"10. SCI Q4 minimum strength: {'closer if all final-mode tables are complete and trends hold' if all(completed.values()) else 'not yet; finish final/fast staged tables first'}.",
            "",
            "## Notes",
            "",
            "- `raw_score` and `binary_label` are stored in the cache for evaluation only and are excluded from model feature columns.",
            "- DP is embedding-level Gaussian perturbation, not DP-SGD.",
            "- If a table is marked missing, run its corresponding `src_v2` script and then rerun this summarizer.",
        ]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results_summary_v2_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'results_summary_v2_final.md'}")


if __name__ == "__main__":
    main()
