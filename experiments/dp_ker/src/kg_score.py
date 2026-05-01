from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def coverage_score(source_text: str, items: list[str]) -> float:
    if not items:
        return 0.0
    source_tokens = set(tokenize(source_text))
    if not source_tokens:
        return 0.0
    ratios: list[float] = []
    for item in items:
        item_tokens = set(tokenize(item))
        if not item_tokens:
            continue
        ratios.append(len(source_tokens & item_tokens) / len(item_tokens))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def criteria_overlap_score(source_text: str, macro_dict: dict[str, Any], micro_dict: dict[str, Any]) -> float:
    source_tokens = set(tokenize(source_text))
    if not source_tokens:
        return 0.0
    weighted_hits = 0.0
    total_weight = 0.0
    for items in (macro_dict or {}, micro_dict or {}):
        for key, weight in items.items():
            total_weight += float(weight or 1.0)
            key_tokens = set(tokenize(str(key)))
            if key_tokens and key_tokens <= source_tokens:
                weighted_hits += float(weight or 1.0)
            elif key_tokens and len(source_tokens & key_tokens) > 0:
                weighted_hits += 0.5 * float(weight or 1.0)
    if total_weight <= 0:
        return 0.0
    return weighted_hits / total_weight


def score_sample(
    candidate_text: str,
    knowledge_text: str,
    minimum_requirements: list[str] | None = None,
    additional_info: str = "",
    macro_dict: dict[str, Any] | None = None,
    micro_dict: dict[str, Any] | None = None,
) -> dict[str, float]:
    base_similarity = jaccard_similarity(candidate_text, knowledge_text)
    requirement_coverage = coverage_score(candidate_text, minimum_requirements or [])
    additional_info_coverage = jaccard_similarity(candidate_text, additional_info)
    criteria_coverage = criteria_overlap_score(candidate_text + "\n" + knowledge_text, macro_dict or {}, micro_dict or {})
    overall = 0.45 * base_similarity + 0.20 * requirement_coverage + 0.10 * additional_info_coverage + 0.25 * criteria_coverage
    return {
        "base_similarity": float(base_similarity),
        "requirement_coverage": float(requirement_coverage),
        "additional_info_coverage": float(additional_info_coverage),
        "criteria_coverage": float(criteria_coverage),
        "overall": float(overall),
    }


def explanation_consistency(
    candidate_text: str,
    knowledge_text: str,
    minimum_requirements: list[str] | None = None,
    additional_info: str = "",
    macro_dict: dict[str, Any] | None = None,
    micro_dict: dict[str, Any] | None = None,
) -> float:
    components = score_sample(
        candidate_text=candidate_text,
        knowledge_text=knowledge_text,
        minimum_requirements=minimum_requirements,
        additional_info=additional_info,
        macro_dict=macro_dict,
        micro_dict=micro_dict,
    )
    return float(
        0.45 * components["base_similarity"]
        + 0.25 * components["requirement_coverage"]
        + 0.15 * components["additional_info_coverage"]
        + 0.15 * components["criteria_coverage"]
    )
