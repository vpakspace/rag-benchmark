"""Benchmark comparison -- compute metrics, generate tables and breakdowns."""

from __future__ import annotations

from typing import Any

from adapters.base import EvalResult


def compute_metrics(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate metrics for a list of eval results."""
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0, "correct": 0, "total": 0,
            "avg_confidence": 0.0, "avg_latency": 0.0,
        }

    correct = sum(1 for r in results if r.passed)
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "avg_confidence": round(sum(r.confidence for r in results) / total, 3),
        "avg_latency": round(sum(r.latency for r in results) / total, 3),
    }


def compare_adapters(results: list[EvalResult]) -> list[dict[str, Any]]:
    """Generate comparison table: one row per adapter+mode combination.

    Returns list of dicts suitable for Streamlit/pandas display.
    """
    # Group by (adapter, mode)
    groups: dict[tuple[str, str], list[EvalResult]] = {}
    for r in results:
        key = (r.adapter, r.mode)
        groups.setdefault(key, []).append(r)

    rows = []
    for (adapter, mode), group in sorted(groups.items()):
        m = compute_metrics(group)
        rows.append({
            "Adapter": adapter,
            "Mode": mode,
            "Accuracy": f"{m['correct']}/{m['total']} ({m['accuracy']:.0%})",
            "Avg Confidence": m["avg_confidence"],
            "Avg Latency (s)": m["avg_latency"],
        })
    return rows


def accuracy_by_type(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """Per-query-type accuracy breakdown.

    Returns dict[query_type -> dict[adapter -> accuracy]].
    """
    # Group by (query_type, adapter)
    groups: dict[tuple[str, str], list[bool]] = {}
    for r in results:
        groups.setdefault((r.query_type, r.adapter), []).append(r.passed)

    breakdown: dict[str, dict[str, float]] = {}
    for (qtype, adapter), passed_list in groups.items():
        breakdown.setdefault(qtype, {})[adapter] = (
            sum(passed_list) / len(passed_list) if passed_list else 0.0
        )
    return breakdown


def accuracy_by_document(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """Per-document accuracy breakdown.

    Returns dict[document -> dict[adapter -> accuracy]].
    """
    groups: dict[tuple[str, str], list[bool]] = {}
    for r in results:
        groups.setdefault((r.document, r.adapter), []).append(r.passed)

    breakdown: dict[str, dict[str, float]] = {}
    for (doc, adapter), passed_list in groups.items():
        breakdown.setdefault(doc, {})[adapter] = (
            sum(passed_list) / len(passed_list) if passed_list else 0.0
        )
    return breakdown
