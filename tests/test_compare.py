"""Tests for metrics computation and comparison tables."""

from __future__ import annotations

import pytest

from adapters.base import EvalResult, BenchmarkRun
from benchmark.compare import (
    compute_metrics,
    compare_adapters,
    accuracy_by_type,
    accuracy_by_document,
)


def _make_results(passed_flags: list[bool], adapter="test", mode="m1") -> list[EvalResult]:
    """Helper: create EvalResult list from pass/fail flags."""
    return [
        EvalResult(
            question_id=i + 1, adapter=adapter, mode=mode, lang="en",
            query_type=["simple", "relation", "multi_hop", "global", "temporal"][i % 5],
            document="en_llm_survey",
            passed=p, answer=f"A{i}", latency=2.0 + i, confidence=0.5 + i * 0.05,
        )
        for i, p in enumerate(passed_flags)
    ]


class TestComputeMetrics:
    def test_all_pass(self):
        results = _make_results([True, True, True])
        m = compute_metrics(results)
        assert m["accuracy"] == 1.0
        assert m["correct"] == 3
        assert m["total"] == 3

    def test_all_fail(self):
        m = compute_metrics(_make_results([False, False]))
        assert m["accuracy"] == 0.0

    def test_mixed(self):
        m = compute_metrics(_make_results([True, True, False, False, True]))
        assert m["accuracy"] == pytest.approx(3 / 5)

    def test_empty(self):
        m = compute_metrics([])
        assert m["accuracy"] == 0.0
        assert m["total"] == 0

    def test_latency_average(self):
        m = compute_metrics(_make_results([True, True]))
        # latencies: 2.0, 3.0 -> avg 2.5
        assert m["avg_latency"] == pytest.approx(2.5)


class TestCompareAdapters:
    def test_two_adapters(self):
        results = (
            _make_results([True, True, True], adapter="rag2", mode="agent")
            + _make_results([True, False, False], adapter="pageindex", mode="tree")
        )
        rows = compare_adapters(results)
        assert len(rows) == 2
        assert any(r["Adapter"] == "rag2" for r in rows)
        assert any(r["Adapter"] == "pageindex" for r in rows)


class TestAccuracyByType:
    def test_breakdown(self):
        results = _make_results([True, False, True, False, True])
        breakdown = accuracy_by_type(results)
        assert "simple" in breakdown
        assert "relation" in breakdown

    def test_per_adapter(self):
        results = (
            _make_results([True, True], adapter="a1")
            + _make_results([False, False], adapter="a2")
        )
        breakdown = accuracy_by_type(results)
        assert breakdown["simple"]["a1"] == 1.0
        assert breakdown["simple"]["a2"] == 0.0


class TestAccuracyByDocument:
    def test_two_documents(self):
        en = _make_results([True, True, True])
        ru = [
            EvalResult(
                question_id=31, adapter="test", mode="m1", lang="ru",
                query_type="simple", document="ru_graph_kb",
                passed=False, answer="A", latency=1.0, confidence=0.5,
            )
        ]
        breakdown = accuracy_by_document(en + ru)
        assert "en_llm_survey" in breakdown
        assert "ru_graph_kb" in breakdown
