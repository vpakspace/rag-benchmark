"""Shared fixtures for rag-benchmark tests."""

from __future__ import annotations

import pytest

from adapters.base import QueryResult, IngestResult, EvalResult, BenchmarkRun


@pytest.fixture
def sample_query_result():
    return QueryResult(
        adapter="test_adapter",
        mode="test_mode",
        question_id=1,
        answer="Test answer about GPT-3 released in 2020.",
        confidence=0.85,
        latency=2.5,
        retries=0,
        sources=["chunk_1"],
    )


@pytest.fixture
def sample_eval_results():
    """10 eval results: 7 passed, 3 failed."""
    results = []
    for i in range(10):
        results.append(EvalResult(
            question_id=i + 1,
            adapter="test_adapter",
            mode="test_mode",
            lang="en",
            query_type=["simple", "relation", "multi_hop", "global", "temporal"][i % 5],
            document="en_llm_survey",
            passed=i < 7,
            answer=f"Answer {i}",
            latency=2.0 + i * 0.5,
            confidence=0.5 + i * 0.05,
        ))
    return results
