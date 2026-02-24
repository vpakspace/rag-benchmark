"""Tests for data models and BaseAdapter interface."""

from __future__ import annotations

import pytest
from datetime import datetime

from adapters.base import (
    BaseAdapter,
    IngestResult,
    QueryResult,
    EvalResult,
    BenchmarkRun,
    Question,
)


class TestIngestResult:
    def test_create_success(self):
        r = IngestResult(
            adapter="rag2", document="en_llm_survey",
            chunks_count=42, duration_seconds=3.5, success=True,
        )
        assert r.success is True
        assert r.chunks_count == 42
        assert r.error is None

    def test_create_failure(self):
        r = IngestResult(
            adapter="rag2", document="en_llm_survey",
            chunks_count=0, duration_seconds=0.1, success=False,
            error="Connection refused",
        )
        assert r.success is False
        assert "Connection" in r.error


class TestQueryResult:
    def test_defaults(self):
        r = QueryResult(
            adapter="pageindex", mode="tree_reasoning",
            question_id=1, answer="The answer", confidence=0.9, latency=1.2,
        )
        assert r.retries == 0
        assert r.sources == []

    def test_with_sources(self):
        r = QueryResult(
            adapter="rag2", mode="agent", question_id=5,
            answer="Answer", confidence=0.7, latency=3.0,
            retries=1, sources=["chunk A", "chunk B"],
        )
        assert len(r.sources) == 2


class TestEvalResult:
    def test_all_fields(self):
        r = EvalResult(
            question_id=1, adapter="agr", mode="hybrid", lang="ru",
            query_type="temporal", document="ru_graph_kb",
            passed=True, answer="Ответ", latency=5.0, confidence=0.8,
        )
        assert r.passed is True
        assert r.lang == "ru"


class TestQuestion:
    def test_bilingual(self):
        q = Question(
            id=1, document="en_llm_survey",
            question_en="What year was GPT-3 released?",
            question_ru="В каком году вышла GPT-3?",
            type="simple",
            keywords=["2020", "GPT-3"],
        )
        assert q.question_en != q.question_ru
        assert q.reference_answer is None


class TestBenchmarkRun:
    def test_create(self):
        run = BenchmarkRun(
            run_id="run_2026-02-24_15-30",
            timestamp=datetime.now(),
            documents=["en_llm_survey", "ru_graph_kb"],
            adapters=["rag2", "pageindex"],
            total_questions=60,
            results=[],
        )
        assert run.total_questions == 60
        assert len(run.results) == 0


class TestBaseAdapterInterface:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseAdapter()

    def test_concrete_adapter_must_implement_all(self):
        class IncompleteAdapter(BaseAdapter):
            name = "test"
            project_path = "/tmp"
            modes = ["m1"]
            supported_langs = ["en"]
            requires_services = []

        with pytest.raises(TypeError):
            IncompleteAdapter()
