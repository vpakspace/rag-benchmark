"""Tests for benchmark runner orchestrator."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from adapters.base import BaseAdapter, IngestResult, QueryResult, Question
from benchmark.runner import load_questions, run_benchmark


class StubAdapter(BaseAdapter):
    name = "stub"
    project_path = "/tmp"
    modes = ["fast", "slow"]
    supported_langs = ["en", "ru"]
    requires_services = []

    def setup(self): pass

    def ingest(self, file_path):
        return IngestResult(adapter="stub", document=file_path, chunks_count=10, duration_seconds=0.5, success=True)

    def query(self, question, mode, lang):
        return QueryResult(
            adapter="stub", mode=mode, question_id=0,
            answer=f"Stub answer for: {question[:30]}",
            confidence=0.8, latency=0.1,
        )

    def cleanup(self): pass


class TestLoadQuestions:
    def test_load_from_file(self, tmp_path):
        q_file = tmp_path / "questions.json"
        q_file.write_text(json.dumps([
            {"id": 1, "document": "doc1", "question_en": "Q?", "question_ru": "В?",
             "type": "simple", "keywords": ["k1"]},
        ]))
        questions = load_questions(q_file)
        assert len(questions) == 1
        assert questions[0].id == 1

    def test_validates_question_schema(self, tmp_path):
        q_file = tmp_path / "questions.json"
        q_file.write_text(json.dumps([{"id": 1}]))  # Missing required fields
        with pytest.raises(Exception):
            load_questions(q_file)


class TestRunBenchmark:
    @patch("benchmark.runner.evaluate_answer", return_value=True)
    def test_basic_run(self, mock_eval):
        adapter = StubAdapter()
        questions = [
            Question(id=1, document="en_llm_survey", question_en="Q1?", question_ru="В1?", type="simple", keywords=["k"]),
            Question(id=2, document="en_llm_survey", question_en="Q2?", question_ru="В2?", type="relation", keywords=["k"]),
        ]
        results = run_benchmark(
            adapters=[adapter],
            questions=questions,
            openai_client=MagicMock(),
            documents={"en_llm_survey": "/tmp/en.txt"},
        )
        assert len(results) > 0
        assert all(r.adapter == "stub" for r in results)

    @patch("benchmark.runner.evaluate_answer", return_value=True)
    def test_modes_filter(self, mock_eval):
        adapter = StubAdapter()
        questions = [
            Question(id=1, document="en_llm_survey", question_en="Q?", question_ru="В?", type="simple", keywords=["k"]),
        ]
        results = run_benchmark(
            adapters=[adapter],
            questions=questions,
            openai_client=MagicMock(),
            documents={"en_llm_survey": "/tmp/en.txt"},
            modes_filter={"stub": ["fast"]},
        )
        modes_used = {r.mode for r in results}
        assert modes_used == {"fast"}

    @patch("benchmark.runner.evaluate_answer", side_effect=Exception("judge error"))
    def test_eval_error_recorded(self, mock_eval):
        adapter = StubAdapter()
        questions = [
            Question(id=1, document="en_llm_survey", question_en="Q?", question_ru="В?", type="simple", keywords=["k"]),
        ]
        results = run_benchmark(
            adapters=[adapter],
            questions=questions,
            openai_client=MagicMock(),
            documents={"en_llm_survey": "/tmp/en.txt"},
        )
        # Should not crash; error results should be marked as not passed
        assert all(r.passed is False for r in results)

    @patch("benchmark.runner.evaluate_answer", return_value=True)
    def test_progress_callback(self, mock_eval):
        adapter = StubAdapter()
        questions = [
            Question(id=1, document="en_llm_survey", question_en="Q?", question_ru="В?", type="simple", keywords=["k"]),
        ]
        progress = []
        run_benchmark(
            adapters=[adapter],
            questions=questions,
            openai_client=MagicMock(),
            documents={"en_llm_survey": "/tmp/en.txt"},
            progress_callback=lambda cur, tot, msg: progress.append((cur, tot)),
        )
        assert len(progress) > 0
