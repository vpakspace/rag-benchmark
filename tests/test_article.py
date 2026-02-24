"""Tests for article generator."""

from __future__ import annotations

from datetime import datetime
from adapters.base import BenchmarkRun, EvalResult
from article.generator import ArticleGenerator


def _make_run() -> BenchmarkRun:
    results = [
        EvalResult(
            question_id=i, adapter="rag2", mode="agent", lang="en",
            query_type="simple", document="en_llm_survey",
            passed=i % 2 == 0, answer="A", latency=2.0, confidence=0.8,
        )
        for i in range(10)
    ]
    return BenchmarkRun(
        run_id="test_run", timestamp=datetime.now(),
        documents=["en_llm_survey"], adapters=["rag2"],
        total_questions=10, results=results,
    )


class TestArticleGenerator:
    def test_generate_returns_markdown(self):
        gen = ArticleGenerator(_make_run(), lang="ru")
        text = gen.generate()
        assert "# " in text  # Has headings
        assert "rag2" in text  # Mentions adapter

    def test_generate_en(self):
        gen = ArticleGenerator(_make_run(), lang="en")
        text = gen.generate()
        assert "Results" in text or "Benchmark" in text

    def test_includes_accuracy(self):
        gen = ArticleGenerator(_make_run(), lang="ru")
        text = gen.generate()
        assert "%" in text  # Has percentage
