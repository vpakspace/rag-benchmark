"""Integration test — end-to-end flow with mock adapters."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from adapters.base import BaseAdapter, IngestResult, QueryResult, Question, BenchmarkRun
from adapters.registry import AdapterRegistry
from benchmark.compare import compare_adapters, accuracy_by_type, accuracy_by_document
from benchmark.runner import run_benchmark, make_benchmark_run, save_results
from article.generator import ArticleGenerator


class MockAdapterA(BaseAdapter):
    name = "mock_a"
    project_path = "/tmp"
    modes = ["fast", "deep"]
    supported_langs = ["en", "ru"]
    requires_services = []

    def setup(self) -> None:
        pass

    def ingest(self, file_path: str) -> IngestResult:
        return IngestResult(
            adapter=self.name, document=Path(file_path).name,
            chunks_count=10, duration_seconds=0.5, success=True,
        )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        # Return correct-ish answers containing keywords
        return QueryResult(
            adapter=self.name, mode=mode, question_id=0,
            answer=f"The answer involves {question.split()[0]} and more details.",
            confidence=0.8, latency=0.1,
        )

    def cleanup(self) -> None:
        pass


class MockAdapterB(BaseAdapter):
    name = "mock_b"
    project_path = "/tmp"
    modes = ["single"]
    supported_langs = ["en"]
    requires_services = []

    def setup(self) -> None:
        pass

    def ingest(self, file_path: str) -> IngestResult:
        return IngestResult(
            adapter=self.name, document=Path(file_path).name,
            chunks_count=5, duration_seconds=0.3, success=True,
        )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        return QueryResult(
            adapter=self.name, mode=mode, question_id=0,
            answer="I don't know", confidence=0.2, latency=0.05,
        )

    def cleanup(self) -> None:
        pass


def _make_questions() -> list[Question]:
    """Create minimal question set for testing."""
    return [
        Question(
            id=1, document="en_llm_survey",
            question_en="How many parameters does GPT-3 have?",
            question_ru="Сколько параметров у GPT-3?",
            type="simple",
            keywords=["175", "billion", "parameters"],
        ),
        Question(
            id=2, document="ru_graph_kb",
            question_en="When was Neo4j first released?",
            question_ru="Когда вышла первая версия Neo4j?",
            type="temporal",
            keywords=["2007", "Neo4j"],
        ),
        Question(
            id=3, document="en_llm_survey",
            question_en="What is the main innovation of the Transformer architecture?",
            question_ru="В чём главная инновация архитектуры Transformer?",
            type="relation",
            keywords=["self-attention", "Transformer", "architecture"],
        ),
    ]


class TestIntegration:
    def test_registry_and_health(self):
        """Test adapter registration and health check flow."""
        registry = AdapterRegistry()
        a = MockAdapterA()
        b = MockAdapterB()
        a.setup()
        b.setup()
        registry.register(a)
        registry.register(b)

        available = registry.list_available()
        assert len(available) == 2
        assert registry.get("mock_a") is a

    def test_full_benchmark_flow(self):
        """End-to-end: questions → run → results → compare → article."""
        # Setup adapters
        adapter_a = MockAdapterA()
        adapter_b = MockAdapterB()
        adapter_a.setup()
        adapter_b.setup()

        questions = _make_questions()

        # Run benchmark (no OpenAI client — will fall back to keyword judge)
        results = run_benchmark(
            adapters=[adapter_a, adapter_b],
            questions=questions,
            openai_client=None,
            documents={
                "en_llm_survey": "/tmp/en.txt",
                "ru_graph_kb": "/tmp/ru.txt",
            },
            lang_mode="native",
        )

        # Should have results for all adapter/mode/question combinations
        # mock_a: 2 modes × 3 questions = 6
        # mock_b: 1 mode × 3 questions = 3
        assert len(results) == 9

        # Create BenchmarkRun
        run = make_benchmark_run(
            results=results,
            adapters=["mock_a", "mock_b"],
            documents=["en_llm_survey", "ru_graph_kb"],
        )
        assert run.total_questions == 3

        # Compare
        rows = compare_adapters(results)
        assert len(rows) >= 2  # At least mock_a and mock_b

        # Accuracy by type
        by_type = accuracy_by_type(results)
        assert "simple" in by_type

        # Accuracy by document
        by_doc = accuracy_by_document(results)
        assert "en_llm_survey" in by_doc

        # Article generation
        gen = ArticleGenerator(run, lang="en")
        article = gen.generate()
        assert "mock_a" in article
        assert "Results" in article or "Benchmark" in article

    def test_save_and_load_results(self):
        """Test saving and loading results JSON."""
        from adapters.base import EvalResult

        results = [
            EvalResult(
                question_id=1, adapter="test", mode="fast",
                lang="en", query_type="simple", document="en_llm_survey",
                passed=True, answer="42", latency=0.1, confidence=0.9,
            )
        ]
        run = make_benchmark_run(
            results=results,
            adapters=["test"],
            documents=["en_llm_survey"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override results dir
            import benchmark.runner as runner_mod
            original_dir = runner_mod.RESULTS_DIR
            runner_mod.RESULTS_DIR = Path(tmpdir)
            try:
                path = save_results(run)
                assert path.exists()

                # Verify JSON is valid
                data = json.loads(path.read_text())
                assert data["run_id"] == run.run_id
                assert len(data["results"]) == 1
            finally:
                runner_mod.RESULTS_DIR = original_dir
