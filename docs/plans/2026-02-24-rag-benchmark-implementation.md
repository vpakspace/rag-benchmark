# RAG Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unified benchmark evaluating 6 RAG projects on 2 bilingual documents (60 questions), with Streamlit dashboard and auto-generated article.

**Architecture:** Adapter Pattern — BaseAdapter ABC with 6 concrete adapters importing project modules via sys.path. 3-level judge (embedding → keywords → LLM). Results persisted as JSON, visualized in Streamlit.

**Tech Stack:** Python 3.12, Pydantic, OpenAI, Streamlit, Plotly, pandas, pytest

---

## Phase 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `adapters/__init__.py`
- Create: `benchmark/__init__.py`
- Create: `article/__init__.py`
- Create: `dashboard/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/documents/.gitkeep`
- Create: `results/.gitkeep`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "rag-benchmark"
version = "0.1.0"
description = "Cross-RAG Benchmark: unified evaluation of 6 RAG projects"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "streamlit>=1.30",
    "plotly>=5.0",
    "pandas>=2.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100
```

**Step 2: Create .env.example**

```bash
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=temporal_kb_2026
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
results/*.json
!results/.gitkeep
.ruff_cache/
.pytest_cache/
*.egg-info/
dist/
build/
```

**Step 4: Create all __init__.py files and directories**

All `__init__.py` files are empty. Create `tests/conftest.py`:

```python
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
```

**Step 5: git init and commit**

```bash
cd ~/rag-benchmark
git init
git add .
git commit -m "feat: project scaffolding"
```

---

### Task 2: Data Models and BaseAdapter

**Files:**
- Create: `adapters/base.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/test_models.py
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
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/rag-benchmark
python -m pytest tests/test_models.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'adapters.base'`

**Step 3: Implement adapters/base.py**

```python
"""Base adapter interface and data models for rag-benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    adapter: str
    document: str
    chunks_count: int
    duration_seconds: float
    success: bool
    error: str | None = None


class QueryResult(BaseModel):
    adapter: str
    mode: str
    question_id: int
    answer: str
    confidence: float
    latency: float
    retries: int = 0
    sources: list[str] = []


class EvalResult(BaseModel):
    question_id: int
    adapter: str
    mode: str
    lang: str
    query_type: str
    document: str
    passed: bool
    answer: str
    latency: float
    confidence: float


class Question(BaseModel):
    id: int
    document: str
    question_en: str
    question_ru: str
    type: str  # simple | relation | multi_hop | global | temporal
    keywords: list[str]
    reference_answer: str | None = None


class BenchmarkRun(BaseModel):
    run_id: str
    timestamp: datetime
    documents: list[str]
    adapters: list[str]
    total_questions: int
    results: list[EvalResult]


# ---------------------------------------------------------------------------
# Base Adapter
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Abstract base for all RAG project adapters."""

    name: str
    project_path: str
    modes: list[str]
    supported_langs: list[str]
    requires_services: list[str]

    @abstractmethod
    def setup(self) -> None:
        """Initialize clients, verify services are available."""

    @abstractmethod
    def ingest(self, file_path: str) -> IngestResult:
        """Ingest a document into the project's storage."""

    @abstractmethod
    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        """Query the project with a question in a given mode."""

    @abstractmethod
    def cleanup(self) -> None:
        """Remove ingested data (indexes, nodes) for clean re-run."""

    def health_check(self) -> bool:
        """Check if required services are available. Override if needed."""
        return True
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_models.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add adapters/base.py tests/test_models.py tests/conftest.py
git commit -m "feat: data models and BaseAdapter ABC"
```

---

### Task 3: Evaluate Module (3-Level Judge)

**Files:**
- Create: `benchmark/evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
"""Tests for 3-level judge evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmark.evaluate import keyword_overlap, is_global_query, evaluate_answer


class TestKeywordOverlap:
    def test_all_found(self):
        assert keyword_overlap("GPT-3 was released in 2020", ["GPT-3", "2020"]) == 1.0

    def test_none_found(self):
        assert keyword_overlap("Unrelated text", ["GPT-3", "2020"]) == 0.0

    def test_partial(self):
        assert keyword_overlap("GPT-3 is a model", ["GPT-3", "2020", "OpenAI"]) == pytest.approx(1 / 3)

    def test_case_insensitive(self):
        assert keyword_overlap("gpt-3 released", ["GPT-3"]) == 1.0

    def test_empty_keywords(self):
        assert keyword_overlap("any text", []) == 0.0

    def test_cyrillic_keywords(self):
        assert keyword_overlap("Neo4j является графовой базой данных", ["графовой", "Neo4j"]) == 1.0


class TestIsGlobalQuery:
    def test_english_list_all(self):
        assert is_global_query("List all frameworks mentioned") is True

    def test_russian_enumerate(self):
        assert is_global_query("Перечисли все компоненты архитектуры") is True

    def test_simple_question(self):
        assert is_global_query("What year was GPT-3 released?") is False

    def test_overview(self):
        assert is_global_query("Give an overview of the system") is True


class TestEvaluateAnswer:
    def test_keyword_pass_simple(self):
        """High keyword overlap → auto-PASS without LLM call."""
        result = evaluate_answer(
            question="What is GPT-3?",
            answer="GPT-3 is a large language model by OpenAI with 175B parameters released in 2020",
            keywords=["GPT-3", "OpenAI", "175B", "2020"],
            openai_client=MagicMock(),  # Should not be called
        )
        assert result is True

    def test_keyword_fail_triggers_llm(self):
        """Low keyword overlap → LLM judge called."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="PASS"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip embedding")

        result = evaluate_answer(
            question="What is GPT-3?",
            answer="It is a neural network",
            keywords=["GPT-3", "OpenAI", "175B", "2020"],
            openai_client=mock_client,
        )
        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_llm_judge_fail(self):
        """LLM judge returns FAIL."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FAIL"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip embedding")

        result = evaluate_answer(
            question="What is GPT-3?",
            answer="I don't know",
            keywords=["GPT-3", "OpenAI", "175B"],
            openai_client=mock_client,
        )
        assert result is False

    def test_embedding_similarity_pass(self):
        """High embedding similarity with reference → auto-PASS."""
        mock_client = MagicMock()
        # Return two identical embeddings → similarity = 1.0
        mock_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=mock_embedding), MagicMock(embedding=mock_embedding)]
        )

        result = evaluate_answer(
            question="What is X?",
            answer="X is Y",
            keywords=[],
            openai_client=mock_client,
            reference_answer="X is Y exactly",
        )
        assert result is True

    def test_global_query_higher_threshold(self):
        """Global queries need 65% keyword overlap (not 40%)."""
        # 2/5 = 40% — passes for simple (≥0.40) but fails for global (needs ≥0.65)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FAIL"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip")

        result = evaluate_answer(
            question="List all components of the system",
            answer="Component A and Component B are present",
            keywords=["A", "B", "C", "D", "E"],
            openai_client=mock_client,
        )
        assert result is False  # 40% < 65% threshold, LLM says FAIL
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_evaluate.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement benchmark/evaluate.py**

Adapt from `~/agentic-graph-rag/benchmark/runner.py:29-135` — extract evaluation logic into standalone module without project-specific imports.

```python
"""3-level evaluation judge for RAG benchmark.

Adapted from agentic-graph-rag benchmark/runner.py.
Standalone — no project-specific imports.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are evaluating a RAG system answer.

Question: {question}
Expected keywords/concepts: {keywords}
System answer: {answer}

IMPORTANT: The answer may be in Russian while keywords are in English, or vice versa.
Match CONCEPTS and meanings, not exact strings.
For example: "ontology" matches "онтология", "graph" matches "граф", etc.

Does the answer correctly address the question and cover the expected concepts?
For enumeration questions (list all, describe all), check at least 50% coverage.
Reply with ONLY one word: PASS or FAIL"""

_JUDGE_PROMPT_REF = """You are evaluating a RAG system answer against a reference.

Question: {question}
Reference answer: {reference_answer}
System answer: {answer}

Does the system answer cover the same THEMES as the reference?
Match meanings and concepts, not exact words. Languages may differ.
For enumeration questions, PASS if at least 50% themes overlap semantically.
Reply ONLY: PASS or FAIL"""

# ---------------------------------------------------------------------------
# Global/comprehensive query detection
# ---------------------------------------------------------------------------

_GLOBAL_RE = re.compile(
    r'\b('
    r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
    r'|list all|describe all|summarize all|overview|every\b'
    r'|все компоненты|все методы|все слои|все решения'
    r'|all components|all layers|all methods|all decisions'
    r')',
    re.IGNORECASE,
)


def is_global_query(query: str) -> bool:
    """Detect global/enumeration queries."""
    return bool(_GLOBAL_RE.search(query))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def keyword_overlap(answer: str, keywords: list[str]) -> float:
    """Return fraction of keywords found in answer (case-insensitive)."""
    if not keywords:
        return 0.0
    lower = answer.lower()
    found = sum(1 for k in keywords if k.lower() in lower)
    return found / len(keywords)


def _embedding_similarity(text_a: str, text_b: str, openai_client: OpenAI) -> float:
    """Cosine similarity between embeddings of two texts."""
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text_a[:8000], text_b[:8000]],
        )
        vec_a = resp.data[0].embedding
        vec_b = resp.data[1].embedding
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    except Exception as e:
        logger.error("Embedding similarity failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    answer: str,
    keywords: list[str],
    openai_client: OpenAI,
    reference_answer: str = "",
) -> bool:
    """Hybrid judge: embedding similarity → keyword overlap → LLM judge."""
    # Fast path 1: embedding similarity with reference ≥ 0.65
    if reference_answer:
        similarity = _embedding_similarity(answer, reference_answer, openai_client)
        if similarity >= 0.65:
            return True

    # Fast path 2: keyword overlap ≥ threshold
    overlap = keyword_overlap(answer, keywords)
    threshold = 0.65 if is_global_query(question) else 0.40
    if overlap >= threshold:
        return True

    # Fallback: LLM judge
    if reference_answer:
        prompt_text = _JUDGE_PROMPT_REF.format(
            question=question,
            reference_answer=reference_answer,
            answer=answer[:2000],
        )
    else:
        prompt_text = _JUDGE_PROMPT.format(
            question=question,
            keywords=", ".join(keywords),
            answer=answer[:2000],
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        verdict = text.split("\n")[-1].strip().upper()
        return verdict == "PASS"
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return False
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_evaluate.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add benchmark/evaluate.py tests/test_evaluate.py
git commit -m "feat: 3-level evaluation judge"
```

---

### Task 4: Compare Module (Metrics)

**Files:**
- Create: `benchmark/compare.py`
- Create: `tests/test_compare.py`

**Step 1: Write the failing tests**

```python
# tests/test_compare.py
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
        # latencies: 2.0, 3.0 → avg 2.5
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
```

**Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_compare.py -v
```

**Step 3: Implement benchmark/compare.py**

```python
"""Benchmark comparison — compute metrics, generate tables and breakdowns."""

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

    Returns dict[query_type → dict[adapter → accuracy]].
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

    Returns dict[document → dict[adapter → accuracy]].
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_compare.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add benchmark/compare.py tests/test_compare.py
git commit -m "feat: metrics computation and comparison tables"
```

---

## Phase 2: Core Pipeline

### Task 5: Adapter Registry

**Files:**
- Create: `adapters/registry.py`
- Create: `tests/test_registry.py`

**Step 1: Write the failing tests**

```python
# tests/test_registry.py
"""Tests for adapter discovery and registry."""

from __future__ import annotations

import pytest
from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters.registry import AdapterRegistry


class DummyAdapter(BaseAdapter):
    name = "dummy"
    project_path = "/tmp/dummy"
    modes = ["mode_a", "mode_b"]
    supported_langs = ["en", "ru"]
    requires_services = []

    def setup(self): pass
    def ingest(self, file_path): return IngestResult(adapter="dummy", document=file_path, chunks_count=5, duration_seconds=0.1, success=True)
    def query(self, question, mode, lang): return QueryResult(adapter="dummy", mode=mode, question_id=0, answer="dummy", confidence=1.0, latency=0.01)
    def cleanup(self): pass


class TestAdapterRegistry:
    def test_register_and_get(self):
        reg = AdapterRegistry()
        adapter = DummyAdapter()
        reg.register(adapter)
        assert reg.get("dummy") is adapter

    def test_get_unknown_returns_none(self):
        reg = AdapterRegistry()
        assert reg.get("unknown") is None

    def test_list_all(self):
        reg = AdapterRegistry()
        reg.register(DummyAdapter())
        names = reg.list_names()
        assert "dummy" in names

    def test_list_available_checks_health(self):
        reg = AdapterRegistry()
        reg.register(DummyAdapter())
        available = reg.list_available()
        assert len(available) == 1

    def test_register_duplicate_overwrites(self):
        reg = AdapterRegistry()
        a1 = DummyAdapter()
        a2 = DummyAdapter()
        reg.register(a1)
        reg.register(a2)
        assert reg.get("dummy") is a2
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement adapters/registry.py**

```python
"""Adapter registry — discover, register, and manage RAG adapters."""

from __future__ import annotations

import logging

from adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for RAG project adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter) -> None:
        """Register an adapter (overwrites if name exists)."""
        self._adapters[adapter.name] = adapter
        logger.info("Registered adapter: %s (%d modes)", adapter.name, len(adapter.modes))

    def get(self, name: str) -> BaseAdapter | None:
        """Get adapter by name."""
        return self._adapters.get(name)

    def list_names(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def list_available(self) -> list[BaseAdapter]:
        """List adapters that pass health_check."""
        available = []
        for adapter in self._adapters.values():
            try:
                if adapter.health_check():
                    available.append(adapter)
                else:
                    logger.warning("Adapter %s failed health check", adapter.name)
            except Exception as e:
                logger.warning("Adapter %s health check error: %s", adapter.name, e)
        return available

    def all(self) -> list[BaseAdapter]:
        """List all registered adapters."""
        return list(self._adapters.values())
```

**Step 4: Run tests — expect ALL PASS**

**Step 5: Commit**

```bash
git add adapters/registry.py tests/test_registry.py
git commit -m "feat: adapter registry"
```

---

### Task 6: Benchmark Runner

**Files:**
- Create: `benchmark/runner.py`
- Create: `tests/test_runner.py`

**Step 1: Write the failing tests**

```python
# tests/test_runner.py
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
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement benchmark/runner.py**

```python
"""Benchmark runner — orchestrate evaluation across adapters and modes."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from adapters.base import BaseAdapter, EvalResult, Question, BenchmarkRun
from benchmark.evaluate import evaluate_answer

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_questions(path: Path | None = None) -> list[Question]:
    """Load and validate benchmark questions from JSON."""
    p = path or QUESTIONS_PATH
    with open(p) as f:
        raw = json.load(f)
    return [Question(**q) for q in raw]


def save_results(run: BenchmarkRun) -> Path:
    """Save benchmark run results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2))
    logger.info("Results saved to %s", path)
    return path


def run_benchmark(
    adapters: list[BaseAdapter],
    questions: list[Question],
    openai_client: OpenAI,
    documents: dict[str, str],
    lang_mode: str = "native",
    modes_filter: dict[str, list[str]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[EvalResult]:
    """Run benchmark across adapters, modes, and questions.

    Args:
        adapters: List of initialized adapters.
        questions: List of Question objects.
        openai_client: OpenAI client for evaluation.
        documents: dict[doc_name → file_path].
        lang_mode: "native" (match doc lang), "cross", or "all".
        modes_filter: Optional dict[adapter_name → list of modes to run].
        progress_callback: Optional fn(current, total, message).

    Returns:
        List of EvalResult objects.
    """
    all_results: list[EvalResult] = []

    # Calculate total work
    total_evals = 0
    for adapter in adapters:
        modes = modes_filter.get(adapter.name, adapter.modes) if modes_filter else adapter.modes
        total_evals += len(modes) * len(questions)

    current = 0

    for adapter in adapters:
        # Health check
        if not adapter.health_check():
            logger.warning("Adapter %s failed health check — skipping", adapter.name)
            continue

        modes = modes_filter.get(adapter.name, adapter.modes) if modes_filter else adapter.modes

        for mode in modes:
            for q in questions:
                current += 1

                # Determine question language based on lang_mode
                if lang_mode == "native":
                    # EN question for EN doc, RU question for RU doc
                    lang = "ru" if "ru_" in q.document else "en"
                elif lang_mode == "cross":
                    lang = "en" if "ru_" in q.document else "ru"
                else:
                    lang = "en"  # TODO: expand for "all" mode

                question_text = q.question_ru if lang == "ru" else q.question_en

                if progress_callback:
                    progress_callback(current, total_evals, f"{adapter.name}/{mode}/Q{q.id}")

                start = time.monotonic()
                try:
                    qr = adapter.query(question_text, mode, lang)
                    latency = round(time.monotonic() - start, 3)

                    try:
                        passed = evaluate_answer(
                            question_text, qr.answer, q.keywords,
                            openai_client, q.reference_answer or "",
                        )
                    except Exception as e:
                        logger.error("Eval error [%s/%s] Q%d: %s", adapter.name, mode, q.id, e)
                        passed = False

                    all_results.append(EvalResult(
                        question_id=q.id, adapter=adapter.name, mode=mode,
                        lang=lang, query_type=q.type, document=q.document,
                        passed=passed, answer=qr.answer,
                        latency=latency, confidence=qr.confidence,
                    ))

                except Exception as e:
                    latency = round(time.monotonic() - start, 3)
                    logger.error("Query error [%s/%s] Q%d: %s", adapter.name, mode, q.id, e)
                    all_results.append(EvalResult(
                        question_id=q.id, adapter=adapter.name, mode=mode,
                        lang=lang, query_type=q.type, document=q.document,
                        passed=False, answer=f"ERROR: {e}",
                        latency=latency, confidence=0.0,
                    ))

        logger.info(
            "Adapter %s complete: %d/%d passed",
            adapter.name,
            sum(1 for r in all_results if r.adapter == adapter.name and r.passed),
            sum(1 for r in all_results if r.adapter == adapter.name),
        )

    return all_results


def make_benchmark_run(
    results: list[EvalResult],
    adapters: list[str],
    documents: list[str],
) -> BenchmarkRun:
    """Create a BenchmarkRun from results."""
    now = datetime.now()
    return BenchmarkRun(
        run_id=f"run_{now.strftime('%Y-%m-%d_%H-%M')}",
        timestamp=now,
        documents=documents,
        adapters=adapters,
        total_questions=len({r.question_id for r in results}),
        results=results,
    )
```

**Step 4: Run tests — expect ALL PASS**

**Step 5: Commit**

```bash
git add benchmark/runner.py tests/test_runner.py
git commit -m "feat: benchmark runner orchestrator"
```

---

## Phase 3: Documents & Questions

### Task 7: Prepare EN Document

**Files:**
- Create: `data/documents/en_llm_survey.txt`
- Create: `scripts/prepare_documents.py`

**Step 1: Create prepare_documents.py**

Script downloads arxiv paper "A Survey of Large Language Models" (2303.18223) and extracts ~10 page excerpt with temporal events.

```python
# scripts/prepare_documents.py
"""Download and prepare benchmark documents."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


def prepare_en_document() -> Path:
    """Download LLM Survey paper (Zhao et al., 2023) and extract excerpt.

    If PDF download fails, uses a pre-written excerpt covering key temporal events.
    """
    output = DATA_DIR / "en_llm_survey.txt"
    if output.exists():
        print(f"EN document already exists: {output}")
        return output

    # Pre-written excerpt covering temporal events for benchmark questions
    # Source: "A Survey of Large Language Models" (Zhao et al., 2023, arXiv:2303.18223)
    print(f"Writing EN document excerpt to {output}")
    # Content will be written by the executing agent using the actual paper
    # Placeholder for plan — actual content ~5-10K words with temporal dimension
    print("EN document prepared.")
    return output


def prepare_ru_document() -> Path:
    """Create Russian technical review of Graph DBs and Knowledge Graphs."""
    output = DATA_DIR / "ru_graph_kb.txt"
    if output.exists():
        print(f"RU document already exists: {output}")
        return output

    print(f"Writing RU document to {output}")
    # Content will be composed by the executing agent
    # ~10 pages, Russian, temporal events from 1999-2024
    print("RU document prepared.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark documents")
    parser.add_argument("--lang", choices=["en", "ru", "all"], default="all")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.lang in ("en", "all"):
        prepare_en_document()
    if args.lang in ("ru", "all"):
        prepare_ru_document()
```

**Step 2: Write EN document excerpt**

Create `data/documents/en_llm_survey.txt` — an excerpt (~8-10 pages, ~6K words) from "A Survey of Large Language Models" covering:
- Introduction to LLMs and their timeline
- GPT series: GPT-1 (2018, 117M) → GPT-2 (2019, 1.5B) → GPT-3 (2020, 175B) → GPT-4 (2023)
- BERT (2018) and bidirectional models
- T5 (2019), PaLM (2022, 540B), LLaMA (2023)
- Scaling laws (Kaplan et al., 2020; Chinchilla, 2022)
- RLHF and InstructGPT (2022)
- ChatGPT launch (November 2022) and impact
- Emergent abilities and benchmarks (MMLU, HumanEval, GSM8K)
- Architecture details (Transformer, attention, positional encoding)
- Training data, tokenization, pre-training objectives

Use Perplexity MCP to fetch factual content from the actual paper and compose the excerpt. Ensure all dates and numbers are accurate.

**Step 3: Commit**

```bash
git add data/documents/en_llm_survey.txt scripts/prepare_documents.py
git commit -m "feat: EN benchmark document (LLM Survey excerpt)"
```

---

### Task 8: Prepare RU Document

**Files:**
- Create: `data/documents/ru_graph_kb.txt`

**Step 1: Compose Russian document**

Create `data/documents/ru_graph_kb.txt` — original Russian technical review (~8-10 pages, ~6K words):

**Required sections with temporal events:**
1. Введение в графовые модели данных (1960s: сетевая модель, 1999: RDF, W3C)
2. RDF и семантический веб (1999-2008: RDF → RDFS → OWL → SPARQL 1.0)
3. Property Graph модель (2007: Neo4j 1.0, 2010: Titan/JanusGraph, 2012: OrientDB)
4. Knowledge Graphs (2012: Google Knowledge Graph, 2012: Wikidata, 2015: Amazon Neptune анонс)
5. Графовые СУБД: сравнение (Neo4j, ArangoDB, Amazon Neptune, Dgraph, TigerGraph)
6. Языки запросов (Cypher vs Gremlin vs SPARQL, 2019: GQL стандарт ISO)
7. Knowledge Graph Embeddings (2013: TransE, 2014: TransR, 2019: RotatE)
8. Graph Neural Networks для KG (2017: R-GCN, 2019: CompGCN)
9. GraphRAG (2024: Microsoft GraphRAG, community summaries, Graphiti, Cognee)
10. Применения (рекомендательные системы, fraud detection, биомедицина, enterprise)

Use Perplexity MCP to verify dates and facts. Write in natural Russian technical style (not translation).

**Step 2: Commit**

```bash
git add data/documents/ru_graph_kb.txt
git commit -m "feat: RU benchmark document (Graph KB review)"
```

---

### Task 9: Generate Questions

**Files:**
- Create: `data/questions.json`
- Create: `scripts/generate_questions.py`
- Create: `tests/test_questions.py`

**Step 1: Write validation tests**

```python
# tests/test_questions.py
"""Tests for questions.json validity."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from benchmark.runner import load_questions


QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "questions.json"


class TestQuestionsFile:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.questions = load_questions(QUESTIONS_PATH)

    def test_total_count(self):
        assert len(self.questions) == 60

    def test_en_doc_count(self):
        en = [q for q in self.questions if q.document == "en_llm_survey"]
        assert len(en) == 30

    def test_ru_doc_count(self):
        ru = [q for q in self.questions if q.document == "ru_graph_kb"]
        assert len(ru) == 30

    def test_all_bilingual(self):
        for q in self.questions:
            assert q.question_en, f"Q{q.id} missing question_en"
            assert q.question_ru, f"Q{q.id} missing question_ru"

    def test_type_distribution_en(self):
        en = [q for q in self.questions if q.document == "en_llm_survey"]
        types = {}
        for q in en:
            types[q.type] = types.get(q.type, 0) + 1
        assert types == {"simple": 7, "relation": 6, "multi_hop": 6, "global": 6, "temporal": 5}

    def test_type_distribution_ru(self):
        ru = [q for q in self.questions if q.document == "ru_graph_kb"]
        types = {}
        for q in ru:
            types[q.type] = types.get(q.type, 0) + 1
        assert types == {"simple": 7, "relation": 6, "multi_hop": 6, "global": 6, "temporal": 5}

    def test_unique_ids(self):
        ids = [q.id for q in self.questions]
        assert len(ids) == len(set(ids))

    def test_all_have_keywords(self):
        for q in self.questions:
            assert len(q.keywords) >= 2, f"Q{q.id} needs at least 2 keywords"

    def test_valid_types(self):
        valid = {"simple", "relation", "multi_hop", "global", "temporal"}
        for q in self.questions:
            assert q.type in valid, f"Q{q.id} has invalid type: {q.type}"
```

**Step 2: Create generate_questions.py helper**

```python
# scripts/generate_questions.py
"""LLM-assisted generation of benchmark questions.

Reads documents and generates bilingual questions with keywords.
Output: data/questions.json
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DOCS_DIR = DATA_DIR / "documents"
OUTPUT = DATA_DIR / "questions.json"


def generate() -> None:
    """Generate 60 bilingual questions based on documents.

    Questions are written manually (or with LLM assistance) to ensure
    quality, correct keywords, and accurate reference answers.
    """
    # Questions are authored during implementation — see Task 9 in plan.
    # This script validates and formats the final JSON.
    questions = json.loads(OUTPUT.read_text())
    print(f"Loaded {len(questions)} questions")

    # Validate
    for q in questions:
        assert "id" in q and "question_en" in q and "question_ru" in q
        assert "type" in q and "keywords" in q and "document" in q

    print("All questions valid!")


if __name__ == "__main__":
    generate()
```

**Step 3: Write questions.json**

Create `data/questions.json` with 60 bilingual questions. Questions MUST be grounded in the actual documents from Tasks 7-8. Each question has keywords from the document text.

**Distribution reminder:**
- EN doc (id 1-30): 7 simple, 6 relation, 6 multi_hop, 6 global, 5 temporal
- RU doc (id 31-60): 7 simple, 6 relation, 6 multi_hop, 6 global, 5 temporal

**Step 4: Run validation tests**

```bash
python -m pytest tests/test_questions.py -v
```

**Step 5: Commit**

```bash
git add data/questions.json scripts/generate_questions.py tests/test_questions.py
git commit -m "feat: 60 bilingual benchmark questions"
```

---

## Phase 4: Adapters

Each adapter follows the same pattern:
1. Write test with mocked project imports
2. Implement adapter with sys.path bridge
3. Run test
4. Commit

### Task 10: PageIndex Adapter (simplest — no external services)

**Files:**
- Create: `adapters/pageindex_adapter.py`
- Create: `tests/test_adapter_pageindex.py`

**Step 1: Write the failing test**

```python
# tests/test_adapter_pageindex.py
"""Tests for PageIndex adapter."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from adapters.pageindex_adapter import PageIndexAdapter


class TestPageIndexAdapter:
    def test_name_and_modes(self):
        adapter = PageIndexAdapter.__new__(PageIndexAdapter)
        assert adapter.name == "pageindex"
        assert adapter.modes == ["tree_reasoning"]
        assert "neo4j" not in adapter.requires_services

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_ingest(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.load_file.return_value = "# Markdown content"
        mock_mod.build_tree.return_value = MagicMock()
        mock_mod.summarize_tree.return_value = MagicMock()
        mock_mod.TreeStore.return_value.save.return_value = "doc_123"
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "pageindex"
        mock_mod.load_file.assert_called_once()

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_query(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.reason_and_answer.return_value = MagicMock(
            answer="The answer is 42", confidence=0.9, hops=2
        )
        mock_mod.TreeStore.return_value.list_trees.return_value = [{"doc_id": "d1"}]
        mock_mod.TreeStore.return_value.load.return_value = MagicMock()
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "tree_reasoning", "en")

        assert result.answer == "The answer is 42"
        assert result.confidence == 0.9

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.TreeStore.return_value.delete_all.return_value = 5
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        adapter.cleanup()  # Should not raise
        mock_mod.TreeStore.return_value.delete_all.assert_called_once()
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

```python
# adapters/pageindex_adapter.py
"""PageIndex adapter — tree-based vectorless RAG."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult

logger = logging.getLogger(__name__)

PAGEINDEX_PATH = Path.home() / "pageindex"


def _import_pageindex() -> ModuleType:
    """Import pageindex modules via sys.path bridge."""
    import importlib

    project = str(PAGEINDEX_PATH)
    if project not in sys.path:
        sys.path.insert(0, project)

    # Clean cached modules to avoid stale imports
    for key in list(sys.modules.keys()):
        if key.startswith("indexing.") or key.startswith("reasoning.") or key.startswith("storage."):
            if PAGEINDEX_PATH.name in getattr(sys.modules[key], "__file__", ""):
                del sys.modules[key]

    mod = ModuleType("pageindex_bridge")
    mod.load_file = importlib.import_module("indexing.loader").load_file
    mod.build_tree = importlib.import_module("indexing.tree_builder").build_tree
    mod.summarize_tree = importlib.import_module("indexing.summarizer").summarize_tree
    mod.TreeStore = importlib.import_module("storage.tree_store").TreeStore
    mod.reason_and_answer = importlib.import_module("reasoning.answerer").reason_and_answer
    return mod


class PageIndexAdapter(BaseAdapter):
    name = "pageindex"
    project_path = str(PAGEINDEX_PATH)
    modes = ["tree_reasoning"]
    supported_langs = ["en", "ru"]
    requires_services = []  # No external services — local JSON store

    def __init__(self) -> None:
        self._mod = None
        self._store = None

    def setup(self) -> None:
        self._mod = _import_pageindex()
        self._store = self._mod.TreeStore()

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            text = self._mod.load_file(file_path)
            tree = self._mod.build_tree(text, filename=Path(file_path).name)
            tree = self._mod.summarize_tree(tree)
            doc_id = self._store.save(tree)
            duration = round(time.monotonic() - start, 3)
            # Count nodes as "chunks" equivalent
            node_count = len(getattr(tree, "nodes", [])) or 1
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=node_count, duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("PageIndex ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        trees = self._store.list_trees()
        if not trees:
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer="No documents indexed", confidence=0.0, latency=0.0,
            )
        tree = self._store.load(doc_id=trees[0]["doc_id"])
        start = time.monotonic()
        result = self._mod.reason_and_answer(question=question, tree=tree, store=self._store)
        latency = round(time.monotonic() - start, 3)
        return QueryResult(
            adapter=self.name, mode=mode, question_id=0,
            answer=result.answer,
            confidence=getattr(result, "confidence", 0.5),
            latency=latency,
        )

    def cleanup(self) -> None:
        if self._store:
            self._store.delete_all()

    def health_check(self) -> bool:
        return PAGEINDEX_PATH.exists()
```

**Step 4: Run tests — expect ALL PASS**

**Step 5: Commit**

```bash
git add adapters/pageindex_adapter.py tests/test_adapter_pageindex.py
git commit -m "feat: PageIndex adapter"
```

---

### Task 11: RAG 2.0 Adapter

**Files:**
- Create: `adapters/rag2_adapter.py`
- Create: `tests/test_adapter_rag2.py`

Same pattern as Task 10. Key differences:
- `requires_services = ["neo4j", "openai"]`
- `modes = ["vector", "reflect", "agent"]`
- Bridge imports: `ingestion.loader`, `ingestion.chunker`, `ingestion.enricher`, `storage.vector_store`, `generation.reflector`, `agent.rag_agent`
- `ingest()`: load_file → chunk_text → enrich_chunks → embed_chunks → store.add_chunks
- `query(mode="vector")`: Retriever.retrieve → generate_answer
- `query(mode="reflect")`: reflect_and_answer
- `query(mode="agent")`: RAGAgent.run
- `cleanup()`: store.clear() or drop bench_rag2_* index
- Neo4j index prefix: `bench_rag2`

**Test**: Mock all rag-2.0 imports, verify ingest/query/cleanup call chains.

**Commit**: `git commit -m "feat: RAG 2.0 adapter"`

---

### Task 12: Agentic Graph RAG Adapter

**Files:**
- Create: `adapters/agentic_graph_rag_adapter.py`
- Create: `tests/test_adapter_agr.py`

Key differences:
- `requires_services = ["neo4j", "openai"]`
- `modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm"]`
- Note: `agent_mangle` excluded (requires PyMangle — WIP on separate branch)
- Bridge imports from `~/agentic-graph-rag/`: `rag_core.*`, `agentic_graph_rag.agent.*`, `agentic_graph_rag.service`
- `ingest()`: Uses PipelineService or direct rag_core ingestion
- `query()`: Uses PipelineService.query(text, mode)
- Neo4j index prefix: `bench_agr`

**Commit**: `git commit -m "feat: Agentic Graph RAG adapter"`

---

### Task 13: RAG Temporal Adapter

**Files:**
- Create: `adapters/rag_temporal_adapter.py`
- Create: `tests/test_adapter_rt.py`

Key differences:
- `requires_services = ["neo4j", "openai"]`
- `modes = ["vector", "kg", "hybrid", "agent"]`
- Bridge imports from `~/rag-temporal/`: DualPipeline, HybridAgent, HybridRetriever
- **Note**: rag-temporal imports rag-2.0 via sys.path — adapter must handle double sys.path
- `ingest()`: DualPipeline.ingest_file (dual-track: vector + KG)
- `query(mode="vector")`: Retriever.retrieve (from rag-2.0)
- `query(mode="kg")`: KGClient.search
- `query(mode="hybrid")`: HybridRetriever.retrieve
- `query(mode="agent")`: HybridAgent.run

**Commit**: `git commit -m "feat: RAG Temporal adapter"`

---

### Task 14: Temporal KB Adapter

**Files:**
- Create: `adapters/temporal_kb_adapter.py`
- Create: `tests/test_adapter_tkb.py`

Key differences:
- `requires_services = ["neo4j", "openai"]`
- `modes = ["structural", "temporal", "hybrid"]`
- Bridge imports from `~/temporal-knowledge-base/`: IngestionPipeline, QueryEngine, ResponseBuilder
- `ingest()`: IngestionPipeline.ingest(content, source, episode_type="text")
- `query()`: QueryEngine.search(query, intent=mode) → ResponseBuilder.build_response
- **Note**: Async API — use `asyncio.run()` or event loop wrapper

**Commit**: `git commit -m "feat: Temporal KB adapter"`

---

### Task 15: Cog-RAG-Cognee Adapter

**Files:**
- Create: `adapters/cog_rag_cognee_adapter.py`
- Create: `tests/test_adapter_crc.py`

Key differences:
- `requires_services = ["neo4j", "ollama"]`
- `modes = ["chunks", "graph", "rag_completion"]`
- Bridge imports from `~/cog-rag-cognee/`: PipelineService
- **Note**: Fully async API — requires asyncio wrapper
- **Note**: Uses Ollama (not OpenAI) — may be slower
- `ingest()`: service.add_file → service.cognify
- `query()`: service.query(query, search_type=mode)
- `health_check()`: Check Ollama + Neo4j availability

**Commit**: `git commit -m "feat: Cog-RAG-Cognee adapter"`

---

## Phase 5: Dashboard & Article

### Task 16: Streamlit Dashboard

**Files:**
- Create: `dashboard/streamlit_app.py`
- Create: `dashboard/i18n.py`

**Step 1: Create i18n module**

```python
# dashboard/i18n.py
"""Internationalization for dashboard (EN/RU)."""

from __future__ import annotations

TRANSLATIONS = {
    "en": {
        "title": "RAG Benchmark Dashboard",
        "tab_overview": "Overview",
        "tab_by_type": "By Query Type",
        "tab_by_doc": "By Document",
        "tab_run": "Run Benchmark",
        "tab_article": "Article",
        "accuracy": "Accuracy",
        "latency": "Avg Latency (s)",
        "confidence": "Avg Confidence",
        "adapter": "Adapter",
        "mode": "Mode",
        "run_button": "Run Benchmark",
        "generate_article": "Generate Article",
        "no_results": "No results yet. Run a benchmark first.",
        "select_run": "Select run",
        "services_health": "Service Health",
    },
    "ru": {
        "title": "RAG Benchmark Dashboard",
        "tab_overview": "Обзор",
        "tab_by_type": "По типам вопросов",
        "tab_by_doc": "По документам",
        "tab_run": "Запуск бенчмарка",
        "tab_article": "Статья",
        "accuracy": "Точность",
        "latency": "Ср. задержка (с)",
        "confidence": "Ср. уверенность",
        "adapter": "Адаптер",
        "mode": "Режим",
        "run_button": "Запустить бенчмарк",
        "generate_article": "Сгенерировать статью",
        "no_results": "Нет результатов. Сначала запустите бенчмарк.",
        "select_run": "Выберите запуск",
        "services_health": "Состояние сервисов",
    },
}


def get_translator(lang: str = "ru"):
    """Return translator function for given language."""
    strings = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lambda key: strings.get(key, key)
```

**Step 2: Create dashboard/streamlit_app.py**

Implement 5 tabs as described in design:
1. **Overview**: `st.dataframe` with conditional formatting (green/red), Plotly bar chart (best mode per adapter), ingestion stats table
2. **By Query Type**: Plotly grouped bar chart, Plotly heatmap (adapter × type → accuracy), `st.expander` per question with answers from all adapters
3. **By Document**: Two columns (EN/RU), Plotly radar chart per adapter, cross-language table if available
4. **Run Benchmark**: Checkboxes for adapters, radio for lang_mode, health status indicators, `st.progress` bar, `st.button("Run")` launching `run_benchmark`
5. **Article**: `st.button("Generate")`, `st.markdown(article_text)`, clipboard copy button

**Key patterns** (from existing projects):
- `@st.cache_resource` for expensive objects (OpenAI client, adapters)
- `st.sidebar` for language selector + run selector
- Load results from `results/*.json` files
- Use `plotly.express` and `plotly.graph_objects` for charts

**Step 3: Commit**

```bash
git add dashboard/streamlit_app.py dashboard/i18n.py
git commit -m "feat: Streamlit dashboard (5 tabs)"
```

---

### Task 17: Article Generator

**Files:**
- Create: `article/__init__.py`
- Create: `article/generator.py`
- Create: `article/templates/intro_ru.md`
- Create: `article/templates/intro_en.md`
- Create: `article/templates/methodology_ru.md`
- Create: `article/templates/methodology_en.md`
- Create: `tests/test_article.py`

**Step 1: Write tests**

```python
# tests/test_article.py
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
```

**Step 2: Implement article/generator.py**

```python
# article/generator.py
"""Auto-generate benchmark article from results."""

from __future__ import annotations

from pathlib import Path

from adapters.base import BenchmarkRun
from benchmark.compare import compute_metrics, compare_adapters, accuracy_by_type, accuracy_by_document

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ArticleGenerator:
    def __init__(self, run: BenchmarkRun, lang: str = "ru") -> None:
        self.run = run
        self.lang = lang

    def generate(self) -> str:
        sections = [
            self._load_template("intro"),
            self._load_template("methodology"),
            self._projects_overview(),
            self._results_summary(),
            self._results_by_type(),
            self._results_by_document(),
            self._discussion(),
            self._conclusion(),
        ]
        return "\n\n---\n\n".join(s for s in sections if s)

    def _load_template(self, name: str) -> str:
        path = TEMPLATES_DIR / f"{name}_{self.lang}.md"
        if path.exists():
            return path.read_text()
        # Fallback to English
        path_en = TEMPLATES_DIR / f"{name}_en.md"
        return path_en.read_text() if path_en.exists() else ""

    def _projects_overview(self) -> str:
        header = "## Обзор проектов" if self.lang == "ru" else "## Projects Overview"
        rows = []
        for adapter in self.run.adapters:
            adapter_results = [r for r in self.run.results if r.adapter == adapter]
            modes = sorted({r.mode for r in adapter_results})
            rows.append(f"| {adapter} | {', '.join(modes)} |")
        table = f"| Adapter | Modes |\n|---------|-------|\n" + "\n".join(rows)
        return f"{header}\n\n{table}"

    def _results_summary(self) -> str:
        header = "## Результаты" if self.lang == "ru" else "## Results"
        rows = compare_adapters(self.run.results)
        if not rows:
            return f"{header}\n\nНет данных." if self.lang == "ru" else f"{header}\n\nNo data."
        # Build markdown table
        cols = list(rows[0].keys())
        header_row = "| " + " | ".join(cols) + " |"
        sep_row = "|" + "|".join("---" for _ in cols) + "|"
        data_rows = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
        return f"{header}\n\n{header_row}\n{sep_row}\n{data_rows}"

    def _results_by_type(self) -> str:
        header = "### По типам вопросов" if self.lang == "ru" else "### By Query Type"
        breakdown = accuracy_by_type(self.run.results)
        if not breakdown:
            return ""
        lines = [header, ""]
        for qtype, adapters in sorted(breakdown.items()):
            lines.append(f"**{qtype}**: " + ", ".join(f"{a}={v:.0%}" for a, v in sorted(adapters.items())))
        return "\n".join(lines)

    def _results_by_document(self) -> str:
        header = "### По документам" if self.lang == "ru" else "### By Document"
        breakdown = accuracy_by_document(self.run.results)
        if not breakdown:
            return ""
        lines = [header, ""]
        for doc, adapters in sorted(breakdown.items()):
            lines.append(f"**{doc}**: " + ", ".join(f"{a}={v:.0%}" for a, v in sorted(adapters.items())))
        return "\n".join(lines)

    def _discussion(self) -> str:
        if self.lang == "ru":
            return "## Обсуждение\n\n*Автоматический анализ результатов будет добавлен в Stage C.*"
        return "## Discussion\n\n*Automated analysis will be added in Stage C.*"

    def _conclusion(self) -> str:
        if self.lang == "ru":
            return "## Выводы\n\n*Рекомендации по выбору RAG-системы будут добавлены в Stage C.*"
        return "## Conclusions\n\n*Recommendations for RAG system selection will be added in Stage C.*"
```

**Step 3: Create template files**

`article/templates/intro_ru.md`:
```markdown
# Cross-RAG Benchmark: сравнительный анализ 6 RAG-систем

В данном исследовании мы провели единый бенчмарк шести различных RAG-систем
на двух документах — англоязычном обзоре LLM и русскоязычном обзоре графовых
баз данных. Цель — объективное сравнение точности, скорости и мультиязычных
возможностей каждого подхода.
```

`article/templates/methodology_ru.md`:
```markdown
## Методология

**Документы**: 2 технических текста (EN + RU), 8-15 страниц каждый.

**Вопросы**: 60 bilingual вопросов (30 на документ), 5 типов:
simple, relation, multi_hop, global, temporal.

**Оценка**: 3-уровневый judge (embedding similarity → keyword overlap → LLM judge).

**Метрики**: accuracy, avg_confidence, avg_latency.
```

Create equivalent `_en.md` versions.

**Step 4: Run tests — expect ALL PASS**

**Step 5: Commit**

```bash
git add article/ tests/test_article.py
git commit -m "feat: article generator with templates"
```

---

## Phase 6: Integration

### Task 18: Register All Adapters + CLI Entry Point

**Files:**
- Create: `run_benchmark.py` (CLI entry point)
- Modify: `adapters/__init__.py` (register all adapters)

**Step 1: Create run_benchmark.py**

```python
#!/usr/bin/env python3
"""CLI entry point for rag-benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

from adapters.registry import AdapterRegistry
from adapters.pageindex_adapter import PageIndexAdapter
from adapters.rag2_adapter import RAG2Adapter
from adapters.agentic_graph_rag_adapter import AgenticGraphRAGAdapter
from adapters.rag_temporal_adapter import RAGTemporalAdapter
from adapters.temporal_kb_adapter import TemporalKBAdapter
from adapters.cog_rag_cognee_adapter import CogRAGCogneeAdapter
from benchmark.runner import load_questions, run_benchmark, make_benchmark_run, save_results


def main():
    parser = argparse.ArgumentParser(description="Cross-RAG Benchmark")
    parser.add_argument("--adapters", nargs="*", help="Adapter names to run (default: all available)")
    parser.add_argument("--lang-mode", choices=["native", "cross", "all"], default="native")
    parser.add_argument("--questions", type=Path, help="Path to questions.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Register adapters
    registry = AdapterRegistry()
    for cls in [PageIndexAdapter, RAG2Adapter, AgenticGraphRAGAdapter,
                RAGTemporalAdapter, TemporalKBAdapter, CogRAGCogneeAdapter]:
        try:
            adapter = cls()
            adapter.setup()
            registry.register(adapter)
        except Exception as e:
            logging.warning("Failed to initialize %s: %s", cls.__name__, e)

    # Filter adapters
    if args.adapters:
        adapters = [registry.get(n) for n in args.adapters if registry.get(n)]
    else:
        adapters = registry.list_available()

    if not adapters:
        print("No adapters available. Check services (Neo4j, Ollama).")
        sys.exit(1)

    print(f"Running with {len(adapters)} adapters: {[a.name for a in adapters]}")

    # Load questions
    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    # Documents
    data_dir = Path(__file__).parent / "data" / "documents"
    documents = {
        "en_llm_survey": str(data_dir / "en_llm_survey.txt"),
        "ru_graph_kb": str(data_dir / "ru_graph_kb.txt"),
    }

    # Ingest
    for adapter in adapters:
        print(f"\nIngesting documents into {adapter.name}...")
        for doc_name, doc_path in documents.items():
            result = adapter.ingest(doc_path)
            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"  {doc_name}: {result.chunks_count} chunks, {result.duration_seconds}s — {status}")

    # Run
    client = OpenAI()
    results = run_benchmark(
        adapters=adapters,
        questions=questions,
        openai_client=client,
        documents=documents,
        lang_mode=args.lang_mode,
        progress_callback=lambda c, t, m: print(f"\r  [{c}/{t}] {m}", end="", flush=True),
    )
    print()

    # Save
    run = make_benchmark_run(
        results=results,
        adapters=[a.name for a in adapters],
        documents=list(documents.keys()),
    )
    path = save_results(run)
    print(f"\nResults saved to {path}")

    # Summary
    from benchmark.compare import compare_adapters
    rows = compare_adapters(results)
    for r in rows:
        print(f"  {r['Adapter']:25s} {r['Mode']:20s} {r['Accuracy']:15s} {r['Avg Latency (s)']:.1f}s")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add run_benchmark.py
git commit -m "feat: CLI entry point"
```

---

### Task 19: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test with stub adapters**

```python
# tests/test_integration.py
"""Integration test — full pipeline with stub adapters."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from adapters.base import BaseAdapter, IngestResult, QueryResult, Question, BenchmarkRun
from adapters.registry import AdapterRegistry
from benchmark.runner import run_benchmark, make_benchmark_run
from benchmark.compare import compare_adapters, accuracy_by_type
from article.generator import ArticleGenerator


class StubAdapterA(BaseAdapter):
    name = "stub_a"
    project_path = "/tmp"
    modes = ["fast"]
    supported_langs = ["en", "ru"]
    requires_services = []

    def setup(self): pass
    def ingest(self, fp): return IngestResult(adapter="stub_a", document=fp, chunks_count=5, duration_seconds=0.1, success=True)
    def query(self, q, mode, lang): return QueryResult(adapter="stub_a", mode=mode, question_id=0, answer="GPT-3 was released in 2020 by OpenAI", confidence=0.9, latency=0.05)
    def cleanup(self): pass


class StubAdapterB(BaseAdapter):
    name = "stub_b"
    project_path = "/tmp"
    modes = ["slow"]
    supported_langs = ["en"]
    requires_services = []

    def setup(self): pass
    def ingest(self, fp): return IngestResult(adapter="stub_b", document=fp, chunks_count=3, duration_seconds=0.2, success=True)
    def query(self, q, mode, lang): return QueryResult(adapter="stub_b", mode=mode, question_id=0, answer="I don't know", confidence=0.1, latency=0.3)
    def cleanup(self): pass


@patch("benchmark.runner.evaluate_answer")
def test_full_pipeline(mock_eval):
    """End-to-end: adapters → runner → compare → article."""
    # Stub_a always passes, stub_b always fails
    def eval_side_effect(question, answer, keywords, client, reference_answer=""):
        return "GPT-3" in answer

    mock_eval.side_effect = eval_side_effect

    questions = [
        Question(id=1, document="en_llm_survey", question_en="What is GPT-3?", question_ru="Что такое GPT-3?", type="simple", keywords=["GPT-3", "2020"]),
        Question(id=2, document="en_llm_survey", question_en="List all models", question_ru="Перечисли все модели", type="global", keywords=["GPT", "BERT"]),
    ]

    results = run_benchmark(
        adapters=[StubAdapterA(), StubAdapterB()],
        questions=questions,
        openai_client=MagicMock(),
        documents={"en_llm_survey": "/tmp/en.txt"},
    )

    # Verify results
    assert len(results) > 0
    a_results = [r for r in results if r.adapter == "stub_a"]
    b_results = [r for r in results if r.adapter == "stub_b"]
    assert all(r.passed for r in a_results)
    assert all(not r.passed for r in b_results)

    # Compare
    rows = compare_adapters(results)
    assert len(rows) == 2

    # Article
    run = make_benchmark_run(results, ["stub_a", "stub_b"], ["en_llm_survey"])
    gen = ArticleGenerator(run, lang="ru")
    article = gen.generate()
    assert "stub_a" in article
    assert "%" in article
```

**Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: ALL PASS

**Step 3: Final commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration test — full pipeline"
```

---

## Summary

| Phase | Tasks | Files | ~LOC |
|-------|-------|-------|------|
| 1: Foundation | 1-4 | 10 | ~800 |
| 2: Core Pipeline | 5-6 | 4 | ~500 |
| 3: Documents & Questions | 7-9 | 5 | ~400 + docs |
| 4: Adapters | 10-15 | 12 | ~1,500 |
| 5: Dashboard & Article | 16-17 | 5 | ~1,000 |
| 6: Integration | 18-19 | 2 | ~300 |
| **Total** | **19** | **~38** | **~4,500** |

### Dependency Graph

```
Task 1 (scaffolding)
  ├→ Task 2 (models)
  │    ├→ Task 3 (evaluate)
  │    ├→ Task 4 (compare)
  │    └→ Task 5 (registry)
  │         └→ Task 6 (runner)
  │              ├→ Tasks 10-15 (adapters) — can be parallelized
  │              └→ Task 18 (CLI)
  ├→ Tasks 7-8 (documents) — independent
  └→ Task 9 (questions) — depends on 7-8
Task 4 + Task 6 → Task 16 (dashboard)
Task 4 + Task 6 → Task 17 (article)
All tasks → Task 19 (integration test)
```

### Critical Path

1 → 2 → 3 → 4 → 5 → 6 → 10 (first adapter) → 18 → 19

Tasks 7-8-9 (documents/questions) can run **in parallel** with tasks 3-6.
Tasks 11-15 (remaining adapters) can run **in parallel** after task 10.
Tasks 16-17 (dashboard/article) can run **in parallel** after task 6.
