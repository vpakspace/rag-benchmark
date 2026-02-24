"""Base adapter interface and data models for rag-benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

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
