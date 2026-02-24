"""RAG Temporal adapter — unified RAG 2.0 + Temporal Knowledge Base."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters._import_utils import prepare_imports

logger = logging.getLogger(__name__)

RT_PATH = Path.home() / "rag-temporal"
RAG2_PATH = Path.home() / "rag-2.0"


def _import_rt() -> ModuleType:
    """Import rag-temporal modules with isolated sys.path.

    rag-temporal uses bridge modules to re-export from rag-2.0,
    so both project paths must be available.
    """
    import importlib

    prepare_imports(str(RT_PATH), extra_paths=[str(RAG2_PATH)])

    mod = ModuleType("rt_bridge")
    mod.DualPipeline = importlib.import_module("ingestion.dual_pipeline").DualPipeline
    mod.HybridRetriever = importlib.import_module("retrieval.hybrid_retriever").HybridRetriever
    mod.HybridAgent = importlib.import_module("agent.hybrid_agent").HybridAgent
    mod.generate_answer = importlib.import_module("generation.generator").generate_answer
    return mod


class RAGTemporalAdapter(BaseAdapter):
    name = "rag_temporal"
    project_path = str(RT_PATH)
    modes = ["vector", "kg", "hybrid", "agent"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "openai"]

    def __init__(self) -> None:
        self._mod = None
        self._pipeline = None
        self._retriever = None

    def setup(self) -> None:
        self._mod = _import_rt()
        self._pipeline = self._mod.DualPipeline()
        self._retriever = self._mod.HybridRetriever(
            kg_client=self._pipeline.kg_client,
            vector_store=self._pipeline.vector_store,
        )

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            result = self._pipeline.ingest_file(file_path)
            duration = round(time.monotonic() - start, 3)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=result.rag_chunks,
                duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("RAG Temporal ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        start = time.monotonic()
        try:
            if mode == "agent":
                agent = self._mod.HybridAgent(self._retriever)
                qa = agent.run(question)
                latency = round(time.monotonic() - start, 3)
                return QueryResult(
                    adapter=self.name, mode=mode, question_id=0,
                    answer=qa.answer,
                    confidence=getattr(qa, "confidence", 0.5),
                    latency=latency,
                )
            else:
                results = self._retriever.retrieve(question, mode=mode)
                qa = self._mod.generate_answer(question, results)
                latency = round(time.monotonic() - start, 3)
                return QueryResult(
                    adapter=self.name, mode=mode, question_id=0,
                    answer=qa.answer,
                    confidence=getattr(qa, "confidence", 0.5),
                    latency=latency,
                )
        except Exception as e:
            latency = round(time.monotonic() - start, 3)
            logger.error("RAG Temporal query error (%s): %s", mode, e)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=f"Error: {e}", confidence=0.0, latency=latency,
            )

    def cleanup(self) -> None:
        if self._pipeline:
            store = getattr(self._pipeline, "vector_store", None)
            if store and hasattr(store, "delete_all"):
                store.delete_all()

    def health_check(self) -> bool:
        return RT_PATH.exists()
