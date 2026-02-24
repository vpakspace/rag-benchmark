"""Agentic Graph RAG adapter — dual-node graph with agent routing."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult

logger = logging.getLogger(__name__)

AGR_PATH = Path.home() / "agentic-graph-rag"


def _import_agr() -> ModuleType:
    """Import agentic-graph-rag modules via sys.path bridge."""
    import importlib

    project = str(AGR_PATH)
    if project not in sys.path:
        sys.path.insert(0, project)

    mod = ModuleType("agr_bridge")
    # rag_core modules for ingestion
    mod.load_file = importlib.import_module("rag_core.loader").load_file
    mod.chunk_text = importlib.import_module("rag_core.chunker").chunk_text
    mod.enrich_chunks = importlib.import_module("rag_core.enricher").enrich_chunks
    mod.embed_chunks = importlib.import_module("rag_core.embedder").embed_chunks
    # Dual-node indexing
    dual = importlib.import_module("agentic_graph_rag.indexing.dual_node")
    mod.create_passage_nodes = dual.create_passage_nodes
    mod.create_phrase_nodes = dual.create_phrase_nodes
    # Service
    mod.PipelineService = importlib.import_module("agentic_graph_rag.service").PipelineService
    return mod


class AgenticGraphRAGAdapter(BaseAdapter):
    name = "agentic_graph_rag"
    project_path = str(AGR_PATH)
    modes = ["agent_pattern", "agent_llm"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "openai"]

    def __init__(self) -> None:
        self._mod = None
        self._service = None

    def setup(self) -> None:
        self._mod = _import_agr()
        self._service = self._mod.PipelineService()

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            text = self._mod.load_file(file_path)
            chunks = self._mod.chunk_text(text)
            chunks = self._mod.enrich_chunks(chunks)
            chunks = self._mod.embed_chunks(chunks)
            passage_count = self._mod.create_passage_nodes(chunks)
            phrase_count = self._mod.create_phrase_nodes(chunks)
            total = (passage_count or 0) + (phrase_count or 0)
            duration = round(time.monotonic() - start, 3)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=total, duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("AGR ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        start = time.monotonic()
        try:
            qa = self._service.query(question, mode=mode)
            latency = round(time.monotonic() - start, 3)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=qa.answer,
                confidence=getattr(qa, "confidence", 0.5),
                latency=latency,
                retries=getattr(qa, "retries", 0),
            )
        except Exception as e:
            latency = round(time.monotonic() - start, 3)
            logger.error("AGR query error (%s): %s", mode, e)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=f"Error: {e}", confidence=0.0, latency=latency,
            )

    def cleanup(self) -> None:
        if self._service and hasattr(self._service, "clear"):
            self._service.clear()

    def health_check(self) -> bool:
        return AGR_PATH.exists()
