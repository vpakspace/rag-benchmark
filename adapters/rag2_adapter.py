"""RAG 2.0 adapter — Neo4j vector + self-reflective + agentic RAG."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult

logger = logging.getLogger(__name__)

RAG2_PATH = Path.home() / "rag-2.0"


def _import_rag2() -> ModuleType:
    """Import rag-2.0 modules via sys.path bridge."""
    import importlib

    project = str(RAG2_PATH)
    if project not in sys.path:
        sys.path.insert(0, project)

    mod = ModuleType("rag2_bridge")
    mod.load_file = importlib.import_module("ingestion.loader").load_file
    mod.chunk_text = importlib.import_module("ingestion.chunker").chunk_text
    mod.enrich_chunks = importlib.import_module("ingestion.enricher").enrich_chunks
    mod.embed_chunks = importlib.import_module("ingestion.enricher").embed_chunks
    mod.VectorStore = importlib.import_module("storage.vector_store").VectorStore
    mod.Retriever = importlib.import_module("retrieval.retriever").Retriever
    mod.generate_answer = importlib.import_module("generation.generator").generate_answer
    mod.reflect_and_answer = importlib.import_module("generation.reflector").reflect_and_answer
    mod.RAGAgent = importlib.import_module("agent.rag_agent").RAGAgent
    return mod


class RAG2Adapter(BaseAdapter):
    name = "rag2"
    project_path = str(RAG2_PATH)
    modes = ["vector", "reflect", "agent"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "openai"]

    def __init__(self) -> None:
        self._mod = None
        self._store = None
        self._retriever = None

    def setup(self) -> None:
        self._mod = _import_rag2()
        self._store = self._mod.VectorStore()
        self._store.init_index()
        self._retriever = self._mod.Retriever(self._store)

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            text = self._mod.load_file(file_path)
            chunks = self._mod.chunk_text(text)
            chunks = self._mod.enrich_chunks(chunks)
            chunks = self._mod.embed_chunks(chunks)
            count = self._store.add_chunks(chunks)
            duration = round(time.monotonic() - start, 3)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=count, duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("RAG2 ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        start = time.monotonic()
        try:
            if mode == "vector":
                results = self._retriever.retrieve(question)
                qa = self._mod.generate_answer(question, results)
            elif mode == "reflect":
                qa = self._mod.reflect_and_answer(question, self._retriever)
            elif mode == "agent":
                agent = self._mod.RAGAgent(self._retriever, self._store)
                qa = agent.run(question)
            else:
                raise ValueError(f"Unknown mode: {mode}")

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
            logger.error("RAG2 query error (%s): %s", mode, e)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=f"Error: {e}", confidence=0.0, latency=latency,
            )

    def cleanup(self) -> None:
        if self._store:
            self._store.delete_all()

    def health_check(self) -> bool:
        return RAG2_PATH.exists()
