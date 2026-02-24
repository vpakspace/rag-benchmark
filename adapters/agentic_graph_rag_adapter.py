"""Agentic Graph RAG adapter — dual-node graph with agent routing."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import ModuleType

from neo4j import GraphDatabase

from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters._import_utils import prepare_imports

logger = logging.getLogger(__name__)

AGR_PATH = Path.home() / "agentic-graph-rag"
AGR_RAG_CORE = AGR_PATH / "packages" / "rag-core"


def _import_agr() -> ModuleType:
    """Import agentic-graph-rag modules with isolated sys.path."""
    import importlib

    prepare_imports(str(AGR_PATH), extra_paths=[str(AGR_RAG_CORE)])

    mod = ModuleType("agr_bridge")
    # rag_core modules for ingestion
    mod.load_file = importlib.import_module("rag_core.loader").load_file
    mod.chunk_text = importlib.import_module("rag_core.chunker").chunk_text
    mod.enrich_chunks = importlib.import_module("rag_core.enricher").enrich_chunks
    mod.embed_chunks = importlib.import_module("rag_core.embedder").embed_chunks
    # Config and client factories
    mod.get_settings = importlib.import_module("rag_core.config").get_settings
    mod.make_openai_client = importlib.import_module("rag_core.config").make_openai_client
    # Dual-node indexing
    dual = importlib.import_module("agentic_graph_rag.indexing.dual_node")
    mod.create_passage_nodes = dual.create_passage_nodes
    mod.create_phrase_nodes = dual.create_phrase_nodes
    # Entity extraction (for phrase nodes)
    skeleton = importlib.import_module("agentic_graph_rag.indexing.skeleton")
    mod.extract_entities_full = skeleton.extract_entities_full
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
        self._driver = None

    def setup(self) -> None:
        self._mod = _import_agr()
        cfg = self._mod.get_settings()
        self._driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        client = self._mod.make_openai_client(cfg)
        self._service = self._mod.PipelineService(self._driver, client)

    def ingest(self, file_path: str) -> IngestResult:
        prepare_imports(str(AGR_PATH), extra_paths=[str(AGR_RAG_CORE)])
        start = time.monotonic()
        try:
            text = self._mod.load_file(file_path)
            chunks = self._mod.chunk_text(text)
            chunks = self._mod.enrich_chunks(chunks)
            chunks = self._mod.embed_chunks(chunks)
            # Create passage nodes (vector index) in Neo4j
            passage_nodes = self._mod.create_passage_nodes(chunks, self._driver)
            passage_count = len(passage_nodes) if passage_nodes else 0
            # Extract entities and create phrase nodes (knowledge graph)
            phrase_count = 0
            try:
                entities, _rels = self._mod.extract_entities_full(chunks)
                if entities:
                    phrase_nodes = self._mod.create_phrase_nodes(
                        entities, self._driver,
                    )
                    phrase_count = len(phrase_nodes) if phrase_nodes else 0
            except Exception as e:
                logger.warning("AGR phrase node creation skipped: %s", e)
            total = passage_count + phrase_count
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
        prepare_imports(str(AGR_PATH), extra_paths=[str(AGR_RAG_CORE)])
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
        if self._driver:
            self._driver.close()

    def health_check(self) -> bool:
        return AGR_PATH.exists()
