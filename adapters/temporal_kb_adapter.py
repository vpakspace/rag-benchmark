"""Temporal Knowledge Base adapter — Graphiti + bi-temporal knowledge graph."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters._import_utils import prepare_imports

logger = logging.getLogger(__name__)

TKB_PATH = Path.home() / "temporal-knowledge-base"


def _import_tkb() -> ModuleType:
    """Import temporal-knowledge-base modules with isolated sys.path."""
    import importlib

    prepare_imports(str(TKB_PATH))

    mod = ModuleType("tkb_bridge")
    # Core
    mod.get_settings = importlib.import_module("core.config").get_settings
    mod.SearchQuery = importlib.import_module("core.models").SearchQuery
    mod.IntentType = importlib.import_module("core.models").IntentType
    # Components for building the pipeline
    mod.Neo4jClient = importlib.import_module("storage.neo4j_client").Neo4jClient
    mod.VectorStore = importlib.import_module("storage.vector_store").VectorStore
    mod.LLMClient = importlib.import_module("generation.llm_client").LLMClient
    mod.TemporalVerifier = importlib.import_module("generation.temporal_verifier").TemporalVerifier
    mod.ResponseBuilder = importlib.import_module("generation.response_builder").ResponseBuilder
    mod.GraphitiClient = importlib.import_module("graphiti_adapter.client").GraphitiClient
    mod.EntityResolver = importlib.import_module("temporal.resolution").EntityResolver
    mod.InvalidationAgent = importlib.import_module("temporal.invalidation_agent").InvalidationAgent
    mod.IngestionPipeline = importlib.import_module("ingestion.pipeline").IngestionPipeline
    mod.QueryEngine = importlib.import_module("retrieval.query_engine").QueryEngine
    return mod


# Map benchmark modes to TKB IntentType values (lowercase — matches enum)
_MODE_TO_INTENT = {
    "structural": "structural",
    "temporal": "temporal",
    "hybrid": "hybrid",
}


class TemporalKBAdapter(BaseAdapter):
    name = "temporal_kb"
    project_path = str(TKB_PATH)
    modes = ["structural", "temporal", "hybrid"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "openai"]

    def __init__(self) -> None:
        self._mod = None
        self._pipeline = None
        self._query_engine = None
        self._response_builder = None
        self._loop = None

    def _run(self, coro):
        """Run async coroutine on a persistent event loop.

        Using a single loop prevents 'Future attached to a different loop'
        errors with Neo4j AsyncDriver.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def setup(self) -> None:
        self._mod = _import_tkb()
        settings = self._mod.get_settings()

        # Build component graph (same order as api/server.py)
        neo4j = self._mod.Neo4jClient(settings.neo4j)
        self._run(neo4j.connect())

        vector_store = self._mod.VectorStore(settings.openai)
        llm = self._mod.LLMClient(settings.openai)
        graphiti = self._mod.GraphitiClient(settings)
        self._run(graphiti.connect())

        entity_resolver = self._mod.EntityResolver(neo4j, vector_store)
        invalidation_agent = self._mod.InvalidationAgent(
            neo4j, vector_store, llm, settings,
        )
        verifier = self._mod.TemporalVerifier(neo4j)

        self._pipeline = self._mod.IngestionPipeline(
            graphiti=graphiti, neo4j=neo4j, vector_store=vector_store,
            llm=llm, invalidation_agent=invalidation_agent,
            entity_resolver=entity_resolver, settings=settings,
        )
        self._query_engine = self._mod.QueryEngine(graphiti, neo4j, llm)
        self._response_builder = self._mod.ResponseBuilder(llm, verifier)

    def ingest(self, file_path: str) -> IngestResult:
        prepare_imports(str(TKB_PATH))
        start = time.monotonic()
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            result = self._run(
                self._pipeline.ingest_episode(
                    content=content,
                    source=Path(file_path).name,
                )
            )
            duration = round(time.monotonic() - start, 3)
            chunks = result.get("chunks", 0) if isinstance(result, dict) else 0
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=chunks, duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("TKB ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        prepare_imports(str(TKB_PATH))
        start = time.monotonic()
        try:
            intent = _MODE_TO_INTENT.get(mode, "HYBRID")
            search_query = self._mod.SearchQuery(
                query=question,
                intent=self._mod.IntentType(intent),
            )
            search_response = self._run(self._query_engine.search(search_query))
            response = self._run(
                self._response_builder.build_response(question, search_response)
            )
            latency = round(time.monotonic() - start, 3)
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=answer,
                confidence=0.7,
                latency=latency,
            )
        except Exception as e:
            latency = round(time.monotonic() - start, 3)
            logger.error("TKB query error (%s): %s", mode, e)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=f"Error: {e}", confidence=0.0, latency=latency,
            )

    def cleanup(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    def health_check(self) -> bool:
        return TKB_PATH.exists()
