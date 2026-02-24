"""Temporal Knowledge Base adapter — Graphiti + bi-temporal knowledge graph."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult

logger = logging.getLogger(__name__)

TKB_PATH = Path.home() / "temporal-knowledge-base"


def _run_async(coro):
    """Run async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _import_tkb() -> ModuleType:
    """Import temporal-knowledge-base modules via sys.path bridge."""
    import importlib

    project = str(TKB_PATH)
    if project not in sys.path:
        sys.path.insert(0, project)

    mod = ModuleType("tkb_bridge")
    mod.IngestionPipeline = importlib.import_module("ingestion.pipeline").IngestionPipeline
    mod.QueryEngine = importlib.import_module("retrieval.query_engine").QueryEngine
    mod.ResponseBuilder = importlib.import_module("generation.response_builder").ResponseBuilder
    mod.get_settings = importlib.import_module("core.config").get_settings
    mod.SearchQuery = importlib.import_module("core.models").SearchQuery
    mod.IntentType = importlib.import_module("core.models").IntentType
    return mod


# Map benchmark modes to TKB IntentType values
_MODE_TO_INTENT = {
    "structural": "STRUCTURAL",
    "temporal": "TEMPORAL",
    "hybrid": "HYBRID",
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

    def setup(self) -> None:
        self._mod = _import_tkb()
        # Components are initialized but connections deferred to first use
        self._pipeline = self._mod.IngestionPipeline()
        self._query_engine = self._mod.QueryEngine()
        self._response_builder = self._mod.ResponseBuilder()

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            result = _run_async(
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
        start = time.monotonic()
        try:
            intent = _MODE_TO_INTENT.get(mode, "HYBRID")
            search_query = self._mod.SearchQuery(
                query=question,
                intent=self._mod.IntentType(intent),
            )
            search_response = _run_async(self._query_engine.search(search_query))
            response = _run_async(
                self._response_builder.build_response(question, search_response)
            )
            latency = round(time.monotonic() - start, 3)
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=answer,
                confidence=0.7,  # TKB doesn't return confidence directly
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
        # TKB uses Graphiti — no simple clear method
        pass

    def health_check(self) -> bool:
        return TKB_PATH.exists()
