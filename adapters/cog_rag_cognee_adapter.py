"""Cog-RAG-Cognee adapter — Cognee SDK semantic memory layer."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult

logger = logging.getLogger(__name__)

CRC_PATH = Path.home() / "cog-rag-cognee"

# Map benchmark mode names to Cognee SearchType values
_MODE_TO_SEARCH_TYPE = {
    "chunks": "CHUNKS",
    "graph": "GRAPH_COMPLETION",
    "rag_completion": "RAG_COMPLETION",
}


def _run_async(coro):
    """Run async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _import_crc() -> ModuleType:
    """Import cog-rag-cognee modules via sys.path bridge."""
    import importlib

    project = str(CRC_PATH)
    if project not in sys.path:
        sys.path.insert(0, project)

    mod = ModuleType("crc_bridge")
    mod.PipelineService = importlib.import_module("cog_rag_cognee.service").PipelineService
    mod.get_settings = importlib.import_module("cog_rag_cognee.config").get_settings
    mod.apply_cognee_env = importlib.import_module("cog_rag_cognee.cognee_setup").apply_cognee_env
    return mod


class CogRAGCogneeAdapter(BaseAdapter):
    name = "cog_rag_cognee"
    project_path = str(CRC_PATH)
    modes = ["chunks", "graph", "rag_completion"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "ollama"]

    def __init__(self) -> None:
        self._mod = None
        self._service = None

    def setup(self) -> None:
        self._mod = _import_crc()
        settings = self._mod.get_settings()
        self._mod.apply_cognee_env(settings)
        self._service = self._mod.PipelineService()

    def ingest(self, file_path: str) -> IngestResult:
        start = time.monotonic()
        try:
            _run_async(self._service.add_file(file_path))
            _run_async(self._service.cognify())
            duration = round(time.monotonic() - start, 3)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=1,  # Cognee doesn't return chunk count directly
                duration_seconds=duration, success=True,
            )
        except Exception as e:
            duration = round(time.monotonic() - start, 3)
            logger.error("CRC ingest error: %s", e)
            return IngestResult(
                adapter=self.name, document=Path(file_path).name,
                chunks_count=0, duration_seconds=duration, success=False, error=str(e),
            )

    def query(self, question: str, mode: str, lang: str) -> QueryResult:
        start = time.monotonic()
        try:
            search_type = _MODE_TO_SEARCH_TYPE.get(mode, "CHUNKS")
            qa = _run_async(self._service.query(question, search_type=search_type))
            latency = round(time.monotonic() - start, 3)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=qa.answer,
                confidence=getattr(qa, "confidence", 0.5),
                latency=latency,
            )
        except Exception as e:
            latency = round(time.monotonic() - start, 3)
            logger.error("CRC query error (%s): %s", mode, e)
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer=f"Error: {e}", confidence=0.0, latency=latency,
            )

    def cleanup(self) -> None:
        if self._service:
            try:
                _run_async(self._service.reset())
            except Exception as e:
                logger.warning("CRC cleanup error: %s", e)

    def health_check(self) -> bool:
        return CRC_PATH.exists()
