"""Cog-RAG-Cognee adapter — Cognee SDK semantic memory layer."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters._import_utils import prepare_imports

logger = logging.getLogger(__name__)

CRC_PATH = Path.home() / "cog-rag-cognee"

# Map benchmark mode names to Cognee SearchType values
_MODE_TO_SEARCH_TYPE = {
    "chunks": "CHUNKS",
    "graph": "GRAPH_COMPLETION",
    "rag_completion": "RAG_COMPLETION",
}


def _import_crc() -> ModuleType:
    """Import cog-rag-cognee modules with isolated sys.path."""
    import importlib

    prepare_imports(str(CRC_PATH))

    mod = ModuleType("crc_bridge")
    mod.get_settings = importlib.import_module("cog_rag_cognee.config").get_settings
    mod.apply_cognee_env = importlib.import_module("cog_rag_cognee.cognee_setup").apply_cognee_env
    return mod


def _import_crc_service() -> type:
    """Import PipelineService separately — after env vars are configured.

    Cognee SDK checks LLM_API_KEY at import time, so env vars must be
    set before importing modules that pull in Cognee.
    """
    import importlib
    return importlib.import_module("cog_rag_cognee.service").PipelineService


class CogRAGCogneeAdapter(BaseAdapter):
    name = "cog_rag_cognee"
    project_path = str(CRC_PATH)
    modes = ["chunks", "graph", "rag_completion"]
    supported_langs = ["en", "ru"]
    requires_services = ["neo4j", "ollama"]

    def __init__(self) -> None:
        self._mod = None
        self._service = None
        self._loop = None

    def _run(self, coro):
        """Run async coroutine on a persistent event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def setup(self) -> None:
        # Phase 1: import config and cognee_setup (lightweight, no Cognee SDK)
        self._mod = _import_crc()
        settings = self._mod.get_settings()

        # Phase 2: override Neo4j credentials to match benchmark container
        # CRC uses GRAPH_DATABASE_PASSWORD, not NEO4J_PASSWORD
        neo4j_pw = os.environ.get("NEO4J_PASSWORD", "temporal_kb_2026")
        os.environ["GRAPH_DATABASE_PASSWORD"] = neo4j_pw
        os.environ["GRAPH_DATABASE_URL"] = os.environ.get(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        os.environ["GRAPH_DATABASE_USERNAME"] = os.environ.get(
            "NEO4J_USER", "neo4j"
        )

        # Phase 3: apply Cognee env vars (LLM_API_KEY, etc.)
        self._mod.apply_cognee_env(settings)

        # Phase 4: now safe to import PipelineService (Cognee SDK env vars are set)
        PipelineService = _import_crc_service()
        self._service = PipelineService()

    def ingest(self, file_path: str) -> IngestResult:
        prepare_imports(str(CRC_PATH))
        start = time.monotonic()
        try:
            self._run(self._service.add_file(file_path))
            self._run(self._service.cognify())
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
        prepare_imports(str(CRC_PATH))
        start = time.monotonic()
        try:
            search_type = _MODE_TO_SEARCH_TYPE.get(mode, "CHUNKS")
            qa = self._run(self._service.query(question, search_type=search_type))
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
                self._run(self._service.reset())
            except Exception as e:
                logger.warning("CRC cleanup error: %s", e)
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    def health_check(self) -> bool:
        return CRC_PATH.exists()
