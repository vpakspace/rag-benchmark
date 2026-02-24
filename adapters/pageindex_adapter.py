"""PageIndex adapter — tree-based vectorless RAG."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import ModuleType

from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters._import_utils import prepare_imports

logger = logging.getLogger(__name__)

PAGEINDEX_PATH = Path.home() / "pageindex"


def _import_pageindex() -> ModuleType:
    """Import pageindex modules with isolated sys.path."""
    import importlib

    prepare_imports(str(PAGEINDEX_PATH))

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
    requires_services = []

    def __init__(self) -> None:
        self._mod = None
        self._store = None

    def setup(self) -> None:
        self._mod = _import_pageindex()
        self._store = self._mod.TreeStore()

    def ingest(self, file_path: str) -> IngestResult:
        prepare_imports(str(PAGEINDEX_PATH))
        start = time.monotonic()
        try:
            text = self._mod.load_file(file_path)
            tree = self._mod.build_tree(text, filename=Path(file_path).name)
            tree = self._mod.summarize_tree(tree)
            doc_id = self._store.save(tree)
            duration = round(time.monotonic() - start, 3)
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
        prepare_imports(str(PAGEINDEX_PATH))
        trees = self._store.list_trees()
        if not trees:
            return QueryResult(
                adapter=self.name, mode=mode, question_id=0,
                answer="No documents indexed", confidence=0.0, latency=0.0,
            )
        tree = self._store.load(doc_id=trees[0]["doc_id"])
        start = time.monotonic()
        result = self._mod.reason_and_answer(question, tree, self._store)
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
