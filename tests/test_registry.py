"""Tests for adapter discovery and registry."""

from __future__ import annotations

import pytest
from adapters.base import BaseAdapter, IngestResult, QueryResult
from adapters.registry import AdapterRegistry


class DummyAdapter(BaseAdapter):
    name = "dummy"
    project_path = "/tmp/dummy"
    modes = ["mode_a", "mode_b"]
    supported_langs = ["en", "ru"]
    requires_services = []

    def setup(self): pass
    def ingest(self, file_path): return IngestResult(adapter="dummy", document=file_path, chunks_count=5, duration_seconds=0.1, success=True)
    def query(self, question, mode, lang): return QueryResult(adapter="dummy", mode=mode, question_id=0, answer="dummy", confidence=1.0, latency=0.01)
    def cleanup(self): pass


class TestAdapterRegistry:
    def test_register_and_get(self):
        reg = AdapterRegistry()
        adapter = DummyAdapter()
        reg.register(adapter)
        assert reg.get("dummy") is adapter

    def test_get_unknown_returns_none(self):
        reg = AdapterRegistry()
        assert reg.get("unknown") is None

    def test_list_all(self):
        reg = AdapterRegistry()
        reg.register(DummyAdapter())
        names = reg.list_names()
        assert "dummy" in names

    def test_list_available_checks_health(self):
        reg = AdapterRegistry()
        reg.register(DummyAdapter())
        available = reg.list_available()
        assert len(available) == 1

    def test_register_duplicate_overwrites(self):
        reg = AdapterRegistry()
        a1 = DummyAdapter()
        a2 = DummyAdapter()
        reg.register(a1)
        reg.register(a2)
        assert reg.get("dummy") is a2
