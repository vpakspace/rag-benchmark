"""Tests for Temporal Knowledge Base adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from adapters.temporal_kb_adapter import TemporalKBAdapter


class TestTemporalKBAdapter:
    def test_name_and_modes(self):
        adapter = TemporalKBAdapter.__new__(TemporalKBAdapter)
        assert adapter.name == "temporal_kb"
        assert set(adapter.modes) == {"structural", "temporal", "hybrid"}
        assert "neo4j" in adapter.requires_services

    @patch("adapters.temporal_kb_adapter._run_async")
    @patch("adapters.temporal_kb_adapter.Path.read_text", return_value="test content")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_ingest(self, mock_import, mock_read, mock_async):
        mock_mod = MagicMock()
        # get_settings returns mock settings with nested neo4j/openai
        mock_settings = MagicMock()
        mock_mod.get_settings.return_value = mock_settings
        # Neo4jClient.connect() and GraphitiClient.connect() are async
        mock_async.return_value = None
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()

        # Now set up ingest result
        mock_async.return_value = {"chunks": 5}
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "temporal_kb"

    @patch("adapters.temporal_kb_adapter._run_async")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_query_structural(self, mock_import, mock_async):
        mock_mod = MagicMock()
        mock_settings = MagicMock()
        mock_mod.get_settings.return_value = mock_settings
        mock_async.return_value = None
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()

        # Set up query responses: first call = search, second = build_response
        mock_async.side_effect = [
            MagicMock(results=[MagicMock()]),  # search response
            {"answer": "structural answer", "sources": []},  # build_response
        ]
        result = adapter.query("What is X?", "structural", "en")

        assert result.answer == "structural answer"

    @patch("adapters.temporal_kb_adapter._run_async")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_cleanup(self, mock_import, mock_async):
        mock_mod = MagicMock()
        mock_mod.get_settings.return_value = MagicMock()
        mock_async.return_value = None
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()
        adapter.cleanup()
