"""Tests for Temporal Knowledge Base adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.temporal_kb_adapter import TemporalKBAdapter


class TestTemporalKBAdapter:
    def test_name_and_modes(self):
        adapter = TemporalKBAdapter.__new__(TemporalKBAdapter)
        assert adapter.name == "temporal_kb"
        assert set(adapter.modes) == {"structural", "temporal", "hybrid"}
        assert "neo4j" in adapter.requires_services

    @patch("adapters.temporal_kb_adapter.prepare_imports")
    @patch("adapters.temporal_kb_adapter.Path.read_text", return_value="test content")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_ingest(self, mock_import, mock_read, mock_prepare):
        mock_mod = MagicMock()
        mock_settings = MagicMock()
        mock_mod.get_settings.return_value = mock_settings
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        # Mock the persistent loop's run_until_complete
        adapter._loop = MagicMock()
        adapter._loop.is_closed.return_value = False
        adapter._loop.run_until_complete.return_value = None

        adapter._mod = mock_mod
        # Simulate setup components
        adapter._pipeline = MagicMock()
        adapter._query_engine = MagicMock()
        adapter._response_builder = MagicMock()

        # Set up ingest result
        adapter._loop.run_until_complete.return_value = {"chunks": 5}
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "temporal_kb"

    @patch("adapters.temporal_kb_adapter.prepare_imports")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_query_structural(self, mock_import, mock_prepare):
        mock_mod = MagicMock()
        mock_settings = MagicMock()
        mock_mod.get_settings.return_value = mock_settings
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter._mod = mock_mod
        adapter._loop = MagicMock()
        adapter._loop.is_closed.return_value = False
        # Set up query responses: first call = search, second = build_response
        adapter._loop.run_until_complete.side_effect = [
            MagicMock(results=[MagicMock()]),  # search response
            {"answer": "structural answer", "sources": []},  # build_response
        ]
        adapter._pipeline = MagicMock()
        adapter._query_engine = MagicMock()
        adapter._response_builder = MagicMock()

        result = adapter.query("What is X?", "structural", "en")

        assert result.answer == "structural answer"

    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.get_settings.return_value = MagicMock()
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        adapter._loop = mock_loop

        adapter.cleanup()

        mock_loop.close.assert_called_once()
        assert adapter._loop is None
