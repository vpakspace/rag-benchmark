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

    @patch("adapters.temporal_kb_adapter.Path.read_text", return_value="test content")
    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_ingest(self, mock_import, mock_read):
        mock_mod = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_episode = AsyncMock(return_value={
            "chunks": 5, "entities_extracted": 3
        })
        mock_mod.IngestionPipeline.return_value = mock_pipeline
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "temporal_kb"

    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_query_structural(self, mock_import):
        mock_mod = MagicMock()
        mock_qe = MagicMock()
        search_response = MagicMock(results=[MagicMock()])
        mock_qe.search = AsyncMock(return_value=search_response)
        mock_mod.QueryEngine.return_value = mock_qe

        mock_rb = MagicMock()
        mock_rb.build_response = AsyncMock(return_value={
            "answer": "structural answer", "facts_used": 2, "sources": []
        })
        mock_mod.ResponseBuilder.return_value = mock_rb
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "structural", "en")

        assert result.answer == "structural answer"

    @patch("adapters.temporal_kb_adapter._import_tkb")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        adapter = TemporalKBAdapter()
        adapter.setup()
        adapter.cleanup()
