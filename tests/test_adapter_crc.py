"""Tests for Cog-RAG-Cognee adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from adapters.cog_rag_cognee_adapter import CogRAGCogneeAdapter


class TestCogRAGCogneeAdapter:
    def test_name_and_modes(self):
        adapter = CogRAGCogneeAdapter.__new__(CogRAGCogneeAdapter)
        assert adapter.name == "cog_rag_cognee"
        assert "chunks" in adapter.modes
        assert "graph" in adapter.modes
        assert "rag_completion" in adapter.modes
        assert "ollama" in adapter.requires_services

    @patch("adapters.cog_rag_cognee_adapter.prepare_imports")
    @patch("adapters.cog_rag_cognee_adapter._import_crc_service")
    @patch("adapters.cog_rag_cognee_adapter._import_crc")
    def test_ingest(self, mock_import, mock_import_svc, mock_prepare):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        mock_svc = MagicMock()
        mock_svc.add_file = AsyncMock(return_value={"status": "added", "chars": 500})
        mock_svc.cognify = AsyncMock(return_value={"status": "ok"})
        mock_import_svc.return_value = MagicMock(return_value=mock_svc)

        adapter = CogRAGCogneeAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "cog_rag_cognee"

    @patch("adapters.cog_rag_cognee_adapter.prepare_imports")
    @patch("adapters.cog_rag_cognee_adapter._import_crc_service")
    @patch("adapters.cog_rag_cognee_adapter._import_crc")
    def test_query_chunks(self, mock_import, mock_import_svc, mock_prepare):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        mock_svc = MagicMock()
        qa = MagicMock(answer="chunk answer", confidence=0.8, mode="CHUNKS")
        mock_svc.query = AsyncMock(return_value=qa)
        mock_import_svc.return_value = MagicMock(return_value=mock_svc)

        adapter = CogRAGCogneeAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "chunks", "en")

        assert result.answer == "chunk answer"

    @patch("adapters.cog_rag_cognee_adapter.prepare_imports")
    @patch("adapters.cog_rag_cognee_adapter._import_crc_service")
    @patch("adapters.cog_rag_cognee_adapter._import_crc")
    def test_query_rag_completion(self, mock_import, mock_import_svc, mock_prepare):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        mock_svc = MagicMock()
        qa = MagicMock(answer="rag answer", confidence=0.9, mode="RAG_COMPLETION")
        mock_svc.query = AsyncMock(return_value=qa)
        mock_import_svc.return_value = MagicMock(return_value=mock_svc)

        adapter = CogRAGCogneeAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "rag_completion", "en")

        assert result.answer == "rag answer"

    @patch("adapters.cog_rag_cognee_adapter.prepare_imports")
    @patch("adapters.cog_rag_cognee_adapter._import_crc_service")
    @patch("adapters.cog_rag_cognee_adapter._import_crc")
    def test_cleanup(self, mock_import, mock_import_svc, mock_prepare):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        mock_svc = MagicMock()
        mock_svc.reset = AsyncMock()
        mock_import_svc.return_value = MagicMock(return_value=mock_svc)

        adapter = CogRAGCogneeAdapter()
        adapter.setup()
        adapter.cleanup()
