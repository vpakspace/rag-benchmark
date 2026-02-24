"""Tests for RAG Temporal adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.rag_temporal_adapter import RAGTemporalAdapter


class TestRAGTemporalAdapter:
    def test_name_and_modes(self):
        adapter = RAGTemporalAdapter.__new__(RAGTemporalAdapter)
        assert adapter.name == "rag_temporal"
        assert set(adapter.modes) == {"vector", "kg", "hybrid", "agent"}
        assert "neo4j" in adapter.requires_services

    @patch("adapters.rag_temporal_adapter._import_rt")
    def test_ingest(self, mock_import):
        mock_mod = MagicMock()
        ingest_result = MagicMock(rag_chunks=5, kg_episodes=3, errors=[])
        mock_mod.DualPipeline.return_value.ingest_file.return_value = ingest_result
        mock_import.return_value = mock_mod

        adapter = RAGTemporalAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.chunks_count == 5

    @patch("adapters.rag_temporal_adapter._import_rt")
    def test_query_hybrid(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.HybridRetriever.return_value.retrieve.return_value = [
            MagicMock(content="fact", score=0.9)
        ]
        mock_mod.generate_answer.return_value = MagicMock(
            answer="hybrid answer", confidence=0.85, retries=0
        )
        mock_import.return_value = mock_mod

        adapter = RAGTemporalAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "hybrid", "en")

        assert result.answer == "hybrid answer"

    @patch("adapters.rag_temporal_adapter._import_rt")
    def test_query_agent(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="agent answer", confidence=0.9, retries=0)
        mock_mod.HybridAgent.return_value.run.return_value = qa
        mock_import.return_value = mock_mod

        adapter = RAGTemporalAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "agent", "en")

        assert result.answer == "agent answer"

    @patch("adapters.rag_temporal_adapter._import_rt")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        adapter = RAGTemporalAdapter()
        adapter.setup()
        adapter.cleanup()
