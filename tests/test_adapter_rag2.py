"""Tests for RAG 2.0 adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.rag2_adapter import RAG2Adapter


class TestRAG2Adapter:
    def test_name_and_modes(self):
        adapter = RAG2Adapter.__new__(RAG2Adapter)
        assert adapter.name == "rag2"
        assert set(adapter.modes) == {"vector", "reflect", "agent"}
        assert "neo4j" in adapter.requires_services
        assert "openai" in adapter.requires_services

    @patch("adapters.rag2_adapter._import_rag2")
    def test_ingest(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.load_file.return_value = "text content"
        mock_mod.chunk_text.return_value = [MagicMock(), MagicMock()]
        mock_mod.enrich_chunks.return_value = [MagicMock(), MagicMock()]
        mock_mod.embed_chunks.return_value = [MagicMock(), MagicMock()]
        mock_mod.VectorStore.return_value.add_chunks.return_value = 2
        mock_import.return_value = mock_mod

        adapter = RAG2Adapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "rag2"
        assert result.chunks_count == 2

    @patch("adapters.rag2_adapter._import_rag2")
    def test_query_vector(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="answer A", confidence=0.85, retries=0)
        mock_mod.generate_answer.return_value = qa
        mock_mod.Retriever.return_value.retrieve.return_value = [MagicMock()]
        mock_import.return_value = mock_mod

        adapter = RAG2Adapter()
        adapter.setup()
        result = adapter.query("What is X?", "vector", "en")

        assert result.answer == "answer A"
        assert result.confidence == 0.85

    @patch("adapters.rag2_adapter._import_rag2")
    def test_query_reflect(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="reflected", confidence=0.9, retries=1)
        mock_mod.reflect_and_answer.return_value = qa
        mock_import.return_value = mock_mod

        adapter = RAG2Adapter()
        adapter.setup()
        result = adapter.query("What is X?", "reflect", "en")

        assert result.answer == "reflected"

    @patch("adapters.rag2_adapter._import_rag2")
    def test_query_agent(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="agent answer", confidence=0.95, retries=0)
        mock_agent_module = MagicMock()
        mock_agent_module.RAGAgent.return_value.run.return_value = qa
        mock_import.return_value = mock_mod

        adapter = RAG2Adapter()
        adapter.setup()
        # Mock importlib.import_module used in agent mode reimport
        with patch("importlib.import_module", return_value=mock_agent_module):
            result = adapter.query("What is X?", "agent", "en")

        assert result.answer == "agent answer"

    @patch("adapters.rag2_adapter._import_rag2")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.VectorStore.return_value.delete_all.return_value = 10
        mock_import.return_value = mock_mod

        adapter = RAG2Adapter()
        adapter.setup()
        adapter.cleanup()
        mock_mod.VectorStore.return_value.delete_all.assert_called_once()
