"""Tests for Agentic Graph RAG adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.agentic_graph_rag_adapter import AgenticGraphRAGAdapter


class TestAgenticGraphRAGAdapter:
    def test_name_and_modes(self):
        adapter = AgenticGraphRAGAdapter.__new__(AgenticGraphRAGAdapter)
        assert adapter.name == "agentic_graph_rag"
        assert "agent_pattern" in adapter.modes
        assert "agent_llm" in adapter.modes
        assert "neo4j" in adapter.requires_services

    @patch("adapters.agentic_graph_rag_adapter._import_agr")
    def test_ingest(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.load_file.return_value = "text"
        mock_mod.chunk_text.return_value = [MagicMock()]
        mock_mod.enrich_chunks.return_value = [MagicMock()]
        mock_mod.embed_chunks.return_value = [MagicMock()]
        mock_mod.create_passage_nodes.return_value = 1
        mock_mod.create_phrase_nodes.return_value = 2
        mock_import.return_value = mock_mod

        adapter = AgenticGraphRAGAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "agentic_graph_rag"

    @patch("adapters.agentic_graph_rag_adapter._import_agr")
    def test_query_agent_pattern(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="pattern answer", confidence=0.8, retries=0)
        mock_mod.PipelineService.return_value.query.return_value = qa
        mock_import.return_value = mock_mod

        adapter = AgenticGraphRAGAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "agent_pattern", "en")

        assert result.answer == "pattern answer"

    @patch("adapters.agentic_graph_rag_adapter._import_agr")
    def test_query_agent_llm(self, mock_import):
        mock_mod = MagicMock()
        qa = MagicMock(answer="llm answer", confidence=0.9, retries=0)
        mock_mod.PipelineService.return_value.query.return_value = qa
        mock_import.return_value = mock_mod

        adapter = AgenticGraphRAGAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "agent_llm", "en")

        assert result.answer == "llm answer"

    @patch("adapters.agentic_graph_rag_adapter._import_agr")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        adapter = AgenticGraphRAGAdapter()
        adapter.setup()
        adapter.cleanup()
