"""Tests for PageIndex adapter."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from adapters.pageindex_adapter import PageIndexAdapter


class TestPageIndexAdapter:
    def test_name_and_modes(self):
        adapter = PageIndexAdapter.__new__(PageIndexAdapter)
        assert adapter.name == "pageindex"
        assert adapter.modes == ["tree_reasoning"]
        assert "neo4j" not in adapter.requires_services

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_ingest(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.load_file.return_value = "# Markdown content"
        mock_mod.build_tree.return_value = MagicMock()
        mock_mod.summarize_tree.return_value = MagicMock()
        mock_mod.TreeStore.return_value.save.return_value = "doc_123"
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        result = adapter.ingest("/tmp/test.txt")

        assert result.success is True
        assert result.adapter == "pageindex"
        mock_mod.load_file.assert_called_once()

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_query(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.reason_and_answer.return_value = MagicMock(
            answer="The answer is 42", confidence=0.9, hops=2
        )
        mock_mod.TreeStore.return_value.list_trees.return_value = [{"doc_id": "d1"}]
        mock_mod.TreeStore.return_value.load.return_value = MagicMock()
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        result = adapter.query("What is X?", "tree_reasoning", "en")

        assert result.answer == "The answer is 42"
        assert result.confidence == 0.9

    @patch("adapters.pageindex_adapter._import_pageindex")
    def test_cleanup(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.TreeStore.return_value.delete_all.return_value = 5
        mock_import.return_value = mock_mod

        adapter = PageIndexAdapter()
        adapter.setup()
        adapter.cleanup()  # Should not raise
        mock_mod.TreeStore.return_value.delete_all.assert_called_once()
