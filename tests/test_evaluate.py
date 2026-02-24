"""Tests for 3-level judge evaluation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from benchmark.evaluate import keyword_overlap, is_global_query, evaluate_answer


class TestKeywordOverlap:
    def test_all_found(self):
        assert keyword_overlap("GPT-3 was released in 2020", ["GPT-3", "2020"]) == 1.0

    def test_none_found(self):
        assert keyword_overlap("Unrelated text", ["GPT-3", "2020"]) == 0.0

    def test_partial(self):
        assert keyword_overlap("GPT-3 is a model", ["GPT-3", "2020", "OpenAI"]) == pytest.approx(1 / 3)

    def test_case_insensitive(self):
        assert keyword_overlap("gpt-3 released", ["GPT-3"]) == 1.0

    def test_empty_keywords(self):
        assert keyword_overlap("any text", []) == 0.0

    def test_cyrillic_keywords(self):
        assert keyword_overlap("Neo4j является графовой базой данных", ["графовой", "Neo4j"]) == 1.0


class TestIsGlobalQuery:
    def test_english_list_all(self):
        assert is_global_query("List all frameworks mentioned") is True

    def test_russian_enumerate(self):
        assert is_global_query("Перечисли все компоненты архитектуры") is True

    def test_simple_question(self):
        assert is_global_query("What year was GPT-3 released?") is False

    def test_overview(self):
        assert is_global_query("Give an overview of the system") is True


class TestEvaluateAnswer:
    def test_keyword_pass_simple(self):
        """High keyword overlap -> auto-PASS without LLM call."""
        result = evaluate_answer(
            question="What is GPT-3?",
            answer="GPT-3 is a large language model by OpenAI with 175B parameters released in 2020",
            keywords=["GPT-3", "OpenAI", "175B", "2020"],
            openai_client=MagicMock(),  # Should not be called
        )
        assert result is True

    def test_keyword_fail_triggers_llm(self):
        """Low keyword overlap -> LLM judge called."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="PASS"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip embedding")

        result = evaluate_answer(
            question="What is GPT-3?",
            answer="It is a neural network",
            keywords=["GPT-3", "OpenAI", "175B", "2020"],
            openai_client=mock_client,
        )
        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_llm_judge_fail(self):
        """LLM judge returns FAIL."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FAIL"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip embedding")

        result = evaluate_answer(
            question="What is GPT-3?",
            answer="I don't know",
            keywords=["GPT-3", "OpenAI", "175B"],
            openai_client=mock_client,
        )
        assert result is False

    def test_embedding_similarity_pass(self):
        """High embedding similarity with reference -> auto-PASS."""
        mock_client = MagicMock()
        # Return two identical embeddings -> similarity = 1.0
        mock_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=mock_embedding), MagicMock(embedding=mock_embedding)]
        )

        result = evaluate_answer(
            question="What is X?",
            answer="X is Y",
            keywords=[],
            openai_client=mock_client,
            reference_answer="X is Y exactly",
        )
        assert result is True

    def test_global_query_higher_threshold(self):
        """Global queries need 65% keyword overlap (not 40%)."""
        # 2/5 = 40% -- passes for simple (>=0.40) but fails for global (needs >=0.65)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="FAIL"))]
        )
        mock_client.embeddings.create.side_effect = Exception("skip")

        result = evaluate_answer(
            question="List all components of the system",
            answer="Alpha module and Beta module are present",
            keywords=["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            openai_client=mock_client,
        )
        assert result is False  # 2/5 = 40% < 65% global threshold, LLM says FAIL
