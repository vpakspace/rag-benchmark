"""3-level evaluation judge for RAG benchmark.

Adapted from agentic-graph-rag benchmark/runner.py.
Standalone -- no project-specific imports.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are evaluating a RAG system answer.

Question: {question}
Expected keywords/concepts: {keywords}
System answer: {answer}

IMPORTANT: The answer may be in Russian while keywords are in English, or vice versa.
Match CONCEPTS and meanings, not exact strings.
For example: "ontology" matches "онтология", "graph" matches "граф", etc.

Does the answer correctly address the question and cover the expected concepts?
For enumeration questions (list all, describe all), check at least 50% coverage.
Reply with ONLY one word: PASS or FAIL"""

_JUDGE_PROMPT_REF = """You are evaluating a RAG system answer against a reference.

Question: {question}
Reference answer: {reference_answer}
System answer: {answer}

Does the system answer cover the same THEMES as the reference?
Match meanings and concepts, not exact words. Languages may differ.
For enumeration questions, PASS if at least 50% themes overlap semantically.
Reply ONLY: PASS or FAIL"""

# ---------------------------------------------------------------------------
# Global/comprehensive query detection
# ---------------------------------------------------------------------------

_GLOBAL_RE = re.compile(
    r'\b('
    r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
    r'|list all|describe all|summarize all|overview|every\b'
    r'|все компоненты|все методы|все слои|все решения'
    r'|all components|all layers|all methods|all decisions'
    r')',
    re.IGNORECASE,
)


def is_global_query(query: str) -> bool:
    """Detect global/enumeration queries."""
    return bool(_GLOBAL_RE.search(query))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def keyword_overlap(answer: str, keywords: list[str]) -> float:
    """Return fraction of keywords found in answer (case-insensitive)."""
    if not keywords:
        return 0.0
    lower = answer.lower()
    found = sum(1 for k in keywords if k.lower() in lower)
    return found / len(keywords)


def _embedding_similarity(text_a: str, text_b: str, openai_client: OpenAI) -> float:
    """Cosine similarity between embeddings of two texts."""
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text_a[:8000], text_b[:8000]],
        )
        vec_a = resp.data[0].embedding
        vec_b = resp.data[1].embedding
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    except Exception as e:
        logger.error("Embedding similarity failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    answer: str,
    keywords: list[str],
    openai_client: OpenAI,
    reference_answer: str = "",
) -> bool:
    """Hybrid judge: embedding similarity -> keyword overlap -> LLM judge."""
    # Fast path 1: embedding similarity with reference >= 0.65
    if reference_answer:
        similarity = _embedding_similarity(answer, reference_answer, openai_client)
        if similarity >= 0.65:
            return True

    # Fast path 2: keyword overlap >= threshold
    overlap = keyword_overlap(answer, keywords)
    threshold = 0.65 if is_global_query(question) else 0.40
    if overlap >= threshold:
        return True

    # Fallback: LLM judge
    if reference_answer:
        prompt_text = _JUDGE_PROMPT_REF.format(
            question=question,
            reference_answer=reference_answer,
            answer=answer[:2000],
        )
    else:
        prompt_text = _JUDGE_PROMPT.format(
            question=question,
            keywords=", ".join(keywords),
            answer=answer[:2000],
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        verdict = text.split("\n")[-1].strip().upper()
        return verdict == "PASS"
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return False
