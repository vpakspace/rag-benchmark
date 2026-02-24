"""Download and prepare benchmark documents."""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


def prepare_en_document() -> Path:
    """Check EN document exists (LLM Survey excerpt).

    Document is pre-written in data/documents/en_llm_survey.txt.
    Source: "A Survey of Large Language Models" (Zhao et al., 2023, arXiv:2303.18223).
    """
    output = DATA_DIR / "en_llm_survey.txt"
    if output.exists():
        size = output.stat().st_size
        print(f"EN document exists: {output} ({size:,} bytes)")
    else:
        print(f"EN document NOT FOUND: {output}")
        print("Please create it manually or run the document preparation agent.")
    return output


def prepare_ru_document() -> Path:
    """Check RU document exists (Graph DBs and Knowledge Graphs review).

    Document is pre-written in data/documents/ru_graph_kb.txt.
    Original Russian technical text covering graph databases, KGs, and GraphRAG.
    """
    output = DATA_DIR / "ru_graph_kb.txt"
    if output.exists():
        size = output.stat().st_size
        print(f"RU document exists: {output} ({size:,} bytes)")
    else:
        print(f"RU document NOT FOUND: {output}")
        print("Please create it manually or run the document preparation agent.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark documents")
    parser.add_argument("--lang", choices=["en", "ru", "all"], default="all")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.lang in ("en", "all"):
        prepare_en_document()
    if args.lang in ("ru", "all"):
        prepare_ru_document()
