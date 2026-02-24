#!/usr/bin/env python3
"""CLI entry point for rag-benchmark."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from adapters.registry import AdapterRegistry
from adapters.pageindex_adapter import PageIndexAdapter
from adapters.rag2_adapter import RAG2Adapter
from adapters.agentic_graph_rag_adapter import AgenticGraphRAGAdapter
from adapters.rag_temporal_adapter import RAGTemporalAdapter
from adapters.temporal_kb_adapter import TemporalKBAdapter
from adapters.cog_rag_cognee_adapter import CogRAGCogneeAdapter
from benchmark.runner import load_questions, run_benchmark, make_benchmark_run, save_results


ALL_ADAPTER_CLASSES = [
    PageIndexAdapter,
    RAG2Adapter,
    AgenticGraphRAGAdapter,
    RAGTemporalAdapter,
    TemporalKBAdapter,
    CogRAGCogneeAdapter,
]


def _load_benchmark_env():
    """Load infrastructure env vars from benchmark .env into os.environ.

    Env vars take precedence over project .env files in Pydantic BaseSettings,
    ensuring all adapters use the same Neo4j / OpenAI credentials.
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main():
    parser = argparse.ArgumentParser(description="Cross-RAG Benchmark")
    parser.add_argument(
        "--adapters", nargs="*",
        help="Adapter names to run (default: all available)",
    )
    parser.add_argument(
        "--lang-mode", choices=["native", "cross", "all"], default="native",
        help="Language mode for questions",
    )
    parser.add_argument(
        "--modes", nargs="*",
        help="Filter specific modes (e.g., vector agent)",
    )
    parser.add_argument(
        "--questions", type=Path,
        help="Path to questions.json (default: data/questions.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show which adapters are available without running",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _load_benchmark_env()

    # Register adapters
    registry = AdapterRegistry()
    for cls in ALL_ADAPTER_CLASSES:
        try:
            adapter = cls()
            if adapter.health_check():
                adapter.setup()
                registry.register(adapter)
                logging.info("Registered: %s (modes: %s)", adapter.name, adapter.modes)
            else:
                logging.warning("Health check failed: %s", cls.__name__)
        except Exception as e:
            logging.warning("Failed to initialize %s: %s", cls.__name__, e)

    # Restore cwd after adapter imports (adapters change cwd to their project dirs)
    os.chdir(Path(__file__).parent)

    # Filter adapters
    if args.adapters:
        adapters = [registry.get(n) for n in args.adapters if registry.get(n)]
    else:
        adapters = registry.list_available()

    if not adapters:
        print("No adapters available. Check services (Neo4j, Ollama).")
        sys.exit(1)

    print(f"\nAvailable adapters ({len(adapters)}):")
    for a in adapters:
        print(f"  - {a.name}: modes={a.modes}, services={a.requires_services}")

    if args.dry_run:
        return

    # Load questions
    questions = load_questions(args.questions)
    print(f"\nLoaded {len(questions)} questions")

    # Documents
    data_dir = Path(__file__).parent / "data" / "documents"
    documents = {
        "en_llm_survey": str(data_dir / "en_llm_survey.txt"),
        "ru_graph_kb": str(data_dir / "ru_graph_kb.txt"),
    }

    # Ingest — skip adapters where all documents fail
    failed_adapters = set()
    for adapter in adapters:
        print(f"\nIngesting documents into {adapter.name}...")
        successes = 0
        for doc_name, doc_path in documents.items():
            result = adapter.ingest(doc_path)
            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"  {doc_name}: {result.chunks_count} chunks, "
                  f"{result.duration_seconds}s — {status}")
            if result.success:
                successes += 1
        if successes == 0:
            print(f"  ** Skipping {adapter.name} — all ingests failed")
            failed_adapters.add(adapter.name)

    adapters = [a for a in adapters if a.name not in failed_adapters]
    if not adapters:
        print("\nNo adapters with successful ingest. Exiting.")
        sys.exit(1)

    # Run benchmark
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        client = None
        logging.warning("OpenAI client not available, LLM judge disabled")

    # Build modes filter if --modes specified
    modes_filter = None
    if args.modes:
        modes_filter = {a.name: args.modes for a in adapters}

    results = run_benchmark(
        adapters=adapters,
        questions=questions,
        openai_client=client,
        documents=documents,
        lang_mode=args.lang_mode,
        modes_filter=modes_filter,
        progress_callback=lambda c, t, m: print(
            f"\r  [{c}/{t}] {m}", end="", flush=True
        ),
    )
    print()

    # Save
    run = make_benchmark_run(
        results=results,
        adapters=[a.name for a in adapters],
        documents=list(documents.keys()),
    )
    path = save_results(run)
    print(f"\nResults saved to {path}")

    # Summary
    from benchmark.compare import compare_adapters
    rows = compare_adapters(results)
    if rows:
        print("\n--- Summary ---")
        for r in rows:
            print(f"  {r['Adapter']:25s} {r['Mode']:20s} "
                  f"{r['Accuracy']:>10s} {r['Avg Latency (s)']:>8.1f}s")


if __name__ == "__main__":
    main()
