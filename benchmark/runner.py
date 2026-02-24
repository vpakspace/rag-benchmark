"""Benchmark runner -- orchestrate evaluation across adapters and modes."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from adapters.base import BaseAdapter, EvalResult, Question, BenchmarkRun
from benchmark.evaluate import evaluate_answer

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_questions(path: Path | None = None) -> list[Question]:
    """Load and validate benchmark questions from JSON."""
    p = path or QUESTIONS_PATH
    with open(p) as f:
        raw = json.load(f)
    return [Question(**q) for q in raw]


def save_results(run: BenchmarkRun) -> Path:
    """Save benchmark run results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2))
    logger.info("Results saved to %s", path)
    return path


def run_benchmark(
    adapters: list[BaseAdapter],
    questions: list[Question],
    openai_client: OpenAI,
    documents: dict[str, str],
    lang_mode: str = "native",
    modes_filter: dict[str, list[str]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[EvalResult]:
    """Run benchmark across adapters, modes, and questions.

    Args:
        adapters: List of initialized adapters.
        questions: List of Question objects.
        openai_client: OpenAI client for evaluation.
        documents: dict[doc_name -> file_path].
        lang_mode: "native" (match doc lang), "cross", or "all".
        modes_filter: Optional dict[adapter_name -> list of modes to run].
        progress_callback: Optional fn(current, total, message).

    Returns:
        List of EvalResult objects.
    """
    all_results: list[EvalResult] = []

    # Calculate total work
    total_evals = 0
    for adapter in adapters:
        modes = modes_filter.get(adapter.name, adapter.modes) if modes_filter else adapter.modes
        total_evals += len(modes) * len(questions)

    current = 0

    for adapter in adapters:
        # Health check
        if not adapter.health_check():
            logger.warning("Adapter %s failed health check -- skipping", adapter.name)
            continue

        modes = modes_filter.get(adapter.name, adapter.modes) if modes_filter else adapter.modes

        for mode in modes:
            for q in questions:
                current += 1

                # Determine question language based on lang_mode
                if lang_mode == "native":
                    # EN question for EN doc, RU question for RU doc
                    lang = "ru" if "ru_" in q.document else "en"
                elif lang_mode == "cross":
                    lang = "en" if "ru_" in q.document else "ru"
                else:
                    lang = "en"  # TODO: expand for "all" mode

                question_text = q.question_ru if lang == "ru" else q.question_en

                if progress_callback:
                    progress_callback(current, total_evals, f"{adapter.name}/{mode}/Q{q.id}")

                start = time.monotonic()
                try:
                    qr = adapter.query(question_text, mode, lang)
                    latency = round(time.monotonic() - start, 3)

                    try:
                        passed = evaluate_answer(
                            question_text, qr.answer, q.keywords,
                            openai_client, q.reference_answer or "",
                        )
                    except Exception as e:
                        logger.error("Eval error [%s/%s] Q%d: %s", adapter.name, mode, q.id, e)
                        passed = False

                    all_results.append(EvalResult(
                        question_id=q.id, adapter=adapter.name, mode=mode,
                        lang=lang, query_type=q.type, document=q.document,
                        passed=passed, answer=qr.answer,
                        latency=latency, confidence=qr.confidence,
                    ))

                except Exception as e:
                    latency = round(time.monotonic() - start, 3)
                    logger.error("Query error [%s/%s] Q%d: %s", adapter.name, mode, q.id, e)
                    all_results.append(EvalResult(
                        question_id=q.id, adapter=adapter.name, mode=mode,
                        lang=lang, query_type=q.type, document=q.document,
                        passed=False, answer=f"ERROR: {e}",
                        latency=latency, confidence=0.0,
                    ))

        logger.info(
            "Adapter %s complete: %d/%d passed",
            adapter.name,
            sum(1 for r in all_results if r.adapter == adapter.name and r.passed),
            sum(1 for r in all_results if r.adapter == adapter.name),
        )

    return all_results


def make_benchmark_run(
    results: list[EvalResult],
    adapters: list[str],
    documents: list[str],
) -> BenchmarkRun:
    """Create a BenchmarkRun from results."""
    now = datetime.now()
    return BenchmarkRun(
        run_id=f"run_{now.strftime('%Y-%m-%d_%H-%M')}",
        timestamp=now,
        documents=documents,
        adapters=adapters,
        total_questions=len({r.question_id for r in results}),
        results=results,
    )
