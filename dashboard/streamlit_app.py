"""RAG Benchmark Dashboard — Streamlit UI (port 8507).

Usage: streamlit run dashboard/streamlit_app.py --server.port 8507
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from adapters.base import BenchmarkRun, EvalResult
from benchmark.compare import (
    compare_adapters,
    compute_metrics,
    accuracy_by_type,
    accuracy_by_document,
)
from article.generator import ArticleGenerator
from dashboard.i18n import get_translator

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_runs() -> dict[str, BenchmarkRun]:
    """Load all benchmark runs from results/*.json."""
    runs = {}
    if not RESULTS_DIR.exists():
        return runs
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            run = BenchmarkRun(**data)
            runs[run.run_id] = run
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
    return runs


def _results_to_df(results: list[EvalResult]) -> pd.DataFrame:
    """Convert EvalResult list to DataFrame."""
    return pd.DataFrame([r.model_dump() for r in results])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_overview(run: BenchmarkRun, t):
    """Overview tab — summary table, bar chart, ingestion stats."""
    st.header(t("tab_overview"))

    results = run.results
    if not results:
        st.warning(t("no_results"))
        return

    # Summary table
    rows = compare_adapters(results)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Best mode per adapter — bar chart
    adapter_best: dict[str, float] = {}
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        groups.setdefault(f"{r.adapter}/{r.mode}", []).append(r)

    chart_data = []
    for key, group in groups.items():
        m = compute_metrics(group)
        adapter, mode = key.split("/", 1)
        chart_data.append({"Adapter": adapter, "Mode": mode, "Accuracy": m["accuracy"]})

    if chart_data:
        fig = px.bar(
            pd.DataFrame(chart_data),
            x="Adapter", y="Accuracy", color="Mode",
            barmode="group",
            title=t("best_performer"),
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Stats
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    col1, col2, col3 = st.columns(3)
    col1.metric(t("total_questions"), total)
    col2.metric(t("passed"), passed)
    col3.metric(t("failed"), total - passed)


def tab_by_type(run: BenchmarkRun, t):
    """By Query Type tab — grouped bar chart, heatmap."""
    st.header(t("tab_by_type"))

    results = run.results
    if not results:
        st.warning(t("no_results"))
        return

    breakdown = accuracy_by_type(results)

    # Grouped bar chart
    chart_rows = []
    for qtype, adapters in sorted(breakdown.items()):
        for adapter, acc in sorted(adapters.items()):
            chart_rows.append({"Type": qtype, "Adapter": adapter, "Accuracy": acc})

    if chart_rows:
        fig = px.bar(
            pd.DataFrame(chart_rows),
            x="Type", y="Accuracy", color="Adapter",
            barmode="group",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    if breakdown:
        adapters_all = sorted({a for qtypes in breakdown.values() for a in qtypes})
        types_all = sorted(breakdown.keys())
        z = [[breakdown.get(qt, {}).get(a, 0) for a in adapters_all] for qt in types_all]

        fig_hm = go.Figure(data=go.Heatmap(
            z=z, x=adapters_all, y=types_all,
            colorscale="RdYlGn", zmin=0, zmax=1,
            text=[[f"{v:.0%}" for v in row] for row in z],
            texttemplate="%{text}",
        ))
        fig_hm.update_layout(title="Accuracy Heatmap: Adapter x Query Type")
        st.plotly_chart(fig_hm, use_container_width=True)


def tab_by_document(run: BenchmarkRun, t):
    """By Document tab — EN vs RU side-by-side."""
    st.header(t("tab_by_doc"))

    results = run.results
    if not results:
        st.warning(t("no_results"))
        return

    breakdown = accuracy_by_document(results)

    col_en, col_ru = st.columns(2)

    for doc, col, label in [
        ("en_llm_survey", col_en, t("en_document")),
        ("ru_graph_kb", col_ru, t("ru_document")),
    ]:
        with col:
            st.subheader(label)
            if doc in breakdown:
                doc_data = breakdown[doc]
                for adapter, acc in sorted(doc_data.items()):
                    st.metric(adapter, f"{acc:.0%}")
            else:
                st.info("N/A")

    # Radar chart per adapter
    adapters_all = sorted({r.adapter for r in results})
    docs = sorted(breakdown.keys())
    if len(docs) >= 2 and adapters_all:
        fig = go.Figure()
        for adapter in adapters_all:
            values = [breakdown.get(doc, {}).get(adapter, 0) for doc in docs]
            values.append(values[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=values, theta=docs + [docs[0]],
                fill="toself", name=adapter,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Adapter Performance by Document",
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_run_benchmark(t):
    """Run Benchmark tab — adapter selection, progress, results."""
    st.header(t("tab_run"))
    st.info("Use CLI: `python run_benchmark.py` to run benchmarks. "
            "Results appear automatically in the dashboard.")


def tab_article(run: BenchmarkRun, t, lang: str):
    """Article tab — generate and display markdown article."""
    st.header(t("tab_article"))

    if st.button(t("generate_article")):
        gen = ArticleGenerator(run, lang=lang)
        article_text = gen.generate()
        st.session_state["article_text"] = article_text

    article_text = st.session_state.get("article_text", "")
    if article_text:
        st.markdown(article_text)
        st.download_button(
            label=t("export_md"),
            data=article_text,
            file_name=f"rag_benchmark_article_{run.run_id}.md",
            mime="text/markdown",
        )
    else:
        st.info(t("no_results"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RAG Benchmark", layout="wide", page_icon="📊")

    # Sidebar
    lang = st.sidebar.radio("Language", ["ru", "en"], index=0)
    t = get_translator(lang)

    st.sidebar.title(t("title"))

    # Load runs
    runs = _load_runs()

    if not runs:
        st.title(t("title"))
        st.warning(t("no_results"))
        tab_run_benchmark(t)
        return

    # Run selector
    run_id = st.sidebar.selectbox(t("select_run"), list(runs.keys()))
    run = runs[run_id]

    st.title(t("title"))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("tab_overview"), t("tab_by_type"), t("tab_by_doc"),
        t("tab_run"), t("tab_article"),
    ])

    with tab1:
        tab_overview(run, t)
    with tab2:
        tab_by_type(run, t)
    with tab3:
        tab_by_document(run, t)
    with tab4:
        tab_run_benchmark(t)
    with tab5:
        tab_article(run, t, lang)


if __name__ == "__main__":
    main()
