# Cross-RAG Benchmark: Design Document

**Date**: 2026-02-24
**Status**: Approved
**Author**: vladspace_ubuntu24 + Claude Opus 4.6

---

## 1. Overview

Unified benchmark for evaluating 6 RAG projects using 2 representative documents (EN + RU).
Compares accuracy, latency, confidence across 5 query types with bilingual questions.

**Output**: Comparison tables, auto-generated article, Streamlit dashboard.

## 2. Projects Under Evaluation

| # | Project | Path | Modes | Storage | Services |
|---|---------|------|-------|---------|----------|
| 1 | agentic-graph-rag | ~/agentic-graph-rag/ | vector, cypher, hybrid, agent_pattern, agent_llm, agent_mangle | Neo4j (Cypher + Vector) | Neo4j, OpenAI |
| 2 | rag-2.0 | ~/rag-2.0/ | vector, reflect, agent | Neo4j Vector Index | Neo4j, OpenAI |
| 3 | pageindex | ~/pageindex/ | tree_reasoning | JSON local files | OpenAI only |
| 4 | cog-rag-cognee | ~/cog-rag-cognee/ | chunks, graph, rag_completion | Neo4j + LanceDB | Ollama, Neo4j |
| 5 | rag-temporal | ~/rag-temporal/ | vector, kg, hybrid, agent | Neo4j Vector + Graphiti KG | Neo4j, OpenAI |
| 6 | temporal-knowledge-base | ~/temporal-knowledge-base/ | structural, temporal, hybrid | Neo4j bi-temporal + Graphiti | Neo4j, OpenAI, Graphiti |

## 3. Architecture

### Adapter Pattern

```
BaseAdapter(ABC)
  setup() → None
  ingest(file_path) → IngestResult
  query(question, mode, lang) → QueryResult
  cleanup() → None
  health_check() → bool
  name / modes / supported_langs / requires_services
```

6 concrete adapters import project modules via sys.path (bridge pattern from rag-temporal).

### Data Flow

```
questions.json (60 bilingual)
    ↓
runner.py (orchestrator)
    ↓ for each adapter × mode × question
    ├─ adapter.query() → QueryResult
    ├─ evaluate() → pass/fail (3-level judge)
    └─ append EvalResult
    ↓
results/run_*.json
    ↓
compare.py → tables, breakdowns
    ↓
dashboard (Streamlit :8507) + article (markdown)
```

### Neo4j Isolation

Prefix-based index naming per adapter:
- `bench_agr_*` (agentic-graph-rag)
- `bench_rag2_*` (rag-2.0)
- `bench_rt_*` (rag-temporal)
- `bench_tkb_*` (temporal-kb)
- `bench_crc_*` (cog-rag-cognee)

cleanup() removes only own indexes/nodes between runs.

## 4. Documents

### EN Document: "A Survey of Large Language Models" (Zhao et al., 2023)
- Stage B: 8-15 pages excerpt (~5-10K words)
- Stage C: full paper (~30 pages)
- Rich temporal dimension: GPT-1 (2018) → GPT-4 (2023)
- Topics: scaling laws, RLHF, benchmarks, architecture evolution

### RU Document: "Графовые базы данных и Knowledge Graphs: обзор"
- Composed original Russian technical text (~10 pages)
- Timeline: RDF (1999) → SPARQL (2008) → Neo4j growth (2010+) → GraphRAG (2024)
- Topics: Neo4j, ArangoDB, Neptune, Dgraph, Knowledge Graphs, applications
- Stage B: 8-15 pages, Stage C: extended version

## 5. Questions

### Structure
```json
{
  "id": 1,
  "document": "en_llm_survey",
  "question_en": "...",
  "question_ru": "...",
  "type": "simple|relation|multi_hop|global|temporal",
  "keywords": ["...", "..."],
  "reference_answer": "..."
}
```

### Distribution (60 total, 30 per document)

| Type | EN doc (1-30) | RU doc (31-60) | Total |
|------|--------------|----------------|-------|
| simple | 7 | 7 | 14 |
| relation | 6 | 6 | 12 |
| multi_hop | 6 | 6 | 12 |
| global | 6 | 6 | 12 |
| temporal | 5 | 5 | 10 |

### Language Modes
- **native**: EN question → EN doc, RU question → RU doc
- **cross**: RU question → EN doc, EN question → RU doc
- **all**: all 4 combinations

## 6. Evaluation (3-Level Judge)

1. **Fast path 1**: Embedding similarity ≥ 0.65 → auto-PASS
2. **Fast path 2**: Keyword overlap ≥ threshold (0.65 global, 0.40 specific) → auto-PASS
3. **Fallback**: LLM judge (gpt-4o-mini)

Reused from agentic-graph-rag benchmark/runner.py.

## 7. Metrics

- **accuracy**: correct / total (pass/fail)
- **avg_confidence**: mean confidence score (0.0-1.0)
- **avg_latency**: mean seconds per query
- Breakdowns: by adapter, by mode, by query_type, by document

## 8. Data Models (Pydantic)

```python
class IngestResult(BaseModel):
    adapter: str
    document: str
    chunks_count: int
    duration_seconds: float
    success: bool
    error: str | None = None

class QueryResult(BaseModel):
    adapter: str
    mode: str
    question_id: int
    answer: str
    confidence: float
    latency: float
    retries: int = 0
    sources: list[str] = []

class EvalResult(BaseModel):
    question_id: int
    adapter: str
    mode: str
    lang: str
    query_type: str
    document: str
    passed: bool
    answer: str
    latency: float
    confidence: float

class BenchmarkRun(BaseModel):
    run_id: str
    timestamp: datetime
    documents: list[str]
    adapters: list[str]
    total_questions: int
    results: list[EvalResult]
```

## 9. Streamlit Dashboard (port 8507)

5 tabs:
1. **Overview** — summary table with heatmap, best performer bar chart, ingestion stats
2. **By Query Type** — grouped bar chart, heatmap adapter × query_type, per-question detail
3. **By Document** — EN vs RU side-by-side, radar chart per adapter, cross-language results
4. **Run Benchmark** — adapter checkboxes, service health, progress bar, live results
5. **Article** — auto-generated markdown article, copy/export buttons

Technologies: Streamlit, Plotly, pandas, i18n (EN/RU).

## 10. Article Generator

Auto-generated from BenchmarkRun data:
1. Introduction — motivation, scope
2. Methodology — documents, questions, judge, metrics
3. Projects Overview — architecture table
4. Results — summary table, by type heatmap, by document, leaders
5. Discussion — strengths, language impact, temporal queries, accuracy vs latency
6. Conclusions — recommendations, roadmap

Templates in article/templates/ (RU + EN).

## 11. Project Structure

```
~/rag-benchmark/
├── adapters/          (8 files, ~1200 LOC)
│   ├── base.py, registry.py
│   └── 6 adapter files
├── benchmark/         (3 files, ~600 LOC)
│   ├── runner.py, evaluate.py, compare.py
├── data/
│   ├── documents/     (EN + RU, stage B + C)
│   └── questions.json (60 bilingual)
├── results/           (auto-generated JSON)
├── article/           (generator + templates, ~400 LOC)
├── dashboard/         (streamlit + i18n, ~800 LOC)
├── scripts/           (CLI, doc prep, question gen, ~300 LOC)
├── tests/             (5 files, ~700 LOC)
└── docs/plans/        (this document)
```

**Total estimate**: ~23 files, ~4,000 LOC

## 12. Dependencies

```toml
[project]
dependencies = [
    "openai", "pydantic", "pydantic-settings",
    "streamlit", "plotly", "pandas", "numpy",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

Project-specific dependencies (neo4j, cognee, graphiti) imported via sys.path from each project.

## 13. Execution Estimate

| Parameter | Value |
|-----------|-------|
| Adapters | 6 |
| Total modes | ~20 |
| Questions | 60 |
| Native run | ~1,200 eval calls |
| All lang modes | ~2,400 eval calls |
| Estimated time (native) | ~30-60 min |
| Query timeout | 120 sec |

## 14. Fault Tolerance

- health_check() before each adapter → skip if unavailable
- Exception per query → record error, continue
- Incremental save after each adapter
- --resume run_id for re-running failed/skipped

## 15. Stages

- **Stage B** (current): 8-15 page documents, 60 questions, native lang mode
- **Stage C** (future): full documents (15-30 pages), cross-language mode, extended metrics (faithfulness, cost, retrieval precision)
