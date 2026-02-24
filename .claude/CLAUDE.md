# RAG Benchmark

Унифицированная платформа для сравнительной оценки 6 RAG-проектов на едином наборе данных и метриках.

**GitHub**: https://github.com/vpakspace/rag-benchmark
**Расположение**: `~/rag-benchmark/`
**Python**: 3.12+, pytest, ruff
**Порт dashboard**: 8507

---

## Архитектура

```
rag-benchmark/
├── adapters/           # 6 адаптеров + base + registry + import utils
├── benchmark/          # runner, evaluate (3-level judge), compare (metrics)
├── article/            # Auto-generate markdown статьи из результатов
│   └── templates/      # RU + EN шаблоны (intro, methodology)
├── dashboard/          # Streamlit UI (5 tabs, i18n EN/RU)
├── data/
│   ├── documents/      # en_llm_survey.txt (44 KB), ru_graph_kb.txt (67 KB)
│   └── questions.json  # 60 билингвальных вопросов (30 EN + 30 RU)
├── results/            # JSON с результатами benchmark runs
├── tests/              # 79 тестов (unit + integration), 100% pass
├── docs/plans/         # Архитектурный дизайн и план реализации
├── pyproject.toml
└── .env                # Credentials (в .gitignore)
```

---

## 6 оцениваемых проектов (адаптеры)

| # | Проект | Адаптер | Режимы | Хранилище |
|---|--------|---------|--------|-----------|
| 1 | **agentic-graph-rag** | `AgenticGraphRAGAdapter` | agent_pattern, agent_llm | Neo4j (Cypher + Vector) |
| 2 | **rag-2.0** | `RAG2Adapter` | vector, reflect, agent | Neo4j Vector Index |
| 3 | **pageindex** | `PageIndexAdapter` | tree_reasoning | JSON (локальные) |
| 4 | **cog-rag-cognee** | `CogRAGCogneeAdapter` | chunks, graph, rag_completion | Neo4j + LanceDB |
| 5 | **rag-temporal** | `RAGTemporalAdapter` | vector, kg, hybrid, agent | Neo4j Vector + Graphiti KG |
| 6 | **temporal-knowledge-base** | `TemporalKBAdapter` | structural, temporal, hybrid | Neo4j bi-temporal + Graphiti |

---

## Ключевые компоненты

### BaseAdapter (adapters/base.py)
Абстрактный интерфейс: `setup()`, `ingest()`, `query()`, `cleanup()`, `health_check()`.
Pydantic модели: `IngestResult`, `QueryResult`, `EvalResult`, `BenchmarkRun`, `Question`.

### Import Isolation (adapters/_import_utils.py)
`prepare_imports(project_path)` — очищает `sys.modules` от конфликтующих пакетов (core, ingestion, retrieval, generation, agent, storage, utils, config, models) перед импортом модулей каждого проекта.

### 3-Level Hybrid Judge (benchmark/evaluate.py)
1. **Embedding similarity** (OpenAI text-embedding-3-small) >= 0.65 → PASS
2. **Keyword overlap** (case-insensitive) >= 0.40 (0.65 для global) → PASS
3. **LLM judge** (gpt-4o-mini) — multilingual theme matching → PASS/FAIL

### Neo4j Isolation
Каждый адаптер использует свой префикс для индексов: `bench_agr_*`, `bench_rag2_*`, `bench_rt_*`, `bench_tkb_*`, `bench_crc_*`.

---

## Данные

### Документы
- **EN**: "A Survey of Large Language Models" (Zhao et al., 2023) — 44 KB
- **RU**: "Графовые БД и Knowledge Graphs: обзор" — 67 KB

### Вопросы (60 total)
- 30 EN + 30 RU, 5 типов: simple (14), relation (12), multi_hop (12), global (12), temporal (10)
- Языковые режимы: native, cross, all

---

## Запуск

### CLI
```bash
python run_benchmark.py                              # Все адаптеры
python run_benchmark.py --adapters rag2 rag_temporal  # Конкретные
python run_benchmark.py --dry-run                     # Проверка доступности
python run_benchmark.py --lang-mode cross             # Кросс-языковой тест
```

### Dashboard
```bash
streamlit run dashboard/streamlit_app.py --server.port 8507
```

### Тесты
```bash
python -m pytest tests/ -v  # 79 tests, 0.4s
```

---

## Инфраструктура

### Neo4j
- **Контейнер**: `osgr-neo4j` (Neo4j 5.26 Community + APOC)
- **Порты**: 7474 (HTTP), 7687 (Bolt)
- **Credentials**: neo4j / temporal_kb_2026
- **Volume**: `osgr-neo4j-data`
- **Hostname**: `osgr-neo4j` (fix DNS resolution)

### .env
```bash
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=temporal_kb_2026
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

---

## Зависимости

**Runtime**: openai, pydantic, pydantic-settings, streamlit, plotly, pandas, numpy
**Dev**: pytest, pytest-asyncio, ruff

---

## Текущий статус

- **79 тестов**: all passing (0.4s)
- **6 адаптеров**: реализованы с полной import isolation
- **Dashboard**: 5 tabs (Overview, By Query Type, By Document, Run Benchmark, Article)
- **Article generator**: auto-generate RU/EN markdown из результатов
- **Stage B** (текущая): 8-15 стр. документов, 60 вопросов, native lang mode
- **Stage C** (планируется): полные документы, cross-language, extended metrics

---

## Коммиты

| Hash | Описание |
|------|----------|
| `c77cf5a` | fix: resolve sys.path conflicts and update adapter APIs for all 6 RAG projects |
| `01c2228` | feat: 60 bilingual benchmark questions (30 EN + 30 RU) |
| `47d2709` | feat: integration tests (end-to-end benchmark flow) |
| `b5bc85e` | feat: CLI entry point for benchmark runner |

---

**Последнее обновление**: 2026-02-24
