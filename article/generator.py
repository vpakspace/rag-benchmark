"""Auto-generate benchmark article from results."""

from __future__ import annotations

from pathlib import Path

from adapters.base import BenchmarkRun
from benchmark.compare import compare_adapters, accuracy_by_type, accuracy_by_document

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ArticleGenerator:
    def __init__(self, run: BenchmarkRun, lang: str = "ru") -> None:
        self.run = run
        self.lang = lang

    def generate(self) -> str:
        sections = [
            self._load_template("intro"),
            self._load_template("methodology"),
            self._projects_overview(),
            self._results_summary(),
            self._results_by_type(),
            self._results_by_document(),
            self._discussion(),
            self._conclusion(),
        ]
        return "\n\n---\n\n".join(s for s in sections if s)

    def _load_template(self, name: str) -> str:
        path = TEMPLATES_DIR / f"{name}_{self.lang}.md"
        if path.exists():
            return path.read_text()
        # Fallback to English
        path_en = TEMPLATES_DIR / f"{name}_en.md"
        return path_en.read_text() if path_en.exists() else ""

    def _projects_overview(self) -> str:
        header = "## Обзор проектов" if self.lang == "ru" else "## Projects Overview"
        rows = []
        for adapter in self.run.adapters:
            adapter_results = [r for r in self.run.results if r.adapter == adapter]
            modes = sorted({r.mode for r in adapter_results})
            rows.append(f"| {adapter} | {', '.join(modes)} |")
        table = "| Adapter | Modes |\n|---------|-------|\n" + "\n".join(rows)
        return f"{header}\n\n{table}"

    def _results_summary(self) -> str:
        header = "## Результаты" if self.lang == "ru" else "## Results"
        rows = compare_adapters(self.run.results)
        if not rows:
            return f"{header}\n\nНет данных." if self.lang == "ru" else f"{header}\n\nNo data."
        # Build markdown table
        cols = list(rows[0].keys())
        header_row = "| " + " | ".join(cols) + " |"
        sep_row = "|" + "|".join("---" for _ in cols) + "|"
        data_rows = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
        return f"{header}\n\n{header_row}\n{sep_row}\n{data_rows}"

    def _results_by_type(self) -> str:
        header = "### По типам вопросов" if self.lang == "ru" else "### By Query Type"
        breakdown = accuracy_by_type(self.run.results)
        if not breakdown:
            return ""
        lines = [header, ""]
        for qtype, adapters in sorted(breakdown.items()):
            lines.append(f"**{qtype}**: " + ", ".join(f"{a}={v:.0%}" for a, v in sorted(adapters.items())))
        return "\n".join(lines)

    def _results_by_document(self) -> str:
        header = "### По документам" if self.lang == "ru" else "### By Document"
        breakdown = accuracy_by_document(self.run.results)
        if not breakdown:
            return ""
        lines = [header, ""]
        for doc, adapters in sorted(breakdown.items()):
            lines.append(f"**{doc}**: " + ", ".join(f"{a}={v:.0%}" for a, v in sorted(adapters.items())))
        return "\n".join(lines)

    def _discussion(self) -> str:
        if self.lang == "ru":
            return "## Обсуждение\n\n*Автоматический анализ результатов будет добавлен в Stage C.*"
        return "## Discussion\n\n*Automated analysis will be added in Stage C.*"

    def _conclusion(self) -> str:
        if self.lang == "ru":
            return "## Выводы\n\n*Рекомендации по выбору RAG-системы будут добавлены в Stage C.*"
        return "## Conclusions\n\n*Recommendations for RAG system selection will be added in Stage C.*"
