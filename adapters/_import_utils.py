"""Import isolation utilities for adapters with conflicting package names."""

from __future__ import annotations

import sys

# Packages that exist in multiple projects and cause sys.modules conflicts
CONFLICTING_PACKAGES = [
    "core", "ingestion", "retrieval", "generation", "agent",
    "storage", "utils", "ui", "config", "models",
]


def prepare_imports(project_path: str, extra_paths: list[str] | None = None):
    """Clear conflicting modules and prioritize project in sys.path.

    Unlike a context manager, this does NOT restore state — the loaded
    modules stay available for constructors and method calls.
    Each adapter calls this before importing, so each adapter gets a
    clean slate for its own project's packages.
    """
    # 1. Clear conflicting cached modules
    for pkg in CONFLICTING_PACKAGES:
        for key in list(sys.modules.keys()):
            if key == pkg or key.startswith(f"{pkg}."):
                del sys.modules[key]

    # 2. Put project first in sys.path
    project_str = str(project_path)
    if project_str in sys.path:
        sys.path.remove(project_str)
    sys.path.insert(0, project_str)

    # 3. Add extra paths (e.g., subpackages, dependency projects)
    for p in extra_paths or []:
        p_str = str(p)
        if p_str in sys.path:
            sys.path.remove(p_str)
        sys.path.insert(1, p_str)
