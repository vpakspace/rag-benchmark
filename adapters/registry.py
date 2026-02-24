"""Adapter registry -- discover, register, and manage RAG adapters."""

from __future__ import annotations

import logging

from adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for RAG project adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter) -> None:
        """Register an adapter (overwrites if name exists)."""
        self._adapters[adapter.name] = adapter
        logger.info("Registered adapter: %s (%d modes)", adapter.name, len(adapter.modes))

    def get(self, name: str) -> BaseAdapter | None:
        """Get adapter by name."""
        return self._adapters.get(name)

    def list_names(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def list_available(self) -> list[BaseAdapter]:
        """List adapters that pass health_check."""
        available = []
        for adapter in self._adapters.values():
            try:
                if adapter.health_check():
                    available.append(adapter)
                else:
                    logger.warning("Adapter %s failed health check", adapter.name)
            except Exception as e:
                logger.warning("Adapter %s health check error: %s", adapter.name, e)
        return available

    def all(self) -> list[BaseAdapter]:
        """List all registered adapters."""
        return list(self._adapters.values())
