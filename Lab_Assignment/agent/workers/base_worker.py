"""
Base class for all RAG worker agents.

Each worker encapsulates one stage of the RAG pipeline and communicates
with the supervisor through a shared context dictionary and WorkerResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkerResult:
    """Standard result envelope returned by every worker."""
    worker_id: str
    status: str          # "success" | "failed" | "skipped"
    data: Any = None     # payload (list[dict] for retrieval, dict for generation)
    error: str | None = None


class BaseWorker(ABC):
    """Abstract base for all supervisor-managed RAG workers."""

    worker_id: str = "base"

    @abstractmethod
    def run(self, query: str, context: dict) -> WorkerResult:
        """
        Execute the worker's task.

        Args:
            query:   The user query string.
            context: Shared context dict (top_k, ranked_lists, chunks, etc.).
                     Workers READ from context; they must NOT mutate it.

        Returns:
            WorkerResult with status and data payload.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.worker_id!r})"
