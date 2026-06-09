"""Worker: dense semantic retrieval via ChromaDB + sentence-transformers."""

from .base_worker import BaseWorker, WorkerResult


class SemanticWorker(BaseWorker):
    worker_id = "semantic"

    def run(self, query: str, context: dict) -> WorkerResult:
        top_k = context.get("top_k", 5)
        try:
            from src.task5_semantic_search import semantic_search
            results = semantic_search(query, top_k=top_k * 2)
            return WorkerResult(self.worker_id, "success", results)
        except Exception as exc:
            return WorkerResult(self.worker_id, "failed", [], str(exc))
