"""Worker: sparse BM25 lexical retrieval."""

from .base_worker import BaseWorker, WorkerResult


class LexicalWorker(BaseWorker):
    worker_id = "lexical"

    def run(self, query: str, context: dict) -> WorkerResult:
        top_k = context.get("top_k", 5)
        try:
            from src.task6_lexical_search import lexical_search
            results = lexical_search(query, top_k=top_k * 2)
            return WorkerResult(self.worker_id, "success", results)
        except Exception as exc:
            return WorkerResult(self.worker_id, "failed", [], str(exc))
