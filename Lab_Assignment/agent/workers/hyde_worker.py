"""Worker: HyDE (Hypothetical Document Embeddings) query expansion + retrieval."""

from .base_worker import BaseWorker, WorkerResult


class HyDEWorker(BaseWorker):
    """
    Generates a hypothetical Vietnamese document from the query, appends it
    to the query for richer embedding, then runs hybrid retrieval + rerank.
    Best for short or keyword-poor queries where direct retrieval struggles.
    """

    worker_id = "hyde"

    def run(self, query: str, context: dict) -> WorkerResult:
        top_k = context.get("top_k", 5)
        try:
            from src.bonus_hyde import hyde_search
            results = hyde_search(query, top_k=top_k)
            return WorkerResult(self.worker_id, "success", results)
        except Exception as exc:
            return WorkerResult(self.worker_id, "failed", [], str(exc))
