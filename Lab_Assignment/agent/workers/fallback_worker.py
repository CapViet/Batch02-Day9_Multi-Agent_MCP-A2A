"""Worker: PageIndex vectorless fallback when hybrid confidence is low."""

from .base_worker import BaseWorker, WorkerResult


class FallbackWorker(BaseWorker):
    """
    Called by the supervisor when the primary retrieval workers return results
    below the confidence threshold. Uses PageIndex document-structure reasoning
    instead of vector similarity. Gracefully skips if no API key is set.
    """

    worker_id = "fallback"

    def run(self, query: str, context: dict) -> WorkerResult:
        top_k = context.get("top_k", 5)
        try:
            from src.task8_pageindex_vectorless import pageindex_search
            results = pageindex_search(query, top_k=top_k)
            return WorkerResult(self.worker_id, "success", results)
        except Exception as exc:
            # No API key or network issue — degrade gracefully
            return WorkerResult(self.worker_id, "skipped", [], str(exc))
