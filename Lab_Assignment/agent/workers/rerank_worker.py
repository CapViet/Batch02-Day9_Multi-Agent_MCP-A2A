"""Worker: RRF merge + MMR reranking over multiple ranked lists."""

from .base_worker import BaseWorker, WorkerResult


class RerankWorker(BaseWorker):
    """
    Expects context["ranked_lists"]: list of list[dict] from upstream workers.
    Runs RRF to merge them, then MMR to select the final top_k.
    """

    worker_id = "rerank"

    def run(self, query: str, context: dict) -> WorkerResult:
        top_k = context.get("top_k", 5)
        ranked_lists: list[list[dict]] = context.get("ranked_lists", [])

        non_empty = [lst for lst in ranked_lists if lst]
        if not non_empty:
            return WorkerResult(self.worker_id, "skipped", [])

        try:
            from src.task7_reranking import rerank, rerank_rrf
            merged = rerank_rrf(non_empty, top_k=top_k * 2)
            for item in merged:
                item.setdefault("source", "hybrid")
            reranked = rerank(query, merged, top_k=top_k)
            for item in reranked:
                item.setdefault("source", "hybrid")
            return WorkerResult(self.worker_id, "success", reranked)
        except Exception as exc:
            flat = [item for lst in non_empty for item in lst]
            return WorkerResult(self.worker_id, "failed", flat[:top_k], str(exc))
