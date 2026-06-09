"""Worker: LLM answer generation with citation (Ollama → OpenAI fallback)."""

from .base_worker import BaseWorker, WorkerResult


class GenerationWorker(BaseWorker):
    """
    Reorders chunks to avoid the "lost in the middle" effect, formats a context
    block, and calls the LLM. If no LLM is available falls back to offline
    extractive answering so the pipeline never returns an empty response.
    """

    worker_id = "generation"

    def run(self, query: str, context: dict) -> WorkerResult:
        chunks: list[dict] = context.get("chunks", [])

        if not chunks:
            return WorkerResult(
                self.worker_id, "success",
                {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.", "used_llm": False},
            )

        from src.task10_generation import (
            _call_llm, _offline_answer, format_context, reorder_for_llm, SYSTEM_PROMPT,
        )

        reordered = reorder_for_llm(chunks)
        ctx_text = format_context(reordered)
        user_msg = f"Context:\n{ctx_text}\n\n---\n\nQuestion: {query}"

        try:
            answer = _call_llm(SYSTEM_PROMPT, user_msg)
            return WorkerResult(self.worker_id, "success", {"answer": answer, "used_llm": True})
        except Exception as exc:
            answer = _offline_answer(query, reordered)
            return WorkerResult(
                self.worker_id, "success",
                {"answer": answer, "used_llm": False, "llm_error": str(exc)},
            )
