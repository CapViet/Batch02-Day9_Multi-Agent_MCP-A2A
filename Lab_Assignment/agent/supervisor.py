"""
RAG Supervisor Agent — orchestrates specialized worker agents.

Pattern
-------
Supervisor/Worker is a multi-agent design where a single orchestrator
(the Supervisor) plans work, dispatches it to self-contained workers,
evaluates intermediate results, and decides whether to re-route or fall
back before assembling the final response.

Decision flow
-------------
1. Classify query  → select retrieval strategy
2. Dispatch retrieval workers (parallel or sequential per strategy)
3. RRF-merge + MMR-rerank via RerankWorker
4. Assess confidence → trigger FallbackWorker if score < threshold
5. Generate answer with citation via GenerationWorker
6. Return result + full execution log for observability

Strategies
----------
"hybrid"       (default) semantic + lexical → RRF → rerank
"lexical_first"          lexical + semantic → RRF → rerank  (named entities)
"hyde"                   HyDE expansion → merged search     (vague/short queries)
"""

from __future__ import annotations

import time
from typing import Any

from .workers.base_worker import WorkerResult
from .workers.fallback_worker import FallbackWorker
from .workers.generation_worker import GenerationWorker
from .workers.hyde_worker import HyDEWorker
from .workers.lexical_worker import LexicalWorker
from .workers.rerank_worker import RerankWorker
from .workers.semantic_worker import SemanticWorker


# ── Routing heuristics ──────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.3  # below this → trigger fallback worker

# Query routing signals
_NEWS_KEYWORDS = frozenset({
    "nghệ sĩ", "ca sĩ", "diễn viên", "người mẫu", "showbiz",
    "bị bắt", "tạm giữ", "truy tố", "sao việt",
})
_LEGAL_KEYWORDS = frozenset({
    "điều", "khoản", "luật", "nghị định", "hình phạt",
    "bộ luật", "quy định", "cai nghiện",
})
_VAGUE_WORD_LIMIT = 6  # queries with ≤ N words and no domain signals → HyDE


# ── Supervisor ───────────────────────────────────────────────────────────────

class RAGSupervisor:
    """
    Supervisor agent that plans and orchestrates RAG worker agents.

    Usage::

        supervisor = RAGSupervisor()

        # Full pipeline (retrieve + generate)
        result = supervisor.run("Hình phạt cho tội tàng trữ ma túy?", top_k=5)
        print(result["answer"])
        print(result["execution_log"])

        # Retrieval only
        chunks = supervisor.search("luật phòng chống ma túy 2021", top_k=5)
    """

    def __init__(self) -> None:
        self._workers: dict[str, Any] = {
            "semantic":   SemanticWorker(),
            "lexical":    LexicalWorker(),
            "rerank":     RerankWorker(),
            "fallback":   FallbackWorker(),
            "hyde":       HyDEWorker(),
            "generation": GenerationWorker(),
        }
        self._log: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str, top_k: int = 5) -> dict:
        """
        Run the full RAG pipeline under supervisor control.

        Returns
        -------
        dict with keys:
            answer        : str
            sources       : list[dict]   retrieved chunks used for generation
            strategy      : str          retrieval strategy chosen
            confidence    : float        best retrieval score (0–1)
            used_fallback : bool         whether PageIndex fallback was triggered
            used_llm      : bool         whether an LLM was called
            execution_log : list[dict]   step-by-step trace for observability
        """
        self._log = []
        context: dict[str, Any] = {"top_k": top_k}

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        strategy = self._classify_query(query)
        self._record("supervisor", "plan", {
            "query": query, "strategy": strategy, "top_k": top_k,
        })

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        chunks, confidence = self._retrieve(query, strategy, context)

        # ── Step 3: Evaluate confidence → fallback if needed ──────────────────
        used_fallback = False
        if confidence < CONFIDENCE_THRESHOLD:
            self._record("supervisor", "decision", {
                "action": "fallback",
                "reason": f"confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}",
            })
            fb_result = self._dispatch("fallback", query, context)
            if fb_result.status == "success" and fb_result.data:
                chunks = fb_result.data
                confidence = max((c.get("score", 0.0) for c in chunks), default=0.0)
                used_fallback = True

        # ── Step 4: Generate ──────────────────────────────────────────────────
        context["chunks"] = chunks
        gen_result = self._dispatch("generation", query, context)
        answer_data: dict = gen_result.data or {}

        self._record("supervisor", "done", {
            "chunks_used": len(chunks),
            "used_fallback": used_fallback,
            "used_llm": answer_data.get("used_llm", False),
        })

        return {
            "answer":        answer_data.get(
                "answer", "Tôi không thể xác minh thông tin này từ nguồn hiện có."
            ),
            "sources":       chunks,
            "strategy":      strategy,
            "confidence":    confidence,
            "used_fallback": used_fallback,
            "used_llm":      answer_data.get("used_llm", False),
            "execution_log": list(self._log),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieval-only path — no generation, returns ranked chunks."""
        self._log = []
        context: dict[str, Any] = {"top_k": top_k}
        strategy = self._classify_query(query)
        chunks, _ = self._retrieve(query, strategy, context)
        return chunks

    # ── Internal: routing ─────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """
        Route a query to the best retrieval strategy.

        Rules (first match wins):
        1. Short query with no domain keywords → "hyde"  (expand before retrieval)
        2. Contains news/celebrity keywords    → "lexical_first"  (BM25 for NEs)
        3. Everything else                     → "hybrid"  (balanced default)
        """
        lower = query.lower()
        words = lower.split()

        is_short = len(words) <= _VAGUE_WORD_LIMIT
        has_news_signal = any(kw in lower for kw in _NEWS_KEYWORDS)
        has_legal_signal = any(kw in lower for kw in _LEGAL_KEYWORDS)

        if is_short and not has_news_signal and not has_legal_signal:
            return "hyde"
        if has_news_signal:
            return "lexical_first"
        return "hybrid"

    # ── Internal: retrieval orchestration ─────────────────────────────────────

    def _retrieve(
        self, query: str, strategy: str, context: dict
    ) -> tuple[list[dict], float]:
        """Dispatch retrieval workers according to the chosen strategy."""

        if strategy == "hyde":
            result = self._dispatch("hyde", query, context)
            chunks = result.data if result.status == "success" else []

        elif strategy == "lexical_first":
            lex = self._dispatch("lexical", query, context)
            sem = self._dispatch("semantic", query, context)
            context["ranked_lists"] = [
                lex.data if lex.status == "success" else [],
                sem.data if sem.status == "success" else [],
            ]
            rr = self._dispatch("rerank", query, context)
            chunks = rr.data if rr.status == "success" else []

        else:  # "hybrid"
            sem = self._dispatch("semantic", query, context)
            lex = self._dispatch("lexical", query, context)
            context["ranked_lists"] = [
                sem.data if sem.status == "success" else [],
                lex.data if lex.status == "success" else [],
            ]
            rr = self._dispatch("rerank", query, context)
            chunks = rr.data if rr.status == "success" else []

        confidence = max((c.get("score", 0.0) for c in chunks), default=0.0)
        return chunks, confidence

    # ── Internal: dispatch + logging ──────────────────────────────────────────

    def _dispatch(self, worker_id: str, query: str, context: dict) -> WorkerResult:
        worker = self._workers[worker_id]
        t0 = time.monotonic()
        result = worker.run(query, context)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        self._record(worker_id, result.status, {
            "items": len(result.data) if isinstance(result.data, list) else None,
            "elapsed_ms": elapsed_ms,
            "error": result.error,
        })
        return result

    def _record(self, actor: str, event: str, payload: dict) -> None:
        self._log.append({"actor": actor, "event": event, **payload})

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"RAGSupervisor(workers={list(self._workers)})"
