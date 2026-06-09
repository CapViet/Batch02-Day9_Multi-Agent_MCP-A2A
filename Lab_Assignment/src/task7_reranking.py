"""
Task 7 — Reranking Module.

Lựa chọn chính: MMR (Maximal Marginal Relevance), tự implement.
    Lý do: chạy hoàn toàn local (tái dùng embedding model ở Task 4, không cần
    tải thêm cross-encoder ~500MB hay gọi API trả phí), dễ giải thích cơ chế
    trong buổi demo, và đặc biệt hữu ích cho corpus của đề bài — nơi nhiều
    chunk pháp luật/tin tức có nội dung gần giống nhau (lặp lại các Điều luật,
    nhiều bài báo cùng nói về 1 vụ án) nên giảm trùng lặp (diversity) giúp
    người dùng thấy nhiều khía cạnh khác nhau hơn là 5 chunk gần như giống hệt.

Ngoài ra implement thêm:
    - RRF (Reciprocal Rank Fusion): dùng để merge kết quả dense + sparse ở Task 9
    - Cross-encoder (tuỳ chọn): dùng model multilingual cục bộ nếu có sẵn
"""

import os

os.environ.setdefault("USE_TF", "0")

import numpy as np


# =============================================================================
# Cross-encoder reranker (tuỳ chọn — cần tải model ~470MB lần đầu)
# =============================================================================

_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (multilingual, chạy local
    qua sentence-transformers.CrossEncoder — model nhận (query, passage) cùng
    lúc nên chấm điểm liên quan chính xác hơn embedding rời rạc).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    pairs = [(query, c["content"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    reranked = [
        {**c, "score": float(s)} for c, s in zip(candidates, scores)
    ]
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance (lựa chọn chính, tự implement)
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    λ=0.7: ưu tiên độ liên quan (relevance) hơn diversity nhưng vẫn đủ để
    loại các chunk gần như trùng lặp nội dung — phù hợp RAG vì câu trả lời
    cuối cùng cần đúng trọng tâm câu hỏi, diversity chỉ là yếu tố phụ.

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR, score = MMR score.
    """
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    relevance_cache = {
        idx: _cosine_sim(query_embedding, candidates[idx]["embedding"])
        for idx in remaining
    }
    mmr_scores: dict[int, float] = {}

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            relevance = relevance_cache[idx]

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        mmr_scores[best_idx] = best_score
        remaining.remove(best_idx)

    return [{**candidates[i], "score": mmr_scores[i]} for i in selected]


# =============================================================================
# RRF — Reciprocal Rank Fusion (dùng để merge dense + sparse ở Task 9)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    k=60: hằng số làm mượt (smoothing) lấy theo paper gốc Cormack et al.
    (2009) "Reciprocal Rank Fusion outperforms Condorcet and individual
    Rank Learning Methods" — giá trị này làm giảm ảnh hưởng của thứ hạng
    quá cao ở 1 ranker đơn lẻ, giúp kết quả merge công bằng giữa các ranker.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "mmr" | "cross_encoder" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Mặc định dùng MMR (xem giải thích lý do ở đầu file) — chạy local, không
    cần API key hay tải thêm model lớn, vẫn cải thiện chất lượng kết quả
    bằng cách giảm trùng lặp giữa các chunk liên quan.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval — cần có 'content';
            nếu đã có sẵn 'embedding' thì dùng luôn, nếu chưa sẽ tự embed
            bằng model ở Task 4 (đảm bảo cùng không gian vector).
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking — "mmr" (default), "cross_encoder", "rrf"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "mmr":
        from .task4_chunking_indexing import embed_texts

        query_embedding = embed_texts([query], is_query=True)[0]

        missing = [c for c in candidates if "embedding" not in c]
        if missing:
            embeddings = embed_texts([c["content"] for c in missing], is_query=False)
            for c, emb in zip(missing, embeddings):
                c["embedding"] = emb

        return rerank_mmr(query_embedding, candidates, top_k=top_k)

    if method == "rrf":
        # RRF gộp NHIỀU ranked lists — gọi rerank_rrf([list1, list2, ...]) trực tiếp.
        return rerank_rrf([candidates], top_k=top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
