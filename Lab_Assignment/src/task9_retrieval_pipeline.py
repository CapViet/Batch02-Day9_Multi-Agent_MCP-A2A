"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5

# Dùng MMR (xem lý do chọn ở src/task7_reranking.py) — chạy hoàn toàn local,
# không cần tải cross-encoder ~500MB hay API key, vẫn cải thiện chất lượng
# bằng cách giảm trùng lặp giữa các chunk gần giống nhau sau khi merge.
RERANK_METHOD = "mmr"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Bước 1: Chạy semantic (dense) + lexical (sparse) search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Bước 2: Merge 2 ranked lists bằng RRF (key theo content — xem task7_reranking.rerank_rrf)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Bước 3: Rerank để chọn ra top_k tốt nhất (giảm trùng lặp — xem RERANK_METHOD)
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Bước 4: Nếu hybrid không tìm được gì đủ tốt (rỗng hoặc best score thấp)
    # → fallback sang PageIndex (vectorless, dựa trên cấu trúc tài liệu thay
    # vì similarity). Bọc try/except vì PageIndex cần API key + network —
    # không có/khi lỗi thì phải "fail gracefully", trả lại kết quả hybrid hiện có
    # thay vì crash toàn bộ pipeline.
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(f"  ⚠ Hybrid best score ({best_score:.3f}) < threshold ({score_threshold}) → fallback PageIndex")
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception as e:
            print(f"  ⚠ PageIndex fallback không khả dụng ({e}) → giữ kết quả hybrid")
            fallback = []

        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
