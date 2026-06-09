"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .task4_chunking_indexing import embed_texts, get_chroma_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine) trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (cao hơn = liên quan hơn)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    # Bước 1: Embed query bằng cùng model + convention "query: " ở Task 4
    query_embedding = embed_texts([query], is_query=True)[0]

    # Bước 2: Query vector store (cosine distance, vì collection được tạo
    # với metadata hnsw:space="cosine")
    n_results = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Bước 3: Chroma trả "cosine distance" = 1 - cosine_similarity
    # → chuyển lại thành similarity (score cao hơn = liên quan hơn)
    output = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        output.append({
            "content": content,
            "score": 1.0 - distance,
            "metadata": dict(metadata),
        })

    output.sort(key=lambda r: r["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
