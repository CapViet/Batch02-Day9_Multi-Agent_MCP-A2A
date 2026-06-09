"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Tokenization: dùng lower().split() — tiếng Việt được viết cách nhau bởi
khoảng trắng ở cấp âm tiết, nên whitespace split là baseline đơn giản và
đủ hiệu quả cho BM25 (không cần thêm dependency nặng như underthesea).
"""

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .task4_chunking_indexing import chunk_documents, load_documents

# Corpus & index được build 1 lần, cache lại cho các lần gọi search sau.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_bm25_index: BM25Okapi | None = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase + whitespace split."""
    return text.lower().split()


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index (k1=1.5, b=0.75 — giá trị mặc định, phù hợp văn bản
        có độ dài không quá chênh lệch như luật/tin tức).
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


def _ensure_index():
    """Load corpus (cùng chunk với Task 4 để tương thích khi merge ở Task 9) & build index nếu chưa có."""
    global _bm25_index
    if _bm25_index is not None:
        return

    if not CORPUS:
        documents = load_documents()
        CORPUS.extend(chunk_documents(documents))

    _bm25_index = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()
    if not CORPUS:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


# =============================================================================
# Bonus — TF-IDF lexical search (khác BM25, +5 bonus nếu giải thích trong demo)
# =============================================================================
#
# So với BM25:
#   - TF-IDF: trọng số mỗi từ = tf * idf, tăng tuyến tính theo tần suất xuất
#     hiện; BM25 có thêm "term saturation" (k1) nên tần suất cao không làm
#     điểm tăng vô hạn, và "length normalization" (b) để văn bản dài không
#     bị lợi thế quá mức.
#   - Cách so khớp: TF-IDF biểu diễn cả query và document thành vector trong
#     không gian từ vựng rồi tính cosine similarity (góc giữa 2 vector); BM25
#     tính điểm trực tiếp theo công thức xác suất, không cần biểu diễn vector.
#   - Hệ quả: TF-IDF/cosine nhạy với độ dài vector (số từ khác nhau) hơn,
#     trong khi BM25 cân bằng tốt hơn giữa văn bản pháp luật dài và tin tức
#     ngắn nhờ length normalization.

def _ensure_tfidf_index():
    """Build TF-IDF index (cosine similarity) trên cùng corpus với BM25."""
    global _tfidf_vectorizer, _tfidf_matrix
    if _tfidf_vectorizer is not None:
        return

    _ensure_index()
    if not CORPUS:
        return

    # token_pattern=None vì đã truyền tokenizer riêng (tránh warning của sklearn)
    _tfidf_vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(doc["content"] for doc in CORPUS)


def tfidf_lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng TF-IDF + cosine similarity — minh hoạ lexical search
    khác BM25 (xem giải thích cơ chế ở trên, dùng cho demo +5 bonus).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # Cosine similarity giữa vector query và document
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_tfidf_index()
    if not CORPUS or _tfidf_vectorizer is None:
        return []

    query_vector = _tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, _tfidf_matrix)[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        if similarities[idx] <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(similarities[idx]),
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
