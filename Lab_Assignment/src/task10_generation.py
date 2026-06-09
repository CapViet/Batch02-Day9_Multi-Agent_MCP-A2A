"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re

os.environ.setdefault("USE_TF", "0")  # tránh transformers cố import TensorFlow/Keras3

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM: ưu tiên Ollama (local, free, không cần API key) — model qwen2.5:3b
# đã có sẵn trên máy, đủ nhanh trên CPU và hỗ trợ tiếng Việt tốt cho tác vụ
# trả lời có trích dẫn. Nếu Ollama không khả dụng, fallback sang OpenAI nếu
# user cung cấp OPENAI_API_KEY trong .env (xem _call_llm()).
OLLAMA_MODEL = "qwen2.5:3b"
OPENAI_MODEL = "gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # chunks[0::2] = rank 1, 3, 5, ... (vị trí lẻ theo thứ hạng — giữ nguyên thứ tự)
    # chunks[1::2] = rank 2, 4, 6, ... (vị trí chẵn — đảo ngược để cái "nhì" nằm cuối cùng)
    # [1,2,3,4,5] -> evens=[1,3,5], odds=[2,4] -> [1,3,5] + [4,2] = [1,3,5,4,2]
    # → #1 (best) ở đầu, #2 (nhì) ở cuối — 2 vị trí LLM nhớ tốt nhất.
    odd_ranked = chunks[0::2]
    even_ranked = chunks[1::2]
    return odd_ranked + even_ranked[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# LLM CALL — local-first (Ollama) với fallback OpenAI
# =============================================================================

def _call_llm(system_prompt: str, user_message: str) -> str:
    """
    Gọi LLM để sinh câu trả lời. Ưu tiên Ollama (local, free, model đã có
    sẵn trên máy — xem OLLAMA_MODEL); nếu Ollama không chạy được (chưa cài/
    chưa start service) thì fallback sang OpenAI nếu user có OPENAI_API_KEY
    trong .env. Nếu cả 2 đều không khả dụng → raise lỗi rõ ràng để
    generate_with_citation() / test có thể bắt và skip một cách an toàn.
    """
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": TEMPERATURE, "top_p": TOP_P},
        )
        return response["message"]["content"]
    except Exception as ollama_err:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"Không gọi được Ollama ({ollama_err}) và cũng không có "
                f"OPENAI_API_KEY trong .env để fallback"
            )
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content


# =============================================================================
# OFFLINE ANSWER — extractive fallback, không gọi LLM
# =============================================================================
#
# Dùng khi không có Ollama/OpenAI key (máy demo của thành viên khác, CI, môi
# trường eval...). Thay vì sinh câu trả lời tự nhiên, hàm này trích câu liên
# quan nhất tới câu hỏi trong mỗi chunk (overlap từ vựng đơn giản) rồi gắn
# citation [Source, Type] đúng format Task 10 yêu cầu. Chất lượng thấp hơn
# LLM thật (câu cụt, ít tự nhiên) nên chỉ là fallback offline, không thay thế
# generate_with_citation().

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _offline_answer(query: str, chunks: list[dict]) -> str:
    """
    Sinh câu trả lời "offline" bằng cách trích 1 câu liên quan nhất từ mỗi
    chunk (theo overlap từ vựng với câu hỏi) và nối lại kèm citation.

    Args:
        query: Câu hỏi của user
        chunks: List of {'content': str, 'metadata': dict, ...} đã reorder

    Returns:
        Chuỗi câu trả lời có citation [Source, Type], hoặc thông báo
        "không thể xác minh" nếu không trích được câu nào phù hợp.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_terms = set(query.lower().split())
    cited_sentences = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Nguồn không rõ")
        doc_type = metadata.get("type", "unknown")

        best_sentence, best_overlap = "", -1
        for raw_sentence in _SENTENCE_SPLIT_RE.split(chunk.get("content", "")):
            sentence = raw_sentence.strip()
            if len(sentence) < 20:
                continue
            overlap = len(query_terms & set(sentence.lower().split()))
            if overlap > best_overlap:
                best_sentence, best_overlap = sentence, overlap

        if best_sentence:
            cited_sentences.append(f"{best_sentence} [{source}, {doc_type}]")

    if not cited_sentences:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return " ".join(cited_sentences)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Bước 1: Retrieve (hybrid semantic+lexical+rerank, fallback PageIndex nếu cần)
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Bước 2: Reorder để tránh "lost in the middle"
    reordered = reorder_for_llm(chunks)

    # Bước 3: Format context kèm source label để LLM trích dẫn được
    context = format_context(reordered)

    # Bước 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Bước 5: Gọi LLM (local-first qua Ollama, fallback OpenAI — xem _call_llm)
    answer = _call_llm(SYSTEM_PROMPT, user_message)

    # Bước 6: Trả về answer kèm sources để hiển thị/kiểm chứng citation
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
