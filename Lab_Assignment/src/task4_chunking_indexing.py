"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn & lý do (xem CONFIGURATION bên dưới):
    - Chunking: RecursiveCharacterTextSplitter, chunk_size=800, overlap=120
    - Embedding: intfloat/multilingual-e5-small (384 dim, multilingual, chạy local)
    - Vector store: ChromaDB (local, persistent, không cần Docker/Cloud)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import os

os.environ.setdefault("USE_TF", "0")  # tránh transformers cố import TensorFlow/Keras3

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / ".chroma"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter — an toàn, tách theo cấu trúc
# (đoạn văn → câu → từ) nên giữ được ngữ cảnh pháp lý (mỗi "Điều" thường
# nằm gọn trong 1-2 chunk thay vì bị cắt giữa câu).
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# CHUNK_SIZE = 800 ký tự: đủ dài để chứa trọn 1 "Điều" luật ngắn hoặc 1 đoạn
# tin tức hoàn chỉnh (giữ ngữ nghĩa), nhưng không quá dài để embedding bị loãng
# (semantic search kém chính xác khi chunk quá to).
CHUNK_SIZE = 800

# CHUNK_OVERLAP = 120 (15% của CHUNK_SIZE): đủ để câu/ý ở ranh giới 2 chunk
# không bị mất ngữ cảnh khi 1 câu bị cắt ngang, mà không tạo quá nhiều trùng lặp.
CHUNK_OVERLAP = 120

# Embedding model: intfloat/multilingual-e5-small
#   - Multilingual, được train cho retrieval, có hỗ trợ tiếng Việt tốt
#   - 384 dimensions → nhẹ, encode nhanh, chạy tốt trên CPU/GPU cá nhân
#   - Cần convention "query: "/"passage: " prefix khi encode (đặc thù họ E5)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Vector store: ChromaDB — chạy local (persistent trên disk), không cần
# Docker/Weaviate Cloud, đơn giản để index & query lại nhiều lần.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# SHARED HELPERS — dùng chung cho Task 4/5 (cùng 1 embedding model & collection)
# =============================================================================

_embedding_model = None


def get_embedding_model():
    """Trả về (cached) SentenceTransformer instance của EMBEDDING_MODEL."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Embed một danh sách text bằng EMBEDDING_MODEL.

    Họ model E5 yêu cầu prefix "query: " cho câu truy vấn và "passage: "
    cho tài liệu được index — giúp model phân biệt vai trò 2 phía khi
    tính cosine similarity (theo khuyến nghị của tác giả model).
    """
    model = get_embedding_model()
    prefix = "query: " if is_query else "passage: "
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def get_chroma_collection():
    """Mở (hoặc tạo) ChromaDB collection lưu trên disk tại CHROMA_DIR."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter (xem giải thích ở CONFIGURATION).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ưu tiên tách theo cấu trúc văn bản pháp luật/tin tức tiếng Việt:
        # đoạn văn → dòng → câu → từ → ký tự (fallback cuối cùng)
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng EMBEDDING_MODEL (xem embed_texts()).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, is_query=False)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks (kèm embedding) vào ChromaDB collection (persistent, local).
    """
    collection = get_chroma_collection()

    ids, embeddings, documents_, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        ids.append(f"{meta['source']}::chunk-{meta['chunk_index']}::{i}")
        embeddings.append(chunk["embedding"])
        documents_.append(chunk["content"])
        metadatas.append({"source": meta["source"], "type": meta["type"], "chunk_index": meta["chunk_index"]})

    # Reset collection trước khi index lại để tránh trùng lặp khi chạy nhiều lần
    try:
        collection_client = collection._client
        collection_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_chroma_collection()

    # Chroma giới hạn batch size khi add — chia nhỏ để an toàn
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents_[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} (collection={COLLECTION_NAME})")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding']) if chunks else 0})")

    collection = index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store — collection count = {collection.count()}")


if __name__ == "__main__":
    run_pipeline()
