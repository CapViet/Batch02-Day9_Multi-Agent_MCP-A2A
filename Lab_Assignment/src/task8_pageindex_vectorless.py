"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
API docs: https://docs.pageindex.ai/api-reference

PageIndex cho phép RAG mà không cần vector store — model "reasoning" trực
tiếp trên cấu trúc cây (mục lục / heading) của tài liệu để chọn ra (các)
node liên quan nhất, thay vì so khớp embedding. Module này gọi thẳng REST
API (base url https://api.pageindex.ai, header xác thực `api_key: <KEY>`)
bằng `requests` — không cần cài thêm SDK riêng.

Quy trình:
    1. upload_documents(): API chỉ nhận file PDF ("Only PDF files are
       supported" — xác nhận bằng cách gọi thử). Vì vậy:
         - Văn bản luật: dùng thẳng PDF gốc trong data/landing/legal/
         - Tin tức: chỉ có sẵn dạng .md → render sang PDF tạm bằng fpdf2
           (cache tại data/.pageindex_pdfs/, idempotent)
       rồi POST /doc/ (multipart, field "file") → nhận về doc_id, cache lại
       vào data/.pageindex_doc_ids.json (idempotent — không upload lại).
    2. pageindex_search(): POST /retrieval/ với {doc_id, query} → poll
       GET /retrieval/{id}/ tới khi status="completed" → trả về
       retrieved_nodes (mỗi node có các đoạn relevant_contents liên quan).

Yêu cầu PAGEINDEX_API_KEY trong file .env (đăng ký tại pageindex.ai để lấy
key). Nếu chưa có key, các hàm sẽ raise lỗi rõ ràng — Task 9 sẽ bắt lỗi này
và bỏ qua PageIndex fallback một cách an toàn (graceful degradation).
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_BASE_URL = "https://api.pageindex.ai"

DATA_DIR = Path(__file__).parent.parent / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
DOC_ID_CACHE_PATH = DATA_DIR / ".pageindex_doc_ids.json"

# PageIndex chỉ nhận PDF — tin tức chỉ có sẵn dạng .md nên cần render tạm
# sang PDF (cache lại, idempotent) trước khi upload. Dùng font Arial có sẵn
# trên Windows vì nó hỗ trợ đầy đủ dấu tiếng Việt (Latin Extended).
PDF_CACHE_DIR = DATA_DIR / ".pageindex_pdfs"
UNICODE_FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def _markdown_to_pdf(md_path: Path, pdf_path: Path):
    """Render 1 file markdown thành PDF đơn giản (idempotent) để upload lên PageIndex."""
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:
        return

    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")
    plain_text = re.sub(r"[#*_`>\[\]()-]", "", text)

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", "", UNICODE_FONT_PATH)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, plain_text)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def _collect_upload_files() -> list[Path]:
    """
    Gom danh sách file PDF cần upload:
      - Văn bản luật: dùng PDF gốc trong data/landing/legal/
      - Tin tức: render .md (data/standardized/news/) sang PDF tạm, cache lại
    """
    files = []

    legal_dir = LANDING_DIR / "legal"
    for pdf_file in sorted(legal_dir.glob("*.pdf")):
        files.append(pdf_file)

    news_dir = STANDARDIZED_DIR / "news"
    for md_file in sorted(news_dir.glob("*.md")):
        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        _markdown_to_pdf(md_file, pdf_path)
        files.append(pdf_path)

    return files


def _headers() -> dict:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được set trong .env — đăng ký tại https://pageindex.ai/ để lấy API key"
        )
    return {"api_key": PAGEINDEX_API_KEY}


def _load_doc_ids() -> dict:
    if DOC_ID_CACHE_PATH.exists():
        return json.loads(DOC_ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_ids(doc_ids: dict):
    DOC_ID_CACHE_PATH.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> dict:
    """
    Upload toàn bộ documents lên PageIndex (xem _collect_upload_files — API
    chỉ nhận PDF nên tin tức được render tạm từ .md sang PDF).

    Idempotent: doc_id của mỗi file được cache vào data/.pageindex_doc_ids.json
    (key = tên file) — chạy lại sẽ bỏ qua các file đã upload thay vì tốn quota.

    Returns:
        dict {filename: doc_id}
    """
    doc_ids = _load_doc_ids()

    for pdf_file in _collect_upload_files():
        if pdf_file.name in doc_ids:
            print(f"↺ Đã upload trước đó: {pdf_file.name} -> {doc_ids[pdf_file.name]}")
            continue

        print(f"Uploading: {pdf_file.name}")
        with open(pdf_file, "rb") as f:
            resp = requests.post(
                f"{PAGEINDEX_BASE_URL}/doc/",
                headers=_headers(),
                files={"file": (pdf_file.name, f, "application/pdf")},
            )
        resp.raise_for_status()
        doc_id = resp.json()["doc_id"]
        doc_ids[pdf_file.name] = doc_id
        _save_doc_ids(doc_ids)
        print(f"  ✓ Uploaded: {pdf_file.name} -> doc_id={doc_id}")

    return doc_ids


def _wait_for_retrieval(retrieval_id: str, timeout: float = 120.0, interval: float = 3.0) -> list:
    """Poll GET /retrieval/{id}/ tới khi status='completed' hoặc hết timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{PAGEINDEX_BASE_URL}/retrieval/{retrieval_id}/", headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "completed":
            return data.get("retrieved_nodes", [])
        time.sleep(interval)
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    PageIndex không trả similarity score (vì không dùng vector) — model
    "reasoning" chọn trực tiếp các node liên quan nhất theo cấu trúc cây
    của tài liệu rồi trả về theo thứ tự liên quan giảm dần. Ta gán
    score = 1/(rank+1) để giữ tương thích với format {'content','score',...}
    dùng chung trong pipeline (Task 9 merge theo score).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    doc_ids = _load_doc_ids()
    if not doc_ids:
        doc_ids = upload_documents()
    if not doc_ids:
        return []

    results = []
    for filename, doc_id in doc_ids.items():
        submit = requests.post(
            f"{PAGEINDEX_BASE_URL}/retrieval/",
            headers=_headers(),
            json={"doc_id": doc_id, "query": query, "thinking": False},
        )
        submit.raise_for_status()
        retrieval_id = submit.json()["retrieval_id"]

        nodes = _wait_for_retrieval(retrieval_id)
        for rank, node in enumerate(nodes):
            chunks = node.get("relevant_contents") or [{"relevant_content": node.get("text", "")}]
            for chunk in chunks:
                content = chunk.get("relevant_content", "").strip()
                if not content:
                    continue
                results.append({
                    "content": content,
                    "score": 1.0 / (rank + 1),
                    "metadata": {
                        "source": filename,
                        "node_id": node.get("node_id"),
                        "title": node.get("title"),
                        "page_index": chunk.get("page_index"),
                    },
                    "source": "pageindex",
                })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
