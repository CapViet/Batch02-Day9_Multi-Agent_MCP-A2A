"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Văn bản pháp luật về ma tuý — nguồn chính thống (cổng thông tin Chính phủ /
# trường đại học công lập), tải trực tiếp dạng PDF gốc.
DOCUMENTS = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma túy 2021 (Luật số 73/2021/QH14)",
        "url": "https://www.sgu.edu.vn/wp-content/uploads/2024/07/8.-Luat-Phong-chong-ma-tuy-nam-2021.pdf",
    },
    {
        "filename": "bo-luat-hinh-su-2015-sua-doi-2017.pdf",
        "title": "Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017) — văn bản hợp nhất, "
                 "Chương XX quy định các tội phạm về ma túy (Điều 247-259)",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
    },
    {
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
        "title": "Nghị định 57/2022/NĐ-CP quy định các danh mục chất ma túy và tiền chất",
        "url": (
            "https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY"
            "68Ekp0Gvvr483flbRJTjcXEdO2_Pu0JyTUYpTyPvovUt0rel_BnYVLCGNsBDgvqb3aPsQBXd_uoyKha7iidNiWTFTqwuUPbqh"
            "AyHavQ~~&file_name=2022_709+%2b+710_57-2022-N%c4%90-CP.pdf"
        ),
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str):
    """Tải 1 file PDF/DOCX về DATA_DIR (bỏ qua nếu đã tồn tại)."""
    filepath = DATA_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"  ↺ Đã có sẵn: {filepath.name} ({filepath.stat().st_size:,} bytes)")
        return

    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    print(f"  ✓ Đã tải: {filepath.name} ({filepath.stat().st_size:,} bytes)")


def collect_all():
    """Tải toàn bộ văn bản pháp luật trong DOCUMENTS."""
    setup_directory()
    for doc in DOCUMENTS:
        print(f"Đang tải: {doc['title']}")
        download_file(doc["url"], doc["filename"])


if __name__ == "__main__":
    collect_all()
