"""
Shared local tokenization helper cho evaluation pipeline.

Cùng quy ước với `_tokenize` ở Task 6 (lowercase + whitespace split — tiếng
Việt được viết cách nhau bởi khoảng trắng ở cấp âm tiết, baseline đơn giản và
đủ dùng cho overlap-based metrics ở evaluation, không cần thêm dependency
nặng như underthesea).
"""


def tokenize(text: str) -> list[str]:
    return text.lower().split()
