# Bài Tập Cộng Điểm — Trả Lời

---

## 1. Vite HTML Demo

File: `demo.html`

Mở trực tiếp trong trình duyệt để demo Stage 4 (không cần API key).  
Chuyển sang tab **Stage 5** khi hệ thống A2A đang chạy (`.\start_all.ps1`).

---

## 2. Latency của hệ thống Stage 5

### Pipeline và thời gian từng bước

```
User
 └─► Customer Agent   (1 LLM call)   ~2-3s
      └─► Law Agent
           ├─ analyze_law    (1 LLM call)   ~3-4s   ← sequential
           ├─ check_routing  (keyword)      <1ms    ← optimized, no LLM
           ├─ Tax Agent      (1 LLM call)   ~3-4s   ┐ parallel
           ├─ Compliance     (1 LLM call)   ~3-4s   ┘ (same time slot)
           └─ aggregate      (1 LLM call)   ~3-4s   ← sequential
 └─► Customer formats response             ~2-3s
```

**Tổng latency ước tính: ~13–18 giây**

> Số chính xác được in ra khi chạy:
> ```
> uv run python test_client.py
> ⏱  End-to-end latency: X.XXs
> ```

---

## 3. Đề Xuất Giảm Latency

### Phương án: Thay LLM routing bằng keyword matching

**Vấn đề:** Node `check_routing` trong `law_agent/graph.py` ban đầu gọi một LLM riêng chỉ để quyết định có cần Tax Agent và Compliance Agent không. Đây là 1 LLM call không cần thiết trên critical path (~2-3 giây).

**Giải pháp:** Thay bằng keyword matching tức thì (< 1ms):

```python
# BEFORE — 1 extra LLM call (~2-3s)
result = await llm.ainvoke([SystemMessage(...), HumanMessage(question)])
parsed = json.loads(result.content)

# AFTER — instant keyword match (<1ms)
needs_tax = any(kw in question.lower() for kw in ["tax", "irs", "thuế", "evasion", ...])
needs_compliance = any(kw in question.lower() for kw in ["compliance", "sec", "sox", ...])
```

**Kết quả:**

| | Trước | Sau |
|---|---|---|
| LLM calls trong pipeline | 6 | 5 |
| check_routing time | ~2-3s | <1ms |
| Tổng latency ước tính | ~18s | ~15s |
| Cải thiện | — | ~17% nhanh hơn |

### Demo so sánh trực tiếp

```powershell
uv run python bonus_latency_demo.py
```

Script chạy cả 2 phiên bản (LLM routing vs keyword routing) liên tiếp và in ra:

```
Version A (LLM routing):      18.XX s
Version B (keyword routing):  15.XX s
Time saved:                    3.XX s  (~17% faster)
```

### Các tối ưu đã áp dụng

1. **`law_agent/graph.py`** — `check_routing` dùng keyword matching thay vì LLM call
2. **`tax_agent/graph.py`** — System prompt rút gọn (< 150 từ) → giảm token generation time
3. **`stages/stage_4_milti_agent/main.py`** — Tương tự, keyword routing thay LLM routing
4. **Tax + Compliance agents chạy song song** (LangGraph `Send` API) — không tăng thêm latency khi thêm agent
