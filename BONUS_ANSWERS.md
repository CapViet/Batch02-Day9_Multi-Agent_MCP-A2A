# Bài Tập Cộng Điểm — Trả Lời

---

## 1. Vite HTML Demo

File: `demo.html`

Mở trực tiếp trong trình duyệt để demo Stage 4 (không cần API key).  
Chuyển sang tab **Stage 5** khi hệ thống A2A đang chạy (`.\start_all.ps1`).

---

## 2. Latency của hệ thống Stage 5

### Pipeline và thời gian từng bước (đo thực tế với Gemini 2.5 Flash)

```
User
 └─► Customer Agent   (1 LLM call)   ~2-3s
      └─► Law Agent
           ├─ analyze_law    (1 LLM call)   6.15s   ← sequential
           ├─ check_routing  (keyword)      <1ms    ← optimized, no LLM
           ├─ Tax Agent      (1 LLM call)   4.15s   ┐ parallel
           ├─ Compliance     (1 LLM call)   2.92s   ┘ (same time slot)
           └─ aggregate      (1 LLM call)   3.86s   ← sequential
 └─► Customer formats response             ~2-3s
```

**Tổng latency đo thực tế (test_client.py): 87.72 giây**

> Đây là thời gian end-to-end toàn bộ hệ thống A2A distributed (5 HTTP hops).  
> Pipeline nội bộ của law_agent (không tính network overhead) mất ~14s với keyword routing.

---

## 3. Đề Xuất Giảm Latency

### Phương án: Thay LLM routing bằng keyword matching

**Vấn đề:** Node `check_routing` trong `law_agent/graph.py` ban đầu gọi một LLM riêng chỉ để quyết định có cần Tax Agent và Compliance Agent không. Với mô hình thinking như Gemini 2.5 Flash, đây là **34.91 giây** lãng phí trên critical path.

**Giải pháp:** Thay bằng keyword matching tức thì (< 1ms):

```python
# BEFORE — 1 extra LLM call (34.91s với Gemini 2.5 Flash thinking model)
result = await llm.ainvoke([SystemMessage(...), HumanMessage(question)])
parsed = json.loads(result.content)

# AFTER — instant keyword match (<1ms)
needs_tax = any(kw in question.lower() for kw in ["tax", "irs", "thuế", "evasion", ...])
needs_compliance = any(kw in question.lower() for kw in ["compliance", "sec", "sox", ...])
```

**Kết quả đo thực tế (`bonus_latency_demo.py` với Gemini 2.5 Flash):**

| | Version A (LLM routing) | Version B (Keyword routing) |
|---|---|---|
| analyze_law | 6.15s | ~6.15s |
| check_routing | **34.91s** ← LLM thinking | **<0.001s** ← instant |
| tax + compliance (parallel) | 4.15s | ~4.15s |
| aggregate | 3.86s | ~3.86s |
| **Tổng** | **49.09s** (đo thực tế) | **~14.2s** (ước tính) |
| LLM calls | 6 | 5 |
| Cải thiện | — | **~71% nhanh hơn** |

> **Ghi chú:** Version B không đo được trực tiếp do rate limit API (20 req/day free tier).  
> Số ~14.2s tính từ: 6.15 + 0.001 + max(4.15, 2.92) + 3.86 = 14.16s.

### Demo so sánh trực tiếp

```powershell
uv run python bonus_latency_demo.py
```

Script chạy cả 2 phiên bản (LLM routing vs keyword routing) liên tiếp và in ra:

```
Version A (LLM routing):      49.09s
Version B (keyword routing):  ~14.16s
Time saved:                   ~34.93s  (~71% faster)
```

### Tại sao tiết kiệm nhiều hơn dự kiến?

Mô hình **thinking** (như Gemini 2.5 Flash, Claude 3.7 Sonnet với extended thinking) có thêm bước "suy nghĩ" nội bộ trước khi trả lời. Với câu hỏi routing đơn giản như "có cần tax agent không?", mô hình vẫn tốn thời gian thinking (~30s) dù câu trả lời chỉ là `{"needs_tax": true}`. Keyword matching bypass hoàn toàn bước này.

### Các tối ưu đã áp dụng

1. **`law_agent/graph.py`** — `check_routing` dùng keyword matching thay vì LLM call
2. **`tax_agent/graph.py`** — System prompt rút gọn (< 150 từ) → giảm token generation time
3. **`stages/stage_4_milti_agent/main.py`** — Tương tự, keyword routing thay LLM routing
4. **Tax + Compliance agents chạy song song** (LangGraph `Send` API) — không tăng thêm latency khi thêm agent
