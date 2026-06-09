# Lab Solutions — Multi-Agent MCP A2A

Giải đáp chi tiết cho tất cả bài tập trong codelab.

---

## Cấu Hình Môi Trường (Setup)

Codelab gốc dùng OpenRouter. Nếu bạn dùng **Gemini API key**, cập nhật như sau:

**`common/llm.py`:**
```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        openai_api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
    )
```

**`.env`:**
```
GOOGLE_API_KEY=AIza...your_key_here
GOOGLE_MODEL=gemini-2.5-flash
REGISTRY_URL=http://localhost:10000
```

> **Lưu ý model:** `gemini-2.0-flash` free tier chỉ ~1500 req/ngày. `gemini-2.5-flash` có quota riêng. Dùng `gemini-2.5-flash` để tránh rate limit.

---

## Stage 1: Direct LLM — Bài Tập 1.1 và 1.2

### Bài Tập 1.1: Thay đổi câu hỏi

Trong `stages/stage_1_direct_llm/main.py`, sửa biến `QUESTION`:

```python
# Ví dụ câu hỏi khác:
QUESTION = "Hợp đồng lao động có thể bị chấm dứt đơn phương không? Điều kiện là gì?"
# hoặc
QUESTION = "What are the penalties for insider trading under SEC regulations?"
```

Chạy lại:
```powershell
.\.venv\Scripts\python.exe stages/stage_1_direct_llm/main.py
```

### Bài Tập 1.2: Thêm temperature control

`temperature` đã được set trong `common/llm.py` (`temperature=0.3`). Để thay đổi:

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        openai_api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,   # 0.0 = deterministic, 1.0 = creative
    )
```

- `temperature=0.0`: câu trả lời ổn định, tái lập được
- `temperature=0.7`: đa dạng hơn, sáng tạo hơn
- `temperature=0.3`: cân bằng — phù hợp cho legal Q&A

---

## Stage 2: RAG + Tools — Bài Tập 2

### Bài Tập: Thêm entry vào LEGAL_KNOWLEDGE

Trong `stages/stage_2_rag_tools/main.py`, thêm vào list `LEGAL_KNOWLEDGE`:

```python
{
    "id": "gdpr_data_breach",
    "keywords": ["gdpr", "data breach", "notification", "personal data", "privacy"],
    "text": (
        "Under GDPR Article 33, data controllers must notify their supervisory authority "
        "of a personal data breach within 72 hours of becoming aware of it. "
        "Article 83 provides fines up to €20 million or 4% of annual global turnover "
        "for serious violations. Data processors must notify controllers without undue delay."
    ),
},
```

### Bài Tập: Thêm tool mới

Thêm tool `check_jurisdiction` vào danh sách tools:

```python
@tool
def check_jurisdiction(country: str) -> str:
    """Kiểm tra luật bảo vệ dữ liệu theo quốc gia."""
    laws = {
        "vietnam": "Luật An ninh mạng 2018, Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
        "eu": "GDPR (General Data Protection Regulation) — hiệu lực từ 2018",
        "usa": "CCPA (California), HIPAA (healthcare), FERPA (education) — không có luật liên bang thống nhất",
        "singapore": "PDPA (Personal Data Protection Act) 2012",
    }
    return laws.get(country.lower(), f"Chưa có thông tin cho {country}")
```

---

## Exercise 2: Tools và Knowledge Base

**File:** `exercises/exercise_2_tools.py`

### Task 1: Thêm entry luật lao động

Entry luật lao động đã được thêm vào `LEGAL_KNOWLEDGE`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
},
```

### Task 2: Tạo tool `check_statute_of_limitations`

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Tool này nhận `case_type` (string) và tra bảng thời hiệu. Cần đăng ký vào `tools` list:

```python
tools = [search_legal_knowledge, check_statute_of_limitations]
llm_with_tools = llm.bind_tools(tools)
```

### Task 3: Test

```powershell
.\.venv\Scripts\python.exe exercises/exercise_2_tools.py
```

Kết quả mong đợi:
```
Câu hỏi: Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?
🔧 Gọi tool: check_statute_of_limitations
✅ Kết quả: Thời hiệu khởi kiện cho vụ án hợp đồng là 4 năm theo UCC § 2-725.
```

**Tại sao LLM chọn đúng tool?** Vì tool description nói "kiểm tra thời hiệu" và câu hỏi hỏi về "thời hiệu". LLM dùng semantic matching để chọn tool phù hợp.

---

## Exercise 4: Multi-Agent với Privacy Agent

**File:** `exercises/exercise_4_multiagent.py`

### Task 1: Implement `privacy_agent`

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
Tập trung: GDPR, CCPA, data breach notification, data subject rights, fines."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

**Key insight:** Return key phải là `privacy_analysis` (khớp với field trong `State` TypedDict).

### Task 2: Thêm conditional routing

Trong hàm `check_routing`, thêm điều kiện cho privacy:

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))

    # Thêm routing cho privacy agent:
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "ccpa", "rò rỉ"]):
        tasks.append(Send("privacy_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]
```

**Tại sao dùng `Send()` thay vì gọi trực tiếp?** `Send()` cho phép LangGraph chạy các agents **song song**. Nếu câu hỏi cần cả tax + privacy, chúng chạy cùng lúc thay vì tuần tự.

### Task 3: Thêm vào graph

```python
def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)       # ← thêm node mới
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges("check_routing", lambda x: x)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")  # ← thêm edge mới
    graph.add_edge("aggregate_results", END)

    return graph.compile()
```

### Task 4: Test với câu hỏi data breach

```powershell
.\.venv\Scripts\python.exe exercises/exercise_4_multiagent.py
```

Câu hỏi test: `"Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?"`

Agents được gọi: `law_agent` → `privacy_agent` + `tax_agent` (song song) → `aggregate_results`

---

## Chạy Toàn Bộ Hệ Thống A2A (Stage 5)

```powershell
# Terminal 1: Registry
.\.venv\Scripts\python.exe -m registry

# Terminal 2: Leaf agents
.\.venv\Scripts\python.exe -m tax_agent
.\.venv\Scripts\python.exe -m compliance_agent

# Terminal 3: Orchestrators
.\.venv\Scripts\python.exe -m law_agent
.\.venv\Scripts\python.exe -m customer_agent

# Terminal 4: Test
.\.venv\Scripts\python.exe test_client.py
```

Hoặc dùng script tự động (Windows):
```powershell
# Chạy tất cả trong background
Start-Process .\.venv\Scripts\python.exe "-m registry"
Start-Sleep 2
Start-Process .\.venv\Scripts\python.exe "-m tax_agent"
Start-Process .\.venv\Scripts\python.exe "-m compliance_agent"
Start-Sleep 3
Start-Process .\.venv\Scripts\python.exe "-m law_agent"
Start-Sleep 3
Start-Process .\.venv\Scripts\python.exe "-m customer_agent"
```

---

## Bài Tập Nâng Cao

### Challenge 1: Financial Agent

Thêm agent phân tích thiệt hại tài chính vào hệ thống:

```python
def financial_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia tài chính và phân tích thiệt hại.

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', '')}

Ước tính thiệt hại tài chính, phí luật sư, và chi phí tuân thủ."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"financial_analysis": response.content}
```

Thêm vào `check_routing`:
```python
if any(kw in question_lower for kw in ["damages", "financial", "penalty", "fine", "thiệt hại"]):
    tasks.append(Send("financial_agent", state))
```

### Challenge 2: Conversation Memory

Dùng `ConversationBufferMemory` hoặc thread state của LangGraph:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = build_graph().compile(checkpointer=memory)

# Mỗi cuộc trò chuyện có thread_id riêng
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke({"question": "..."}, config=config)
```

### Challenge 3: Custom Tool gọi API thực

```python
import httpx

@tool
async def search_vietnamese_law(keyword: str) -> str:
    """Tra cứu văn bản pháp luật Việt Nam từ thuvienphapluat.vn."""
    # Đây là ví dụ minh họa — cần API key thực
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.thuvienphapluat.vn/search",
            params={"q": keyword, "lang": "vi"}
        )
    return resp.json().get("summary", "Không tìm thấy")
```

### Challenge 4: Error Handling + Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def robust_llm_call(llm, messages):
    """LLM call với tự động retry khi gặp rate limit."""
    try:
        return await llm.ainvoke(messages)
    except Exception as e:
        print(f"LLM call failed: {e}. Retrying...")
        raise
```

---

## Câu Hỏi Thường Gặp

**Q: `ModuleNotFoundError: No module named 'common'`**
A: Chạy từ thư mục root của project, hoặc thêm vào đầu file:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

**Q: Agent không call tool mà trả lời trực tiếp?**
A: Thêm vào system prompt: `"Always use the available tools before answering. Do not rely on training data alone."`

**Q: Gemini trả về JSON trong markdown code block (` ```json ... ``` `)?**
A: Strip trước khi parse:
```python
raw = result.content.strip().lstrip("```json").rstrip("```").strip()
parsed = json.loads(raw)
```

**Q: Rate limit 429 từ Gemini API?**
A: Free tier `gemini-2.5-flash` có giới hạn 20 req/ngày. Chờ hết ngày hoặc dùng key khác. Paid tier không có giới hạn này.

**Q: Agents không register vào Registry?**
A: Đảm bảo thứ tự khởi động: Registry trước, rồi leaf agents (tax, compliance), rồi orchestrators (law, customer). Thêm `Start-Sleep 3` giữa các bước.
