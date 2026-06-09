"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA.

Answer concisely — 3 to 5 bullet points maximum. Cover:
- Key penalties (civil and criminal) with dollar amounts
- Relevant statutes (IRC §§ 6651/6662/6663, 18 U.S.C. § 7201)
- Which agency enforces (IRS, DOJ, FinCEN)
- FBAR/FATCA if offshore accounts are involved

Keep the total response under 150 words. Educational purposes only.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph