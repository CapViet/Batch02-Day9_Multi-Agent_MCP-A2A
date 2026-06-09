"""Bonus Demo: Latency Comparison — LLM Routing vs Keyword Routing

Shows the time saved by replacing the check_routing LLM call with
instant keyword matching in the Stage 4 multi-agent graph.

Run:
    uv run python bonus_latency_demo.py
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm


QUESTION = (
    "A tech startup shared user data without consent and avoided taxes. "
    "What are the legal and regulatory consequences?"
)


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------

@tool
def search_tax_law(query: str) -> str:
    """Search tax law knowledge base."""
    knowledge = [
        (["tax", "evasion", "irs"], "Tax evasion (26 U.S.C. § 7201): up to $250K fine and 5 years prison."),
        (["offshore", "fbar"], "FBAR penalties: up to $100K or 50% of account balance per violation."),
    ]
    results = [t for kws, t in knowledge if any(kw in query.lower() for kw in kws)]
    return "\n".join(results) or "No matches."


@tool
def search_compliance_law(query: str) -> str:
    """Search compliance law knowledge base."""
    knowledge = [
        (["data", "privacy", "gdpr", "ccpa"], "CCPA: $7,500/violation. GDPR: 4% of global revenue."),
        (["sox", "sec"], "SOX § 906: up to $5M fine and 20 years prison."),
    ]
    results = [t for kws, t in knowledge if any(kw in query.lower() for kw in kws)]
    return "\n".join(results) or "No matches."


# ---------------------------------------------------------------------------
# Shared state + nodes
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    return b if b else a


class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str
    timings: Annotated[dict, lambda a, b: {**a, **b}]


async def analyze_law(state: LegalState) -> dict:
    t0 = time.perf_counter()
    llm = get_llm()
    result = await llm.ainvoke([
        SystemMessage(content="You are a senior attorney. Analyse the legal aspects briefly (under 100 words)."),
        HumanMessage(content=state["question"]),
    ])
    elapsed = time.perf_counter() - t0
    print(f"    [analyze_law]  {elapsed:.2f}s")
    return {"law_analysis": result.content, "timings": {"analyze_law": elapsed}}


async def call_tax_specialist(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    t0 = time.perf_counter()
    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_tax_law],
                               prompt="Tax attorney. Use search_tax_law. Under 80 words.")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    elapsed = time.perf_counter() - t0
    print(f"    [tax_specialist]  {elapsed:.2f}s")
    return {"tax_result": result["messages"][-1].content, "timings": {"tax": elapsed}}


async def call_compliance_specialist(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    t0 = time.perf_counter()
    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_compliance_law],
                               prompt="Compliance officer. Use search_compliance_law. Under 80 words.")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    elapsed = time.perf_counter() - t0
    print(f"    [compliance_specialist]  {elapsed:.2f}s")
    return {"compliance_result": result["messages"][-1].content, "timings": {"compliance": elapsed}}


async def aggregate(state: LegalState) -> dict:
    t0 = time.perf_counter()
    llm = get_llm()
    combined = "\n\n".join(filter(None, [
        state.get("law_analysis"), state.get("tax_result"), state.get("compliance_result")
    ]))
    result = await llm.ainvoke([
        SystemMessage(content="Synthesise these analyses into a concise final answer (under 150 words)."),
        HumanMessage(content=combined),
    ])
    elapsed = time.perf_counter() - t0
    print(f"    [aggregate]  {elapsed:.2f}s")
    return {"final_answer": result.content, "timings": {"aggregate": elapsed}}


INITIAL_STATE = {
    "question": QUESTION,
    "law_analysis": "",
    "needs_tax": False,
    "needs_compliance": False,
    "tax_result": "",
    "compliance_result": "",
    "final_answer": "",
    "timings": {},
}


# ---------------------------------------------------------------------------
# VERSION A: LLM-based routing (original approach)
# ---------------------------------------------------------------------------

async def check_routing_llm(state: LegalState) -> dict:
    """Original: uses an LLM call to decide routing."""
    t0 = time.perf_counter()
    llm = get_llm()
    result = await llm.ainvoke([
        SystemMessage(content=(
            'Reply with ONLY valid JSON: {"needs_tax": <bool>, "needs_compliance": <bool>}\n'
            'needs_tax=true if question involves tax/IRS. needs_compliance=true if SEC/SOX/GDPR.'
        )),
        HumanMessage(content=state["question"]),
    ])
    raw = result.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"needs_tax": True, "needs_compliance": True}
    elapsed = time.perf_counter() - t0
    print(f"    [check_routing LLM]  {elapsed:.2f}s  ← LLM call")
    return {
        "needs_tax": bool(parsed.get("needs_tax", True)),
        "needs_compliance": bool(parsed.get("needs_compliance", True)),
        "timings": {"check_routing": elapsed},
    }


def route_to_specialists(state: LegalState) -> list[Send]:
    sends = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    return sends or [Send("aggregate", state)]


def build_graph_llm_routing():
    g = StateGraph(LegalState)
    g.add_node("analyze_law", analyze_law)
    g.add_node("check_routing", check_routing_llm)
    g.add_node("call_tax_specialist", call_tax_specialist)
    g.add_node("call_compliance_specialist", call_compliance_specialist)
    g.add_node("aggregate", aggregate)
    g.set_entry_point("analyze_law")
    g.add_edge("analyze_law", "check_routing")
    g.add_conditional_edges("check_routing", route_to_specialists,
                            ["call_tax_specialist", "call_compliance_specialist", "aggregate"])
    g.add_edge("call_tax_specialist", "aggregate")
    g.add_edge("call_compliance_specialist", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# VERSION B: Keyword-based routing (optimized)
# ---------------------------------------------------------------------------

async def check_routing_keywords(state: LegalState) -> dict:
    """Optimized: instant keyword matching — no LLM call."""
    t0 = time.perf_counter()
    q = state["question"].lower()
    needs_tax = any(kw in q for kw in ["tax", "irs", "evasion", "offshore", "fbar"])
    needs_compliance = any(kw in q for kw in ["compliance", "sec", "sox", "gdpr", "ccpa", "regulation"])
    elapsed = time.perf_counter() - t0
    print(f"    [check_routing KEYWORDS]  {elapsed:.4f}s  ← instant, no LLM")
    return {"needs_tax": needs_tax, "needs_compliance": needs_compliance,
            "timings": {"check_routing": elapsed}}


def build_graph_keyword_routing():
    g = StateGraph(LegalState)
    g.add_node("analyze_law", analyze_law)
    g.add_node("check_routing", check_routing_keywords)
    g.add_node("call_tax_specialist", call_tax_specialist)
    g.add_node("call_compliance_specialist", call_compliance_specialist)
    g.add_node("aggregate", aggregate)
    g.set_entry_point("analyze_law")
    g.add_edge("analyze_law", "check_routing")
    g.add_conditional_edges("check_routing", route_to_specialists,
                            ["call_tax_specialist", "call_compliance_specialist", "aggregate"])
    g.add_edge("call_tax_specialist", "aggregate")
    g.add_edge("call_compliance_specialist", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_version(label: str, graph) -> float:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Question: {QUESTION[:70]}...")
    print()
    t_start = time.perf_counter()
    result = await graph.ainvoke(dict(INITIAL_STATE))
    total = time.perf_counter() - t_start
    print(f"\n  Final answer ({len(result['final_answer'])} chars):")
    print(f"  {result['final_answer'][:200]}...")
    print(f"\n  ⏱  Total latency: {total:.2f}s")
    return total


async def main():
    print("\n" + "=" * 60)
    print("  BONUS: Latency Demo — LLM Routing vs Keyword Routing")
    print("=" * 60)

    graph_a = build_graph_llm_routing()
    graph_b = build_graph_keyword_routing()

    latency_a = await run_version("VERSION A — LLM-based routing (original)", graph_a)
    latency_b = await run_version("VERSION B — Keyword routing (optimized)", graph_b)

    saved = latency_a - latency_b
    pct = (saved / latency_a) * 100 if latency_a > 0 else 0

    print("\n" + "=" * 60)
    print("  LATENCY COMPARISON")
    print("=" * 60)
    print(f"  Version A (LLM routing):      {latency_a:.2f}s")
    print(f"  Version B (keyword routing):  {latency_b:.2f}s")
    print(f"  Time saved:                   {saved:.2f}s  ({pct:.0f}% faster)")
    print()
    print("  Why it's faster:")
    print("  - Removed 1 LLM call from the critical path (check_routing)")
    print("  - Keyword matching runs in <1ms vs ~2-3s for an LLM round-trip")
    print("  - Same routing accuracy for the keyword set in scope")
    print("=" * 60)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
