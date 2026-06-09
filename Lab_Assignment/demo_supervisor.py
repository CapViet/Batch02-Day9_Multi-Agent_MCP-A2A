"""
CLI demo: RAG Supervisor/Worker pattern.

Run from the lab_assignments/ directory:
    python demo_supervisor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure lab_assignments/ is on the path so both agent/ and src/ resolve
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.supervisor import RAGSupervisor


DEMO_QUERIES = [
    # Hybrid strategy: clear legal question with domain keywords
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    # Lexical-first strategy: named entities / news
    "Ca sĩ nào bị bắt vì sử dụng ma tuý năm 2024?",
    # HyDE strategy: short vague query
    "cai nghiện",
]


def _divider(title: str, char: str = "─", width: int = 70) -> None:
    pad = max(0, width - len(title) - 2)
    print(f"\n{char * (pad // 2)} {title} {char * (pad - pad // 2)}\n")


def _print_log(log: list[dict]) -> None:
    print("  Execution log:")
    for step in log:
        actor = step["actor"].ljust(12)
        event = step["event"].ljust(10)
        extras = {k: v for k, v in step.items() if k not in ("actor", "event") and v is not None}
        print(f"    [{actor}] {event}  {extras}")


def main() -> None:
    supervisor = RAGSupervisor()

    _divider("RAG Supervisor / Worker Demo", "═")
    print(f"Workers registered: {list(supervisor._workers)}\n")

    for query in DEMO_QUERIES:
        _divider(f"Query: {query[:55]}{'...' if len(query) > 55 else ''}")

        result = supervisor.run(query, top_k=3)

        print(f"  Strategy  : {result['strategy']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Fallback  : {result['used_fallback']}")
        print(f"  LLM used  : {result['used_llm']}")
        print(f"  Sources   : {len(result['sources'])} chunks\n")

        _print_log(result["execution_log"])

        print(f"\n  Answer (first 300 chars):\n  {result['answer'][:300]}...")

    _divider("Done", "═")


if __name__ == "__main__":
    main()
