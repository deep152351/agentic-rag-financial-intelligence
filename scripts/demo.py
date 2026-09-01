"""Interactive / scripted walkthrough of the pipeline, with the full trace shown.

    python -m scripts.demo                      # scripted showcase questions
    python -m scripts.demo --interactive        # ask your own
    python -m scripts.demo -q "Apple FY2024 operating margin"

Prints what the agent actually did on each question -- resolved constraints,
entity-resolution decisions, retrieval waves, budget spend -- not just the answer.
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap

from app.llm import get_llms
from app.orchestrator import Orchestrator
from app.schemas import AskRequest
from app.vectorstore import get_store

SHOWCASE = [
    "What was Apple's operating margin in FY2024?",
    "Compare Microsoft and Nvidia on revenue growth from FY2023 to FY2025.",
    "How has Tesla's free cash flow trended since 2021?",
    "What is my portfolio's exposure by sector?",
    "What is my look-through share of FY2024 net income?",
    "Which of my holdings had operating margin decline in FY2024?",
    "Should I buy Nvidia stock right now?",
]

RULE = "=" * 78


def show(response, question: str) -> None:
    print(f"\n{RULE}\nQ: {question}\n{RULE}")
    print(textwrap.fill(response.answer, width=78,
                        replace_whitespace=False)[:1800])

    trace = response.trace
    if not trace:
        return

    c = trace.resolved_constraints
    print(f"\n  intent            {trace.intent}")
    print(f"  constraints       tickers={c.tickers or '-'} "
          f"years={c.fiscal_years or '-'} metrics={c.metrics or '-'}")

    strict = [r for r in trace.entity_resolution if r["applied_as"] == "strict_filter"]
    soft = [r for r in trace.entity_resolution if r["applied_as"] == "soft_hint"]
    if strict:
        print("  resolved          " + ", ".join(
            f"{r['surface']!r}->{r['resolved']}({r['score']:.0f})" for r in strict[:6]))
    if soft:
        print("  demoted to hint   " + ", ".join(
            f"{r['surface']!r}->{r['resolved']}({r['score']:.0f})" for r in soft[:4]))

    for turn in trace.turns:
        print(f"  turn {turn.turn}            {len(turn.subqueries)} sub-queries, "
              f"{turn.docs_retrieved} docs ({turn.new_docs} new), "
              f"coverage {turn.coverage:.0%}, {turn.elapsed_ms:.0f}ms -> {turn.action}")

    b = trace.budget
    print(f"  budget            {b.turns_used}/{b.turns_allowed} turns, "
          f"{b.searches_issued} searches, {b.llm_calls} LLM calls, "
          f"{b.elapsed_ms:.0f}ms / {b.deadline_ms:.0f}ms  [{b.stop_reason}]")
    if trace.portfolio_tools_used:
        print(f"  portfolio tools   {', '.join(trace.portfolio_tools_used)}")
    print(f"  grounded          {response.grounded}"
          + (f"  UNGROUNDED: {response.ungrounded_figures}" if response.ungrounded_figures else ""))
    if response.citations:
        print(f"  citations         {', '.join(c.doc_id for c in response.citations[:6])}")


async def run(args) -> int:
    print(f"LLM configuration: {get_llms().describe()}")
    if get_llms().is_stub:
        print("(no API key configured -- deterministic planner + stub synthesizer;\n"
              " retrieval, filtering, fusion, budgeting and validation are all live)")

    orchestrator = Orchestrator()
    questions = [args.question] if args.question else SHOWCASE

    if args.interactive:
        print("\nType a question, or 'quit' to exit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"quit", "exit", "q", ""}:
                break
            show(await orchestrator.answer(AskRequest(question=question)), question)
    else:
        for question in questions:
            show(await orchestrator.answer(AskRequest(question=question)), question)

    await get_store().close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="ask a single question")
    ap.add_argument("-i", "--interactive", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
