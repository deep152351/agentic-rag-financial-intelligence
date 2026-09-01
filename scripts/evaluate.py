"""Evaluation harness: Recall@10, constraint compliance, groundedness, latency.

    python -m scripts.evaluate                    # retrieval metrics (no LLM needed)
    python -m scripts.evaluate --ablation         # + dense-only / sparse-only / no-filter
    python -m scripts.evaluate --with-answers     # + synthesis, groundedness, budget
    python -m scripts.evaluate --limit 30 --json results.json

What each number means
----------------------
**Recall@10** -- of the documents that *must* be retrieved to answer the question,
what share appear in the fused top 10. Macro-averaged over cases, so a case
needing three documents is not worth three times a case needing one.

**Constraint compliance** -- two different things, reported separately because
they fail for different reasons:

*   *Filter compliance* is the share of returned documents that satisfy the gold
    constraints. It answers "when the system filtered, did it filter correctly?"
    Low values mean the retrieval let through the wrong company or year.
*   *Resolution accuracy* compares the constraints the system resolved against
    the constraints the question actually implies, field by field. It answers
    "did the resolver understand the question?" Precision below 1.0 means an
    over-constraint (a filter nothing needed, which silently removes good
    documents); recall below 1.0 means an under-constraint (a filter that should
    have been applied and was not).

Separating them matters: a system that applies no filters at all scores perfect
filter compliance and zero resolution recall.

**Groundedness** -- share of answers whose every figure traces back to the
retrieved context (see ``app.validator``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.config import DATA_DIR, get_settings
from app.entities import get_resolver
from app.llm import get_llms
from app.orchestrator import Orchestrator
from app.planner import build_planner
from app.retrieval import HybridRetriever, doc_satisfies
from app.schemas import AskRequest, Constraints, Plan, RetrievedDoc
from app.vectorstore import get_store

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int = 10) -> float | None:
    if not gold:
        return None
    top = set(retrieved[:k])
    return len(set(gold) & top) / len(set(gold))


def hit_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int = 10) -> float | None:
    if not gold:
        return None
    return 1.0 if set(gold) & set(retrieved[:k]) else 0.0


def mrr(gold: Sequence[str], retrieved: Sequence[str]) -> float | None:
    if not gold:
        return None
    goldset = set(gold)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in goldset:
            return 1.0 / i
    return 0.0


def prf(predicted: Sequence[Any], expected: Sequence[Any]) -> tuple[float, float, float]:
    """Precision / recall / F1 for one constraint field."""
    p_set, e_set = set(predicted), set(expected)
    if not p_set and not e_set:
        return 1.0, 1.0, 1.0
    precision = len(p_set & e_set) / len(p_set) if p_set else (1.0 if not e_set else 0.0)
    recall = len(p_set & e_set) / len(e_set) if e_set else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@dataclass
class CaseResult:
    case_id: str
    family: str
    question: str
    recall10: float | None
    hit10: float | None
    mrr: float | None
    filter_compliance: float | None
    intent_ok: bool
    resolution: dict[str, tuple[float, float, float]]
    exact_constraints: bool
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)
    grounded: bool | None = None
    ungrounded: list[str] = field(default_factory=list)
    turns: int | None = None
    tools_ok: bool | None = None


def _mean(values: Sequence[float | None]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(self, top_k: int) -> None:
        self.settings = get_settings()
        self.top_k = top_k
        self.retriever = HybridRetriever(settings=self.settings)
        self.planner = build_planner(get_llms(self.settings).planner, self.settings)
        self.resolver = get_resolver()

    async def plan_for(self, question: str) -> Plan:
        outcome = self.planner.plan(question)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome.plan

    @staticmethod
    def merge_constraints(plan: Plan) -> Constraints:
        merged = Constraints()
        for sq in plan.subqueries:
            merged.tickers.extend(sq.constraints.tickers)
            merged.fiscal_years.extend(sq.constraints.fiscal_years)
            merged.metrics.extend(sq.constraints.metrics)
            merged.doc_types.extend(sq.constraints.doc_types)
        for f in ("tickers", "fiscal_years", "metrics", "doc_types"):
            setattr(merged, f, list(dict.fromkeys(getattr(merged, f))))
        return merged

    async def run_case(self, case: dict, branches=("dense", "sparse"),
                       apply_filters: bool = True) -> CaseResult:
        started = time.perf_counter()
        plan = await self.plan_for(case["question"])
        resolved = self.merge_constraints(plan)
        gold = case.get("gold_constraints") or {}

        docs: list[RetrievedDoc] = []
        if plan.intent != "out_of_scope":
            fusion = await self.retriever.search(
                plan.subqueries, top_k=self.top_k,
                branches=branches, apply_filters=apply_filters,
            )
            docs = fusion.docs
        retrieved = [d.doc_id for d in docs]
        latency = (time.perf_counter() - started) * 1000

        gold_constraints = Constraints(
            tickers=gold.get("tickers", []),
            fiscal_years=gold.get("fiscal_years", []),
            metrics=gold.get("metrics", []),
            doc_types=gold.get("doc_types", []),
        )
        # measured against the gold constraints, not the ones we resolved --
        # grading a filter against itself would always score 100%
        compliance = None
        if docs and not gold_constraints.is_empty():
            compliance = sum(doc_satisfies(d, gold_constraints) for d in docs) / len(docs)

        resolution = {
            "ticker": prf(resolved.tickers, gold.get("tickers", [])),
            "fiscal_year": prf(resolved.fiscal_years, gold.get("fiscal_years", [])),
            "metric": prf(resolved.metrics, gold.get("metrics", [])),
        }
        exact = all(f == 1.0 for _, _, f in resolution.values())

        tools_ok = None
        if case.get("expected_tools"):
            tools_ok = bool(set(case["expected_tools"]) & set(plan.portfolio_tools))

        return CaseResult(
            case_id=case["id"],
            family=case["family"],
            question=case["question"],
            recall10=recall_at_k(case["gold_doc_ids"], retrieved, self.top_k),
            hit10=hit_at_k(case["gold_doc_ids"], retrieved, self.top_k),
            mrr=mrr(case["gold_doc_ids"], retrieved),
            filter_compliance=compliance,
            intent_ok=plan.intent == case["expected_intent"],
            resolution=resolution,
            exact_constraints=exact,
            latency_ms=latency,
            retrieved=retrieved,
            tools_ok=tools_ok,
        )


async def run_answers(cases: Sequence[dict], top_k: int) -> list[CaseResult]:
    """Full end-to-end pass, including synthesis and grounding validation."""
    orchestrator = Orchestrator()
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        response = await orchestrator.answer(
            AskRequest(question=case["question"], top_k=top_k, include_trace=True)
        )
        latency = (time.perf_counter() - started) * 1000
        trace = response.trace
        retrieved = trace.retrieved_doc_ids if trace else []
        gold = case.get("gold_constraints") or {}
        resolution = {
            "ticker": prf(trace.resolved_constraints.tickers if trace else [], gold.get("tickers", [])),
            "fiscal_year": prf(trace.resolved_constraints.fiscal_years if trace else [], gold.get("fiscal_years", [])),
            "metric": prf(trace.resolved_constraints.metrics if trace else [], gold.get("metrics", [])),
        }
        results.append(CaseResult(
            case_id=case["id"], family=case["family"], question=case["question"],
            recall10=recall_at_k(case["gold_doc_ids"], retrieved, top_k),
            hit10=hit_at_k(case["gold_doc_ids"], retrieved, top_k),
            mrr=mrr(case["gold_doc_ids"], retrieved),
            # compliance needs per-document metadata, which the API response does
            # not return; it is measured in the retrieval-only path above
            filter_compliance=None,
            intent_ok=bool(trace and trace.intent == case["expected_intent"]),
            resolution=resolution,
            exact_constraints=all(f == 1.0 for _, _, f in resolution.values()),
            latency_ms=latency,
            retrieved=retrieved,
            grounded=response.grounded,
            ungrounded=response.ungrounded_figures,
            turns=trace.budget.turns_used if trace else None,
            tools_ok=(bool(set(case["expected_tools"]) & set(trace.portfolio_tools_used))
                      if case.get("expected_tools") and trace else None),
        ))
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def summarise(results: Sequence[CaseResult], label: str) -> dict[str, Any]:
    retrieval_cases = [r for r in results if r.recall10 is not None]
    latencies = [r.latency_ms for r in results]

    def field_mean(name: str, idx: int) -> float:
        return _mean([r.resolution[name][idx] for r in results])

    summary = {
        "label": label,
        "cases": len(results),
        "recall_at_10": _mean([r.recall10 for r in retrieval_cases]),
        "hit_at_10": _mean([r.hit10 for r in retrieval_cases]),
        "mrr": _mean([r.mrr for r in retrieval_cases]),
        # None (rendered "n/a"), not 0.0, when nothing in this pass measured it --
        # the end-to-end path has no per-document metadata to check against
        "filter_compliance": (_mean([r.filter_compliance for r in results])
                              if any(r.filter_compliance is not None for r in results)
                              else None),
        "intent_accuracy": sum(r.intent_ok for r in results) / len(results) if results else 0.0,
        "constraint_resolution": {
            field: {
                "precision": field_mean(field, 0),
                "recall": field_mean(field, 1),
                "f1": field_mean(field, 2),
            }
            for field in ("ticker", "fiscal_year", "metric")
        },
        "exact_constraint_match": sum(r.exact_constraints for r in results) / len(results) if results else 0.0,
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": (statistics.quantiles(latencies, n=20)[18]
                    if len(latencies) > 20 else max(latencies, default=0.0)),
            "mean": _mean(latencies),
        },
    }
    graded = [r for r in results if r.grounded is not None]
    if graded:
        summary["groundedness"] = sum(r.grounded for r in graded) / len(graded)
        summary["mean_turns"] = _mean([r.turns for r in graded])
    tool_cases = [r for r in results if r.tools_ok is not None]
    if tool_cases:
        summary["tool_selection_accuracy"] = sum(r.tools_ok for r in tool_cases) / len(tool_cases)
    return summary


def by_family(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    families: dict[str, list[CaseResult]] = {}
    for r in results:
        families.setdefault(r.family, []).append(r)
    def maybe_mean(values):
        """None, not 0.0, when a family has nothing to score -- out-of-scope and
        portfolio cases have no gold documents, and reporting them as 0% recall
        reads as a failure rather than as not-applicable."""
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    return {
        family: {
            "n": len(rows),
            "recall_at_10": maybe_mean([r.recall10 for r in rows]),
            "filter_compliance": maybe_mean([r.filter_compliance for r in rows]),
            "intent_accuracy": sum(r.intent_ok for r in rows) / len(rows),
        }
        for family, rows in sorted(families.items())
    }


def print_report(summary: dict[str, Any], families: dict[str, dict] | None = None) -> None:
    print(f"\n{'=' * 74}\n{summary['label']}  ({summary['cases']} cases)\n{'=' * 74}")
    print(f"  Recall@10                {_pct(summary['recall_at_10'])}")
    print(f"  Hit@10                   {_pct(summary['hit_at_10'])}")
    print(f"  MRR                      {summary['mrr']:.3f}")
    compliance = summary["filter_compliance"]
    print(f"  Constraint compliance    "
          f"{_pct(compliance) if compliance is not None else '  n/a':>6}   "
          f"(retrieved docs satisfying the gold filters)")
    print(f"  Intent accuracy          {_pct(summary['intent_accuracy'])}")
    print(f"  Exact constraint match   {_pct(summary['exact_constraint_match'])}")
    print("  Constraint resolution    " + " " * 5 + "precision  recall     F1")
    for field_name, scores in summary["constraint_resolution"].items():
        print(f"    {field_name:<22}{_pct(scores['precision'])}  "
              f"{_pct(scores['recall'])}  {_pct(scores['f1'])}")
    if "groundedness" in summary:
        print(f"  Groundedness             {_pct(summary['groundedness'])}")
        print(f"  Mean turns used          {summary['mean_turns']:.2f}")
    if "tool_selection_accuracy" in summary:
        print(f"  Portfolio tool selection {_pct(summary['tool_selection_accuracy'])}")
    lat = summary["latency_ms"]
    print(f"  Latency p50/p95/mean     {lat['p50']:.0f} / {lat['p95']:.0f} / {lat['mean']:.0f} ms")

    if families:
        print(f"\n  {'family':<16}{'n':>4}{'recall@10':>12}{'compliance':>13}{'intent':>9}")
        for family, scores in families.items():
            recall = _pct(scores["recall_at_10"]) if scores["recall_at_10"] is not None else "n/a"
            compliance = (_pct(scores["filter_compliance"])
                          if scores["filter_compliance"] is not None else "n/a")
            print(f"  {family:<16}{scores['n']:>4}{recall:>12}"
                  f"{compliance:>13}{_pct(scores['intent_accuracy']):>9}")


async def main_async(args) -> int:
    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    cases = payload["cases"][: args.limit] if args.limit else payload["cases"]
    print(f"loaded {len(cases)} cases from {args.eval_set}")
    print(f"LLM: {get_llms().describe()}")

    output: dict[str, Any] = {}

    if args.with_answers:
        results = await run_answers(cases, args.top_k)
        summary = summarise(results, "END-TO-END (retrieval + synthesis + grounding)")
        print_report(summary, by_family(results))
        output["end_to_end"] = summary
    else:
        evaluator = Evaluator(args.top_k)
        results = [await evaluator.run_case(c) for c in cases]
        summary = summarise(results, "RETRIEVAL — hybrid (BGE + BM25) fused with RRF, filters on")
        print_report(summary, by_family(results))
        output["hybrid_rrf"] = summary

        if args.ablation:
            print("\n\n### ABLATION GRID " + "#" * 56)
            print("Two factors, crossed: retrieval branch x metadata filtering.\n"
                  "The filtered rows show what the deployed system does; the unfiltered\n"
                  "rows isolate what fusion contributes when ranking has to do the work.")

            grid: dict[tuple[str, bool], dict] = {}
            branch_sets = [(("dense",), "dense-only  (BGE)"),
                           (("sparse",), "sparse-only (BM25)"),
                           (("dense", "sparse"), "hybrid+RRF")]
            for use_filters in (True, False):
                for branches, name in branch_sets:
                    if (branches, use_filters) == (("dense", "sparse"), True):
                        grid[(name, use_filters)] = summary  # already computed
                        continue
                    rows = [await evaluator.run_case(c, branches, use_filters) for c in cases]
                    grid[(name, use_filters)] = summarise(rows, f"{name} / filters={'on' if use_filters else 'off'}")

            header = f"  {'configuration':<22}{'filters':>9}{'recall@10':>12}{'hit@10':>9}{'MRR':>8}{'compliance':>12}{'p50 ms':>9}"
            print(f"\n{'=' * 84}\n{header}\n{'=' * 84}")
            for (name, use_filters), s in grid.items():
                print(f"  {name:<22}{('on' if use_filters else 'off'):>9}"
                      f"{_pct(s['recall_at_10']):>12}{_pct(s['hit_at_10']):>9}"
                      f"{s['mrr']:>8.3f}"
                      f"{(_pct(s['filter_compliance']) if s['filter_compliance'] is not None else 'n/a'):>12}"
                      f"{s['latency_ms']['p50']:>9.0f}")
                output[f"{name}|filters={use_filters}"] = s

            unfiltered = {n: grid[(n, False)] for _, n in branch_sets}
            fused, dense, sparse = (unfiltered["hybrid+RRF"],
                                    unfiltered["dense-only  (BGE)"],
                                    unfiltered["sparse-only (BM25)"])
            filtered_fused = grid[("hybrid+RRF", True)]
            print(f"\n{'=' * 84}\nWhat each component actually buys\n{'=' * 84}")
            print(f"  RRF fusion vs dense alone   (filters off)  "
                  f"{(fused['recall_at_10'] - dense['recall_at_10']) * 100:+5.1f} pp recall@10, "
                  f"{fused['mrr'] - dense['mrr']:+.3f} MRR")
            print(f"  RRF fusion vs sparse alone  (filters off)  "
                  f"{(fused['recall_at_10'] - sparse['recall_at_10']) * 100:+5.1f} pp recall@10, "
                  f"{fused['mrr'] - sparse['mrr']:+.3f} MRR")
            print(f"  Metadata filtering on top of fusion        "
                  f"{(filtered_fused['recall_at_10'] - fused['recall_at_10']) * 100:+5.1f} pp recall@10, "
                  f"{(filtered_fused['filter_compliance'] - fused['filter_compliance']) * 100:+5.1f} pp compliance"
                  if filtered_fused['filter_compliance'] is not None
                  and fused['filter_compliance'] is not None else "")

        failures = sorted((r for r in results if (r.recall10 or 1.0) < 1.0),
                          key=lambda r: r.recall10 or 0.0)[:8]
        if failures:
            print(f"\n{'=' * 74}\nWorst retrieval cases\n{'=' * 74}")
            for r in failures:
                print(f"  [{r.recall10:.2f}] {r.question[:78]}")

    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    await get_store().close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the retrieval and answer pipeline")
    ap.add_argument("--eval-set", default=str(DATA_DIR / "eval" / "eval_set.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--ablation", action="store_true", help="also run branch/filter ablations")
    ap.add_argument("--with-answers", action="store_true",
                    help="run the full agent including synthesis and grounding checks")
    ap.add_argument("--json", help="write the summary to this path")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
