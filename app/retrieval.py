"""Parallel hybrid retrieval with Reciprocal Rank Fusion.

Shape of one retrieval wave
---------------------------
A plan carries N sub-queries. Each sub-query is searched on two independent
branches -- BGE dense and BM25 sparse -- so a wave issues ``2N`` searches. All of
them are dispatched concurrently with ``asyncio.gather``: the wave costs about as
long as its slowest single search rather than the sum of all of them, which is
what keeps a six-sub-query comparison inside the turn budget.

Why RRF rather than score blending
----------------------------------
Cosine similarity lives in [-1, 1] and BM25 is an unbounded positive score whose
scale depends on corpus statistics. Any weighted sum of the two needs
normalisation constants that drift as the corpus changes. RRF throws the
magnitudes away and keeps only the ranks::

    score(d) = sum over branches b of  weight_b / (k + rank_b(d))

with ``k = 60``. A document ranked #1 by either branch scores 1/61; agreement
across branches compounds. The same operator fuses across sub-queries, so a
document that answers several parts of a decomposed question rises above one that
answers a single part very well -- which is exactly the ranking a multi-part
financial question wants.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from qdrant_client import models

from .config import Settings, get_settings
from .embeddings import Encoders, get_encoders
from .schemas import Constraints, RetrievedDoc, SubQuery
from .vectorstore import VectorStore, build_filter, get_store

logger = logging.getLogger(__name__)

#: dense and sparse contribute equally; BM25 carries exact tickers and figures,
#: BGE carries paraphrase. Neither dominates on this corpus.
BRANCH_WEIGHTS = {"dense": 1.0, "sparse": 1.0}


@dataclass
class BranchResult:
    subquery: str
    branch: str
    doc_ids: list[str]
    elapsed_ms: float
    error: str | None = None


@dataclass
class FusionResult:
    docs: list[RetrievedDoc]
    branch_results: list[BranchResult] = field(default_factory=list)
    searches_issued: int = 0
    elapsed_ms: float = 0.0

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, str, Sequence[str]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fuse ``(subquery, branch, ordered_doc_ids)`` triples into one ranking.

    Returns ``{doc_id: fused_score}``. Pure and dependency-free so the eval and
    the unit tests can exercise the fusion in isolation.
    """
    weights = weights or BRANCH_WEIGHTS
    fused: dict[str, float] = {}
    for _subquery, branch, doc_ids in ranked_lists:
        weight = weights.get(branch, 1.0)
        for rank, doc_id in enumerate(doc_ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


def _cache_payloads(sink: dict[str, dict], points: Iterable[models.ScoredPoint]) -> None:
    for p in points:
        payload = p.payload or {}
        doc_id = payload.get("doc_id")
        if doc_id:
            sink[str(doc_id)] = payload


class HybridRetriever:
    def __init__(self, store: VectorStore | None = None, encoders: Encoders | None = None,
                 settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.encoders = encoders or get_encoders()

    # -- one branch --------------------------------------------------------

    async def _dense_branch(self, subquery: str, vector: list[float],
                            flt: models.Filter | None, limit: int,
                            payloads: dict[str, dict]) -> BranchResult:
        t0 = time.perf_counter()
        try:
            points = await self.store.search_dense(vector, flt, limit)
            ids = [str(p.payload.get("doc_id")) for p in points]
            _cache_payloads(payloads, points)
            return BranchResult(subquery, "dense", ids, (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.warning("dense branch failed for %r: %s", subquery, exc)
            return BranchResult(subquery, "dense", [], (time.perf_counter() - t0) * 1000, str(exc))

    async def _sparse_branch(self, subquery: str, vector, flt: models.Filter | None,
                             limit: int, payloads: dict[str, dict]) -> BranchResult:
        t0 = time.perf_counter()
        try:
            points = await self.store.search_sparse(vector, flt, limit)
            ids = [str(p.payload.get("doc_id")) for p in points]
            _cache_payloads(payloads, points)
            return BranchResult(subquery, "sparse", ids, (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.warning("sparse branch failed for %r: %s", subquery, exc)
            return BranchResult(subquery, "sparse", [], (time.perf_counter() - t0) * 1000, str(exc))

    # -- one wave ----------------------------------------------------------

    async def search(self, subqueries: Sequence[SubQuery], top_k: int | None = None,
                     deadline_s: float | None = None,
                     branches: Sequence[str] = ("dense", "sparse"),
                     apply_filters: bool = True) -> FusionResult:
        """Run every (sub-query x branch) search concurrently, then RRF-fuse.

        ``branches`` and ``apply_filters`` exist for the evaluation ablations --
        running one branch alone, or disabling metadata filtering, is how the
        contribution of each half is actually measured rather than asserted.
        """
        # Payloads are collected per call, never on self: one Orchestrator (and so
        # one HybridRetriever) serves every concurrent request, and instance state
        # here would let two in-flight searches overwrite each other's documents.
        payloads: dict[str, dict] = {}
        top_k = top_k or self.settings.top_k
        limit = self.settings.candidates_per_branch
        started = time.perf_counter()

        if not subqueries:
            return FusionResult(docs=[], elapsed_ms=0.0)

        texts = [sq.text for sq in subqueries]
        # encoding is CPU-bound; keep it off the event loop
        want_dense = "dense" in branches
        want_sparse = "sparse" in branches
        dense_vectors, sparse_vectors = await asyncio.gather(
            asyncio.to_thread(self.encoders.embed_queries, texts) if want_dense
            else asyncio.sleep(0, result=[None] * len(texts)),
            asyncio.to_thread(self.encoders.embed_queries_sparse, texts) if want_sparse
            else asyncio.sleep(0, result=[None] * len(texts)),
        )

        tasks: list[asyncio.Task] = []
        for i, sq in enumerate(subqueries):
            flt = build_filter(sq.constraints) if apply_filters else None
            if want_dense:
                tasks.append(asyncio.create_task(
                    self._dense_branch(sq.text, dense_vectors[i], flt, limit, payloads)))
            if want_sparse:
                tasks.append(asyncio.create_task(
                    self._sparse_branch(sq.text, sparse_vectors[i], flt, limit, payloads)))

        timeout = deadline_s if deadline_s is not None else self.settings.turn_deadline_s
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            logger.warning("%d/%d searches cut off by the turn deadline", len(pending), len(tasks))

        branch_results = [t.result() for t in done if not t.cancelled()]
        # asyncio.wait returns a set; restore deterministic order for the trace
        order = {(sq.text, b): i for i, sq in enumerate(subqueries) for b in ("dense", "sparse")}
        branch_results.sort(key=lambda r: (order.get((r.subquery, r.branch), 999), r.branch))

        fused = reciprocal_rank_fusion(
            [(r.subquery, r.branch, r.doc_ids) for r in branch_results],
            k=self.settings.rrf_k,
        )

        # per-doc provenance for the trace
        ranks: dict[str, dict[str, int]] = {}
        matched: dict[str, set[str]] = {}
        for r in branch_results:
            for rank, doc_id in enumerate(r.doc_ids, start=1):
                ranks.setdefault(doc_id, {})
                ranks[doc_id][r.branch] = min(ranks[doc_id].get(r.branch, 10**6), rank)
                matched.setdefault(doc_id, set()).add(r.subquery)

        docs: list[RetrievedDoc] = []
        for doc_id, score in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]:
            payload = payloads.get(doc_id, {})
            docs.append(RetrievedDoc(
                doc_id=doc_id,
                text=payload.get("text", ""),
                score=score,
                doc_type=payload.get("doc_type", "unknown"),
                ticker=payload.get("ticker"),
                company=payload.get("company"),
                sector=payload.get("sector"),
                fiscal_year=payload.get("fiscal_year"),
                fiscal_years=payload.get("fiscal_years") or [],
                metrics=payload.get("metrics") or [],
                source=payload.get("source", ""),
                branch_ranks=ranks.get(doc_id, {}),
                matched_subqueries=sorted(matched.get(doc_id, set())),
            ))

        return FusionResult(
            docs=docs,
            branch_results=branch_results,
            searches_issued=len(tasks),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    # -- coverage ----------------------------------------------------------

    @staticmethod
    def coverage(subqueries: Sequence[SubQuery], docs: Sequence[RetrievedDoc]) -> float:
        """Share of the plan's information needs that retrieved something usable.

        A sub-query counts as covered when at least one returned document
        satisfies every strict constraint it asked for. This is the signal the
        orchestrator spends its turn budget on: low coverage is the only reason
        to pay for another retrieval wave.
        """
        if not subqueries:
            return 1.0
        covered = 0
        for sq in subqueries:
            if any(_doc_satisfies(doc, sq.constraints) for doc in docs):
                covered += 1
        return covered / len(subqueries)


def _doc_satisfies(doc: RetrievedDoc, c: Constraints) -> bool:
    """Does one document satisfy every strict constraint? Also the eval's compliance test."""
    if c.tickers and (doc.ticker or "").upper() not in {t.upper() for t in c.tickers}:
        return False
    if c.fiscal_years:
        # match the same field the Qdrant filter matched on
        covered = set(doc.fiscal_years) or ({doc.fiscal_year} if doc.fiscal_year else set())
        if not covered & set(c.fiscal_years):
            return False
    if c.sectors and doc.sector not in c.sectors:
        return False
    if c.doc_types and doc.doc_type not in c.doc_types:
        return False
    if c.metrics and not (set(c.metrics) & set(doc.metrics)):
        return False
    return True


def doc_satisfies(doc: RetrievedDoc, constraints: Constraints) -> bool:
    return _doc_satisfies(doc, constraints)
