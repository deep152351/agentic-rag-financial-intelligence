"""Qdrant hybrid store: one collection carrying both a dense and a sparse vector.

The collection holds two named vectors per point -- ``dense`` (BGE-small, cosine)
and ``sparse`` (BM25) -- so a single point can be reached by either branch and the
two branches can be searched independently and fused by the caller.

Constraint translation lives here too: ``build_filter`` turns a
``Constraints`` object into a Qdrant ``Filter``. The mapping is deliberately
boring -- every strict field becomes a ``MatchAny`` over an indexed payload key --
because that is what makes constraint compliance measurable in the eval.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Iterable, Sequence

from qdrant_client import AsyncQdrantClient, models

from .config import Settings, get_settings
from .embeddings import SparseVector, get_encoders
from .schemas import Constraints

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

#: payload keys that get a Qdrant index; these are exactly the filterable fields
INDEXED_KEYWORD_FIELDS = ("ticker", "sector", "doc_type", "statement", "metrics", "account_id")
INDEXED_INTEGER_FIELDS = ("fiscal_years",)


def build_filter(constraints: Constraints) -> models.Filter | None:
    """Constraints -> Qdrant filter. Only *strict* fields become conditions."""
    must: list[models.Condition] = []

    def any_of(key: str, values: Sequence[Any]) -> None:
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))

    any_of("ticker", [t.upper() for t in constraints.tickers])
    any_of("sector", constraints.sectors)
    any_of("doc_type", constraints.doc_types)
    any_of("statement", constraints.statements)
    any_of("account_id", constraints.account_ids)
    # `metrics` is a list payload; MatchAny on a list field means "intersects"
    any_of("metrics", constraints.metrics)
    # `fiscal_years` is a list too, so a range query and a point query look the
    # same to the caller -- one MatchAny covers "FY2024" and "FY2019 to FY2023".
    any_of("fiscal_years", constraints.fiscal_years)

    return models.Filter(must=must) if must else None


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.collection
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            s = self.settings
            if s.uses_embedded_qdrant:
                s.qdrant_path.mkdir(parents=True, exist_ok=True)
                logger.info("qdrant: embedded at %s", s.qdrant_path)
                self._client = AsyncQdrantClient(path=str(s.qdrant_path))
            else:
                logger.info("qdrant: remote %s", s.qdrant_url)
                self._client = AsyncQdrantClient(
                    url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=30
                )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- lifecycle ---------------------------------------------------------

    async def exists(self) -> bool:
        return await self.client.collection_exists(self.collection)

    async def recreate(self, dense_dim: int) -> None:
        if await self.exists():
            await self.client.delete_collection(self.collection)
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF is what makes this BM25 rather than raw term frequency.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field in INDEXED_KEYWORD_FIELDS:
            await self.client.create_payload_index(
                self.collection, field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        for field in INDEXED_INTEGER_FIELDS:
            await self.client.create_payload_index(
                self.collection, field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

    async def upsert(self, points: Iterable[models.PointStruct]) -> None:
        await self.client.upsert(collection_name=self.collection, points=list(points))

    async def count(self) -> int:
        result = await self.client.count(self.collection, exact=True)
        return result.count

    # -- search branches ---------------------------------------------------

    async def search_dense(self, vector: list[float], flt: models.Filter | None,
                           limit: int) -> list[models.ScoredPoint]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return response.points

    async def search_sparse(self, vector: SparseVector, flt: models.Filter | None,
                            limit: int) -> list[models.ScoredPoint]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return response.points

    async def health(self) -> dict[str, Any]:
        try:
            ok = await self.exists()
            return {
                "backend": "embedded" if self.settings.uses_embedded_qdrant else "remote",
                "collection": self.collection,
                "exists": ok,
                "points": await self.count() if ok else 0,
            }
        except Exception as exc:  # pragma: no cover - health must never raise
            return {"backend": "unknown", "collection": self.collection, "error": str(exc)}


@functools.lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()


def point_from_chunk(idx: int, chunk: dict, dense: list[float],
                     sparse: SparseVector) -> models.PointStruct:
    payload = {k: v for k, v in chunk.items() if k != "text"}
    payload["text"] = chunk["text"]
    return models.PointStruct(
        id=idx,
        vector={
            DENSE: dense,
            SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
        },
        payload=payload,
    )


__all__ = [
    "DENSE", "SPARSE", "VectorStore", "build_filter", "get_store",
    "point_from_chunk", "get_encoders",
]
