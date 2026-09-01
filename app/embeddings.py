"""Dense (BGE) and sparse (BM25) encoders.

Both models are loaded once per process and reused. Encoding is CPU-bound, so
callers on the async path push it through ``asyncio.to_thread`` rather than
blocking the event loop.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from fastembed import SparseTextEmbedding, TextEmbedding

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class Encoders:
    """Lazily-initialised dense + sparse embedding pair."""

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None

    @property
    def dense(self) -> TextEmbedding:
        if self._dense is None:
            logger.info("loading dense model %s", self.dense_model_name)
            self._dense = TextEmbedding(model_name=self.dense_model_name)
        return self._dense

    @property
    def sparse(self) -> SparseTextEmbedding:
        if self._sparse is None:
            logger.info("loading sparse model %s", self.sparse_model_name)
            self._sparse = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse

    @functools.cached_property
    def dense_dim(self) -> int:
        return len(next(iter(self.dense.embed(["dimension probe"]))))

    # -- documents ---------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self.dense.embed(list(texts))]

    def embed_documents_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in self.sparse.embed(list(texts))
        ]

    # -- queries -----------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.dense.query_embed(text))).tolist()

    def embed_query_sparse(self, text: str) -> SparseVector:
        v = next(iter(self.sparse.query_embed(text)))
        return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())

    def embed_queries(self, texts: Iterable[str]) -> list[list[float]]:
        return [v.tolist() for v in self.dense.query_embed(list(texts))]

    def embed_queries_sparse(self, texts: Iterable[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in self.sparse.query_embed(list(texts))
        ]

    def warm_up(self) -> None:
        """Force both model downloads/loads up front, off the request path."""
        self.embed_query("warm up")
        self.embed_query_sparse("warm up")


@functools.lru_cache(maxsize=1)
def get_encoders() -> Encoders:
    settings = get_settings()
    return Encoders(settings.dense_model, settings.sparse_model)
