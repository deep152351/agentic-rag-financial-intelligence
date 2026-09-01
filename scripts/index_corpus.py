"""Index the SEC corpus and the portfolio book into Qdrant as a hybrid collection.

    python -m scripts.index_corpus              # recreate and index everything
    python -m scripts.index_corpus --batch 128

Each chunk is written once with two vectors: BGE-small dense and BM25 sparse.
Payload fields listed in ``vectorstore.INDEXED_*`` get a Qdrant index so the
retrieval filters are served from the index rather than by post-scan.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from app.config import DATA_DIR
from app.embeddings import get_encoders
from app.vectorstore import get_store, point_from_chunk

CORPUS_FILES = ("corpus.jsonl", "portfolio_corpus.jsonl")


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for name in CORPUS_FILES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  !! {name} missing -- run its build script first")
            continue
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        chunks.extend(rows)
        print(f"  {name}: {len(rows)} chunks")
    return chunks


async def run(batch_size: int) -> int:
    encoders = get_encoders()
    store = get_store()

    print("loading corpus...")
    chunks = load_chunks()
    if not chunks:
        print("nothing to index")
        return 1

    doc_ids = [c["doc_id"] for c in chunks]
    if len(set(doc_ids)) != len(doc_ids):
        raise SystemExit("duplicate doc_id in corpus -- fix the builders before indexing")

    print(f"loading embedding models ({encoders.dense_model_name} / {encoders.sparse_model_name})...")
    t0 = time.perf_counter()
    dim = encoders.dense_dim
    print(f"  dense dimension: {dim} ({time.perf_counter() - t0:.1f}s)")

    await store.recreate(dim)
    print(f"collection '{store.collection}' recreated")

    started = time.perf_counter()
    indexed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        dense = await asyncio.to_thread(encoders.embed_documents, texts)
        sparse = await asyncio.to_thread(encoders.embed_documents_sparse, texts)
        points = [
            point_from_chunk(start + i, chunk, dense[i], sparse[i])
            for i, chunk in enumerate(batch)
        ]
        await store.upsert(points)
        indexed += len(points)
        print(f"  indexed {indexed}/{len(chunks)}", end="\r", flush=True)

    elapsed = time.perf_counter() - started
    total = await store.count()
    print(f"\nindexed {indexed} chunks in {elapsed:.1f}s ({indexed / elapsed:.0f}/s)")
    print(f"collection now holds {total} points")
    await store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    return asyncio.run(run(args.batch))


if __name__ == "__main__":
    raise SystemExit(main())
