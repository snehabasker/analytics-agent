"""Vector retrieval over the semantic layer, backed by Qdrant.

What gets embedded: one document per metric, dimension, few-shot example and categorical
value ("Rock" is a genre). At question time the top-k hits ground the LLM in the *right*
slice of the layer instead of pasting everything into the prompt, and value documents let
it map "jazz" to the exact stored string "Jazz".

The default embedder is deliberately a deterministic hashed bag of words + character
trigrams: no model download, no network, reproducible in CI. It satisfies the small
`Embedder` protocol, so swapping in a neural embedder is a one-class change.

Optional Cohere backends (`pip install cohere`, `COHERE_API_KEY`):
  * `CohereEmbedder` - Embed with asymmetric document/query input types.
  * `CohereReranker` - second stage: vector search over-fetches a candidate pool per kind,
    then ONE rerank call over the union reorders every pool for that question.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .semantic import SemanticLayer

COLLECTION = "semantic_layer"


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

# Optional embedder extensions, looked up with getattr so HashingEmbedder stays minimal:
#   embed_query(text)   - asymmetric query encoding (defaults to embed)
#   embed_batch(texts)  - batched document encoding (defaults to a loop over embed)


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return (index into documents, relevance score), best first."""
        ...


def _with_retry(fn, attempts: int = 6):
    """Back off on rate limits (trial Cohere keys are heavily throttled)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # cohere raises TooManyRequestsError; don't import cohere at module load
            limited = "TooManyRequests" in type(e).__name__ or "429" in str(e)
            if not limited or i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i * 2))


class CohereEmbedder:
    def __init__(self, model: str = "embed-v4.0", dim: int = 1024, api_key: str | None = None):
        import cohere
        self.client = cohere.ClientV2(api_key=api_key or os.environ["COHERE_API_KEY"])
        self.model, self.dim = model, dim
        self.calls = 0

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):  # API max batch size
            chunk = texts[i:i + 96]
            self.calls += 1
            r = _with_retry(lambda: self.client.embed(
                texts=chunk, model=self.model, input_type=input_type,
                embedding_types=["float"], output_dimension=self.dim))
            out.extend(r.embeddings.float_)
        return out

    def embed(self, text: str) -> list[float]:
        return self._embed([text], "search_document")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "search_query")[0]


class CohereReranker:
    def __init__(self, model: str = "rerank-v3.5", api_key: str | None = None):
        import cohere
        self.client = cohere.ClientV2(api_key=api_key or os.environ["COHERE_API_KEY"])
        self.model = model
        self.calls = 0

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        self.calls += 1
        r = _with_retry(lambda: self.client.rerank(model=self.model, query=query, documents=documents))
        return [(x.index, x.relevance_score) for x in r.results]


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"^{w}$"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        vec = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass
class Hit:
    kind: str  # metric | dimension | example | value
    name: str
    text: str
    score: float
    payload: dict[str, Any]


class SemanticIndex:
    # Per-kind slice sizes handed to the prompt; examples come last.
    CONTEXT_QUOTAS: tuple[tuple[str, int], ...] = (("metric", 4), ("dimension", 4), ("value", 4), ("example", 3))

    def __init__(self, layer: SemanticLayer, embedder: Embedder | None = None,
                 qdrant_url: str | None = None, values: dict[str, list[str]] | None = None,
                 reranker: Reranker | None = None, rerank_pool: int = 25):
        self.layer = layer
        self.embedder = embedder or HashingEmbedder()
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
        self._build(values or {})

    def _documents(self, values: dict[str, list[str]]) -> list[tuple[str, str, str, dict[str, Any]]]:
        docs = []
        for m in self.layer.metrics:
            text = f"metric {m.name}: {m.description} Also called: {', '.join(m.synonyms)}"
            docs.append(("metric", m.name, text, {"metric": m.name}))
        for d in self.layer.dimensions:
            text = f"dimension {d.name}: {d.description} Also called: {', '.join(s.lstrip('~') for s in d.synonyms)}"
            docs.append(("dimension", d.name, text, {"dimension": d.name}))
        for ex in self.layer.examples:
            docs.append(("example", ex.question, f"example question: {ex.question}", {"question": ex.question, "query": ex.query}))
        for dim, vals in values.items():
            for v in vals:
                docs.append(("value", f"{dim}={v}", f"{dim} value {v}", {"dimension": dim, "value": v}))
        return docs

    def _build(self, values: dict[str, list[str]]) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
        docs = self._documents(values)
        texts = [d[2] for d in docs]
        batch = getattr(self.embedder, "embed_batch", None)
        vectors = batch(texts) if batch else [self.embedder.embed(t) for t in texts]
        self.client.upsert(COLLECTION, points=[
            PointStruct(id=i, vector=vec, payload={"kind": kind, "name": name, "text": text, **payload})
            for i, ((kind, name, text, payload), vec) in enumerate(zip(docs, vectors))
        ])
        self.size = len(docs)

    def _query_vector(self, question: str) -> list[float]:
        return getattr(self.embedder, "embed_query", self.embedder.embed)(question)

    def _vector_search(self, qvec: list[float], k: int, kinds: tuple[str, ...] | None) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        flt = Filter(must=[FieldCondition(key="kind", match=MatchAny(any=list(kinds)))]) if kinds else None
        res = self.client.query_points(COLLECTION, query=qvec, limit=k, query_filter=flt)
        return [Hit(p.payload["kind"], p.payload["name"], p.payload["text"], p.score, p.payload) for p in res.points]

    def _rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        ranked = self.reranker.rerank(question, [h.text for h in hits])
        return [Hit(hits[i].kind, hits[i].name, hits[i].text, score, hits[i].payload) for i, score in ranked]

    def search(self, question: str, k: int = 8, kinds: tuple[str, ...] | None = None) -> list[Hit]:
        qvec = self._query_vector(question)
        if self.reranker is None:
            return self._vector_search(qvec, k, kinds)
        return self._rerank(question, self._vector_search(qvec, max(k, self.rerank_pool), kinds))[:k]

    def retrieve(self, question: str, quotas: tuple[tuple[str, int], ...] | None = None) -> list[Hit]:
        """Top-k hits per kind. With a reranker: one query embedding, a candidate pool per kind,
        and a single rerank call over the union (Cohere bills per 100 docs, so this is one unit)."""
        quotas = quotas or self.CONTEXT_QUOTAS
        qvec = self._query_vector(question)
        if self.reranker is None:
            return [h for kind, k in quotas for h in self._vector_search(qvec, k, (kind,))]
        pool = [h for kind, k in quotas for h in self._vector_search(qvec, max(k, self.rerank_pool), (kind,))]
        ranked = self._rerank(question, pool)
        return [h for kind, k in quotas for h in [r for r in ranked if r.kind == kind][:k]]

    def context_for(self, question: str) -> list[str]:
        """Render retrieved hits as compact prompt lines, grouped so examples come last."""
        hits = self.retrieve(question)
        lines = []
        for h in hits:
            if h.kind == "example":
                lines.append(f"EXAMPLE: {h.payload['question']} -> {h.payload['query']}")
            elif h.kind == "value":
                lines.append(f"VALUE: dimension '{h.payload['dimension']}' has the exact value {h.payload['value']!r}")
            else:
                lines.append(h.text)
        return lines
