"""Vector retrieval over the semantic layer, backed by Qdrant.

What gets embedded: one document per metric, dimension, few-shot example and categorical
value ("Rock" is a genre). At question time the top-k hits ground the LLM in the *right*
slice of the layer instead of pasting everything into the prompt, and value documents let
it map "jazz" to the exact stored string "Jazz".

The embedder is deliberately a deterministic hashed bag of words + character trigrams:
no model download, no network, reproducible in CI. It satisfies the small `Embedder`
protocol, so swapping in a neural embedder is a one-class change.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .semantic import SemanticLayer

COLLECTION = "semantic_layer"


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


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
    def __init__(self, layer: SemanticLayer, embedder: Embedder | None = None,
                 qdrant_url: str | None = None, values: dict[str, list[str]] | None = None):
        self.layer = layer
        self.embedder = embedder or HashingEmbedder()
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
        self.client.upsert(COLLECTION, points=[
            PointStruct(id=i, vector=self.embedder.embed(text), payload={"kind": kind, "name": name, "text": text, **payload})
            for i, (kind, name, text, payload) in enumerate(docs)
        ])
        self.size = len(docs)

    def search(self, question: str, k: int = 8, kinds: tuple[str, ...] | None = None) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        flt = Filter(must=[FieldCondition(key="kind", match=MatchAny(any=list(kinds)))]) if kinds else None
        res = self.client.query_points(COLLECTION, query=self.embedder.embed(question), limit=k, query_filter=flt)
        return [Hit(p.payload["kind"], p.payload["name"], p.payload["text"], p.score, p.payload) for p in res.points]

    def context_for(self, question: str) -> list[str]:
        """Render retrieved hits as compact prompt lines, grouped so examples come last."""
        hits = self.search(question, k=4, kinds=("metric",)) + self.search(question, k=4, kinds=("dimension",)) \
            + self.search(question, k=4, kinds=("value",)) + self.search(question, k=3, kinds=("example",))
        lines = []
        for h in hits:
            if h.kind == "example":
                lines.append(f"EXAMPLE: {h.payload['question']} -> {h.payload['query']}")
            elif h.kind == "value":
                lines.append(f"VALUE: dimension '{h.payload['dimension']}' has the exact value {h.payload['value']!r}")
            else:
                lines.append(h.text)
        return lines
