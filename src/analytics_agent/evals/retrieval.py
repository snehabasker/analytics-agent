"""Retrieval A/B: does the prompt context contain the semantic-layer items each golden question needs?

    python -m analytics_agent.evals.retrieval                     # every arm whose deps/keys exist
    python -m analytics_agent.evals.retrieval --arms hashing cohere_embed_rerank

Arms:
    hashing               current setup: hashed BoW + char-trigram vectors
    hashing_rerank        hashing first stage (pool 25/kind) + Cohere Rerank
    cohere_embed          Cohere Embed (search_document / search_query)
    cohere_embed_rerank   Cohere Embed first stage + Cohere Rerank
    random                control: random vectors. Must score near chance, proving the metric can fail.

Scored per kind (metric / dimension / value) on the SAME slice sizes `context_for` hands the LLM:
    recall@k  share of needed items present in the slice
    hit@1     top-ranked item is a needed one
    MRR       1 / rank of the first needed item (over the full candidate ranking)
plus `grounded`: every needed item of every kind is in the context (the end-to-end-relevant number).

This isolates retrieval. The end-to-end eval (`evals.run`) can't show it with the heuristic planner,
which ignores retrieved context; run `evals.run --llm claude` with ANALYTICS_EMBEDDER/RERANKER set for that.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import yaml

from ..agent import _categorical_values
from ..config import ROOT, Settings
from ..db import Warehouse
from ..retrieval import CohereEmbedder, CohereReranker, HashingEmbedder, SemanticIndex
from ..semantic import SemanticLayer

HERE = Path(__file__).parent
KINDS = (("metric", "m"), ("dimension", "d"), ("value", "v"))
ARMS = ("hashing", "hashing_rerank", "cohere_embed", "cohere_embed_rerank", "random")


class RandomEmbedder:
    dim = 64

    def embed(self, text: str) -> list[float]:
        rng = random.Random(text)
        return [rng.gauss(0, 1) for _ in range(self.dim)]


def _cases() -> list[dict]:
    golden = {c["id"]: c for c in yaml.safe_load((HERE / "golden.yaml").read_text(encoding="utf-8"))}
    labels = yaml.safe_load((HERE / "retrieval_labels.yaml").read_text(encoding="utf-8"))
    missing = set(labels) - set(golden)
    assert not missing, f"labels for unknown golden ids: {missing}"
    return [{"id": i, "question": golden[i]["question"], "category": golden[i]["category"], **lab} for i, lab in labels.items()]


def _key(h) -> str:
    return h.payload["metric"] if h.kind == "metric" else h.payload["dimension"] if h.kind == "dimension" \
        else f"{h.payload['dimension']}={h.payload['value']}"


def _build(arm: str, layer: SemanticLayer, values: dict, s: Settings) -> SemanticIndex:
    embedder = {"hashing": HashingEmbedder, "hashing_rerank": HashingEmbedder, "random": RandomEmbedder}.get(arm)
    embedder = embedder() if embedder else CohereEmbedder(s.cohere_embed_model)
    reranker = CohereReranker(s.cohere_rerank_model) if arm.endswith("_rerank") else None
    return SemanticIndex(layer, embedder=embedder, values=values, reranker=reranker)


def evaluate(index: SemanticIndex, cases: list[dict]) -> dict:
    quota = dict(index.CONTEXT_QUOTAS)
    # Deep ranking for MRR; its first `quota[kind]` items are exactly what context_for would emit.
    deep = (("metric", 50), ("dimension", 50), ("value", index.rerank_pool))
    per_case, lat = [], []
    for c in cases:
        t = time.perf_counter()
        hits = index.retrieve(c["question"], deep)
        lat.append((time.perf_counter() - t) * 1000)
        row = {"id": c["id"], "question": c["question"], "grounded": True, "missing": []}
        for kind, short in KINDS:
            need = set(c.get(short, []))
            if not need:
                continue
            ranked = [_key(h) for h in hits if h.kind == kind]
            ctx = ranked[:quota[kind]]
            first = next((i for i, k in enumerate(ranked) if k in need), None)
            row[kind] = {"recall": len(need & set(ctx)) / len(need), "hit1": float(bool(ranked) and ranked[0] in need),
                         "rr": 0.0 if first is None else 1 / (first + 1), "top": ctx}
            gone = sorted(need - set(ctx))
            if gone:
                row["grounded"] = False
                row["missing"] += gone
        per_case.append(row)

    summary = {"n": len(cases), "grounded": statistics.mean(r["grounded"] for r in per_case),
               "latency_ms_p50": statistics.median(lat), "latency_ms_mean": statistics.mean(lat)}
    for kind, _ in KINDS:
        rows = [r[kind] for r in per_case if kind in r]
        summary[kind] = {"n": len(rows), **{m: statistics.mean(x[m] for x in rows) for m in ("recall", "hit1", "rr")}}
    for attr in ("embedder", "reranker"):
        calls = getattr(getattr(index, attr), "calls", None)
        if calls is not None:
            summary[f"{attr}_api_calls"] = calls
    return {"summary": summary, "cases": per_case}


def _markdown(results: dict[str, dict], skipped: dict[str, str], s: Settings) -> str:
    out = ["# Retrieval A/B: hashing vs Cohere Embed / Rerank", "",
           f"Cohere models: `{s.cohere_embed_model}`, `{s.cohere_rerank_model}`. "
           f"Slice sizes = what `context_for` puts in the prompt (metric 4 of 5, dimension 4 of 12, value 4 of 218).", "",
           "| arm | grounded | metric hit@1 | metric MRR | dim recall@4 | dim hit@1 | dim MRR | value recall@4 | value hit@1 | value MRR | p50 ms |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in results.items():
        x = r["summary"]
        m, d, v = x["metric"], x["dimension"], x["value"]
        out.append(f"| {arm} | {x['grounded']:.0%} | {m['hit1']:.0%} | {m['rr']:.2f} | {d['recall']:.0%} | {d['hit1']:.0%} | "
                   f"{d['rr']:.2f} | {v['recall']:.0%} | {v['hit1']:.0%} | {v['rr']:.2f} | {x['latency_ms_p50']:.0f} |")
    for arm, why in skipped.items():
        out.append(f"| {arm} | skipped: {why} |" + " |" * 9)
    out += ["", "## Ungrounded questions per arm (needed items missing from the prompt context)", ""]
    for arm, r in results.items():
        bad = [f"`{c['id']}` missing {', '.join(c['missing'])}" for c in r["cases"] if not c["grounded"]]
        out.append(f"- **{arm}** ({len(bad)}): " + ("; ".join(bad) if bad else "none"))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "retrieval_ab")
    args = ap.parse_args()

    s = Settings()
    layer = SemanticLayer.load()
    wh = Warehouse(s.db_path, s.max_rows, s.query_timeout_s)
    values = _categorical_values(wh, layer)
    cases = _cases()

    results, skipped = {}, {}
    for arm in args.arms:
        if arm.startswith(("cohere", "hashing_rerank")) and not os.getenv("COHERE_API_KEY"):
            skipped[arm] = "COHERE_API_KEY not set"
            continue
        t = time.perf_counter()
        index = _build(arm, layer, values, s)
        res = evaluate(index, cases)
        res["summary"]["index_build_s"] = time.perf_counter() - t - res["summary"]["latency_ms_mean"] * len(cases) / 1000
        results[arm] = res
        sm = res["summary"]
        print(f"{arm:22s} grounded={sm['grounded']:.0%}  metric hit@1={sm['metric']['hit1']:.0%}  "
              f"dim recall@4={sm['dimension']['recall']:.0%}  value recall@4={sm['value']['recall']:.0%} "
              f"value MRR={sm['value']['rr']:.2f}  p50={sm['latency_ms_p50']:.0f}ms")
    for arm, why in skipped.items():
        print(f"{arm:22s} SKIPPED ({why})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"results": results, "skipped": skipped}, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(_markdown(results, skipped, s), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
