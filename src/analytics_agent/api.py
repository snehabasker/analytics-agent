"""FastAPI service."""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import AgentResponse, AnalyticsAgent, build_agent
from .config import Settings


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    include_trace: bool = False


class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts = {"requests": 0, "ok": 0, "refused": 0, "error": 0}
        self.total_latency_ms = 0.0

    def record(self, r: AgentResponse) -> None:
        with self._lock:
            self.counts["requests"] += 1
            self.counts[r.status] += 1
            self.total_latency_ms += r.latency_ms

    def snapshot(self) -> dict:
        n = self.counts["requests"]
        return {**self.counts, "avg_latency_ms": round(self.total_latency_ms / n, 1) if n else 0.0}


def create_app(agent: AnalyticsAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = agent or build_agent(Settings())
        app.state.stats = Stats()
        yield
        app.state.agent.close()

    app = FastAPI(title="Analytics Agent", version="0.1.0", lifespan=lifespan,
                  description="Natural-language analytics over a governed semantic layer.")

    @app.get("/health")
    def health(request: Request) -> dict:
        a: AnalyticsAgent = request.app.state.agent
        return {"status": "ok", "llm": a.llm.name, "index_size": a.index.size}

    @app.get("/semantic-layer")
    def semantic_layer(request: Request) -> dict:
        layer = request.app.state.agent.layer
        return {"metrics": [{"name": m.name, "description": m.description} for m in layer.metrics],
                "dimensions": [{"name": d.name, "type": d.type, "description": d.description} for d in layer.dimensions]}

    @app.post("/ask", response_model=AgentResponse)
    def ask(body: AskRequest, request: Request) -> AgentResponse:
        try:
            r = request.app.state.agent.ask(body.question, include_trace=body.include_trace)
        except Exception as e:  # last-resort guard: never leak a stack trace to clients
            raise HTTPException(status_code=500, detail=f"internal error: {type(e).__name__}") from e
        request.app.state.stats.record(r)
        return r

    @app.get("/stats")
    def stats(request: Request) -> dict:
        return request.app.state.stats.snapshot()

    return app


app = create_app()
