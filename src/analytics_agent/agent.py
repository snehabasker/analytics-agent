"""The LangGraph agent.

    screen -> retrieve -> plan -> validate -> execute -> respond
                           ^         |           |
                           +---------+-----------+   (bounded self-repair on error)

Every node is wrapped in a trace span. Refusals short-circuit straight to `respond`.
"""
from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from pydantic import ValidationError

from .config import Settings
from .db import QueryError, QueryResult, Warehouse
from .guard import screen_input, validate_sql
from .llm import LLM, ClaudeLLM, HeuristicLLM, Plan, PlanError
from .retrieval import CohereEmbedder, CohereReranker, HashingEmbedder, SemanticIndex
from .semantic import SemanticError, SemanticLayer, SemanticQuery
from .tracing import span, start_trace

Status = Literal["ok", "refused", "error"]


class State(TypedDict, total=False):
    question: str
    context: list[str]
    plan: Plan | None
    sql: str | None
    params: list[Any]
    result: QueryResult | None
    error: str | None
    attempts: int
    status: Status
    refusal_reason: str | None
    answer: str


class AgentResponse(BaseModel):
    question: str
    status: Status
    answer: str
    plan_kind: str | None = None
    sql: str | None = None
    params: list[Any] = []
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False
    attempts: int = 0
    refusal_reason: str | None = None
    llm: str
    latency_ms: float
    trace_id: str
    trace: list[dict[str, Any]] | None = None


class AnalyticsAgent:
    def __init__(self, settings: Settings, layer: SemanticLayer, warehouse: Warehouse, index: SemanticIndex, llm: LLM):
        self.settings, self.layer, self.warehouse, self.index, self.llm = settings, layer, warehouse, index, llm
        self._allowed = {t.lower() for t in layer.allowed_tables}
        self._pii = {c.lower() for c in layer.pii_columns}
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes

    def _screen(self, s: State) -> State:
        with span("screen") as sp:
            v = screen_input(s["question"])
            sp.set(ok=v.ok, reason=v.reason)
        return {} if v.ok else {"status": "refused", "refusal_reason": v.reason}

    def _retrieve(self, s: State) -> State:
        with span("retrieve") as sp:
            ctx = self.index.context_for(s["question"])
            sp.set(hits=len(ctx), index_size=self.index.size)
        return {"context": ctx}

    def _plan(self, s: State) -> State:
        with span("plan", attempt=s.get("attempts", 0), llm=self.llm.name) as sp:
            try:
                plan = self.llm.plan(s["question"], s.get("context", []), s.get("error"))
            except PlanError as e:
                sp.set(plan_error=str(e))
                return {"error": str(e), "attempts": s.get("attempts", 0) + 1, "plan": None}
            sp.set(kind=plan.kind, reason=plan.reason)
        if plan.kind == "refuse":
            return {"plan": plan, "status": "refused", "refusal_reason": plan.reason or "refused by planner", "error": None}
        return {"plan": plan, "error": None}

    def _validate(self, s: State) -> State:
        plan = s["plan"]
        with span("validate", kind=plan.kind) as sp:
            try:
                if plan.kind == "semantic":
                    if plan.semantic_query is None:
                        raise SemanticError("kind='semantic' requires semantic_query")
                    sql, params = self.layer.compile(plan.semantic_query)
                else:
                    if not plan.sql:
                        raise SemanticError("kind='sql' requires sql")
                    sql, verdict = validate_sql(plan.sql, self._allowed, self._pii, self.settings.max_rows)
                    params = []
                    if not verdict.ok:
                        raise SemanticError(verdict.reason)
            except (SemanticError, ValidationError) as e:
                sp.set(rejected=str(e))
                return {"error": f"validation failed: {e}", "attempts": s.get("attempts", 0) + 1}
            sp.set(sql=sql)
        return {"sql": sql, "params": params, "error": None}

    def _execute(self, s: State) -> State:
        with span("execute", sql=s["sql"]) as sp:
            try:
                res = self.warehouse.execute(s["sql"], s.get("params"))
            except QueryError as e:
                sp.set(db_error=str(e))
                return {"error": f"execution failed: {e}", "attempts": s.get("attempts", 0) + 1}
            sp.set(rows=res.row_count, truncated=res.truncated)
        return {"result": res, "error": None, "status": "ok"}

    def _respond(self, s: State) -> State:
        with span("respond", status=s.get("status")):
            if s.get("status") == "refused":
                return {"answer": f"I can't help with that: {s.get('refusal_reason')}", "status": "refused"}
            if s.get("error"):
                return {"answer": "I couldn't produce a reliable answer after several attempts. "
                                  f"Last problem: {s['error']}", "status": "error"}
            return {"answer": self.llm.narrate(s["question"], s["result"], s["plan"]), "status": "ok"}

    # ------------------------------------------------------------------ routing

    def _after_screen(self, s: State) -> str:
        return "respond" if s.get("status") == "refused" else "retrieve"

    def _after_plan(self, s: State) -> str:
        if s.get("status") == "refused":
            return "respond"
        if s.get("error"):
            return "plan" if s.get("attempts", 0) <= self.settings.max_repair_attempts else "respond"
        return "validate"

    def _after_step(self, ok_next: str):
        def route(s: State) -> str:
            if s.get("error"):
                return "plan" if s.get("attempts", 0) <= self.settings.max_repair_attempts else "respond"
            return ok_next
        return route

    def _build_graph(self):
        g = StateGraph(State)
        for name, fn in [("screen", self._screen), ("retrieve", self._retrieve), ("plan", self._plan),
                         ("validate", self._validate), ("execute", self._execute), ("respond", self._respond)]:
            g.add_node(name, fn)
        g.add_edge(START, "screen")
        g.add_conditional_edges("screen", self._after_screen, ["retrieve", "respond"])
        g.add_edge("retrieve", "plan")
        g.add_conditional_edges("plan", self._after_plan, ["plan", "validate", "respond"])
        g.add_conditional_edges("validate", self._after_step("execute"), ["plan", "execute", "respond"])
        g.add_conditional_edges("execute", self._after_step("respond"), ["plan", "respond"])
        g.add_edge("respond", END)
        return g.compile()

    # ------------------------------------------------------------------ public API

    def ask(self, question: str, include_trace: bool = False) -> AgentResponse:
        t0 = time.perf_counter()
        with start_trace(self.settings.trace_file) as trace:
            with span("ask", question=question[:200], llm=self.llm.name):
                final: State = self.graph.invoke({"question": question, "attempts": 0})
        res: QueryResult | None = final.get("result") if final.get("status") == "ok" else None
        plan: Plan | None = final.get("plan")
        return AgentResponse(
            question=question, status=final["status"], answer=final["answer"],
            plan_kind=plan.kind if plan else None, sql=final.get("sql") if res else None,
            params=final.get("params", []) if res else [],
            columns=res.columns if res else [], rows=res.rows if res else [],
            row_count=res.row_count if res else 0, truncated=res.truncated if res else False,
            attempts=final.get("attempts", 0), refusal_reason=final.get("refusal_reason"),
            llm=self.llm.name, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            trace_id=trace.trace_id, trace=trace.export() if include_trace else None,
        )

    def close(self) -> None:
        self.warehouse.close()


# --------------------------------------------------------------------------- factory

def _schema_text(wh: Warehouse, layer: SemanticLayer) -> str:
    pii = {c.lower() for c in layer.pii_columns}
    lines = []
    for t in layer.allowed_tables:
        cols = wh.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [t])
        names = [r[0] for r in cols.rows if r[0].lower() not in pii]
        lines.append(f"- {t}({', '.join(names)})")
    return "\n".join(lines)


def _categorical_values(wh: Warehouse, layer: SemanticLayer) -> dict[str, list[str]]:
    out = {}
    for d in layer.dimensions:
        if d.categorical:
            sql, _ = layer.compile(SemanticQuery(metrics=["revenue"], dimensions=[d.name], limit=1000))
            out[d.name] = [str(r[0]) for r in wh.execute(sql).rows]
    return out


def make_embedder(settings: Settings):
    if settings.embedder == "cohere":
        return CohereEmbedder(settings.cohere_embed_model)
    if settings.embedder != "hashing":
        raise ValueError(f"unknown embedder {settings.embedder!r}")
    return HashingEmbedder()


def make_reranker(settings: Settings):
    if settings.reranker == "cohere":
        return CohereReranker(settings.cohere_rerank_model)
    if settings.reranker != "none":
        raise ValueError(f"unknown reranker {settings.reranker!r}")
    return None


def build_agent(settings: Settings | None = None, llm: LLM | None = None, layer: SemanticLayer | None = None) -> AnalyticsAgent:
    settings = settings or Settings()
    if not settings.db_path.exists():
        raise FileNotFoundError(f"warehouse not found at {settings.db_path}; run `make data`")
    layer = layer or SemanticLayer.load()
    wh = Warehouse(settings.db_path, settings.max_rows, settings.query_timeout_s)
    values = _categorical_values(wh, layer)
    index = SemanticIndex(layer, embedder=make_embedder(settings), qdrant_url=settings.qdrant_url,
                          values=values, reranker=make_reranker(settings))
    if llm is None:
        if settings.resolved_llm() == "claude":
            llm = ClaudeLLM(layer, _schema_text(wh, layer), settings.model, settings.narrate_model)
        else:
            llm = HeuristicLLM(layer, values)
    return AnalyticsAgent(settings, layer, wh, index, llm)
