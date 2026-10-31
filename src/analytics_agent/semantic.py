"""Semantic layer: typed definitions plus a compiler from SemanticQuery to parameterised SQL.

Design rule: every identifier in the emitted SQL comes from the YAML layer, and every
user-supplied value is bound as a parameter. The LLM can therefore only choose *names*,
never write SQL text, on the governed path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_LAYER = Path(__file__).with_name("semantic_layer.yaml")
MAX_LIMIT = 1000
DEFAULT_LIMIT = 100


class SemanticError(ValueError):
    """The query references something the layer does not define."""


class Metric(BaseModel):
    name: str
    description: str
    expr: str
    requires: list[str] = []
    round: int | None = None
    format: str = "number"
    synonyms: list[str] = []


class Dimension(BaseModel):
    name: str
    alias: str
    expr: str
    type: Literal["string", "number", "time"] = "string"
    categorical: bool = False
    description: str = ""
    synonyms: list[str] = []


class Join(BaseModel):
    alias: str
    table: str
    parent: str
    condition: str


class Example(BaseModel):
    question: str
    query: dict[str, Any]


class Filter(BaseModel):
    dimension: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains"] = "="
    value: str | int | float | list[str | int | float]


class SemanticQuery(BaseModel):
    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = []
    filters: list[Filter] = []
    order_by: str | None = None
    descending: bool | None = None
    limit: int | None = None

    @field_validator("limit")
    @classmethod
    def _limit_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("limit must be >= 1")
        return v


class SemanticLayer(BaseModel):
    base: dict[str, str]
    joins: list[Join]
    allowed_tables: list[str]
    pii_columns: list[str]
    metrics: list[Metric]
    dimensions: list[Dimension]
    examples: list[Example] = []

    @classmethod
    def load(cls, path: Path | str = DEFAULT_LAYER) -> "SemanticLayer":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def metric(self, name: str) -> Metric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise SemanticError(f"unknown metric {name!r}; available: {[m.name for m in self.metrics]}")

    def dimension(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise SemanticError(f"unknown dimension {name!r}; available: {[d.name for d in self.dimensions]}")

    # --- compilation -------------------------------------------------------------

    def _ancestors(self, alias: str) -> set[str]:
        by_alias = {j.alias: j for j in self.joins}
        out: set[str] = set()
        while alias != self.base["alias"]:
            if alias not in by_alias:
                raise SemanticError(f"alias {alias!r} is not joinable from the base table")
            out.add(alias)
            alias = by_alias[alias].parent
        return out

    def compile(self, q: SemanticQuery) -> tuple[str, list[Any]]:
        metrics = [self.metric(n) for n in q.metrics]
        dims = [self.dimension(n) for n in q.dimensions]
        filters = [(self.dimension(f.dimension), f) for f in q.filters]

        needed: set[str] = set()
        for m in metrics:
            for a in m.requires:
                needed |= self._ancestors(a)
        for d in dims + [d for d, _ in filters]:
            needed |= self._ancestors(d.alias)

        select = [f'{d.expr} AS "{d.name}"' for d in dims]
        for m in metrics:
            expr = f"ROUND({m.expr}, {m.round})" if m.round is not None else m.expr
            select.append(f'{expr} AS "{m.name}"')

        sql = [f"SELECT {', '.join(select)}", f"FROM {self.base['table']} {self.base['alias']}"]
        for j in self.joins:  # declaration order guarantees parents come first
            if j.alias in needed:
                sql.append(f"JOIN {j.table} {j.alias} ON {j.condition}")

        params: list[Any] = []
        where = []
        for d, f in filters:
            clause, p = _filter_clause(d, f)
            where.append(clause)
            params.extend(p)
        if where:
            sql.append("WHERE " + " AND ".join(where))
        if dims:
            sql.append("GROUP BY " + ", ".join(str(i + 1) for i in range(len(dims))))

        order, desc = self._order(q, metrics, dims)
        sql.append(f'ORDER BY "{order}" {"DESC" if desc else "ASC"}')
        sql.append(f"LIMIT {min(q.limit or DEFAULT_LIMIT, MAX_LIMIT)}")
        return "\n".join(sql), params

    def _order(self, q: SemanticQuery, metrics: list[Metric], dims: list[Dimension]) -> tuple[str, bool]:
        if q.order_by:
            if q.order_by not in {m.name for m in metrics} | {d.name for d in dims}:
                raise SemanticError(f"order_by {q.order_by!r} must be one of the selected metrics/dimensions")
            default_desc = q.order_by in {m.name for m in metrics}
            return q.order_by, default_desc if q.descending is None else q.descending
        time_dims = [d for d in dims if d.type == "time"]
        if time_dims:  # time series read chronologically unless told otherwise
            return time_dims[0].name, bool(q.descending)
        return metrics[0].name, True if q.descending is None else q.descending

    # --- prompt / retrieval helpers ----------------------------------------------

    def catalog_text(self) -> str:
        lines = ["METRICS:"]
        lines += [f"- {m.name}: {m.description}" for m in self.metrics]
        lines.append("DIMENSIONS:")
        lines += [f"- {d.name} ({d.type}): {d.description}" for d in self.dimensions]
        return "\n".join(lines)


def _filter_clause(d: Dimension, f: Filter) -> tuple[str, list[Any]]:
    if f.op == "in":
        vals = f.value if isinstance(f.value, list) else [f.value]
        if not vals:
            raise SemanticError("'in' filter needs at least one value")
        return f"{d.expr} IN ({', '.join('?' for _ in vals)})", list(vals)
    if isinstance(f.value, list):
        raise SemanticError(f"operator {f.op!r} takes a single value")
    if f.op == "contains":
        return f"{d.expr} ILIKE ?", [f"%{f.value}%"]
    return f"{d.expr} {f.op} ?", [f.value]
