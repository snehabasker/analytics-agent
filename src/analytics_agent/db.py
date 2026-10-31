"""Read-only DuckDB executor with a row cap and a wall-clock timeout."""
from __future__ import annotations

import datetime as dt
import decimal
import threading
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False


class QueryError(RuntimeError):
    """Execution failed; the message is safe to feed back to the LLM for repair."""


class Warehouse:
    def __init__(self, path: Path, max_rows: int = 200, timeout_s: float = 10.0):
        # read_only + no external access: even a bypassed validator cannot write or read files.
        self._con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def execute(self, sql: str, params: list[Any] | None = None) -> QueryResult:
        cur = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            fetched = cur.fetchmany(self.max_rows + 1)
        except duckdb.InterruptException as e:
            raise QueryError(f"query exceeded {self.timeout_s}s timeout") from e
        except duckdb.Error as e:
            raise QueryError(str(e).splitlines()[0]) from e
        finally:
            timer.cancel()
            cur.close()
        truncated = len(fetched) > self.max_rows
        rows = [[_jsonable(v) for v in r] for r in fetched[: self.max_rows]]
        return QueryResult(columns=cols, rows=rows, row_count=len(rows), truncated=truncated)

    def distinct_values(self, expr: str, from_sql: str, limit: int = 500) -> list[str]:
        res = self.execute(f"SELECT DISTINCT {expr} AS v FROM {from_sql} WHERE {expr} IS NOT NULL LIMIT {limit}")
        return [str(r[0]) for r in res.rows]

    def close(self) -> None:
        self._con.close()


def _jsonable(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v
