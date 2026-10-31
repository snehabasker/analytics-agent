"""Load the Chinook dataset (a real digital-music-store schema) into DuckDB.

Source: https://github.com/lerocha/chinook-database  (downloaded by `make data`).
Tables are renamed to snake_case so the semantic layer reads cleanly.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "Chinook_Sqlite.sqlite"
DST = ROOT / "data" / "warehouse.duckdb"


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run `python scripts/download_data.py` first", file=sys.stderr)
        return 1
    DST.unlink(missing_ok=True)
    src = sqlite3.connect(SRC)
    dst = duckdb.connect(str(DST))
    tables = [r[0] for r in src.execute("select name from sqlite_master where type='table'")]
    for t in tables:
        cur = src.execute(f'select * from "{t}"')
        cols = [snake(d[0]) for d in cur.description]
        rows = cur.fetchall()
        col_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        types = _infer_types(rows, len(cols))
        ddl = ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types))
        dst.execute(f"create table {snake(t)} ({ddl})")
        if rows:
            dst.executemany(f"insert into {snake(t)} ({col_sql}) values ({placeholders})", rows)
        print(f"{snake(t):<16} {len(rows):>6} rows")
    dst.close()
    return 0


def _infer_types(rows: list[tuple], n: int) -> list[str]:
    out = []
    for i in range(n):
        vals = [r[i] for r in rows if r[i] is not None]
        if vals and all(isinstance(v, int) for v in vals):
            out.append("BIGINT")
        elif vals and all(isinstance(v, (int, float)) for v in vals):
            out.append("DOUBLE")
        elif vals and all(isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", v) for v in vals):
            out.append("TIMESTAMP")
        else:
            out.append("VARCHAR")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
