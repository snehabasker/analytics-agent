"""Safety layers: input screening and raw-SQL validation.

Defence in depth. Input screening is a cheap heuristic and *not* the security boundary;
the boundary is (1) the AST validator below, (2) the read-only, no-external-access
DuckDB connection, and (3) never executing anything the LLM wrote without validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now\b",
    r"\bjailbreak\b",
    r"developer mode",
    r"act as (an? )?(unrestricted|unfiltered|dan)\b",
    r"\b(drop|truncate|delete from|alter|insert into|update)\s+(table\s+)?\w+",
    r"(export|dump|list|show)\b.*\b(emails?|phone numbers?|addresses|passwords?)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I)

_BLOCKED_FUNCS = {"glob", "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_text",
                  "read_blob", "sniff_csv", "query", "query_table", "duckdb_settings", "pragma_database_list"}
_FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                    exp.Command, exp.Copy, exp.Set, exp.Use, exp.Merge, exp.Pragma, exp.Attach)


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


def screen_input(question: str) -> Verdict:
    if not question.strip():
        return Verdict(False, "empty question")
    if len(question) > 1000:
        return Verdict(False, "question too long (max 1000 characters)")
    if m := _INJECTION_RE.search(question):
        return Verdict(False, f"blocked by input screening: {m.group(0)!r}")
    return Verdict(True)


def validate_sql(sql: str, allowed_tables: set[str], pii_columns: set[str], max_limit: int) -> tuple[str, Verdict]:
    """Return (safe_sql, verdict). safe_sql has a LIMIT enforced; it is only meaningful if verdict.ok."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        return sql, Verdict(False, f"SQL parse error: {str(e).splitlines()[0]}")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return sql, Verdict(False, "exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        return sql, Verdict(False, f"only SELECT queries are allowed (got {type(tree).__name__})")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return sql, Verdict(False, f"forbidden operation: {type(node).__name__}")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if not t.name:
            return sql, Verdict(False, "table functions are not allowed")
        if t.name.lower() not in allowed_tables | ctes:
            return sql, Verdict(False, f"table {t.name!r} is not in the allow-list")
    for f in tree.find_all(exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if name in _BLOCKED_FUNCS or name.startswith("read_"):
            return sql, Verdict(False, f"function {name!r} is not allowed")
    for s in tree.find_all(exp.Star):
        if not isinstance(s.parent, exp.Count):
            return sql, Verdict(False, "SELECT * is not allowed; list columns explicitly")
    for c in tree.find_all(exp.Column):
        if c.name.lower() in pii_columns:
            return sql, Verdict(False, f"column {c.name!r} is restricted (PII)")

    limit = tree.args.get("limit")
    if limit is None or not _limit_ok(limit, max_limit):
        tree = tree.limit(max_limit)
    return tree.sql(dialect="duckdb"), Verdict(True)


def _limit_ok(limit: exp.Limit, max_limit: int) -> bool:
    try:
        return int(limit.expression.name) <= max_limit
    except (ValueError, AttributeError):
        return False
