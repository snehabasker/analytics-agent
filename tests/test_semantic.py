import pytest

from analytics_agent.semantic import SemanticError, SemanticQuery


def q(**kw):
    return SemanticQuery(**kw)


def test_only_needed_joins_are_emitted(layer):
    sql, _ = layer.compile(q(metrics=["revenue"]))
    assert "JOIN" not in sql
    sql, _ = layer.compile(q(metrics=["revenue"], dimensions=["artist"]))
    # artist is three hops from the fact table: track -> album -> artist
    for table in ("track", "album", "artist"):
        assert f"JOIN {table} " in sql
    assert "JOIN genre" not in sql


def test_values_are_bound_never_interpolated(layer):
    evil = "Rock'; DROP TABLE customer; --"
    sql, params = layer.compile(q(metrics=["revenue"], filters=[{"dimension": "genre", "value": evil}]))
    assert evil not in sql and params == [evil]


def test_unknown_names_are_rejected_with_helpful_message(layer):
    with pytest.raises(SemanticError, match="available"):
        layer.compile(q(metrics=["profit"]))
    with pytest.raises(SemanticError):
        layer.compile(q(metrics=["revenue"], dimensions=["shoe_size"]))


def test_order_by_must_be_selected(layer):
    with pytest.raises(SemanticError):
        layer.compile(q(metrics=["revenue"], dimensions=["genre"], order_by="units_sold"))


def test_time_series_defaults_to_chronological(layer):
    sql, _ = layer.compile(q(metrics=["revenue"], dimensions=["year"]))
    assert 'ORDER BY "year" ASC' in sql


def test_limit_is_capped(layer):
    sql, _ = layer.compile(q(metrics=["revenue"], dimensions=["track"], limit=10**6))
    assert sql.rstrip().endswith("LIMIT 1000")


def test_in_filter_expands_placeholders(layer):
    sql, params = layer.compile(q(metrics=["revenue"], filters=[{"dimension": "year", "op": "in", "value": [2023, 2024]}]))
    assert "IN (?, ?)" in sql and params == [2023, 2024]
