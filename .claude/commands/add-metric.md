---
description: Add a metric to the semantic layer the safe way
argument-hint: <metric name and business definition>
---
Add this metric to the semantic layer: $ARGUMENTS

Steps:
1. Inspect real columns first (`describe` the tables in `data/warehouse.duckdb`); do not guess names.
2. Add it to `src/analytics_agent/semantic_layer.yaml` (expr, description, format, synonyms, `requires` if needed).
3. Add >=2 cases to `src/analytics_agent/evals/golden.yaml` with reference SQL written against the raw tables.
4. Mutation-check: temporarily break the metric's expr, confirm the new cases fail, restore it.
5. Run `pytest` and the eval gate; report the numbers.
