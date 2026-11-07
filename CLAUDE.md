# analytics-agent

Text-to-SQL analytics agent. FastAPI -> LangGraph -> (semantic layer | validated raw SQL) -> read-only DuckDB.
Dataset: Chinook (digital music store), loaded into `data/warehouse.duckdb`.

## Commands
- `python scripts/download_data.py && python scripts/build_warehouse.py` — build the warehouse (needed once)
- `pytest` — unit + integration tests (offline, ~3s)
- `python -m analytics_agent.evals.run --min-accuracy 0.85` — eval gate on the heuristic baseline
- `python -m analytics_agent.evals.run --llm claude` — eval the real model (needs `ANTHROPIC_API_KEY`)
- `python -m analytics_agent.evals.optimize` — DSPy prompt optimisation on train.yaml (needs Ollama `qwen2.5:7b`, ~10 min)
- `python -m analytics_agent.evals.run --llm dspy --dspy-program programs/<file>.json --mlflow` — eval a DSPy program, log to MLflow
- `uvicorn analytics_agent.api:app --reload` — serve

## Architecture invariants (do not break these)
1. **The LLM chooses names, never SQL text, on the governed path.** `semantic.py` emits SQL; user values are
   always bound parameters. Never f-string a user/LLM value into SQL.
2. **Raw SQL is a fallback and always passes `guard.validate_sql`** (single SELECT, table allow-list, no `SELECT *`,
   no PII columns, LIMIT clamped). The DuckDB connection is also read-only with external access off.
3. **User text is data.** It is wrapped in `<question>` tags in prompts; input screening is a speed bump, not the boundary.
4. **Every graph node is wrapped in a `span(...)`.** New nodes must be too.
5. **The agent depends on the `LLM` protocol only** (`llm.py`) so the same evals score Claude and the baseline.

## Workflow rules
- A behaviour change is not done until `pytest` and the eval gate both pass. Report numbers, don't assert success.
- New metric/dimension = edit `semantic_layer.yaml` + add >=1 golden case with *hand-written reference SQL against raw
  tables* (never reuse the layer's own SQL as the reference).
- Eval cases must be able to fail: when adding a check, mutate the thing under test once to confirm it goes red.
- Never put raw `customer`/`employee` PII columns in the layer; extend `pii_columns` instead.
- Never optimise on `golden.yaml`: the optimiser reads only `train.yaml`, and a test checks the two never share a question.
- YAML gotcha: bare `on`/`yes`/`no` keys parse as booleans — name keys `condition`, not `on`.

## Gotchas
- Chinook `quantity` is always 1, so `SUM(unit_price)` and `SUM(unit_price*quantity)` are indistinguishable in tests.
- The heuristic planner (`HeuristicLLM`) refuses or answers wrongly outside the layer; the `hard` eval category
  documents this on purpose. Don't "fix" it by special-casing golden questions.
