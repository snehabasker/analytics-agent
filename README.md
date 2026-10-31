# analytics-agent

Ask questions about a real dataset in plain English; get a verified SQL answer, with the SQL, the rows and a trace.

**FastAPI + LangGraph agent · semantic layer · Qdrant retrieval · DuckDB · eval suite with CI gate · tracing · Docker**
Dataset: [Chinook](https://github.com/lerocha/chinook-database) digital music store (412 invoices, 2,240 lines, 3,503 tracks, 2021–2025).

```
question ─► screen ─► retrieve (Qdrant) ─► plan (Claude, forced tool call) ─► validate ─► execute ─► respond
              │                                ▲                │               │
              └─ refuse                        └── repair ◄─────┴───────────────┘  (bounded)

  plan = semantic query (names only → compiled, parameterised SQL)   ← preferred, governed
       | raw SQL  (single SELECT, AST-validated, allow-listed, PII-blocked, LIMIT-clamped)
       | refuse
```

## Quick start

```bash
pip install -e ".[dev]"
python scripts/download_data.py && python scripts/build_warehouse.py
pytest                                              # 45 tests, offline
python -m analytics_agent.evals.run                 # eval report -> reports/
uvicorn analytics_agent.api:app                     # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Top 5 genres by units sold", "include_trace": true}'
```

Without `ANTHROPIC_API_KEY` it runs the deterministic heuristic planner (free, offline). With a key it uses Claude
(`ANALYTICS_MODEL`, default `claude-sonnet-5-5`; narration uses `claude-haiku-5-5`).
Docker: `docker compose up --build` (adds a Qdrant server).

## What's in it

| Piece | Where | Notes |
|---|---|---|
| Semantic layer | [semantic_layer.yaml](src/analytics_agent/semantic_layer.yaml), [semantic.py](src/analytics_agent/semantic.py) | 5 metrics, 11 dimensions, join graph; compiler emits only the joins needed; values bound as params |
| Agent graph | [agent.py](src/analytics_agent/agent.py) | LangGraph, bounded self-repair, early refusal |
| Planner | [llm.py](src/analytics_agent/llm.py) | Claude via forced tool call + prompt caching; heuristic baseline behind the same protocol |
| Retrieval | [retrieval.py](src/analytics_agent/retrieval.py) | Qdrant over metrics, dimensions, few-shots and exact categorical values |
| Guardrails | [guard.py](src/analytics_agent/guard.py), [db.py](src/analytics_agent/db.py) | injection screen, SQL AST validator, PII deny-list, read-only DuckDB, timeout, row cap |
| Tracing | [tracing.py](src/analytics_agent/tracing.py) | span per node, token usage, JSONL export, `trace_id` in every response |
| API | [api.py](src/analytics_agent/api.py) | `/ask`, `/health`, `/semantic-layer`, `/stats` |
| Evals | [evals/](src/analytics_agent/evals) | 43 cases, execution accuracy vs hand-written reference SQL, safety recall, false-refusal rate |
| CI | [ci.yml](.github/workflows/ci.yml) | tests + eval gate on every push; optional Claude eval on dispatch |

## Results (heuristic baseline — see caveats)

| Category | Pass rate |
|---|---|
| simple metric / group-by / filter / top-N / time series | 100% |
| safety (9 must-refuse: injection, destructive, PII, off-topic) | 100% |
| benign look-alikes wrongly refused | 0% |
| **hard (HAVING, ratios, multi-period)** | **0%** |
| **Overall execution accuracy** | **88.2%** |

The `hard` row is deliberate: outside the layer the baseline returns **confidently wrong answers** instead of
refusing. That gap is what the raw-SQL path and Claude are for. Reports: [reports/eval_heuristic.md](reports/eval_heuristic.md).

**Not yet measured:** Claude's score. The Claude path is tested against a mocked SDK client but has not run against the
live API in this build — run `python -m analytics_agent.evals.run --llm claude` and add the number here.
Docker files and the remote-Qdrant path are written but unbuilt/untested on this machine.

## Built with Claude Code

How Claude was used, what it got wrong and how that was caught, and what is unverified:
**[docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md)**. Design rationale: [docs/DECISIONS.md](docs/DECISIONS.md).
Project memory for the tool: [CLAUDE.md](CLAUDE.md), [.claude/commands/](.claude/commands).
