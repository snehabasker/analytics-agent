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
pytest                                              # 46 tests, offline
python -m analytics_agent.evals.run                 # eval report -> reports/
python -m analytics_agent.evals.retrieval           # retrieval A/B -> reports/retrieval_ab/
python -m analytics_agent.evals.run --mlflow        # same eval, also tracked in MLflow (mlflow ui --backend-store-uri sqlite:///mlflow.db)
uvicorn analytics_agent.api:app                     # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Top 5 genres by units sold", "include_trace": true}'
```

Without `ANTHROPIC_API_KEY` it runs the deterministic heuristic planner (free, offline). With a key it uses Claude
(`ANALYTICS_MODEL`, default `claude-sonnet-5-5`; narration uses `claude-haiku-5-5`).
Docker: `docker compose up --build` (adds a Qdrant server).

Retrieval defaults to an offline hashing embedder. To use Cohere instead (`pip install -e ".[cohere]"`, `COHERE_API_KEY`):
`ANALYTICS_EMBEDDER=cohere` (recommended, see results below) and optionally `ANALYTICS_RERANKER=cohere`.
Models are set with `COHERE_EMBED_MODEL` (default `embed-v4.0`) and `COHERE_RERANK_MODEL` (default `rerank-v3.5`).

## What's in it

| Piece | Where | Notes |
|---|---|---|
| Semantic layer | [semantic_layer.yaml](src/analytics_agent/semantic_layer.yaml), [semantic.py](src/analytics_agent/semantic.py) | 5 metrics, 11 dimensions, join graph; compiler emits only the joins needed; values bound as params |
| Agent graph | [agent.py](src/analytics_agent/agent.py) | LangGraph, bounded self-repair, early refusal |
| Planner | [llm.py](src/analytics_agent/llm.py), [dspy_planner.py](src/analytics_agent/dspy_planner.py) | Claude via forced tool call + prompt caching; DSPy module (any LiteLLM model, default local Ollama) with optimisable demos; heuristic baseline; all behind the same protocol |
| Prompt optimisation | [evals/optimize.py](src/analytics_agent/evals/optimize.py), [train.yaml](src/analytics_agent/evals/train.yaml) | DSPy `BootstrapFewShotWithRandomSearch` on a separate 31-case training set, metric = execution match through the agent's own validate/execute nodes |
| Experiment tracking | [evals/tracking.py](src/analytics_agent/evals/tracking.py) | MLflow: params, metrics, per-case table, reports and optimised program per run; DSPy autolog on optimisation runs |
| Retrieval | [retrieval.py](src/analytics_agent/retrieval.py) | Qdrant over metrics, dimensions, few-shots and exact categorical values; hashing (offline) or Cohere Embed, optional Cohere Rerank |
| Guardrails | [guard.py](src/analytics_agent/guard.py), [db.py](src/analytics_agent/db.py) | injection screen, SQL AST validator, PII deny-list, read-only DuckDB, timeout, row cap |
| Tracing | [tracing.py](src/analytics_agent/tracing.py) | span per node, token usage, JSONL export, `trace_id` in every response |
| API | [api.py](src/analytics_agent/api.py) | `/ask`, `/health`, `/semantic-layer`, `/stats` |
| Evals | [evals/](src/analytics_agent/evals) | 43 cases, execution accuracy vs hand-written reference SQL, safety recall, false-refusal rate; retrieval A/B on hand-labelled needed items |
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

## Retrieval: hashing vs Cohere Embed / Rerank

Does the prompt context contain every metric, dimension and categorical value a question needs? Measured on the 34
answerable golden questions, labelled by hand from the reference SQL
([retrieval_labels.yaml](src/analytics_agent/evals/retrieval_labels.yaml)), scored on the exact slices the planner sees.

| Retriever | All needed items in context | Value recall@4 | p50 latency |
|---|---|---|---|
| Hashing (default, offline) | 74% | 86% | 9 ms |
| Hashing + Cohere Rerank | 91% | 71% | 295 ms |
| **Cohere Embed** | **97%** | **100%** | 360 ms |
| Cohere Embed + Rerank | 91% | 86% | 652 ms |
| Random vectors (control) | 15% | 14% | – |

Cohere Embed fixes the lexical misses (e.g. "United States" → stored value `USA`). Rerank *hurts* it: it scores each
candidate against the whole question, so on multi-intent questions it favours the main intent and pushes out the
filter (e.g. drops `country=Germany` for "top 3 genres by revenue in Germany"). Caveats: 34 questions (one question ≈ 3
points), and the golden questions overlap the layer's synonyms, which favours hashing. This measures retrieval only, not
answer accuracy. Full report: [reports/retrieval_ab/report.md](reports/retrieval_ab/report.md).

## Prompt optimisation with DSPy (tracked in MLflow)

The planner is also a DSPy module ([dspy_planner.py](src/analytics_agent/dspy_planner.py)). It is optimised on
[train.yaml](src/analytics_agent/evals/train.yaml), 31 new hand-labelled questions that appear in neither the golden
set nor the layer's few-shot examples (a test enforces this). The 43 golden cases are only ever used for the numbers
below, so they are held out. Student model: **qwen2.5:7b on local Ollama** (RTX 5050 laptop GPU, ~2.5 s per plan).

```bash
pip install -e ".[dspy,mlflow]"
python -m analytics_agent.evals.optimize                                   # ~9 min, saves programs/planner_qwen25-7b.json
python -m analytics_agent.evals.run --llm dspy --dspy-program programs/planner_qwen25-7b.json --mlflow
```

Execution accuracy on the 43 golden cases (34 answerable):

| Planner | Overall | filter (7) | time (2) | benign (2) | hard (4) | Safety recall |
|---|---|---|---|---|---|---|
| Heuristic baseline (no LLM) | 88.2% | 100% | 100% | 100% | 0% | 100% |
| qwen2.5:7b, DSPy zero-shot | 67.6% | 43% | 0% | 50% | 0% | 100% |
| qwen2.5:7b, DSPy optimised (self-bootstrapped demos) | 76.5% | 43% | **100%** | 100% | 0% | 100% |
| qwen2.5:7b, DSPy optimised (demos from Cohere Command A as teacher) | **82.4%** | **86%** | 50% | 100% | 0% | 100% |

What the numbers say:

- **Optimisation helps, by fixing output shape.** Zero-shot, the model puts filter values into the GROUP BY ("Rock revenue
  in 2024" → rows per year), doesn't order time series and adds columns nobody asked for. Two bootstrapped demos lift
  accuracy by 9 to 15 points. Validation scores during optimisation went from 35.7% to 71.4%.
- **Demo sets trade categories off against each other.** The teacher's winning demo set includes a filter question
  (filter 43% → 86%) but no time-series one (time 100% → 50%). With 2 demos per program, which questions they cover
  matters more than who wrote them.
- **`hard` stays at 0%.** Neither winning program contains a raw-SQL demo, and without one the 7B model never picks
  the SQL path for HAVING/share/comparison questions. Next step: force at least one `kind: sql` demo per program, or try
  `MIPROv2`/GEPA, which also rewrite the instructions.
- **Safety never regressed** (100% recall, 0 false refusals) across all optimised programs.
- **Still below the heuristic baseline (88.2%)** on this layer-shaped golden set. The case for an LLM planner is the
  questions the heuristic cannot parse, which this suite only samples through `hard`.

Caveats: 34 answerable cases, so one question is about 3 points; one seed per configuration; train/val split of 17/14.
Every run (params, metrics, per-case table, program JSON, DSPy optimisation traces) is in MLflow; the reports are in
[reports/](reports).

**Not yet measured:** Claude's score. The Claude path is tested against a mocked SDK client but has not run against the
live API in this build — run `python -m analytics_agent.evals.run --llm claude` and add the number here.
Docker files and the remote-Qdrant path are written but unbuilt/untested on this machine.

## Built with Claude Code

How Claude was used, what it got wrong and how that was caught, and what is unverified:
**[docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md)**. Design rationale: [docs/DECISIONS.md](docs/DECISIONS.md).
Project memory for the tool: [CLAUDE.md](CLAUDE.md), [.claude/commands/](.claude/commands).
