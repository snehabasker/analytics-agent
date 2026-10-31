# Design decisions

**1. Semantic layer first, raw SQL as fallback.** Free-form text-to-SQL fails silently (wrong join, wrong grain, a
metric defined three ways). Named metrics compile to one trusted definition, so "revenue" means the same thing
everywhere and the model's job shrinks from *writing SQL* to *choosing names*. Raw SQL stays for questions the layer
can't express, behind an AST validator.

**2. DuckDB + Chinook.** Embedded, zero-ops, fast analytical engine; Chinook is a real multi-table star-ish schema
(invoice_line → track → album → artist/genre) so the join graph is non-trivial. Swapping to Postgres means replacing
`Warehouse` and the `strftime`/`EXTRACT` expressions in the YAML.

**3. Qdrant for retrieval, with a hashing embedder.** Retrieval narrows the prompt to relevant metrics, dimensions,
few-shot examples and exact categorical values. In-process Qdrant by default (`:memory:`), a server via `QDRANT_URL`.
The embedder is deterministic so CI needs no model download; the `Embedder` protocol makes upgrading trivial.
Trade-off: lexical, not semantic — paraphrases rely on the synonym lists.

**4. LangGraph for control flow.** The repair loop, early refusal and per-node tracing are explicit graph edges rather
than hidden in a prompt, which makes each independently testable (`tests/test_agent.py`).

**5. Two planners behind one protocol.** `ClaudeLLM` for production, `HeuristicLLM` for free deterministic CI and as
an honest baseline. Evals run against either; CI gates on the baseline, Claude evals run on demand.

**6. Execution accuracy, not string match.** Cases pass when the result *set* equals the reference SQL's, so equivalent
queries pass and wrong-but-plausible ones fail. Reference SQL is written against raw tables to avoid circular checks.

**7. Own tracing instead of a vendor SDK.** OpenTelemetry-shaped spans to JSONL: no dependency, trivially swapped for
OTLP/Langfuse later.

**8. Model choice.** Planner: `claude-sonnet-5-5` (structured-output reliability on a hard step). Narration:
`claude-haiku-5-5` (summarising already-computed rows). Both env-configurable.
