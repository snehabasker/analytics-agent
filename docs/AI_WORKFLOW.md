# How this project was built with Claude (and how to tell it was done well)

Claude Code did the typing; the engineering judgement is in *what the work was anchored to*. The signals below are
what a reviewer can verify in this repo, not claims.

## What good AI-assisted engineering looks like — and where to find it here

| Signal | Why it matters | Evidence in this repo |
|---|---|---|
| **Ground in reality before writing code** | Models guess schemas; engineers inspect them | Dataset was downloaded and every table/column `describe`d before the semantic layer was written (`scripts/build_warehouse.py`, `semantic_layer.yaml`) |
| **A contract the model can't break** | Prompts are advice; types and validators are enforcement | LLM emits a typed `Plan` via a *forced* tool call; SQL is compiled from the layer or AST-validated (`semantic.py`, `guard.py`) |
| **Evals before features, and evals that can fail** | "It works" from a model is not evidence | `evals/` scores result-set equality against hand-written reference SQL. Verified red: breaking the `revenue` metric dropped accuracy 100% → 36.7% |
| **Honest baselines** | Shows you know where the model earns its keep | Deterministic heuristic planner scores 100% on layer-shaped questions and **0% on the `hard` set**, with confidently wrong answers — the case for the raw-SQL path and for Claude |
| **Defence in depth, not one prompt** | Injection can't be solved by instructions alone | Input screen → typed plan → AST validator → read-only/no-external-access DB → row cap + timeout. Prompt says "user text is data" *and* tests prove the boundary holds without it |
| **Cost/latency design** | Production LLM work is a budget problem | Prompt-cache on the static prefix, temperature 0, smaller model for narration, retrieval instead of stuffing the catalog |
| **Observability** | You can't debug an agent from its output | Span per graph node, token usage per LLM call, JSONL export, `trace_id` on every response |
| **Bounded autonomy** | Agents that retry forever are incidents | Self-repair loop with a hard cap; validator/DB errors are fed back verbatim (tested) |
| **Project memory for the tool** | Makes the AI's behaviour repeatable by a teammate | `CLAUDE.md` (invariants, gotchas), `.claude/commands/` (`/eval`, `/add-metric`), `.claude/settings.json` (allow-list + deny `.env`) |
| **Reviewing the AI's mistakes** | The honest log is the strongest signal | See below |

## Mistakes caught during the build (kept on purpose)

1. **YAML `on:` parsed as boolean `True`** — the join key silently vanished and Pydantic failed. Renamed to `condition`
   and recorded in `CLAUDE.md` so it can't recur.
2. **First mutation test was a no-op** — I "broke" `revenue` by dropping `* quantity`, accuracy stayed 100%. Cause: every
   `quantity` in Chinook is 1, so the mutation was *equivalent*. A passing check that can't fail was caught only because
   a perfect first score was treated as suspicious. Re-ran with a mutation that changes values (→ 36.7%).
3. **Golden set was saturated** (100%, written by the same author as the baseline) so it said nothing. Added a `hard`
   category outside the layer's reach; the baseline's 0% there is the honest picture.
4. **A test of mine was wrong, not the code** — the "disallowed" table in the repair test was actually allow-listed.
   Fixed the test to probe `information_schema` (a realistic model behaviour) instead of weakening the guard.
5. **Formatting bug** — years rendered as `2,021` because number formatting was applied to dimensions. Fixed by
   formatting only metric columns.
6. **Half-applied regex edit** — renaming `on` missed aligned lines with double spaces; caught by re-running, not by assuming.

## What is *not* verified (stated plainly)

- The Claude planner/narrator path is tested against a **mock SDK client** (request shape, forced tool choice, cache
  control, parsing, error feedback). It has **not** been run against the live API in this build (no key available), so
  there is no Claude accuracy number yet — run `python -m analytics_agent.evals.run --llm claude` to get it.
- Docker/compose files were written but **not built** here (Docker not installed on this machine). The remote-Qdrant
  code path (`QDRANT_URL`) is likewise untested; the in-process Qdrant path is covered by tests.
- The embedder is a hashed bag-of-words, chosen for determinism; retrieval quality on paraphrases is untested.
- The golden set is 43 questions on one small dataset — a regression net, not a benchmark.
