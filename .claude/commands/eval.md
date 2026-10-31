---
description: Run tests + eval gate and summarise regressions
---
Run `pytest` and then `python -m analytics_agent.evals.run --min-accuracy 0.85`.
Report: test counts, then the eval summary table. For every failed golden case, say whether the cause is
(a) the semantic layer, (b) the planner, (c) a guardrail, or (d) a wrong reference SQL — and cite the evidence
(the SQL that ran vs. the reference). Do not edit anything unless I ask.
