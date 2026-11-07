# Eval report (dspy:ollama_chat/qwen2.5:7b+planner_qwen25-7b_teacher-command-a-03-2025)

- Execution accuracy: **82.4%**
- Safety recall (must-refuse cases refused): **100.0%**
- False-refusal rate (benign questions blocked): **0.0%**
- Latency p50/p95: 2640.5 / 8347.9 ms
- Avg repair attempts: 0.12

| Category | Pass rate |
|---|---|
| benign | 100% |
| filter | 86% |
| group_by | 100% |
| hard | 0% |
| safety | 100% |
| simple_metric | 100% |
| time | 50% |
| top_n | 100% |

## Failures (6)

- `usa_units_2025` — Units sold in the United States in 2025
  - MISMATCH want=[(86,)]... got=[(None,)]...
- `quarterly_revenue_2024` — Quarterly revenue in 2024
  - MISMATCH want=[('2024-Q1', 112.86), ('2024-Q2', 112.86), ('2024-Q3', 133.95)]... got=[(477.53,)]...
- `hard_having` — Which genres earned more than $100 in revenue?
  - status=error: I couldn't produce a reliable answer after several attempts. Last problem: execution failed: Binder 
- `hard_share` — What share of total revenue comes from Rock?
  - MISMATCH want=[(35.5,)]... got=[(826.65,)]...
- `hard_compare_years` — Compare revenue in 2024 versus 2023
  - MISMATCH want=[(2023, 469.58), (2024, 477.53)]... got=[]...
- `hard_every_year` — How many customers made a purchase in every year of data?
  - MISMATCH want=[(9,)]... got=[(59,)]...
