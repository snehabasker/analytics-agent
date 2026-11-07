# Eval report (dspy:ollama_chat/qwen2.5:7b+planner_qwen25-7b)

- Execution accuracy: **76.5%**
- Safety recall (must-refuse cases refused): **100.0%**
- False-refusal rate (benign questions blocked): **0.0%**
- Latency p50/p95: 2520.0 / 3542.9 ms
- Avg repair attempts: 0.09

| Category | Pass rate |
|---|---|
| benign | 100% |
| filter | 43% |
| group_by | 100% |
| hard | 0% |
| safety | 100% |
| simple_metric | 100% |
| time | 100% |
| top_n | 100% |

## Failures (8)

- `rock_2024` — Revenue from Rock tracks in 2024
  - MISMATCH want=[(162.36,)]... got=[(2024, 'Rock', 162.36)]...
- `brazil_revenue` — What was the revenue in Brazil?
  - MISMATCH want=[(190.1,)]... got=[('Brazil', 190.1)]...
- `orders_2023` — How many orders were placed in 2023?
  - MISMATCH want=[(83,)]... got=[('2023-01', 7), ('2023-02', 7), ('2023-03', 7)]...
- `usa_units_2025` — Units sold in the United States in 2025
  - MISMATCH want=[(86,)]... got=[(29,)]...
- `hard_having` — Which genres earned more than $100 in revenue?
  - status=error: I couldn't produce a reliable answer after several attempts. Last problem: validation failed: unknow
- `hard_share` — What share of total revenue comes from Rock?
  - MISMATCH want=[(35.5,)]... got=[('Rock', 826.65)]...
- `hard_compare_years` — Compare revenue in 2024 versus 2023
  - MISMATCH want=[(2023, 469.58), (2024, 477.53)]... got=[(2021, 449.46), (2022, 481.45), (2023, 469.58)]...
- `hard_every_year` — How many customers made a purchase in every year of data?
  - MISMATCH want=[(9,)]... got=[(2021, 46), (2022, 46), (2023, 47)]...
