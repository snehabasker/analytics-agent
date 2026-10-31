# Eval report (heuristic)

- Execution accuracy: **88.2%**
- Safety recall (must-refuse cases refused): **100.0%**
- False-refusal rate (benign questions blocked): **0.0%**
- Latency p50/p95: 31.5 / 90.7 ms
- Avg repair attempts: 0

| Category | Pass rate |
|---|---|
| benign | 100% |
| filter | 100% |
| group_by | 100% |
| hard | 0% |
| safety | 100% |
| simple_metric | 100% |
| time | 100% |
| top_n | 100% |

## Failures (4)

- `hard_having` — Which genres earned more than $100 in revenue?
  - MISMATCH want=[('Alternative & Punk',), ('Metal',), ('Latin',)]... got=[('Rock', 826.65), ('Latin', 382.14), ('Metal', 261.36)]...
- `hard_share` — What share of total revenue comes from Rock?
  - MISMATCH want=[(35.5,)]... got=[(826.65,)]...
- `hard_compare_years` — Compare revenue in 2024 versus 2023
  - MISMATCH want=[(2023, 469.58), (2024, 477.53)]... got=[(None,)]...
- `hard_every_year` — How many customers made a purchase in every year of data?
  - MISMATCH want=[(9,)]... got=[(2021, 46), (2022, 46), (2023, 47)]...
