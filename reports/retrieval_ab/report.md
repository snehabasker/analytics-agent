# Retrieval A/B: hashing vs Cohere Embed / Rerank

Cohere models: `embed-v4.0`, `rerank-v3.5`. Slice sizes = what `context_for` puts in the prompt (metric 4 of 5, dimension 4 of 12, value 4 of 218).

| arm | grounded | metric hit@1 | metric MRR | dim recall@4 | dim hit@1 | dim MRR | value recall@4 | value hit@1 | value MRR | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| hashing | 74% | 76% | 0.88 | 76% | 66% | 0.76 | 86% | 86% | 0.86 | 9 |
| hashing_rerank | 91% | 91% | 0.96 | 95% | 79% | 0.89 | 71% | 71% | 0.72 | 295 |
| cohere_embed | 97% | 88% | 0.94 | 98% | 83% | 0.91 | 100% | 100% | 1.00 | 360 |
| cohere_embed_rerank | 91% | 91% | 0.96 | 95% | 79% | 0.89 | 86% | 86% | 0.86 | 652 |
| random | 15% | 29% | 0.53 | 28% | 3% | 0.22 | 14% | 0% | 0.04 | 32 |

## Ungrounded questions per arm (needed items missing from the prompt context)

- **hashing** (9): `rock_2024` missing year; `brazil_revenue` missing country; `usa_units_2025` missing country, year, country=USA; `revenue_since_2024` missing year; `jazz_by_year` missing genre; `top_genre_in_germany` missing country; `monthly_orders_2023` missing year; `benign_show` missing country; `hard_compare_years` missing year
- **hashing_rerank** (3): `usa_units_2025` missing country, country=USA; `top_genre_in_germany` missing country, country=Germany; `monthly_orders_2023` missing year
- **cohere_embed** (1): `rock_2024` missing year
- **cohere_embed_rerank** (3): `usa_units_2025` missing country; `top_genre_in_germany` missing country, country=Germany; `monthly_orders_2023` missing year
- **random** (29): `customer_count` missing customer_count; `units_total` missing units_sold; `aov_total` missing avg_order_value; `revenue_by_country` missing country; `units_by_genre` missing units_sold, genre; `revenue_by_media_type` missing media_type; `revenue_by_rep` missing revenue, support_rep; `aov_by_year` missing year; `revenue_units_by_year` missing units_sold, year; `revenue_year_media` missing media_type; `rock_2024` missing genre, year, genre=Rock; `brazil_revenue` missing country=Brazil; `orders_2023` missing year; `usa_units_2025` missing country, country=USA; `revenue_since_2024` missing revenue; `jazz_by_year` missing genre, year, genre=Jazz; `top5_genres_units` missing units_sold, genre; `top3_customers` missing customer; `best_country` missing revenue; `top10_artists` missing artist; `busiest_year` missing year; `top_genre_in_germany` missing genre; `monthly_orders_2023` missing month, year; `quarterly_revenue_2024` missing year; `benign_list` missing genre; `benign_show` missing country, country=Canada; `hard_having` missing genre; `hard_share` missing genre, genre=Rock; `hard_every_year` missing year
