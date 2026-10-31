.PHONY: install data test eval serve

install:
	pip install -e ".[dev]"
data:
	python scripts/download_data.py && python scripts/build_warehouse.py
test:
	pytest
eval:
	python -m analytics_agent.evals.run --llm heuristic --min-accuracy 0.85
serve:
	uvicorn analytics_agent.api:app --reload
