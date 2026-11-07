.PHONY: install data test eval serve optimize eval-dspy mlflow-ui

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
optimize:
	python -m analytics_agent.evals.optimize
eval-dspy:
	python -m analytics_agent.evals.run --llm dspy --dspy-program programs/planner_qwen25-7b.json --mlflow
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
