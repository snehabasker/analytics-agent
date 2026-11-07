"""DSPy planner + optimiser plumbing, fully offline (DSPy's DummyLM stands in for the model)."""
from pathlib import Path

import pytest
import yaml

dspy = pytest.importorskip("dspy")

from analytics_agent.agent import AnalyticsAgent, _schema_text  # noqa: E402
from analytics_agent.dspy_planner import DSPyLLM, prediction_to_plan  # noqa: E402
from analytics_agent.evals import optimize  # noqa: E402
from analytics_agent.evals.run import GOLDEN  # noqa: E402
from analytics_agent.llm import PlanError  # noqa: E402

P = dspy.Prediction


@pytest.fixture(scope="module")
def agent():
    # Own instance: other test modules may close the shared session agent's DuckDB connection.
    from analytics_agent.agent import build_agent
    from analytics_agent.config import Settings
    a = build_agent(Settings(llm="heuristic"))
    yield a
    a.close()


def _dspy_agent(agent, answers: list[dict]) -> AnalyticsAgent:
    lm = dspy.utils.DummyLM([{"reasoning": "r", **a} for a in answers])
    llm = DSPyLLM(agent.layer, _schema_text(agent.warehouse, agent.layer), lm=lm)
    return AnalyticsAgent(agent.settings, agent.layer, agent.warehouse, agent.index, llm)


# ------------------------------------------------------------------ parsing model output into a Plan

def test_semantic_json_parsed_and_validated():
    plan = prediction_to_plan(P(kind="semantic", semantic_query='```json\n{"metrics": ["revenue"], "dimensions": ["year"]}\n```',
                                sql="", reason="x"))
    assert plan.kind == "semantic" and plan.semantic_query.dimensions == ["year"]


def test_sql_fence_stripped():
    plan = prediction_to_plan(P(kind="sql", semantic_query="", sql="```sql\nSELECT 1\n```", reason="x"))
    assert plan.sql == "SELECT 1"


@pytest.mark.parametrize("pred", [
    P(kind="semantic", semantic_query="", sql="", reason=""),            # empty query
    P(kind="semantic", semantic_query="{not json", sql="", reason=""),   # bad JSON
    P(kind="semantic", semantic_query='{"metrics": []}', sql="", reason=""),  # fails SemanticQuery (min 1 metric)
    P(kind="sql", semantic_query="", sql="none", reason=""),             # no SQL
    P(kind="banana", semantic_query="", sql="", reason=""),              # unknown kind
])
def test_unusable_output_is_plan_error(pred):
    with pytest.raises(PlanError):
        prediction_to_plan(pred)


# ------------------------------------------------------------------ through the real agent graph

def test_dspy_planner_answers_through_semantic_layer(agent):
    a = _dspy_agent(agent, [{"kind": "semantic", "semantic_query": '{"metrics": ["revenue"], "dimensions": ["year"]}',
                             "sql": "", "reason": "by year"}])
    r = a.ask("Revenue by year")
    assert r.status == "ok" and r.plan_kind == "semantic" and r.row_count == 5
    assert a.llm.name.startswith("dspy:") and a.llm.name.endswith("+zero-shot")


def test_dspy_raw_sql_still_goes_through_guard(agent):
    # PII column -> validate_sql rejects -> repair loop -> second answer refuses
    a = _dspy_agent(agent, [{"kind": "sql", "semantic_query": "", "sql": "SELECT email FROM customer", "reason": "x"},
                            {"kind": "refuse", "semantic_query": "", "sql": "", "reason": "private data"}])
    r = a.ask("emails of customers")
    assert r.status == "refused" and r.attempts == 1 and not r.rows


def test_parse_failure_feeds_repair_loop(agent):
    a = _dspy_agent(agent, [{"kind": "semantic", "semantic_query": "{oops", "sql": "", "reason": "x"},
                            {"kind": "semantic", "semantic_query": '{"metrics": ["order_count"]}', "sql": "", "reason": "x"}])
    r = a.ask("How many orders have been placed?")
    assert r.status == "ok" and r.attempts == 1 and r.rows == [[412]]


# ------------------------------------------------------------------ optimiser: data hygiene + metric

def _train():
    return yaml.safe_load(optimize.TRAIN.read_text(encoding="utf-8"))


def test_train_set_is_disjoint_from_golden_and_layer_examples(layer):
    golden = {c["question"].strip().lower() for c in yaml.safe_load(Path(GOLDEN).read_text(encoding="utf-8"))}
    examples = {e.question.strip().lower() for e in layer.examples}
    train = {c["question"].strip().lower() for c in _train()}
    assert not train & golden
    assert not train & examples


def test_train_references_execute(agent):
    for c in _train():
        if c.get("expect") != "refused":
            agent.warehouse.execute(c["reference_sql"].strip())


def test_split_is_deterministic_stratified_and_disjoint():
    tr, va = optimize.split(_train())
    tr2, va2 = optimize.split(list(reversed(_train())))
    assert [c["id"] for c in tr] == [c["id"] for c in tr2] and [c["id"] for c in va] == [c["id"] for c in va2]
    assert not {c["id"] for c in tr} & {c["id"] for c in va}
    from collections import Counter
    multi = {cat for cat, n in Counter(c["category"] for c in _train()).items() if n >= 2}
    assert multi <= {c["category"] for c in tr} and multi <= {c["category"] for c in va}  # stratified


def test_metric_scores_by_execution_not_text(agent):
    metric = optimize.make_metric(agent)
    case = next(c for c in _train() if c["id"] == "t_revenue_2022")
    ex = dspy.Example(case=case)
    right = P(kind="semantic", reason="", sql="",
              semantic_query='{"metrics": ["revenue"], "filters": [{"dimension": "year", "op": "=", "value": 2022}]}')
    extra_col = P(kind="semantic", reason="", sql="",
                  semantic_query='{"metrics": ["revenue"], "dimensions": ["year"], "filters": [{"dimension": "year", "op": "=", "value": 2022}]}')
    raw_sql = P(kind="sql", reason="", semantic_query="",
                sql="SELECT ROUND(SUM(il.unit_price*il.quantity),2) FROM invoice_line il JOIN invoice i "
                    "ON i.invoice_id=il.invoice_id WHERE year(i.invoice_date)=2022")
    assert metric(ex, right) and metric(ex, raw_sql)
    assert not metric(ex, extra_col)                       # right number, wrong shape -> fail
    assert not metric(ex, P(kind="refuse", reason="", sql="", semantic_query=""))
    assert not metric(ex, P(kind="semantic", reason="", sql="", semantic_query="{bad"))


def test_metric_refusal_cases(agent):
    metric = optimize.make_metric(agent)
    case = next(c for c in _train() if c.get("expect") == "refused")
    ex = dspy.Example(case=case)
    assert metric(ex, P(kind="refuse", reason="", sql="", semantic_query=""))
    assert not metric(ex, P(kind="semantic", reason="", sql="", semantic_query='{"metrics": ["revenue"]}'))
