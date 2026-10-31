import pytest

from analytics_agent.agent import build_agent
from analytics_agent.config import ROOT, Settings
from analytics_agent.semantic import SemanticLayer

if not (ROOT / "data" / "warehouse.duckdb").exists():
    pytest.skip("warehouse missing: run `make data`", allow_module_level=True)


@pytest.fixture(scope="session")
def layer() -> SemanticLayer:
    return SemanticLayer.load()


@pytest.fixture(scope="session")
def agent():
    a = build_agent(Settings(llm="heuristic"))
    yield a
    a.close()
