"""Unit tests for the EvaluationEngine."""

import pytest
from groundtruth.schema import Passage, GoldenQuery, EvaluationDataset
from groundtruth.evaluator import EvaluationEngine


class MockRetriever:
    """Mock retriever for deterministic testing."""

    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self.mapping.get(query, [])[:top_k]


def test_evaluation_engine_aggregation():
    p1 = Passage(id="p1", title="Case 1", category="contracts", text="Contract text")
    p2 = Passage(id="p2", title="Case 2", category="torts", text="Tort text")

    q1 = GoldenQuery(id="q1", query="query 1", category="contracts", relevant_passages={"p1": 3})
    q2 = GoldenQuery(id="q2", query="query 2", category="torts", relevant_passages={"p2": 3})

    dataset = EvaluationDataset(
        version="1.0",
        passages=[p1, p2],
        queries=[q1, q2],
    )

    # Mock: q1 finds p1 at rank 1 (recall=1.0, mrr=1.0)
    # q2 finds nothing (recall=0.0, mrr=0.0) -> failure
    mock = MockRetriever({
        "query 1": ["p1", "pX"],
        "query 2": ["pY", "pZ"],
    })

    engine = EvaluationEngine(dataset)
    run = engine.evaluate(mock, configuration_name="MockConfig")

    assert run.total_queries == 2
    assert run.overall_metrics.mean_recall_at_10 == 0.5
    assert run.overall_metrics.mean_mrr == 0.5
    assert run.overall_metrics.failure_count == 1
    assert run.overall_metrics.failure_rate == 0.5
    assert len(run.failing_queries) == 1
    assert run.failing_queries[0].query_id == "q2"

    # Check category slicing
    assert "contracts" in run.category_metrics
    assert run.category_metrics["contracts"].mean_recall_at_10 == 1.0
    assert run.category_metrics["contracts"].failure_count == 0

    assert "torts" in run.category_metrics
    assert run.category_metrics["torts"].mean_recall_at_10 == 0.0
    assert run.category_metrics["torts"].failure_count == 1
