"""CI Regression Gate automated tests.

Verifies:
1. A valid production candidate satisfies the gate rules and passes.
2. A deliberately bad change fails the gate and blocks the pull request.
"""

from pathlib import Path
import pytest
from groundtruth.schema import EvaluationDataset, Passage
from groundtruth.retriever import DenseRetriever, BM25Retriever
from groundtruth.regression_gate import run_regression_gate


@pytest.fixture(scope="module")
def dataset() -> EvaluationDataset:
    dataset_path = Path("data") / "golden_eval_v1.json"
    return EvaluationDataset.load_from_json(dataset_path)


class DegradedCandidateRetriever:
    """Simulates a flawed PR (e.g. buggy tokenizer, broken embeddings, or bad index).

    Truncates all searches to only return 1 irrelevant passage, causing
    Recall@10 to collapse from 0.898 to near zero.
    """

    def __init__(self, fallback_id: str):
        self.fallback_id = fallback_id

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        # Maliciously bad retrieval: returns only 1 hardcoded irrelevant document
        return [self.fallback_id]


def test_production_candidate_passes_regression_gate(dataset):
    """Test that a healthy production candidate passes the CI regression gate."""
    # Dense retriever achieves ~0.907 Recall@10, comfortably above 0.888 hard floor
    candidate = DenseRetriever(dataset.passages)

    result = run_regression_gate(
        candidate_retriever=candidate,
        candidate_name="PR_Branch_Dense_Semantic",
        dataset=dataset,
        baseline_path="data/baseline_metrics.json",
    )

    assert result.passed is True, result.summary
    assert result.candidate_recall >= 0.888
    print(f"\n{result.summary}")


def test_deliberately_bad_change_fails_regression_gate(dataset):
    """DEMONSTRATION: Prove that a bad PR drops Recall@10 and FAILS the gate."""
    # Simulate a PR where someone introduced a buggy component
    bad_candidate = DegradedCandidateRetriever(fallback_id=dataset.passages[0].id)

    result = run_regression_gate(
        candidate_retriever=bad_candidate,
        candidate_name="PR_Branch_Deliberately_Degraded",
        dataset=dataset,
        baseline_path="data/baseline_metrics.json",
    )

    # Verify that the gate caught the regression
    assert result.passed is False
    assert result.delta < -result.max_allowed_drop
    print(f"\n[DEMO GATE BLOCK CONFIRMED]: {result.summary}")

    # In CI, this assertion raises an AssertionError and exits with code 1, failing the build!
    with pytest.raises(AssertionError, match="CI GATE BLOCKED"):
        if not result.passed:
            raise AssertionError(f"CI GATE BLOCKED: {result.summary}")
