"""Unit tests for generation scorers and failure triager."""

import pytest
from groundtruth.schema import Passage
from groundtruth.generation import (
    FaithfulnessScorer,
    AnswerRelevanceScorer,
    FailureTriager,
    FailureMode,
)


@pytest.fixture
def twombly_passage() -> Passage:
    return Passage(
        id="p_twombly",
        title="Bell Atlantic Corp. v. Twombly",
        category="procedural_motion",
        text="Rule 8 requires factual allegations sufficient to raise a right to relief above the speculative level, establishing the plausibility standard.",
    )


class TestFaithfulnessScorer:
    def test_faithful_grounded_answer(self, twombly_passage):
        scorer = FaithfulnessScorer()
        answer = "Under Bell Atlantic Corp. v. Twombly, Rule 8 requires factual allegations that raise a right to relief above the speculative level."
        score = scorer.compute_faithfulness(answer, [twombly_passage])
        assert score == 1.0

    def test_hallucinated_unsupported_answer(self, twombly_passage):
        scorer = FaithfulnessScorer()
        # Answer makes up claims not in the Twombly passage
        answer = "The defendant must provide ten notarized affidavits signed by the governor within twenty days of filing."
        score = scorer.compute_faithfulness(answer, [twombly_passage])
        assert score < 0.5


class TestFailureTriager:
    def test_triage_retrieval_failure(self):
        triager = FailureTriager()
        # Precedent was missed: Recall@10 = 0.0
        mode, diagnosis = triager.triage(
            recall_at_10=0.0,
            faithfulness_score=0.9,
            relevance_score=0.8,
        )
        assert mode == FailureMode.RETRIEVAL_FAILURE
        assert "Retrieval Failure" in diagnosis

    def test_triage_generation_hallucination(self):
        triager = FailureTriager()
        # Precedent retrieved successfully (Recall=1.0), but generator hallucinated
        mode, diagnosis = triager.triage(
            recall_at_10=1.0,
            faithfulness_score=0.2,
            relevance_score=0.8,
        )
        assert mode == FailureMode.GENERATION_HALLUCINATION
        assert "Generation Failure (Hallucination)" in diagnosis

    def test_triage_success(self):
        triager = FailureTriager()
        mode, diagnosis = triager.triage(
            recall_at_10=1.0,
            faithfulness_score=0.95,
            relevance_score=0.9,
        )
        assert mode == FailureMode.SUCCESS
        assert "Success" in diagnosis
