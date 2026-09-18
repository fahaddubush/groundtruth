"""Unit tests for retrieval metrics."""

import math
import pytest
from groundtruth.metrics import recall_at_k, reciprocal_rank, ndcg_at_k


class TestRecallAtK:
    def test_perfect_recall_all_found_within_k(self):
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
        relevant = {"doc_A", "doc_B"}
        # Both doc_A and doc_B are in top 2
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_partial_recall(self):
        retrieved = ["doc_A", "doc_X", "doc_B", "doc_Y"]
        relevant = {"doc_A", "doc_B"}
        # In top 2: only doc_A is retrieved -> 1 / 2 = 0.5
        assert recall_at_k(retrieved, relevant, k=2) == 0.5
        # In top 3: doc_A and doc_B are retrieved -> 2 / 2 = 1.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self):
        retrieved = ["doc_X", "doc_Y", "doc_Z"]
        relevant = {"doc_A", "doc_B"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_greater_than_retrieved_length(self):
        retrieved = ["doc_A"]
        relevant = {"doc_A", "doc_B"}
        # Only 1 retrieved item exists, k=10
        assert recall_at_k(retrieved, relevant, k=10) == 0.5

    def test_empty_relevant_ids(self):
        retrieved = ["doc_A", "doc_B"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, k=5) == 0.0


class TestReciprocalRank:
    def test_first_rank_match(self):
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevant = {"doc_A"}
        # Rank is 1 -> 1 / 1 = 1.0
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_rank_match(self):
        retrieved = ["doc_X", "doc_A", "doc_C"]
        relevant = {"doc_A"}
        # Rank is 2 -> 1 / 2 = 0.5
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_fourth_rank_match(self):
        retrieved = ["doc_X", "doc_Y", "doc_Z", "doc_A"]
        relevant = {"doc_A"}
        # Rank is 4 -> 1 / 4 = 0.25
        assert reciprocal_rank(retrieved, relevant) == 0.25

    def test_not_in_retrieved_list(self):
        retrieved = ["doc_X", "doc_Y", "doc_Z"]
        relevant = {"doc_A"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_with_k_cutoff_excluding_match(self):
        retrieved = ["doc_X", "doc_Y", "doc_A"]
        relevant = {"doc_A"}
        # doc_A is at rank 3, but k=2 cuts off before rank 3
        assert reciprocal_rank(retrieved, relevant, k=2) == 0.0
        # with k=3 it is found
        assert reciprocal_rank(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_multiple_relevant_only_first_matters(self):
        retrieved = ["doc_X", "doc_B", "doc_A"]
        relevant = {"doc_A", "doc_B"}
        # doc_B is at rank 2, doc_A is at rank 3. RR only cares about the first one (rank 2)!
        assert reciprocal_rank(retrieved, relevant) == 0.5


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # Relevance: doc_A=3, doc_B=2, doc_C=1
        # Retrieved order matches ideal order: [doc_A, doc_B, doc_C]
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevance = {"doc_A": 3, "doc_B": 2, "doc_C": 1}
        assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)

    def test_suboptimal_ranking(self):
        # Retrieved order is inverted: [doc_C, doc_B, doc_A]
        retrieved = ["doc_C", "doc_B", "doc_A"]
        relevance = {"doc_A": 3, "doc_B": 2, "doc_C": 1}
        # DCG = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^3 - 1)/log2(4)
        #     = 1/1 + 3/1.58496 + 7/2 = 1 + 1.89279 + 3.5 = 6.39279
        # IDCG = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4)
        #      = 7/1 + 3/1.58496 + 1/2 = 7 + 1.89279 + 0.5 = 9.39279
        # nDCG = 6.39279 / 9.39279 ≈ 0.6806
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(6.39279 / 9.39279, rel=1e-3)

    def test_zero_relevance(self):
        retrieved = ["doc_X", "doc_Y"]
        relevance = {"doc_A": 3, "doc_B": 2}
        assert ndcg_at_k(retrieved, relevance, k=2) == 0.0

    def test_empty_relevance(self):
        retrieved = ["doc_A", "doc_B"]
        relevance = {}
        assert ndcg_at_k(retrieved, relevance, k=2) == 0.0

    def test_k_cutoff(self):
        # Highest relevance doc_A is at rank 3, but k=2 cuts off before rank 3
        retrieved = ["doc_C", "doc_B", "doc_A"]
        relevance = {"doc_A": 3, "doc_B": 1, "doc_C": 0}
        # At k=2, only doc_C and doc_B are considered.
        # DCG@2: doc_C (rel=0) -> 0; doc_B (rel=1) -> 1/log2(3) = 0.6309
        # IDCG@2: top 2 ideal are doc_A (rel=3) and doc_B (rel=1)
        # IDCG@2 = 7/log2(2) + 1/log2(3) = 7 + 0.6309 = 7.6309
        # nDCG@2 = 0.6309 / 7.6309 ≈ 0.0827
        score = ndcg_at_k(retrieved, relevance, k=2)
        assert score == pytest.approx(0.63093 / 7.63093, rel=1e-3)
