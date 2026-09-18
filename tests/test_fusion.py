"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest
from groundtruth.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_example_trace(self):
        # Dense: doc_A at rank 1, doc_B at rank 2, doc_C at rank 3
        # BM25: doc_B at rank 1, doc_D at rank 2, doc_A at rank 3
        dense = ["doc_A", "doc_B", "doc_C"]
        bm25 = ["doc_B", "doc_D", "doc_A"]

        # doc_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.03252 (1st)
        # doc_A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.03226 (2nd)
        # doc_D: 1/(60+2) ≈ 0.01613 (3rd)
        # doc_C: 1/(60+3) ≈ 0.01587 (4th)
        fused = reciprocal_rank_fusion([dense, bm25], c=60)
        assert fused == ["doc_B", "doc_A", "doc_D", "doc_C"]

    def test_single_list(self):
        single = ["doc_1", "doc_2", "doc_3"]
        fused = reciprocal_rank_fusion([single])
        assert fused == ["doc_1", "doc_2", "doc_3"]

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_no_duplicates_in_output(self):
        list1 = ["doc_A", "doc_B", "doc_C"]
        list2 = ["doc_C", "doc_A", "doc_B"]
        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) == 3
        assert len(set(fused)) == 3
