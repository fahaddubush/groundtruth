"""Unit tests for Groundtruth data schemas."""

import tempfile
from pathlib import Path
import pytest
from groundtruth.schema import Passage, GoldenQuery, EvaluationDataset


class TestSchema:
    def test_passage_creation(self):
        p = Passage(
            id="doc_001",
            title="Smith v. Jones",
            category="contract_dispute",
            text="Under Delaware law, mutual assent requires an offer and acceptance.",
            metadata={"jurisdiction": "Delaware", "year": 2018},
        )
        assert p.id == "doc_001"
        assert p.category == "contract_dispute"
        assert p.metadata["year"] == 2018

    def test_golden_query_properties(self):
        q = GoldenQuery(
            id="q_01",
            query="What is required for mutual assent under Delaware contract law?",
            category="contract_dispute",
            relevant_passages={"doc_001": 3, "doc_002": 2},
        )
        assert q.relevant_ids == {"doc_001", "doc_002"}
        assert q.relevance_scores == {"doc_001": 3, "doc_002": 2}

    def test_evaluation_dataset_serialization_roundtrip(self):
        p1 = Passage(
            id="p1",
            title="Case One",
            category="ip",
            text="Patent claim construction is a matter of law.",
        )
        p2 = Passage(
            id="p2",
            title="Case Two",
            category="contracts",
            text="Liquidated damages must not constitute a penalty.",
        )
        q1 = GoldenQuery(
            id="q1",
            query="Markman hearing claim construction rules",
            category="ip",
            relevant_passages={"p1": 3},
        )

        dataset = EvaluationDataset(
            version="1.0",
            passages=[p1, p2],
            queries=[q1],
        )

        assert dataset.get_all_categories() == ["ip"]
        assert dataset.get_passage_by_id("p1") == p1
        assert dataset.get_passage_by_id("missing") is None
        assert len(dataset.get_queries_by_category("ip")) == 1

        # Test save and load JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_dataset.json"
            dataset.save_to_json(temp_path)

            loaded = EvaluationDataset.load_from_json(temp_path)
            assert loaded.version == "1.0"
            assert len(loaded.passages) == 2
            assert len(loaded.queries) == 1
            assert loaded.queries[0].id == "q1"
            assert loaded.queries[0].relevant_ids == {"p1"}

    def test_golden_dataset_file_integrity(self):
        dataset_path = Path("data") / "golden_eval_v1.json"
        assert dataset_path.exists(), "golden_eval_v1.json should exist"

        dataset = EvaluationDataset.load_from_json(dataset_path)
        assert len(dataset.passages) == 100
        assert len(dataset.queries) == 100
        assert len(dataset.get_all_categories()) == 5

        passage_ids = {p.id for p in dataset.passages}
        assert len(passage_ids) == 100, "All passage IDs must be unique"

        # Every referenced passage must exist in the corpus!
        for q in dataset.queries:
            for pid in q.relevant_ids:
                assert pid in passage_ids, f"Query {q.id} references non-existent passage {pid}"
