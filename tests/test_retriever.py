"""Integration tests for all 4 retrieval configurations."""

import pytest
from groundtruth.schema import Passage
from groundtruth.retriever import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    RerankedRetriever,
)


@pytest.fixture(scope="module")
def mini_corpus() -> list[Passage]:
    return [
        Passage(
            id="p_twombly",
            title="Bell Atlantic Corp. v. Twombly",
            category="procedural_motion",
            text="Rule 8 requires factual allegations sufficient to raise a right to relief above the speculative level, establishing the plausibility standard.",
        ),
        Passage(
            id="p_palsgraf",
            title="Palsgraf v. Long Island R.R.",
            category="tort_liability",
            text="A duty of care in negligence extends only to foreseeable victims within the zone of reasonably foreseeable danger.",
        ),
        Passage(
            id="p_markman",
            title="Markman v. Westview Instruments",
            category="intellectual_property",
            text="Patent claim construction is exclusively a question of law for the court rather than the jury.",
        ),
    ]


def test_bm25_retriever(mini_corpus):
    bm25 = BM25Retriever(mini_corpus)
    # Search for an exact term in Twombly
    results = bm25.retrieve("plausibility standard speculative level", top_k=2)
    assert len(results) == 2
    assert results[0] == "p_twombly"


def test_dense_retriever(mini_corpus):
    dense = DenseRetriever(mini_corpus)
    # Semantic search with conceptual phrasing (different words than the passage)
    results = dense.retrieve("interpreting patent claims is for the judge not jurors", top_k=1)
    assert len(results) == 1
    assert results[0] == "p_markman"


def test_hybrid_retriever(mini_corpus):
    dense = DenseRetriever(mini_corpus)
    bm25 = BM25Retriever(mini_corpus)
    hybrid = HybridRetriever(dense, bm25)

    results = hybrid.retrieve("zone of danger negligence duty of care", top_k=1)
    assert len(results) == 1
    assert results[0] == "p_palsgraf"


def test_reranked_retriever(mini_corpus):
    dense = DenseRetriever(mini_corpus)
    bm25 = BM25Retriever(mini_corpus)
    hybrid = HybridRetriever(dense, bm25)
    reranked = RerankedRetriever(hybrid, mini_corpus, candidate_k=3)

    results = reranked.retrieve("pleading standard under federal civil rule 8", top_k=1)
    assert len(results) == 1
    assert results[0] == "p_twombly"
