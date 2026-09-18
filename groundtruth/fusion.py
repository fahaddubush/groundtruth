"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Combines rankings from multiple search systems (e.g. BM25 and Dense Embeddings)
without requiring score calibration or normalization.
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    c: int = 60,
) -> list[str]:
    """Merge multiple ranked lists of document IDs using Reciprocal Rank Fusion.

    Formula:
        RRF_score(doc) = sum(1 / (c + rank_m(doc))) for each retriever m
        where rank is 1-indexed.

    Args:
        ranked_lists: List of ranked document ID lists from different retrievers
                      (e.g., [dense_doc_ids, bm25_doc_ids]).
        c: Smoothing constant to penalize low ranks without over-penalizing (default 60).

    Returns:
        Consolidated list of document IDs, sorted in descending order of RRF score.
    """
    scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (c + rank)

    return sorted(scores.keys(), key=lambda doc: scores[doc], reverse=True)
