"""Retrieval Evaluation Metrics.

This module contains the foundational metrics used to evaluate search and retrieval systems:
- Recall@k
- Reciprocal Rank (and Mean Reciprocal Rank - MRR)
- Discounted Cumulative Gain & Normalized DCG (nDCG@k)
"""

from typing import Sequence, Set, Any
import math


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any] | Set[Any],
    k: int,
) -> float:
    """Calculate Recall@k for a single query.

    Recall@k measures the fraction of relevant documents that were successfully
    retrieved in the top-k results.

    Formula:
        Recall@k = |Retrieved_top_k ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ranked list of document/chunk IDs returned by the retriever.
        relevant_ids: Set or list of document IDs known to be relevant (ground truth).
        k: Cutoff rank (e.g., 5, 10, 20).

    Returns:
        float between 0.0 and 1.0. If relevant_ids is empty, returns 0.0.
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    
    hits = 0
    for doc_id in top_k:
        if doc_id in relevant_ids:
            hits+=1
    
    return hits / len(relevant_ids)

def reciprocal_rank(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any] | Set[Any],
    k: int | None = None,
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Reciprocal Rank is 1 / rank of the FIRST relevant document found.
    If no relevant document is found within top-k (or the entire list if k is None),
    the score is 0.0.

    Formula:
        RR = 1 / rank_of_first_relevant_item  (where rank is 1-indexed)

    Example:
        If the first relevant document is at index 0 (rank 1) -> RR = 1 / 1 = 1.0
        If the first relevant document is at index 2 (rank 3) -> RR = 1 / 3 = 0.3333
        If no relevant document is in retrieved_ids -> RR = 0.0

    Args:
        retrieved_ids: Ranked list of document/chunk IDs returned by the retriever.
        relevant_ids: Set or list of document IDs known to be relevant.
        k: Optional cutoff rank. If provided, only search within retrieved_ids[:k].

    Returns:
        float between 0.0 and 1.0.
    """
    top_k = retrieved_ids[:k] if k is not None else retrieved_ids

    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevance_scores: dict[Any, float | int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank k (nDCG@k).

    Measures ranking quality with graded relevance and logarithmic position discounting.

    Args:
        retrieved_ids: Ranked list of document/chunk IDs returned by retriever.
        relevance_scores: Mapping from doc_id -> ground truth relevance score (e.g. 0, 1, 2, 3).
        k: Cutoff rank.

    Returns:
        float between 0.0 and 1.0. If ideal DCG is 0.0, returns 0.0.
    """
    def _dcg(scores: list[float], cutoff: int) -> float:
        total = 0.0
        for rank, score in enumerate(scores[:cutoff], start=1):
            if score > 0:
                numerator = 2.0 ** score - 1.0
                denominator = math.log2(rank + 1)
                total += numerator / denominator
        return total

    actual_scores = [float(relevance_scores.get(doc_id, 0.0)) for doc_id in retrieved_ids[:k]]
    actual_dcg = _dcg(actual_scores, k)

    ideal_scores = sorted([float(v) for v in relevance_scores.values()], reverse=True)
    ideal_dcg = _dcg(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg
