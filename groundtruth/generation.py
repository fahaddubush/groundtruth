"""Generation evaluation and failure triaging layer.

Separates retrieval failures from generation failures:
- Faithfulness: Are the claims in the generated answer strictly grounded in the retrieved context?
- Answer Relevance: Does the generated answer address the attorney's query?
- Failure Triaging: Pinpoints whether a failure is caused by retrieval (precedent missing)
  or generation (hallucination / unfaithful synthesis).
"""

from enum import Enum
import re
from typing import Sequence
import numpy as np
from pydantic import BaseModel, Field

from groundtruth.schema import Passage


class FailureMode(str, Enum):
    """Diagnostic categorization of RAG failure modes."""

    SUCCESS = "SUCCESS"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GENERATION_HALLUCINATION = "GENERATION_HALLUCINATION"
    GENERATION_IRRELEVANT = "GENERATION_IRRELEVANT"


class RAGGenerationOutput(BaseModel):
    """Represents a generated answer alongside its retrieval context."""

    query: str
    retrieved_passages: list[Passage]
    answer: str
    retrieval_recall_at_10: float
    faithfulness_score: float = 0.0
    answer_relevance_score: float = 0.0
    triage_result: FailureMode = FailureMode.SUCCESS
    diagnosis: str = ""


class FaithfulnessScorer:
    """Evaluates whether claims in the generated answer are grounded in the retrieved context.

    Breaks the answer into atomic factual sentences/claims and verifies
    whether each claim is supported by the retrieved context.
    """

    def __init__(self, support_threshold: float = 0.55):
        self.support_threshold = support_threshold

    def extract_claims(self, answer: str) -> list[str]:
        """Split generated answer into atomic claims (sentences)."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        return claims if claims else [answer.strip()]

    def score_claim_support(self, claim: str, context: str) -> bool:
        """Determines if a claim has lexical and semantic support in the context."""
        claim_tokens = set(re.findall(r"\w+", claim.lower()))
        # Remove common stopwords for cleaner legal overlap
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "over", "after",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "shall", "should", "will", "would",
            "that", "this", "these", "those", "it", "its", "as", "if"
        }
        content_tokens = claim_tokens - stopwords
        if not content_tokens:
            return True

        context_tokens = set(re.findall(r"\w+", context.lower()))
        overlap = len(content_tokens & context_tokens) / len(content_tokens)
        return overlap >= self.support_threshold

    def compute_faithfulness(self, answer: str, context_passages: Sequence[Passage]) -> float:
        """Computes faithfulness score: fraction of claims supported by the context."""
        claims = self.extract_claims(answer)
        if not claims:
            return 1.0

        full_context = "\n".join([f"{p.title} {p.text}" for p in context_passages])
        if not full_context.strip():
            return 0.0

        supported_claims = sum(
            1 for claim in claims if self.score_claim_support(claim, full_context)
        )
        return float(supported_claims / len(claims))


class AnswerRelevanceScorer:
    """Evaluates whether the generated answer directly addresses the query."""

    def compute_relevance(self, query: str, answer: str) -> float:
        """Computes keyword relevance overlap between query and answer."""
        query_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {
            "what", "is", "the", "under", "for", "to", "in", "of", "and", "or", "a", "an",
            "does", "can", "how", "when", "where", "which", "are", "why"
        }
        key_query_terms = query_words - stopwords
        if not key_query_terms:
            return 1.0

        answer_words = set(re.findall(r"\w+", answer.lower()))
        overlap = len(key_query_terms & answer_words)
        return float(min(1.0, overlap / max(1, len(key_query_terms) * 0.4)))


class FailureTriager:
    """Classifies RAG pipeline outputs into distinct retrieval vs generation failure modes."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.70,
        relevance_threshold: float = 0.40,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold

    def triage(
        self,
        recall_at_10: float,
        faithfulness_score: float,
        relevance_score: float,
    ) -> tuple[FailureMode, str]:
        """Classify the root cause of an answer failure."""
        # 1. Did retrieval fail to bring back the critical precedent?
        if recall_at_10 == 0.0:
            return (
                FailureMode.RETRIEVAL_FAILURE,
                "Retrieval Failure: The required legal precedent was not present in the top-10 retrieved passages. The generator had insufficient context.",
            )

        # 2. Precedent was retrieved, but did the LLM hallucinate or state unsupported claims?
        if faithfulness_score < self.faithfulness_threshold:
            return (
                FailureMode.GENERATION_HALLUCINATION,
                f"Generation Failure (Hallucination): Precedents were successfully retrieved (Recall@10 = {recall_at_10:.2f}), but the answer made claims not grounded in the source text (Faithfulness = {faithfulness_score:.2f}).",
            )

        # 3. Did the LLM drift off-topic?
        if relevance_score < self.relevance_threshold:
            return (
                FailureMode.GENERATION_IRRELEVANT,
                f"Generation Failure (Irrelevance): Precedents were retrieved, but the answer failed to address the specific attorney query (Relevance = {relevance_score:.2f}).",
            )

        # 4. Both retrieval and generation succeeded
        return (
            FailureMode.SUCCESS,
            "Success: Both retrieval and generation were faithful and relevant.",
        )
