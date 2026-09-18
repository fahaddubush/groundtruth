"""Retrievers for Groundtruth evaluation.

Implements:
1. DenseRetriever (Vector search via FastEmbed ONNX embeddings)
2. BM25Retriever (Lexical BM25 keyword search via rank-bm25)
3. HybridRetriever (Reciprocal Rank Fusion of Dense + BM25)
4. RerankedRetriever (Cross-Encoder 2-stage reranking via FlashRank)
"""

from typing import Protocol
import re
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from groundtruth.schema import Passage
from groundtruth.fusion import reciprocal_rank_fusion


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class Retriever(Protocol):
    """Protocol defining the search interface."""

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Retrieve top_k passage IDs for the given query."""
        ...


class DenseRetriever:
    """Dense vector retriever using FastEmbed ONNX bi-encoder."""

    def __init__(
        self,
        passages: list[Passage],
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.passages = passages
        self.passage_ids = [p.id for p in passages]
        self.embed_model = TextEmbedding(model_name=model_name)

        # Precompute embeddings for the entire corpus
        # Combine title + text for richer legal semantic representation
        corpus_texts = [f"{p.title}\n{p.text}" for p in passages]
        embeddings_gen = self.embed_model.embed(corpus_texts)
        self.corpus_embeddings = np.array(list(embeddings_gen), dtype=np.float32)

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.corpus_embeddings /= norms

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_embed = np.array(list(self.embed_model.embed([query])), dtype=np.float32)[0]
        q_norm = np.linalg.norm(query_embed)
        if q_norm > 0:
            query_embed /= q_norm

        # Cosine similarity via matrix multiplication
        scores = np.dot(self.corpus_embeddings, query_embed)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.passage_ids[i] for i in top_indices]


class BM25Retriever:
    """Lexical keyword retriever using BM25Okapi."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.passage_ids = [p.id for p in passages]

        corpus_texts = [f"{p.title} {p.text}" for p in passages]
        tokenized_corpus = [tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return self.passage_ids[:top_k]

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.passage_ids[i] for i in top_indices]


class HybridRetriever:
    """Hybrid retriever combining Dense and BM25 via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        fusion_c: int = 60,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.fusion_c = fusion_c

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        # Retrieve candidate pool from both systems (2x top_k, min 20)
        candidate_k = max(top_k * 2, 20)
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)

        # Merge rankings using our RRF algorithm
        fused = reciprocal_rank_fusion([dense_results, bm25_results], c=self.fusion_c)
        return fused[:top_k]


class RerankedRetriever:
    """Two-stage retriever: base retriever candidate pool + Cross-Encoder reranker."""

    def __init__(
        self,
        base_retriever: Retriever,
        passages: list[Passage],
        candidate_k: int = 25,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
    ):
        self.base_retriever = base_retriever
        self.passage_lookup = {p.id: p for p in passages}
        self.candidate_k = candidate_k
        self.ranker = Ranker(model_name=model_name)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        # Stage 1: Get initial candidate pool
        initial_candidates = self.base_retriever.retrieve(query, top_k=self.candidate_k)
        if not initial_candidates:
            return []

        # Stage 2: Prepare inputs for FlashRank cross-encoder
        flashrank_passages = [
            {
                "id": pid,
                "text": f"{self.passage_lookup[pid].title}\n{self.passage_lookup[pid].text}",
            }
            for pid in initial_candidates
            if pid in self.passage_lookup
        ]

        rerank_request = RerankRequest(query=query, passages=flashrank_passages)
        reranked_results = self.ranker.rerank(rerank_request)

        # FlashRank returns a list of dicts sorted by score descending
        reranked_ids = [res["id"] for res in reranked_results]
        return reranked_ids[:top_k]
