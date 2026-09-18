"""Enterprise-grade retrieval evaluation engine.

Evaluates retrieval configurations against the golden dataset, reporting:
- Recall@5 and Recall@10
- MRR (Mean Reciprocal Rank)
- nDCG@10 (Normalized Discounted Cumulative Gain)
- Latency (Mean, p50, p95)
- Category-sliced metrics breakdown
- Hard failure tracking (Recall@10 == 0)
"""

from pathlib import Path
import time
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from groundtruth.schema import EvaluationDataset, GoldenQuery
from groundtruth.metrics import recall_at_k, reciprocal_rank, ndcg_at_k
from groundtruth.retriever import Retriever


class QueryEvaluationResult(BaseModel):
    """Evaluation result for an individual query."""

    query_id: str
    query_text: str
    category: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float
    latency_ms: float
    is_failure: bool = Field(
        description="True if none of the ground truth documents were retrieved in top-10"
    )


class CategoryMetrics(BaseModel):
    """Aggregated retrieval metrics for a single practice area or overall."""

    category: str
    num_queries: int
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_10: float
    mean_latency_ms: float
    p95_latency_ms: float
    failure_count: int
    failure_rate: float


class BenchmarkRun(BaseModel):
    """Complete evaluation run for a specific retrieval configuration."""

    configuration_name: str
    description: str = ""
    timestamp: float = Field(default_factory=time.time)
    total_queries: int
    overall_metrics: CategoryMetrics
    category_metrics: dict[str, CategoryMetrics]
    query_results: list[QueryEvaluationResult]
    failing_queries: list[QueryEvaluationResult]


class BenchmarkComparisonReport(BaseModel):
    """Comparison report across multiple retrieval configurations."""

    runs: list[BenchmarkRun]

    def to_markdown_table(self) -> str:
        """Generate a comparison table in Markdown format."""
        lines = [
            "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean Latency (ms) | p95 Latency (ms) | Failures (Rec@10=0) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for run in self.runs:
            o = run.overall_metrics
            lines.append(
                f"| **{run.configuration_name}** | "
                f"{o.mean_recall_at_5:.3f} | "
                f"{o.mean_recall_at_10:.3f} | "
                f"{o.mean_mrr:.3f} | "
                f"{o.mean_ndcg_at_10:.3f} | "
                f"{o.mean_latency_ms:.1f} ms | "
                f"{o.p95_latency_ms:.1f} ms | "
                f"{o.failure_count}/{run.total_queries} ({o.failure_rate:.1%}) |"
            )
        return "\n".join(lines)

    def to_category_markdown_table(self, category: str) -> str:
        """Generate a sliced comparison table for a specific query category."""
        lines = [
            f"### Category Breakdown: `{category}`",
            "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Failures |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for run in self.runs:
            if category in run.category_metrics:
                c = run.category_metrics[category]
                lines.append(
                    f"| **{run.configuration_name}** | "
                    f"{c.mean_recall_at_5:.3f} | "
                    f"{c.mean_recall_at_10:.3f} | "
                    f"{c.mean_mrr:.3f} | "
                    f"{c.mean_ndcg_at_10:.3f} | "
                    f"{c.failure_count}/{c.num_queries} |"
                )
        return "\n".join(lines)


class EvaluationEngine:
    """Enterprise evaluation engine that runs benchmarks across retrievers."""

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    def evaluate(
        self,
        retriever: Retriever,
        configuration_name: str,
        description: str = "",
        top_k: int = 10,
    ) -> BenchmarkRun:
        """Run full evaluation suite on a retriever against the golden dataset."""
        query_results: list[QueryEvaluationResult] = []

        for q in self.dataset.queries:
            # Measure query latency with high resolution monotonic timer
            start_time = time.perf_counter()
            retrieved_ids = retriever.retrieve(q.query, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Compute retrieval metrics
            r5 = recall_at_k(retrieved_ids, q.relevant_ids, k=5)
            r10 = recall_at_k(retrieved_ids, q.relevant_ids, k=10)
            mrr = reciprocal_rank(retrieved_ids, q.relevant_ids, k=top_k)
            ndcg = ndcg_at_k(retrieved_ids, q.relevance_scores, k=top_k)
            is_failure = (r10 == 0.0)

            query_results.append(
                QueryEvaluationResult(
                    query_id=q.id,
                    query_text=q.query,
                    category=q.category,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=sorted(list(q.relevant_ids)),
                    recall_at_5=r5,
                    recall_at_10=r10,
                    reciprocal_rank=mrr,
                    ndcg_at_10=ndcg,
                    latency_ms=latency_ms,
                    is_failure=is_failure,
                )
            )

        # Aggregate overall metrics
        overall_metrics = self._aggregate_metrics("OVERALL", query_results)

        # Aggregate per-category metrics
        category_metrics: dict[str, CategoryMetrics] = {}
        for category in self.dataset.get_all_categories():
            cat_results = [r for r in query_results if r.category == category]
            category_metrics[category] = self._aggregate_metrics(category, cat_results)

        failing_queries = [r for r in query_results if r.is_failure]

        return BenchmarkRun(
            configuration_name=configuration_name,
            description=description,
            total_queries=len(query_results),
            overall_metrics=overall_metrics,
            category_metrics=category_metrics,
            query_results=query_results,
            failing_queries=failing_queries,
        )

    def _aggregate_metrics(
        self,
        category: str,
        results: list[QueryEvaluationResult],
    ) -> CategoryMetrics:
        """Compute statistical averages and percentiles for a subset of queries."""
        if not results:
            return CategoryMetrics(
                category=category,
                num_queries=0,
                mean_recall_at_5=0.0,
                mean_recall_at_10=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_10=0.0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                failure_count=0,
                failure_rate=0.0,
            )

        rec5 = [r.recall_at_5 for r in results]
        rec10 = [r.recall_at_10 for r in results]
        mrrs = [r.reciprocal_rank for r in results]
        ndcgs = [r.ndcg_at_10 for r in results]
        latencies = [r.latency_ms for r in results]
        failures = sum(1 for r in results if r.is_failure)

        return CategoryMetrics(
            category=category,
            num_queries=len(results),
            mean_recall_at_5=float(np.mean(rec5)),
            mean_recall_at_10=float(np.mean(rec10)),
            mean_mrr=float(np.mean(mrrs)),
            mean_ndcg_at_10=float(np.mean(ndcgs)),
            mean_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            failure_count=failures,
            failure_rate=float(failures / len(results)),
        )
