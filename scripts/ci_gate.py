"""CI Gate Runner CLI.

Can be run directly in GitHub Actions or any CI/CD pipeline:
    python scripts/ci_gate.py --candidate [hybrid|dense|bm25|bad]

Exits with code 0 on pass, code 1 on regression.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from groundtruth.schema import EvaluationDataset
from groundtruth.retriever import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    RerankedRetriever,
)
from groundtruth.regression_gate import run_regression_gate


class BadRetriever:
    """Simulates a flawed PR with catastrophic regression."""
    def __init__(self, fallback_id: str):
        self.fallback_id = fallback_id

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return [self.fallback_id]


def main():
    parser = argparse.ArgumentParser(description="Groundtruth CI Regression Gate")
    parser.add_argument(
        "--candidate",
        choices=["reranked", "hybrid", "dense", "bm25", "bad"],
        default="reranked",
        help="Retriever candidate to evaluate against baseline",
    )
    parser.add_argument(
        "--baseline",
        default="data/baseline_metrics.json",
        help="Path to baseline metrics file",
    )
    args = parser.parse_args()

    dataset_path = Path("data") / "golden_eval_v1.json"
    if not dataset_path.exists():
        print(f"Error: Golden dataset not found at {dataset_path}")
        sys.exit(1)

    dataset = EvaluationDataset.load_from_json(dataset_path)

    if args.candidate == "reranked":
        dense = DenseRetriever(dataset.passages)
        bm25 = BM25Retriever(dataset.passages)
        hybrid = HybridRetriever(dense, bm25)
        retriever = RerankedRetriever(hybrid, dataset.passages, candidate_k=25)
        name = "Production_Hybrid_Reranked"
    elif args.candidate == "hybrid":
        dense = DenseRetriever(dataset.passages)
        bm25 = BM25Retriever(dataset.passages)
        retriever = HybridRetriever(dense, bm25)
        name = "PR_Candidate_Hybrid_RRF"
    elif args.candidate == "dense":
        retriever = DenseRetriever(dataset.passages)
        name = "PR_Candidate_Dense"
    elif args.candidate == "bm25":
        retriever = BM25Retriever(dataset.passages)
        name = "PR_Candidate_BM25"
    elif args.candidate == "bad":
        retriever = BadRetriever(dataset.passages[0].id)
        name = "PR_Candidate_Flawed_Buggy_Code"

    print(f"\nEvaluating CI Regression Gate for candidate: {name}...")
    result = run_regression_gate(
        candidate_retriever=retriever,
        candidate_name=name,
        dataset=dataset,
        baseline_path=args.baseline,
    )

    print("\n" + "=" * 80)
    print(result.summary)
    print("=" * 80 + "\n")

    if not result.passed:
        print("[FAIL] CI Gate blocked this pull request due to regression. Exit Code 1.\n")
        sys.exit(1)
    else:
        print("[SUCCESS] CI Gate passed. Safe to merge! Exit Code 0.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
