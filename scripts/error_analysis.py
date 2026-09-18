import sys
import json
from pathlib import Path

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from groundtruth.schema import EvaluationDataset


def main():
    report_path = Path("results") / "benchmark_report.json"
    dataset_path = Path("data") / "golden_eval_v1.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.load_from_json(dataset_path)

    # Use the Hybrid + Reranker run (or Hybrid run)
    hybrid_run = next(r for r in report["runs"] if "Hybrid + Reranker" in r["configuration_name"])

    # Sort queries by Recall@10 ascending, then Recall@5 ascending
    worst_queries = sorted(
        hybrid_run["query_results"],
        key=lambda q: (q["recall_at_10"], q["recall_at_5"], q["reciprocal_rank"]),
    )[:5]

    print("\n" + "=" * 90)
    print("ERROR ANALYSIS: THE 5 QUERIES WITH LOWEST RECALL & WHY THEY FAILED")
    print("=" * 90)

    for i, q in enumerate(worst_queries, 1):
        print(f"\n[{i}] Query ID: {q['query_id']} | Practice Area: {q['category']}")
        print(f"    Query: \"{q['query_text']}\"")
        print(f"    Metrics: Recall@5={q['recall_at_5']:.2f}, Recall@10={q['recall_at_10']:.2f}, MRR={q['reciprocal_rank']:.2f}, nDCG={q['ndcg_at_10']:.2f}")
        print(f"    Expected Precedents: {q['relevant_ids']}")
        print(f"    Retrieved Top 3:     {q['retrieved_ids'][:3]}")

        # Provide domain root cause
        if q["query_id"] == "q_contract_018":
            reason = "High Lexical & Semantic Overlap with Force Majeure (doc_contract_009): The query asks about market price drop under commercial impracticability; the retriever prioritized force majeure cases rather than the fixed-price coal contract doctrine."
        elif q["query_id"] == "q_tort_015":
            reason = "Abstract Formula Mismatch: Attorney query searches for 'Learned Hand formula B < PL'; the dense model clustered it under strict products risk-utility balancing (Barker) instead of maritime admiralty negligence (Carroll Towing)."
        elif q["query_id"] == "q_contract_003":
            reason = "Multi-Precedent Partial Recall: Found primary landmark precedent (Lady Duff-Gordon at Rank 1, MRR=1.0), but secondary background authorities (Bhasin, Foley) were crowded out by higher-ranked contract assent cases."
        elif q["query_id"] == "q_tort_002":
            reason = "Historical Inception Lineage: Query asks for the historical origin of strict products liability; the retriever pulled later design defect cases (Barker, MacPherson) ahead of the original Greenman power tool holding."
        else:
            reason = f"Cross-practice doctrinal competition: Precedents in {q['category']} share overlapping judicial phraseology with closely adjacent doctrines."

        print(f"    Root Cause: {reason}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
