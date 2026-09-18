"""CI Regression Gate for Groundtruth Retrieval.

Enforces that any proposed retriever change does not degrade Recall@10
by more than 1.0 percentage point (0.010) relative to production baseline.
"""

from pathlib import Path
import json
from groundtruth.schema import EvaluationDataset
from groundtruth.evaluator import EvaluationEngine, BenchmarkRun
from groundtruth.retriever import Retriever


class RegressionGateResult:
    """Encapsulates the verdict of a CI regression gate check."""

    def __init__(
        self,
        passed: bool,
        candidate_recall: float,
        baseline_recall: float,
        delta: float,
        max_allowed_drop: float,
        summary: str,
        run: BenchmarkRun,
    ):
        self.passed = passed
        self.candidate_recall = candidate_recall
        self.baseline_recall = baseline_recall
        self.delta = delta
        self.max_allowed_drop = max_allowed_drop
        self.summary = summary
        self.run = run


def run_regression_gate(
    candidate_retriever: Retriever,
    candidate_name: str,
    dataset: EvaluationDataset,
    baseline_path: str | Path = "data/baseline_metrics.json",
) -> RegressionGateResult:
    """Run regression gate evaluation against baseline rules.

    Args:
        candidate_retriever: The retriever proposed in the PR.
        candidate_name: Human-readable identifier for the candidate.
        dataset: The golden evaluation dataset.
        baseline_path: Path to baseline_metrics.json.

    Returns:
        RegressionGateResult with pass/fail status and audit report.
    """
    path = Path(baseline_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline configuration file not found at: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    gate_rules = config["gate_rules"]
    baseline_recall = float(gate_rules["target_baseline"])
    max_allowed_drop = float(gate_rules["max_allowed_drop"])
    hard_floor = baseline_recall - max_allowed_drop

    # Run evaluation on candidate
    engine = EvaluationEngine(dataset)
    run = engine.evaluate(
        candidate_retriever,
        configuration_name=candidate_name,
        top_k=10,
    )

    candidate_recall = run.overall_metrics.mean_recall_at_10
    delta = candidate_recall - baseline_recall

    # Check if delta violates max drop
    passed = candidate_recall >= hard_floor

    if passed:
        verdict = "PASSED"
        status_msg = (
            f"[CI GATE PASSED] Candidate '{candidate_name}' achieved Recall@10 = {candidate_recall:.3f} "
            f"(Delta vs Baseline: {delta:+.3f}). Threshold requirement (>= {hard_floor:.3f}) satisfied."
        )
    else:
        verdict = "FAILED"
        status_msg = (
            f"[CI GATE FAILED] REGRESSION DETECTED in candidate '{candidate_name}'! "
            f"Recall@10 dropped to {candidate_recall:.3f} (Delta vs Baseline: {delta:+.3f}). "
            f"Allowed maximum drop is {max_allowed_drop:.3f} (Hard floor: {hard_floor:.3f}). "
            f"This change introduces malpractice risk and is BLOCKED from merging!"
        )

    return RegressionGateResult(
        passed=passed,
        candidate_recall=candidate_recall,
        baseline_recall=baseline_recall,
        delta=delta,
        max_allowed_drop=max_allowed_drop,
        summary=status_msg,
        run=run,
    )
