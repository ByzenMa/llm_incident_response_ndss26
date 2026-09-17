"""Compare recovery-only and risk-aware response strategy experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from model_test_generation import RECOVERY_ONLY_STRATEGY, RISK_AWARE_STRATEGY
from response_evaluation import (
    LabelSimilarityEvaluator,
    ResponseEvaluator,
    SafetyEvaluationReport,
    SentenceTransformerSimilarity,
    SimilarityEvaluationReport,
    load_evaluation_records,
)
from response_model_comparison import validate_paired_records


@dataclass
class ScoreGap:
    recovery_only: float
    risk_aware: float
    risk_aware_minus_recovery_only: float


@dataclass
class ErrorRateGap:
    recovery_only: float
    risk_aware: float
    reduction_from_recovery_to_risk_aware: float


@dataclass
class StrategyCoverage:
    action_count: int
    recovery_action_rate: float
    target_coverage: float
    evidence_coverage: float
    precondition_coverage: float
    risk_coverage: float
    rollback_coverage: float


@dataclass
class RecoveryRiskComparisonReport:
    paired_record_count: int
    recovery_only_similarity: SimilarityEvaluationReport
    risk_aware_similarity: SimilarityEvaluationReport
    recovery_only_safety: SafetyEvaluationReport
    risk_aware_safety: SafetyEvaluationReport
    recovery_only_coverage: StrategyCoverage
    risk_aware_coverage: StrategyCoverage
    response_action_accuracy_gap: ScoreGap
    evidence_accuracy_gap: ScoreGap
    mean_semantic_similarity_gap: ScoreGap
    incorrect_command_rate_gap: ErrorRateGap
    unsafe_action_rate_gap: ErrorRateGap
    incomplete_action_rate_gap: ErrorRateGap


def validate_strategy_modes(recovery_records: Sequence[dict], risk_records: Sequence[dict]) -> None:
    if any(record.get("response_strategy") != RECOVERY_ONLY_STRATEGY for record in recovery_records):
        raise ValueError("Every recovery arm record must have response_strategy='recovery_only'.")
    if any(record.get("response_strategy") != RISK_AWARE_STRATEGY for record in risk_records):
        raise ValueError("Every risk-aware arm record must have response_strategy='risk_aware'.")
    for name, records in ((RECOVERY_ONLY_STRATEGY, recovery_records), (RISK_AWARE_STRATEGY, risk_records)):
        if any(not isinstance(record.get("post_processing"), dict) for record in records):
            raise ValueError(f"Every {name} record must contain a post_processing report.")


def _all_recorded_actions(records: Sequence[dict]) -> List[dict]:
    actions: List[dict] = []
    for record in records:
        report = record["post_processing"]
        for wrapped in list(report.get("actions", [])) + list(report.get("blocked_actions", [])):
            actions.append(wrapped.get("action", {}))
    return actions


def _coverage(records: Sequence[dict]) -> StrategyCoverage:
    actions = _all_recorded_actions(records)
    count = len(actions)

    def rate(predicate) -> float:
        return sum(bool(predicate(action)) for action in actions) / count if count else 0.0

    return StrategyCoverage(
        action_count=count,
        recovery_action_rate=rate(lambda action: action.get("action_type") == "recovery"),
        target_coverage=rate(lambda action: action.get("target")),
        evidence_coverage=rate(lambda action: action.get("evidence")),
        precondition_coverage=rate(lambda action: action.get("precondition")),
        risk_coverage=rate(lambda action: action.get("risk")),
        rollback_coverage=rate(lambda action: action.get("rollback")),
    )


def _score_gap(recovery: float, risk_aware: float) -> ScoreGap:
    return ScoreGap(recovery, risk_aware, risk_aware - recovery)


def _error_gap(recovery: float, risk_aware: float) -> ErrorRateGap:
    return ErrorRateGap(recovery, risk_aware, recovery - risk_aware)


class RecoveryRiskComparator:
    def __init__(
        self,
        similarity_evaluator: Optional[LabelSimilarityEvaluator] = None,
        safety_evaluator: Optional[ResponseEvaluator] = None,
    ) -> None:
        self.similarity_evaluator = similarity_evaluator or LabelSimilarityEvaluator()
        self.safety_evaluator = safety_evaluator or ResponseEvaluator()

    def compare(self, recovery_records: Sequence[dict], risk_records: Sequence[dict]) -> RecoveryRiskComparisonReport:
        validate_paired_records(recovery_records, risk_records)
        validate_strategy_modes(recovery_records, risk_records)
        recovery_similarity = self.similarity_evaluator.evaluate(recovery_records, progress_label="recovery-only")
        risk_similarity = self.similarity_evaluator.evaluate(risk_records, progress_label="risk-aware")
        recovery_safety = self.safety_evaluator.evaluate_recorded_post_processing(recovery_records)
        risk_safety = self.safety_evaluator.evaluate_recorded_post_processing(risk_records)
        return RecoveryRiskComparisonReport(
            len(recovery_records),
            recovery_similarity,
            risk_similarity,
            recovery_safety,
            risk_safety,
            _coverage(recovery_records),
            _coverage(risk_records),
            _score_gap(recovery_similarity.response_action_accuracy.average, risk_similarity.response_action_accuracy.average),
            _score_gap(recovery_similarity.evidence_accuracy.average, risk_similarity.evidence_accuracy.average),
            _score_gap(recovery_similarity.mean_semantic_similarity.average, risk_similarity.mean_semantic_similarity.average),
            _error_gap(recovery_safety.incorrect_command_rate.rate, risk_safety.incorrect_command_rate.rate),
            _error_gap(recovery_safety.unsafe_action_rate.rate, risk_safety.unsafe_action_rate.rate),
            _error_gap(recovery_safety.incomplete_action_rate.rate, risk_safety.incomplete_action_rate.rate),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recovery-only and risk-aware prediction outputs.")
    parser.add_argument("--recovery-only-output", type=Path, required=True)
    parser.add_argument("--risk-aware-output", type=Path, required=True)
    parser.add_argument("--semantic-model")
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorer = SentenceTransformerSimilarity(args.semantic_model) if args.semantic_model else None
    report = RecoveryRiskComparator(
        similarity_evaluator=LabelSimilarityEvaluator(
            semantic_scorer=scorer,
            show_progress=args.show_progress,
            progress_interval=args.progress_interval,
        )
    ).compare(
        load_evaluation_records(args.recovery_only_output),
        load_evaluation_records(args.risk_aware_output),
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
