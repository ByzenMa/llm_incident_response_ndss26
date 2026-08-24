"""Compare base and KG-RAG-fine-tuned models with a two-stage protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from response_evaluation import (
    LabelSimilarityEvaluator,
    MetricResult,
    ResponseEvaluator,
    SentenceTransformerSimilarity,
    SimilarityEvaluationReport,
    SafetyEvaluationReport,
    load_evaluation_records,
)


@dataclass
class SimilarityMetricComparison:
    baseline_score: float
    candidate_score: float
    absolute_improvement: float
    relative_improvement: Optional[float]


@dataclass
class StageOneComparison:
    baseline: SimilarityEvaluationReport
    candidate: SimilarityEvaluationReport
    response_action_accuracy: SimilarityMetricComparison
    evidence_accuracy: SimilarityMetricComparison
    mean_semantic_similarity: SimilarityMetricComparison


@dataclass
class StageTwoComparison:
    baseline_without_post_processing: SafetyEvaluationReport
    candidate_with_post_processing: SafetyEvaluationReport
    baseline_ignored_incorrect_command_rate: MetricResult
    baseline_ignored_unsafe_action_rate: MetricResult
    candidate_recorded_incorrect_command_rate: MetricResult
    candidate_recorded_incomplete_action_rate: MetricResult


@dataclass
class ModelComparisonReport:
    baseline_name: str
    candidate_name: str
    paired_record_count: int
    stage_one_label_similarity: StageOneComparison
    stage_two_post_processing: StageTwoComparison


def _record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "record_id", "incident_id", "prompt_id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def validate_paired_records(
    baseline_records: Sequence[Dict[str, Any]], candidate_records: Sequence[Dict[str, Any]]
) -> None:
    """Ensure both models use the same ordered test labels and source IDs."""
    if len(baseline_records) != len(candidate_records):
        raise ValueError("Baseline and candidate files must contain the same number of records.")
    for index, (baseline, candidate) in enumerate(zip(baseline_records, candidate_records)):
        baseline_id = _record_id(baseline)
        candidate_id = _record_id(candidate)
        if baseline_id is not None and candidate_id is not None and baseline_id != candidate_id:
            raise ValueError(
                f"Record {index} is not paired: baseline ID {baseline_id!r} != candidate ID {candidate_id!r}."
            )
        baseline_label = baseline.get("expected_answer")
        candidate_label = candidate.get("expected_answer")
        if baseline_label is not None and candidate_label is not None and baseline_label != candidate_label:
            raise ValueError(f"Record {index} has different expected_answer labels between models.")


def validate_post_processing_modes(
    baseline_records: Sequence[Dict[str, Any]], candidate_records: Sequence[Dict[str, Any]]
) -> None:
    """Enforce stage-2 conditions: base off, KG-RAG candidate on and recorded."""
    if any(record.get("post_processing_enabled") is not False for record in baseline_records):
        raise ValueError("Stage 2 requires every baseline record to have post_processing_enabled=false.")
    if any(record.get("post_processing_enabled") is not True for record in candidate_records):
        raise ValueError("Stage 2 requires every candidate record to have post_processing_enabled=true.")
    if any(record.get("post_processing") is None for record in candidate_records):
        raise ValueError("Stage 2 requires every candidate record to contain its post_processing report.")


def _compare_similarity(baseline: float, candidate: float) -> SimilarityMetricComparison:
    improvement = candidate - baseline
    return SimilarityMetricComparison(
        baseline,
        candidate,
        improvement,
        improvement / baseline if baseline else None,
    )


class ModelResponseComparator:
    """Run label-similarity and post-processing stages over paired predictions."""

    def __init__(
        self,
        similarity_evaluator: Optional[LabelSimilarityEvaluator] = None,
        safety_evaluator: Optional[ResponseEvaluator] = None,
    ) -> None:
        self.similarity_evaluator = similarity_evaluator or LabelSimilarityEvaluator()
        self.safety_evaluator = safety_evaluator or ResponseEvaluator()

    def compare(
        self,
        baseline_records: Sequence[Dict[str, Any]],
        candidate_records: Sequence[Dict[str, Any]],
        baseline_name: str = "base",
        candidate_name: str = "kg_rag_finetuned",
    ) -> ModelComparisonReport:
        validate_paired_records(baseline_records, candidate_records)
        validate_post_processing_modes(baseline_records, candidate_records)
        baseline_similarity = self.similarity_evaluator.evaluate(baseline_records)
        candidate_similarity = self.similarity_evaluator.evaluate(candidate_records)
        baseline_safety = self.safety_evaluator.evaluate(baseline_records)
        candidate_safety = self.safety_evaluator.evaluate_recorded_post_processing(candidate_records)
        stage_one = StageOneComparison(
            baseline_similarity,
            candidate_similarity,
            _compare_similarity(
                baseline_similarity.response_action_accuracy.average,
                candidate_similarity.response_action_accuracy.average,
            ),
            _compare_similarity(
                baseline_similarity.evidence_accuracy.average,
                candidate_similarity.evidence_accuracy.average,
            ),
            _compare_similarity(
                baseline_similarity.mean_semantic_similarity.average,
                candidate_similarity.mean_semantic_similarity.average,
            ),
        )
        stage_two = StageTwoComparison(
            baseline_safety,
            candidate_safety,
            baseline_safety.incorrect_command_rate,
            baseline_safety.unsafe_action_rate,
            candidate_safety.incorrect_command_rate,
            candidate_safety.incomplete_action_rate,
        )
        return ModelComparisonReport(
            baseline_name,
            candidate_name,
            len(baseline_records),
            stage_one,
            stage_two,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage base vs KG-RAG model evaluation.")
    parser.add_argument("--baseline-input", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--baseline-name", default="base")
    parser.add_argument("--candidate-name", default="kg_rag_finetuned")
    parser.add_argument("--semantic-model", help="Optional SentenceTransformer model for semantic similarity.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    semantic_scorer = SentenceTransformerSimilarity(args.semantic_model) if args.semantic_model else None
    report = ModelResponseComparator(
        similarity_evaluator=LabelSimilarityEvaluator(semantic_scorer=semantic_scorer)
    ).compare(
        load_evaluation_records(args.baseline_input),
        load_evaluation_records(args.candidate_input),
        args.baseline_name,
        args.candidate_name,
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
