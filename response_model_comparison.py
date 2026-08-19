"""Compare base-model and KG-RAG-fine-tuned response safety metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from response_evaluation import EvaluationReport, ResponseEvaluator, load_evaluation_records
from response_post_processor import GenerationPostProcessor


@dataclass
class MetricComparison:
    baseline_rate: float
    candidate_rate: float
    absolute_change: float
    absolute_improvement: float
    relative_reduction: Optional[float]


@dataclass
class ModelComparisonReport:
    baseline_name: str
    candidate_name: str
    paired_record_count: int
    baseline: EvaluationReport
    candidate: EvaluationReport
    hallucinated_action_rate: MetricComparison
    incorrect_command_rate: MetricComparison
    unsafe_action_rate: MetricComparison


def _record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "record_id", "incident_id", "prompt_id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def validate_paired_records(
    baseline_records: Sequence[Dict[str, Any]], candidate_records: Sequence[Dict[str, Any]]
) -> None:
    """Ensure both models were evaluated on the same ordered prompt set."""

    if len(baseline_records) != len(candidate_records):
        raise ValueError("Baseline and candidate files must contain the same number of records.")
    for index, (baseline, candidate) in enumerate(zip(baseline_records, candidate_records)):
        baseline_id = _record_id(baseline)
        candidate_id = _record_id(candidate)
        if baseline_id is not None and candidate_id is not None and baseline_id != candidate_id:
            raise ValueError(
                f"Record {index} is not paired: baseline ID {baseline_id!r} != candidate ID {candidate_id!r}."
            )


def _compare_metric(baseline_rate: float, candidate_rate: float) -> MetricComparison:
    change = candidate_rate - baseline_rate
    improvement = baseline_rate - candidate_rate
    relative_reduction = improvement / baseline_rate if baseline_rate else None
    return MetricComparison(
        baseline_rate=baseline_rate,
        candidate_rate=candidate_rate,
        absolute_change=change,
        absolute_improvement=improvement,
        relative_reduction=relative_reduction,
    )


class ModelResponseComparator:
    """Evaluate and compare two models over paired generated responses."""

    def __init__(self, evaluator: Optional[ResponseEvaluator] = None) -> None:
        self.evaluator = evaluator or ResponseEvaluator()

    def compare(
        self,
        baseline_records: Sequence[Dict[str, Any]],
        candidate_records: Sequence[Dict[str, Any]],
        baseline_name: str = "base",
        candidate_name: str = "kg_rag_finetuned",
    ) -> ModelComparisonReport:
        validate_paired_records(baseline_records, candidate_records)
        baseline = self.evaluator.evaluate(baseline_records)
        candidate = self.evaluator.evaluate(candidate_records)
        return ModelComparisonReport(
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            paired_record_count=len(baseline_records),
            baseline=baseline,
            candidate=candidate,
            hallucinated_action_rate=_compare_metric(
                baseline.hallucinated_action_rate.rate, candidate.hallucinated_action_rate.rate
            ),
            incorrect_command_rate=_compare_metric(
                baseline.incorrect_command_rate.rate, candidate.incorrect_command_rate.rate
            ),
            unsafe_action_rate=_compare_metric(
                baseline.unsafe_action_rate.rate, candidate.unsafe_action_rate.rate
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and KG-RAG-fine-tuned model safety metrics.")
    parser.add_argument("--baseline-input", type=Path, required=True, help="Base-model JSON/JSONL generations.")
    parser.add_argument("--candidate-input", type=Path, required=True, help="Fine-tuned-model JSON/JSONL generations.")
    parser.add_argument("--baseline-name", default="base", help="Display name for the baseline model.")
    parser.add_argument("--candidate-name", default="kg_rag_finetuned", help="Display name for the candidate model.")
    parser.add_argument("--output", type=Path, help="Optional path for the comparison JSON report.")
    parser.add_argument("--allowed-cve", action="append", default=[], help="Trusted CVE identifier; can be repeated.")
    parser.add_argument("--allow-external-cves", action="store_true", help="Do not flag context-external CVEs.")
    args = parser.parse_args()

    evaluator = ResponseEvaluator(
        GenerationPostProcessor(
            allowed_cves=args.allowed_cve,
            require_context_cve_match=not args.allow_external_cves,
        )
    )
    report = ModelResponseComparator(evaluator).compare(
        load_evaluation_records(args.baseline_input),
        load_evaluation_records(args.candidate_input),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
