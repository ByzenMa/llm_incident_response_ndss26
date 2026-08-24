"""Two-stage evaluation for incident-response model generations.

Stage 1 compares each generation with its labelled answer. Stage 2 inspects
command, policy, and action-completeness findings.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from response_post_processor import GenerationPostProcessor


@dataclass
class MetricResult:
    numerator: int
    denominator: int
    rate: float


@dataclass
class AverageMetric:
    total: float
    count: int
    average: float


@dataclass
class SimilarityRecordEvaluation:
    record_index: int
    action_accuracy: float
    evidence_accuracy: float
    semantic_similarity: float


@dataclass
class SimilarityEvaluationReport:
    record_count: int
    response_action_accuracy: AverageMetric
    evidence_accuracy: AverageMetric
    mean_semantic_similarity: AverageMetric
    record_evaluations: List[SimilarityRecordEvaluation]


@dataclass
class SafetyActionEvaluation:
    record_index: int
    action_index: int
    action: Dict[str, Any]
    unsafe: bool
    incomplete: bool
    command_count: int
    incorrect_command_count: int
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SafetyEvaluationReport:
    record_count: int
    action_count: int
    command_count: int
    incorrect_command_rate: MetricResult
    unsafe_action_rate: MetricResult
    incomplete_action_rate: MetricResult
    action_evaluations: List[SafetyActionEvaluation]


@dataclass
class TwoStageEvaluationReport:
    stage_one_label_similarity: SimilarityEvaluationReport
    stage_two_safety: SafetyEvaluationReport


def _rate(numerator: int, denominator: int) -> MetricResult:
    return MetricResult(numerator, denominator, numerator / denominator if denominator else 0.0)


def _average(values: Sequence[float]) -> AverageMetric:
    total = float(sum(values))
    return AverageMetric(total, len(values), total / len(values) if values else 0.0)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _record_generation(record: Dict[str, Any]) -> Any:
    for key in ("generation", "generated_response", "response", "actions"):
        if key in record:
            return record[key]
    raise ValueError("Each evaluation record must contain generation, generated_response, response, or actions.")


def _record_label(record: Dict[str, Any]) -> Any:
    for key in ("expected_answer", "label", "reference", "answer"):
        if key in record:
            return record[key]
    raise ValueError("Stage-1 evaluation requires expected_answer, label, reference, or answer in every record.")


def lexical_semantic_similarity(left: str, right: str) -> float:
    """Dependency-free cosine baseline used when no embedding model is selected."""
    left_counts = Counter(re.findall(r"[A-Za-z0-9_.:/-]+", left.lower()))
    right_counts = Counter(re.findall(r"[A-Za-z0-9_.:/-]+", right.lower()))
    if not left_counts and not right_counts:
        return 1.0
    dot = sum(value * right_counts[token] for token, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class SentenceTransformerSimilarity:
    """Optional language-model embedding scorer loaded only when requested."""

    def __init__(self, model_name_or_path: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path)

    def __call__(self, left: str, right: str) -> float:
        embeddings = self.model.encode([left, right], normalize_embeddings=True)
        score = sum(float(a) * float(b) for a, b in zip(embeddings[0], embeddings[1]))
        return max(-1.0, min(1.0, score))


def _set_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    predicted_counts = Counter(value for value in predicted if value)
    expected_counts = Counter(value for value in expected if value)
    denominator = max(sum(predicted_counts.values()), sum(expected_counts.values()))
    if denominator == 0:
        return 1.0
    matches = sum((predicted_counts & expected_counts).values())
    return matches / denominator


def _evidence_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    predicted_tokens = set(re.findall(r"[A-Za-z0-9_.:/-]+", " ".join(predicted).lower()))
    expected_tokens = set(re.findall(r"[A-Za-z0-9_.:/-]+", " ".join(expected).lower()))
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0
    return 2 * len(predicted_tokens & expected_tokens) / (len(predicted_tokens) + len(expected_tokens))


class LabelSimilarityEvaluator:
    """Stage 1: compare generated actions and evidence with labelled answers."""

    def __init__(
        self,
        processor: Optional[GenerationPostProcessor] = None,
        semantic_scorer: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.processor = processor or GenerationPostProcessor()
        self.semantic_scorer = semantic_scorer or lexical_semantic_similarity

    def evaluate(self, records: Iterable[Dict[str, Any]]) -> SimilarityEvaluationReport:
        details: List[SimilarityRecordEvaluation] = []
        for record_index, record in enumerate(records):
            generation = _record_generation(record)
            label = _record_label(record)
            predicted_actions = self.processor.parse_actions(generation)
            expected_actions = self.processor.parse_actions(label)
            action_accuracy = _set_accuracy(
                [action.action_type for action in predicted_actions],
                [action.action_type for action in expected_actions],
            )
            evidence_accuracy = _evidence_accuracy(
                [item for action in predicted_actions for item in action.evidence],
                [item for action in expected_actions for item in action.evidence],
            )
            semantic_similarity = float(self.semantic_scorer(_as_text(generation), _as_text(label)))
            details.append(
                SimilarityRecordEvaluation(
                    record_index, action_accuracy, evidence_accuracy, semantic_similarity
                )
            )
        return SimilarityEvaluationReport(
            record_count=len(details),
            response_action_accuracy=_average([item.action_accuracy for item in details]),
            evidence_accuracy=_average([item.evidence_accuracy for item in details]),
            mean_semantic_similarity=_average([item.semantic_similarity for item in details]),
            record_evaluations=details,
        )


class ResponseEvaluator:
    """Stage 2: calculate command, unsafe-action, and incomplete-action rates."""

    def __init__(self, processor: Optional[GenerationPostProcessor] = None) -> None:
        self.processor = processor or GenerationPostProcessor()

    def evaluate(self, records: Iterable[Dict[str, Any]]) -> SafetyEvaluationReport:
        details: List[SafetyActionEvaluation] = []
        record_count = 0
        incorrect_commands = 0
        command_count = 0
        unsafe_actions = 0
        incomplete_actions = 0
        for record_index, record in enumerate(records):
            record_count += 1
            context = record.get("kg_context", record.get("security_context"))
            for action_index, action in enumerate(self.processor.parse_actions(_record_generation(record))):
                findings = self.processor.validate_action(action, kg_context=context)
                unsafe = any(
                    finding.category == "policy_constraint" and finding.severity == "error"
                    for finding in findings
                )
                incomplete = any(finding.category == "action_completeness" for finding in findings)
                invalid_commands = {
                    finding.evidence
                    for finding in findings
                    if finding.category == "command_syntax" and finding.evidence
                }
                incorrect_count = sum(command in invalid_commands for command in action.command)
                command_count += len(action.command)
                incorrect_commands += incorrect_count
                unsafe_actions += int(unsafe)
                incomplete_actions += int(incomplete)
                details.append(
                    SafetyActionEvaluation(
                        record_index,
                        action_index,
                        asdict(action),
                        unsafe,
                        incomplete,
                        len(action.command),
                        incorrect_count,
                        [asdict(finding) for finding in findings],
                    )
                )
        action_count = len(details)
        return SafetyEvaluationReport(
            record_count,
            action_count,
            command_count,
            _rate(incorrect_commands, command_count),
            _rate(unsafe_actions, action_count),
            _rate(incomplete_actions, action_count),
            details,
        )

    def evaluate_recorded_post_processing(
        self, records: Iterable[Dict[str, Any]]
    ) -> SafetyEvaluationReport:
        """Calculate stage-2 metrics only from post-processing reports on disk."""
        details: List[SafetyActionEvaluation] = []
        record_count = 0
        command_count = 0
        incorrect_commands = 0
        unsafe_actions = 0
        incomplete_actions = 0
        for record_index, record in enumerate(records):
            record_count += 1
            report = record.get("post_processing")
            if not isinstance(report, dict):
                raise ValueError("Recorded evaluation requires a post_processing object in every record.")
            wrapped_actions = list(report.get("actions", [])) + list(report.get("blocked_actions", []))
            for action_index, wrapped in enumerate(wrapped_actions):
                action = wrapped.get("action", {})
                findings = wrapped.get("findings", [])
                commands = list(action.get("command", []))
                invalid_commands = {
                    finding.get("evidence")
                    for finding in findings
                    if finding.get("category") == "command_syntax" and finding.get("evidence")
                }
                incorrect_count = sum(command in invalid_commands for command in commands)
                unsafe = any(
                    finding.get("category") == "policy_constraint" and finding.get("severity") == "error"
                    for finding in findings
                )
                incomplete = any(finding.get("category") == "action_completeness" for finding in findings)
                command_count += len(commands)
                incorrect_commands += incorrect_count
                unsafe_actions += int(unsafe)
                incomplete_actions += int(incomplete)
                details.append(
                    SafetyActionEvaluation(
                        record_index,
                        action_index,
                        action,
                        unsafe,
                        incomplete,
                        len(commands),
                        incorrect_count,
                        findings,
                    )
                )
        action_count = len(details)
        return SafetyEvaluationReport(
            record_count,
            action_count,
            command_count,
            _rate(incorrect_commands, command_count),
            _rate(unsafe_actions, action_count),
            _rate(incomplete_actions, action_count),
            details,
        )


def load_evaluation_records(path: Path) -> Sequence[Dict[str, Any]]:
    """Load a JSON array/object or newline-delimited JSON evaluation file."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("records", [payload])
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Evaluation input must be a JSON object, array of objects, or JSONL objects.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage incident-response generation evaluation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--semantic-model",
        help="Optional local/HF SentenceTransformer model for language-model embedding similarity.",
    )
    args = parser.parse_args()
    records = load_evaluation_records(args.input)
    semantic_scorer = SentenceTransformerSimilarity(args.semantic_model) if args.semantic_model else None
    report = TwoStageEvaluationReport(
        LabelSimilarityEvaluator(semantic_scorer=semantic_scorer).evaluate(records),
        ResponseEvaluator().evaluate(records),
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
