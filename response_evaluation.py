"""Evaluate generated incident-response plans with deterministic safety metrics.

The evaluator reuses the generation post-processor so offline evaluation and
the generation-time safety gate apply the same rules.  Rates are calculated at
the natural unit of each metric: actions for hallucination and safety, and
commands for command correctness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from response_post_processor import GenerationPostProcessor, ValidationFinding


@dataclass
class MetricResult:
    numerator: int
    denominator: int
    rate: float


@dataclass
class ActionEvaluation:
    record_index: int
    action_index: int
    action: Dict[str, Any]
    hallucinated: bool
    unsafe: bool
    command_count: int
    incorrect_command_count: int
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationReport:
    record_count: int
    action_count: int
    command_count: int
    hallucinated_action_rate: MetricResult
    incorrect_command_rate: MetricResult
    unsafe_action_rate: MetricResult
    action_evaluations: List[ActionEvaluation]


def _is_hallucination(finding: ValidationFinding) -> bool:
    """Return whether a finding represents an unsupported generated claim.

    Invalid/untrusted CVEs and ATT&CK IDs not linked to the retrieved incident
    graph count as hallucinations. Merely omitting an attack-stage reference
    does not, because omission is a completeness issue rather than fabrication.
    """

    return finding.category == "cve_authenticity" or (
        finding.category == "attack_path" and bool(finding.evidence)
    )


def _rate(numerator: int, denominator: int) -> MetricResult:
    return MetricResult(numerator, denominator, numerator / denominator if denominator else 0.0)


class ResponseEvaluator:
    """Compute aggregate quality and safety rates over generated responses."""

    def __init__(self, processor: Optional[GenerationPostProcessor] = None) -> None:
        self.processor = processor or GenerationPostProcessor()

    def evaluate(self, records: Iterable[Dict[str, Any]]) -> EvaluationReport:
        details: List[ActionEvaluation] = []
        record_count = 0
        hallucinated_actions = 0
        unsafe_actions = 0
        command_count = 0
        incorrect_commands = 0

        for record_index, record in enumerate(records):
            record_count += 1
            generation = _record_generation(record)
            context = record.get("kg_context", record.get("security_context"))
            actions = self.processor.parse_actions(generation)
            for action_index, action in enumerate(actions):
                findings = self.processor.validate_action(action, kg_context=context)
                hallucinated = any(_is_hallucination(finding) for finding in findings)
                unsafe = any(
                    finding.category == "policy_constraint" and finding.severity == "error"
                    for finding in findings
                )
                invalid_command_values = {
                    finding.evidence
                    for finding in findings
                    if finding.category == "command_syntax" and finding.evidence
                }
                action_command_count = len(action.command)
                action_incorrect_commands = sum(command in invalid_command_values for command in action.command)

                hallucinated_actions += int(hallucinated)
                unsafe_actions += int(unsafe)
                command_count += action_command_count
                incorrect_commands += action_incorrect_commands
                details.append(
                    ActionEvaluation(
                        record_index=record_index,
                        action_index=action_index,
                        action=asdict(action),
                        hallucinated=hallucinated,
                        unsafe=unsafe,
                        command_count=action_command_count,
                        incorrect_command_count=action_incorrect_commands,
                        findings=[asdict(finding) for finding in findings],
                    )
                )

        action_count = len(details)
        return EvaluationReport(
            record_count=record_count,
            action_count=action_count,
            command_count=command_count,
            hallucinated_action_rate=_rate(hallucinated_actions, action_count),
            incorrect_command_rate=_rate(incorrect_commands, command_count),
            unsafe_action_rate=_rate(unsafe_actions, action_count),
            action_evaluations=details,
        )


def _record_generation(record: Dict[str, Any]) -> Any:
    for key in ("generation", "generated_response", "response", "actions"):
        if key in record:
            return record[key]
    raise ValueError("Each evaluation record must contain generation, generated_response, response, or actions.")


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
    parser = argparse.ArgumentParser(description="Evaluate generated incident-response actions.")
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL records with generation and optional kg_context.")
    parser.add_argument("--output", type=Path, help="Optional path for the detailed JSON evaluation report.")
    parser.add_argument("--allowed-cve", action="append", default=[], help="Trusted CVE identifier; can be repeated.")
    parser.add_argument("--allow-external-cves", action="store_true", help="Do not treat context-external CVEs as findings.")
    args = parser.parse_args()

    evaluator = ResponseEvaluator(
        GenerationPostProcessor(
            allowed_cves=args.allowed_cve,
            require_context_cve_match=not args.allow_external_cves,
        )
    )
    report_json = json.dumps(asdict(evaluator.evaluate(load_evaluation_records(args.input))), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
