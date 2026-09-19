"""Run a controlled no-verifier versus verifier ablation experiment.

The same saved generations are evaluated twice. The no-verifier arm releases
all actions and measures errors offline; the verifier arm applies the existing
deterministic post-processor before release. This isolates verifier effects from
model sampling differences.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from response_evaluation import MetricResult, ResponseEvaluator, SafetyEvaluationReport, load_evaluation_records
from response_post_processor import GenerationPostProcessor


@dataclass
class NoVerifierArm:
    released_action_count: int
    ignored_incorrect_command_rate: MetricResult
    ignored_unsafe_action_rate: MetricResult
    ignored_incomplete_action_rate: MetricResult


@dataclass
class VerifierArm:
    accepted_action_count: int
    blocked_action_count: int
    blocking_rate: float
    recorded_incorrect_command_rate: MetricResult
    recorded_unsafe_action_rate: MetricResult
    recorded_incomplete_action_rate: MetricResult
    incorrect_commands_flagged: int
    unsafe_actions_prevented: int
    unsafe_actions_released: int
    incomplete_actions_flagged: int
    blocked_record_ids: List[str]


@dataclass
class VerifierAblationReport:
    record_count: int
    generation_identity_preserved: bool
    no_verifier: NoVerifierArm
    verifier: VerifierArm
    unsafe_action_release_reduction: int
    verified_safety_details: SafetyEvaluationReport


def _record_id(record: Dict[str, Any], index: int) -> str:
    for key in ("id", "record_id", "source_index", "incident_id", "prompt_id"):
        if record.get(key) is not None:
            return str(record[key])
    return str(index)


def _has_policy_error(wrapped_action: Dict[str, Any]) -> bool:
    return any(
        finding.get("category") == "policy_constraint" and finding.get("severity") == "error"
        for finding in wrapped_action.get("findings", [])
    )


class VerifierAblationExperiment:
    """Apply both ablation arms to identical generation text."""

    def __init__(self, processor: Optional[GenerationPostProcessor] = None) -> None:
        self.processor = processor or GenerationPostProcessor()
        self.safety_evaluator = ResponseEvaluator(self.processor)

    def run(
        self, records: Sequence[Dict[str, Any]]
    ) -> tuple[VerifierAblationReport, List[Dict[str, Any]]]:
        # Always derive both arms from the immutable original generation. Any
        # previously recorded verifier state is deliberately ignored and is
        # replaced only in the deep-copied verifier arm below.
        no_verifier_safety = self.safety_evaluator.evaluate(records)
        verified_records = copy.deepcopy(list(records))
        accepted_count = 0
        blocked_count = 0
        unsafe_prevented = 0
        unsafe_released = 0
        blocked_record_ids: List[str] = []

        for index, record in enumerate(verified_records):
            generation = record.get("generation")
            if generation is None:
                raise ValueError(f"Record {index} does not contain generation.")
            kg_context = record.get("kg_context", record.get("security_context"))
            result = asdict(self.processor.process(generation, kg_context=kg_context))
            accepted = result.get("actions", [])
            blocked = result.get("blocked_actions", [])
            accepted_count += len(accepted)
            blocked_count += len(blocked)
            unsafe_prevented += sum(_has_policy_error(action) for action in blocked)
            unsafe_released += sum(_has_policy_error(action) for action in accepted)
            if blocked:
                blocked_record_ids.append(_record_id(record, index))
            record["post_processing_enabled"] = True
            record["post_processing"] = result

        verified_safety = self.safety_evaluator.evaluate_recorded_post_processing(verified_records)
        total_verified_actions = accepted_count + blocked_count
        report = VerifierAblationReport(
            record_count=len(records),
            generation_identity_preserved=all(
                original.get("generation") == verified.get("generation")
                for original, verified in zip(records, verified_records)
            ),
            no_verifier=NoVerifierArm(
                released_action_count=no_verifier_safety.action_count,
                ignored_incorrect_command_rate=no_verifier_safety.incorrect_command_rate,
                ignored_unsafe_action_rate=no_verifier_safety.unsafe_action_rate,
                ignored_incomplete_action_rate=no_verifier_safety.incomplete_action_rate,
            ),
            verifier=VerifierArm(
                accepted_action_count=accepted_count,
                blocked_action_count=blocked_count,
                blocking_rate=blocked_count / total_verified_actions if total_verified_actions else 0.0,
                recorded_incorrect_command_rate=verified_safety.incorrect_command_rate,
                recorded_unsafe_action_rate=verified_safety.unsafe_action_rate,
                recorded_incomplete_action_rate=verified_safety.incomplete_action_rate,
                incorrect_commands_flagged=verified_safety.incorrect_command_rate.numerator,
                unsafe_actions_prevented=unsafe_prevented,
                unsafe_actions_released=unsafe_released,
                incomplete_actions_flagged=verified_safety.incomplete_action_rate.numerator,
                blocked_record_ids=blocked_record_ids,
            ),
            unsafe_action_release_reduction=(
                no_verifier_safety.unsafe_action_rate.numerator - unsafe_released
            ),
            verified_safety_details=verified_safety,
        )
        return report, verified_records


def save_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no-verifier and verifier arms on identical generations.")
    parser.add_argument("--input", type=Path, required=True, help="Raw JSON/JSONL prediction records.")
    parser.add_argument("--output", type=Path, required=True, help="Verifier ablation report JSON.")
    parser.add_argument("--verified-output", type=Path, help="Optional copied JSONL records with verifier reports.")
    parser.add_argument("--allowed-cve", action="append", default=[])
    parser.add_argument("--allow-external-cves", action="store_true")
    args = parser.parse_args()
    processor = GenerationPostProcessor(
        allowed_cves=args.allowed_cve,
        require_context_cve_match=not args.allow_external_cves,
    )
    report, verified_records = VerifierAblationExperiment(processor).run(
        load_evaluation_records(args.input)
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json + "\n", encoding="utf-8")
    if args.verified_output:
        save_records(args.verified_output, verified_records)
    print(report_json)


if __name__ == "__main__":
    main()
