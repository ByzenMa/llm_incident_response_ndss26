from dataclasses import asdict

import pytest

from response_model_comparison import (
    ModelResponseComparator,
    validate_paired_records,
    validate_post_processing_modes,
)
from response_post_processor import GenerationPostProcessor


def _record(record_id, generation, expected, post_processing_enabled):
    processor = GenerationPostProcessor()
    report = asdict(processor.process(generation)) if post_processing_enabled else None
    return {
        "id": record_id,
        "generation": generation,
        "expected_answer": expected,
        "post_processing_enabled": post_processing_enabled,
        "post_processing": report,
    }


def test_two_stage_comparison_reports_similarity_and_post_processing_statistics():
    label_one = {"action_type": "containment", "target": "host=web-01", "evidence": "firewall log"}
    label_two = {"action_type": "investigation", "target": "host=web-02", "evidence": "audit log"}
    baseline = [
        _record(
            "p1",
            {"action_type": "eradication", "target": "host=web-01", "command": "rm -rf /", "evidence": "guess"},
            label_one,
            False,
        ),
        _record(
            "p2",
            {"action_type": "investigation", "command": "unknown-tool --scan web-02"},
            label_two,
            False,
        ),
    ]
    candidate = [
        _record("p1", label_one, label_one, True),
        _record("p2", label_two, label_two, True),
    ]

    report = ModelResponseComparator().compare(baseline, candidate)

    assert report.paired_record_count == 2
    assert report.stage_one_label_similarity.response_action_accuracy.absolute_improvement == 0.5
    assert report.stage_one_label_similarity.evidence_accuracy.candidate_score == 1.0
    assert report.stage_two_post_processing.baseline_ignored_incorrect_command_rate.rate == 0.5
    assert report.stage_two_post_processing.baseline_ignored_unsafe_action_rate.rate == 0.5
    assert report.stage_two_post_processing.candidate_recorded_incorrect_command_rate.rate == 0.0
    assert report.stage_two_post_processing.candidate_recorded_incomplete_action_rate.rate == 0.0


def test_comparison_rejects_unpaired_inputs_and_labels():
    with pytest.raises(ValueError, match="same number"):
        validate_paired_records([], [{"generation": "review"}])
    with pytest.raises(ValueError, match="not paired"):
        validate_paired_records([{"id": "p1"}], [{"id": "p2"}])
    with pytest.raises(ValueError, match="different expected_answer"):
        validate_paired_records(
            [{"id": "p1", "expected_answer": "one"}],
            [{"id": "p1", "expected_answer": "two"}],
        )


def test_comparison_enforces_base_off_and_candidate_on_post_processing():
    with pytest.raises(ValueError, match="baseline"):
        validate_post_processing_modes([{"post_processing_enabled": True}], [{"post_processing_enabled": True, "post_processing": {}}])
    with pytest.raises(ValueError, match="candidate"):
        validate_post_processing_modes([{"post_processing_enabled": False}], [{"post_processing_enabled": False}])
