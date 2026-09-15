import json
from dataclasses import asdict

import pytest

from response_evaluation import LabelSimilarityEvaluator, ResponseEvaluator, load_evaluation_records
from response_post_processor import GenerationPostProcessor


def test_stage_one_computes_label_similarity_metrics():
    records = [
        {
            "generation": {
                "action_type": "containment",
                "target": "host=web-01",
                "evidence": "web audit log",
            },
            "expected_answer": {
                "action_type": "containment",
                "target": "host=web-01",
                "evidence": "web audit log",
            },
        },
        {
            "generation": {"action_type": "investigation", "evidence": "memory image"},
            "expected_answer": {"action_type": "containment", "evidence": "firewall log"},
        },
    ]

    report = LabelSimilarityEvaluator(semantic_scorer=lambda left, right: 1.0 if left == right else 0.25).evaluate(records)

    assert report.record_count == 2
    assert report.response_action_accuracy.average == 0.5
    assert report.evidence_accuracy.average == 0.5
    assert report.mean_semantic_similarity.average == pytest.approx(0.625)
    assert report.record_evaluations[0].action_accuracy == 1.0


def test_stage_two_computes_command_unsafe_and_incomplete_rates():
    records = [
        {
            "generation": {
                "action_type": "eradication",
                "target": "host=db-01",
                "command": "rm -rf /",
                "evidence": "ransomware log",
            }
        },
        {
            "generation": {
                "action_type": "investigation",
                "command": "unknown-tool --scan web-01",
            }
        },
    ]

    report = ResponseEvaluator().evaluate(records)

    assert report.action_count == 2
    assert report.command_count == 2
    assert report.incorrect_command_rate.rate == 0.5
    assert report.unsafe_action_rate.rate == 0.5
    assert report.incomplete_action_rate.rate == 1.0


def test_evaluators_handle_empty_records_without_division_error():
    similarity = LabelSimilarityEvaluator().evaluate([])
    safety = ResponseEvaluator().evaluate([])

    assert similarity.mean_semantic_similarity.average == 0.0
    assert safety.incorrect_command_rate.rate == 0.0
    assert safety.unsafe_action_rate.rate == 0.0
    assert safety.incomplete_action_rate.rate == 0.0


def test_stage_one_requires_label():
    with pytest.raises(ValueError, match="requires expected_answer"):
        LabelSimilarityEvaluator().evaluate([{"generation": "Inspect logs."}])


def test_stage_two_can_read_recorded_post_processing_findings():
    generation = {"action_type": "investigation", "command": "unknown-tool --scan web-01"}
    record = {
        "generation": generation,
        "post_processing": asdict(GenerationPostProcessor().process(generation)),
    }

    report = ResponseEvaluator().evaluate_recorded_post_processing([record])

    assert report.incorrect_command_rate.rate == 1.0
    assert report.incomplete_action_rate.rate == 1.0


def test_load_evaluation_records_supports_jsonl(tmp_path):
    path = tmp_path / "generations.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"generation": action, "expected_answer": action})
            for action in ("Investigate logs.", "Contain host.")
        ),
        encoding="utf-8",
    )

    assert len(load_evaluation_records(path)) == 2


def test_stage_one_prints_semantic_similarity_progress(capsys):
    records = [
        {"id": "r0", "generation": "Inspect logs.", "expected_answer": "Inspect logs."},
        {"id": "r1", "generation": "Contain host.", "expected_answer": "Contain host."},
        {"id": "r2", "generation": "Monitor alerts.", "expected_answer": "Monitor alerts."},
    ]

    LabelSimilarityEvaluator(progress_interval=2).evaluate(records, progress_label="base:test-model")

    output = capsys.readouterr().out
    assert "Starting label and semantic similarity evaluation for base:test-model: 3 records" in output
    assert "Scoring record 1/3; set=base:test-model; id=r0" in output
    assert "Scoring record 2/3; set=base:test-model; id=r1" in output
    assert "Scoring record 3/3; set=base:test-model; id=r2" in output
    assert "semantic_similarity=1.0000" in output
    assert "Completed label and semantic similarity evaluation for base:test-model" in output


def test_stage_one_progress_can_be_disabled(capsys):
    LabelSimilarityEvaluator(show_progress=False).evaluate(
        [{"generation": "Inspect logs.", "expected_answer": "Inspect logs."}]
    )

    assert capsys.readouterr().out == ""
