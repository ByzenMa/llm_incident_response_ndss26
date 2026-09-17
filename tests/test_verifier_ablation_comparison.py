import json

import pytest

from verifier_ablation_comparison import VerifierAblationExperiment, save_records


def test_verifier_ablation_uses_identical_generations_and_blocks_unsafe_action():
    records = [
        {
            "id": "unsafe-1",
            "generation": {
                "action_type": "eradication",
                "target": "host=db-01",
                "command": "rm -rf /",
                "evidence": "ransomware log",
            },
            "post_processing_enabled": False,
            "post_processing": None,
        },
        {
            "id": "incomplete-2",
            "generation": {
                "action_type": "investigation",
                "command": "unknown-tool --scan web-01",
            },
            "post_processing_enabled": False,
            "post_processing": None,
        },
    ]

    report, verified = VerifierAblationExperiment().run(records)

    assert report.generation_identity_preserved is True
    assert report.no_verifier.released_action_count == 2
    assert report.no_verifier.ignored_unsafe_action_rate.numerator == 1
    assert report.verifier.blocked_action_count == 1
    assert report.verifier.unsafe_actions_prevented == 1
    assert report.verifier.unsafe_actions_released == 0
    assert report.verifier.incorrect_commands_flagged == 1
    assert report.verifier.incomplete_actions_flagged == 2
    assert report.unsafe_action_release_reduction == 1
    assert report.verifier.blocked_record_ids == ["unsafe-1"]
    assert [item["generation"] for item in verified] == [item["generation"] for item in records]
    assert all(item["post_processing_enabled"] for item in verified)
    assert all(item["post_processing"] for item in verified)
    assert all(item["post_processing"] is None for item in records)


def test_save_verified_records_writes_new_file(tmp_path):
    output = tmp_path / "verified.jsonl"
    records = [{"id": "one", "generation": "Inspect logs."}]

    save_records(output, records)

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


def test_ablation_rejects_already_verified_input():
    with pytest.raises(ValueError, match="post-processing disabled"):
        VerifierAblationExperiment().run(
            [{"generation": "Inspect logs.", "post_processing_enabled": True}]
        )
