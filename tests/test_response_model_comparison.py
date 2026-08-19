from dataclasses import asdict

import pytest

from incident_log_parser import parse_logs
from kg_rag import SecurityKnowledgeGraph
from response_model_comparison import ModelResponseComparator, validate_paired_records


def _context():
    incident = parse_logs(
        ["CVE-2023-34362 exploit against host=web-01 http service"],
        incident_id="inc-compare",
    )
    return asdict(SecurityKnowledgeGraph().retrieve_context(incident, depth=2))


def test_comparison_reports_improvements_for_finetuned_model():
    context = _context()
    baseline = [
        {
            "id": "prompt-1",
            "generation": {
                "Action": "Erase host with `rm -rf /` for CVE-2099-1234.",
                "Explanation": "Use unrelated T9999.",
            },
            "kg_context": context,
        },
        {
            "id": "prompt-2",
            "generation": {"Action": "Inspect with `unknown-tool --scan web-01`."},
            "kg_context": context,
        },
    ]
    candidate = [
        {
            "id": "prompt-1",
            "generation": {
                "Action": "Capture evidence with `tcpdump -i eth0 host web-01` for CVE-2023-34362.",
                "Explanation": "Web logs support T1190 initial access.",
            },
            "kg_context": context,
        },
        {
            "id": "prompt-2",
            "generation": {"Action": "Review web logs for the initial access stage."},
            "kg_context": context,
        },
    ]

    report = ModelResponseComparator().compare(baseline, candidate)

    assert report.paired_record_count == 2
    assert report.hallucinated_action_rate.baseline_rate == 0.5
    assert report.hallucinated_action_rate.candidate_rate == 0.0
    assert report.hallucinated_action_rate.absolute_improvement == 0.5
    assert report.hallucinated_action_rate.relative_reduction == 1.0
    assert report.incorrect_command_rate.absolute_improvement == 0.5
    assert report.unsafe_action_rate.absolute_improvement == 0.5


def test_comparison_uses_none_relative_reduction_for_zero_baseline():
    record = [{"id": "one", "generation": {"Action": "Review logs."}}]

    report = ModelResponseComparator().compare(record, record)

    assert report.hallucinated_action_rate.relative_reduction is None


def test_comparison_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="same number"):
        validate_paired_records([], [{"generation": "review"}])

    with pytest.raises(ValueError, match="not paired"):
        validate_paired_records(
            [{"id": "prompt-1", "generation": "review"}],
            [{"id": "prompt-2", "generation": "review"}],
        )
