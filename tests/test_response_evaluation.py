import json
from dataclasses import asdict

import pytest

from incident_log_parser import parse_logs
from kg_rag import SecurityKnowledgeGraph
from response_evaluation import ResponseEvaluator, load_evaluation_records


def _context(log):
    incident = parse_logs([log], incident_id="inc-eval")
    return asdict(SecurityKnowledgeGraph().retrieve_context(incident, depth=2))


def test_evaluator_computes_action_and_command_rates():
    context = _context("CVE-2023-34362 exploit against host=web-01 http service")
    records = [
        {
            "generation": {
                "Action": "Collect evidence with `tcpdump -i eth0 host web-01` for CVE-2023-34362.",
                "Explanation": "T1190 initial access observed in web logs.",
            },
            "kg_context": context,
        },
        {
            "generation": {
                "Action": "Remove data with `rm -rf /` for CVE-2099-1234.",
                "Explanation": "Unrelated T9999 technique.",
            },
            "kg_context": context,
        },
        {
            "generation": {
                "Action": "Inspect host with `madeup-tool --scan web-01`.",
                "Explanation": "Review the initial access stage.",
            },
            "kg_context": context,
        },
    ]

    report = ResponseEvaluator().evaluate(records)

    assert report.record_count == 3
    assert report.action_count == 3
    assert report.command_count == 3
    assert report.hallucinated_action_rate.numerator == 1
    assert report.hallucinated_action_rate.rate == pytest.approx(1 / 3)
    assert report.incorrect_command_rate.numerator == 1
    assert report.incorrect_command_rate.rate == pytest.approx(1 / 3)
    assert report.unsafe_action_rate.numerator == 1
    assert report.unsafe_action_rate.rate == pytest.approx(1 / 3)
    assert report.action_evaluations[1].hallucinated is True
    assert report.action_evaluations[1].unsafe is True


def test_evaluator_handles_empty_records_without_division_error():
    report = ResponseEvaluator().evaluate([])

    assert report.hallucinated_action_rate.rate == 0.0
    assert report.incorrect_command_rate.rate == 0.0
    assert report.unsafe_action_rate.rate == 0.0


def test_load_evaluation_records_supports_jsonl(tmp_path):
    path = tmp_path / "generations.jsonl"
    path.write_text(
        "\n".join(json.dumps({"generation": {"Action": action}}) for action in ("Investigate logs.", "Contain host.")),
        encoding="utf-8",
    )

    records = load_evaluation_records(path)

    assert len(records) == 2
