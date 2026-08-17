import json

from action_schema_parser import ActionSchemaParser


def test_parse_json_action_answer_extracts_core_schema():
    text = json.dumps(
        {
            "Action": "Acquire full disk and memory images of 10.20.11.42 and export DNS, firewall, and NetFlow logs.",
            "Explanation": "Capturing images and logs secures evidence for later analysis.",
        }
    )

    parsed = ActionSchemaParser().parse(text)

    assert parsed.action_type == "evidence_acquisition"
    assert parsed.target == ["10.20.11.42"]
    assert "images" in parsed.evidence[0]
    assert parsed.source_action.startswith("Acquire full disk")


def test_parse_explicit_structured_fields_are_preserved():
    parsed = ActionSchemaParser().parse(
        {
            "action_type": "containment",
            "target": "host:web-01, 192.0.2.10",
            "command": "sudo ufw deny from 192.0.2.10; systemctl restart nginx",
            "precondition": "Confirm the IP is malicious.",
            "risk": "May block legitimate traffic.",
            "rollback": "Remove the deny rule.",
            "evidence": "firewall log, IDS alert",
        }
    )

    assert parsed.action_type == "containment"
    assert parsed.target == ["host:web-01", "192.0.2.10"]
    assert parsed.command == ["sudo ufw deny from 192.0.2.10", "systemctl restart nginx"]
    assert parsed.precondition == "Confirm the IP is malicious."
    assert parsed.risk == "May block legitimate traffic."
    assert parsed.rollback == "Remove the deny rule."
    assert parsed.evidence == ["firewall log", "IDS alert"]
