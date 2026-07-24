from dataclasses import asdict

from incident_log_parser import parse_logs
from kg_rag import SecurityKnowledgeGraph
from response_post_processor import GenerationPostProcessor


def _context_for_log(log):
    incident = parse_logs([log], incident_id="inc-post-1")
    return asdict(SecurityKnowledgeGraph().retrieve_context(incident, depth=2))


def test_post_processor_accepts_context_aligned_safe_action():
    context = _context_for_log("CVE-2023-34362 exploit against host=moveit-01 http service followed by upload exfil")
    generation = {
        "Action": "Run `tcpdump -i eth0 host moveit-01` and block CVE-2023-34362 exploit traffic for host=moveit-01.",
        "Explanation": "Evidence from web logs maps to T1190 initial access. Rollback by removing the temporary capture and firewall rule.",
    }

    result = GenerationPostProcessor().process(generation, kg_context=context)

    assert result.accepted is True
    assert result.blocked_actions == []
    assert all(finding["severity"] != "error" for finding in result.findings)


def test_post_processor_blocks_destructive_command_without_rollback():
    context = _context_for_log("ransomware impact on host=db-01 smb service")
    generation = {"Action": "Remove malware with `rm -rf /` on host=db-01.", "Explanation": "Impact stage."}

    result = GenerationPostProcessor().process(generation, kg_context=context)

    assert result.accepted is False
    categories = {finding["category"] for finding in result.findings}
    assert "policy_constraint" in categories
    assert result.blocked_actions


def test_post_processor_flags_malformed_and_out_of_context_cves():
    context = _context_for_log("CVE-2023-34362 exploit against host=moveit-01 http service")
    generation = {
        "Action": "Patch CVE-2099-1234 and investigate CVE-20-12345 on host=moveit-01.",
        "Explanation": "Rollback by reverting the package update if service health checks fail.",
    }

    result = GenerationPostProcessor().process(generation, kg_context=context)

    cve_findings = [finding for finding in result.findings if finding["category"] == "cve_authenticity"]
    assert any(finding["severity"] == "error" for finding in cve_findings)
    assert any("Malformed CVE" in finding["message"] for finding in cve_findings)
