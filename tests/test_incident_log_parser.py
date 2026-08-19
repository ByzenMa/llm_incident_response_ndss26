from incident_log_parser import extract_cves, extract_iocs, parse_logs


def test_parse_logs_extracts_incident_json_fields():
    incident = parse_logs(
        [
            "critical ssh brute force login failed src=203.0.113.10 host=web-01 service=sshd CVE-2024-12345",
            "powershell payload downloaded from http://evil.example/a.ps1 account=alice",
        ],
        incident_id="inc-test",
    )

    assert incident.incident_id == "inc-test"
    assert "CVE-2024-12345" in incident.cves
    assert "web-01" in incident.assets
    assert "alice" in incident.assets
    assert "ssh" in incident.services
    assert "initial_access" in incident.attack_stages
    assert "execution" in incident.attack_stages
    assert incident.severity == "critical"


def test_extract_iocs_deduplicates_url_domains():
    iocs = extract_iocs("callback http://c2.example/payload from 198.51.100.4 hash d41d8cd98f00b204e9800998ecf8427e")
    values = {(ioc.type, ioc.value) for ioc in iocs}

    assert ("url", "http://c2.example/payload") in values
    assert ("ipv4", "198.51.100.4") in values
    assert ("md5", "d41d8cd98f00b204e9800998ecf8427e") in values
    assert ("domain", "c2.example") not in values


def test_extract_cves_normalizes_case():
    assert extract_cves("exploit cve-2023-34362 and CVE-2023-34362") == ["CVE-2023-34362"]
