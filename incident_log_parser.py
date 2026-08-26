"""Parse security logs and alerts into structured incident JSON.

This module implements part 2 of the incident-response pipeline: extracting
IOCs, CVEs, assets, services, and likely attack stages from raw logs/alerts.
It intentionally uses deterministic rules so it can run offline on commodity
systems and serve as a transparent pre-processor before action generation or
KG-RAG retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_ATTACK_STAGE_KEYWORDS: Dict[str, Sequence[str]] = {
    "reconnaissance": ("scan", "nmap", "enumerat", "probe", "discovery", "recon"),
    "initial_access": ("login failed", "brute", "phish", "exploit", "public-facing", "webshell", "valid account"),
    "execution": ("powershell", "cmd.exe", "bash", "exec", "spawn", "script", "payload"),
    "persistence": ("scheduled task", "cron", "service install", "registry run", "startup", "backdoor"),
    "privilege_escalation": ("sudo", "uac", "privilege", "root", "administrator", "setuid", "token"),
    "defense_evasion": ("disable", "clear log", "tamper", "obfuscat", "encodedcommand", "bypass"),
    "credential_access": ("credential", "password", "hash", "lsass", "mimikatz", "secrets", "token"),
    "discovery": ("whoami", "hostname", "net user", "ipconfig", "ifconfig", "systeminfo", "ldap"),
    "lateral_movement": ("rdp", "ssh", "psexec", "winrm", "smb", "remote service", "lateral"),
    "collection": ("archive", "collect", "compress", "staging", "clipboard", "screenshot"),
    "command_and_control": ("c2", "beacon", "callback", "dns tunnel", "tor", "command and control"),
    "exfiltration": ("exfil", "upload", "mega", "dropbox", "s3", "ftp", "data transfer"),
    "impact": ("encrypt", "ransom", "wipe", "delete shadow", "ddos", "deface", "destruct"),
}

_SERVICE_HINTS: Dict[str, Sequence[str]] = {
    "ssh": ("ssh", "sshd", "port 22"),
    "rdp": ("rdp", "mstsc", "port 3389"),
    "http": ("http", "apache", "nginx", "iis", "port 80"),
    "https": ("https", "tls", "ssl", "port 443"),
    "dns": ("dns", "bind", "port 53"),
    "smb": ("smb", "samba", "port 445"),
    "database": ("mysql", "postgres", "mssql", "mongodb", "redis", "database"),
    "email": ("smtp", "imap", "pop3", "exchange", "mail"),
    "kubernetes": ("kubernetes", "kubectl", "kubelet", "pod"),
    "aws": ("aws", "cloudtrail", "ec2", "iam", "s3"),
}

_IOC_PATTERNS: Dict[str, str] = {
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "ipv6": r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b",
    "domain": r"\b(?:[a-zA-Z0-9-]+\.)+[A-Za-z]{2,}\b",
    "url": r"https?://[^\s,;\]')\"]+",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "sha256": r"\b[A-Fa-f0-9]{64}\b",
    "sha1": r"\b[A-Fa-f0-9]{40}\b",
    "md5": r"\b[A-Fa-f0-9]{32}\b",
    "file_path": r"(?:[A-Za-z]:\\[^\s,;]+|/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+)",
}

_CVE_PATTERN = r"\bCVE-\d{4}-\d{4,7}\b"
_ASSET_PATTERNS = (
    r"\b(?:host|hostname|src_host|dst_host|asset|server|endpoint|node|workstation|instance|pod)[:= ]+([A-Za-z0-9._:@/-]+)",
    r"\b(?:user|account|principal|username)[:= ]+([A-Za-z0-9._@/-]+)",
)


@dataclass
class IOC:
    type: str
    value: str


@dataclass
class IncidentJSON:
    incident_id: str
    observed_at: str
    summary: str
    iocs: List[IOC] = field(default_factory=list)
    cves: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    attack_stages: List[str] = field(default_factory=list)
    severity: str = "unknown"
    source: str = "log_parser"
    raw_events: List[str] = field(default_factory=list)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = value.strip().strip(".,;()[]{}'")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_iocs(text: str) -> List[IOC]:
    """Extract common indicator types from log text."""
    indicators: List[IOC] = []
    for ioc_type, pattern in _IOC_PATTERNS.items():
        for value in _unique(re.findall(pattern, text)):
            indicators.append(IOC(type=ioc_type, value=value))
    # Avoid double-counting domains embedded in URLs and email addresses.
    url_domains = {re.sub(r"^https?://", "", i.value).split("/")[0].lower() for i in indicators if i.type == "url"}
    email_domains = {i.value.split("@", 1)[1].lower() for i in indicators if i.type == "email"}
    return [
        ioc
        for ioc in indicators
        if not (ioc.type == "domain" and ioc.value.lower() in url_domains | email_domains)
    ]


def extract_cves(text: str) -> List[str]:
    return _unique(match.upper() for match in re.findall(_CVE_PATTERN, text, flags=re.IGNORECASE))


def extract_assets(text: str) -> List[str]:
    assets: List[str] = []
    for pattern in _ASSET_PATTERNS:
        assets.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return _unique(assets)


def infer_services(text: str) -> List[str]:
    lowered = text.lower()
    return [service for service, hints in _SERVICE_HINTS.items() if any(hint in lowered for hint in hints)]


def infer_attack_stages(text: str) -> List[str]:
    lowered = text.lower()
    return [stage for stage, hints in _ATTACK_STAGE_KEYWORDS.items() if any(hint in lowered for hint in hints)]


def infer_severity(text: str, cves: Sequence[str], stages: Sequence[str]) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("critical", "ransom", "exfil", "domain admin", "root compromise")):
        return "critical"
    if cves or any(stage in stages for stage in ("lateral_movement", "credential_access", "impact")):
        return "high"
    if any(token in lowered for token in ("warning", "failed", "denied", "suspicious")):
        return "medium"
    return "low" if text.strip() else "unknown"


def parse_logs(logs: Iterable[str], incident_id: Optional[str] = None) -> IncidentJSON:
    events = [line.strip() for line in logs if line and line.strip()]
    text = "\n".join(events)
    cves = extract_cves(text)
    stages = infer_attack_stages(text)
    return IncidentJSON(
        incident_id=incident_id or f"incident-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        observed_at=datetime.now(timezone.utc).isoformat(),
        summary=(events[0][:240] if events else "No log events supplied."),
        iocs=extract_iocs(text),
        cves=cves,
        assets=extract_assets(text),
        services=infer_services(text),
        attack_stages=stages,
        severity=infer_severity(text, cves, stages),
        raw_events=events,
    )


def incident_to_dict(incident: IncidentJSON) -> Dict[str, Any]:
    return asdict(incident)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured incident JSON from logs or alerts.")
    parser.add_argument("--text", help="Single log/alert string to parse.")
    parser.add_argument("--input", type=Path, help="Text file containing log/alert lines.")
    parser.add_argument("--incident-id", help="Optional stable incident identifier.")
    args = parser.parse_args()

    if args.input:
        logs = args.input.read_text(encoding="utf-8").splitlines()
    elif args.text:
        logs = [args.text]
    else:
        parser.error("Provide --text or --input.")
    print(json.dumps(incident_to_dict(parse_logs(logs, args.incident_id)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
