"""Post-process generated incident-response actions with safety validators.

The generation-time post-processing stage is designed to run after a fine-tuned
model produces a response plan, especially when the model was fine-tuned with
KG-RAG-enriched examples. It normalizes actions into the action schema and
checks CVE plausibility, command syntax, policy constraints, and attack-path
consistency against the retrieved security context.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from action_schema_parser import ActionSchemaParser, ParsedAction

_CVE_PATTERN = re.compile(r"\bCVE-(\d{4})-(\d{4,7})\b", re.IGNORECASE)
_MITRE_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
_VALID_SHELL_STARTERS = {
    "auditctl",
    "aws",
    "az",
    "chmod",
    "chown",
    "cp",
    "dd",
    "docker",
    "find",
    "gcloud",
    "grep",
    "iptables",
    "journalctl",
    "kubectl",
    "kill",
    "mv",
    "netsh",
    "passwd",
    "pkill",
    "powershell",
    "rm",
    "rsync",
    "scp",
    "service",
    "ssh",
    "sudo",
    "systemctl",
    "tcpdump",
    "tshark",
    "ufw",
    "usermod",
    "volatility",
    "wevtutil",
    "yara",
}
_HIGH_RISK_PATTERNS = (
    (re.compile(r"\brm\s+-rf\s+/(?:\s|$)", re.IGNORECASE), "recursive deletion of filesystem root"),
    (re.compile(r"\bdd\b[^\n;]*\bof=/dev/(?:sd|nvme|xvd|vd)", re.IGNORECASE), "raw disk overwrite"),
    (re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b", re.IGNORECASE), "filesystem formatting"),
    (re.compile(r"\bshutdown\b|\breboot\b", re.IGNORECASE), "host shutdown or reboot"),
    (re.compile(r"\biptables\s+-F\b|\bufw\s+disable\b", re.IGNORECASE), "firewall policy flush or disable"),
    (re.compile(r"\bkill\s+-9\s+1\b", re.IGNORECASE), "terminating init/system process"),
)
_DESTRUCTIVE_ACTION_TYPES = {"eradication", "recovery"}


@dataclass
class ValidationFinding:
    category: str
    severity: str
    message: str
    evidence: str = ""


@dataclass
class PostProcessedAction:
    action: Dict[str, Any]
    accepted: bool
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PostProcessResult:
    accepted: bool
    actions: List[Dict[str, Any]]
    blocked_actions: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    summary: str


def _context_cves(kg_context: Optional[Dict[str, Any]]) -> Set[str]:
    if not kg_context:
        return set()
    incident = kg_context.get("incident", {})
    cves = {str(cve).upper() for cve in incident.get("cves", [])}
    for node in kg_context.get("nodes", []):
        if node.get("type") == "cve":
            cves.add(str(node.get("id") or node.get("name")).upper())
    return cves


def _context_attack_ids(kg_context: Optional[Dict[str, Any]]) -> Set[str]:
    if not kg_context:
        return set()
    attack_ids: Set[str] = set()
    for node in kg_context.get("nodes", []):
        if node.get("type") == "mitre_attack":
            technique = node.get("properties", {}).get("technique_id") or node.get("id", "").replace("attack:", "")
            if technique:
                attack_ids.add(str(technique).upper())
    return attack_ids


def _context_stages(kg_context: Optional[Dict[str, Any]]) -> Set[str]:
    if not kg_context:
        return set()
    stages = {str(stage).lower() for stage in kg_context.get("incident", {}).get("attack_stages", [])}
    for node in kg_context.get("nodes", []):
        if node.get("type") == "attack_stage":
            stages.add(str(node.get("name") or node.get("id", "")).lower())
    return stages


class GenerationPostProcessor:
    """Validate and gate generated response actions before execution or review."""

    def __init__(self, allowed_cves: Optional[Iterable[str]] = None, require_context_cve_match: bool = True) -> None:
        self.parser = ActionSchemaParser()
        self.allowed_cves = {cve.upper() for cve in allowed_cves or []}
        self.require_context_cve_match = require_context_cve_match

    def process(self, generation: Any, kg_context: Optional[Dict[str, Any]] = None) -> PostProcessResult:
        parsed_actions = self._parse_actions(generation)
        accepted: List[PostProcessedAction] = []
        blocked: List[PostProcessedAction] = []
        all_findings: List[ValidationFinding] = []
        for action in parsed_actions:
            findings = self.validate_action(action, kg_context=kg_context)
            all_findings.extend(findings)
            action_blocked = any(f.severity == "error" for f in findings)
            wrapped = PostProcessedAction(
                action=asdict(action),
                accepted=not action_blocked,
                findings=[asdict(f) for f in findings],
            )
            if action_blocked:
                blocked.append(wrapped)
            else:
                accepted.append(wrapped)
        result_ok = not blocked and not any(f.severity == "error" for f in all_findings)
        return PostProcessResult(
            accepted=result_ok,
            actions=[asdict(item) for item in accepted],
            blocked_actions=[asdict(item) for item in blocked],
            findings=[asdict(finding) for finding in all_findings],
            summary=self._summarize(result_ok, len(parsed_actions), len(blocked), all_findings),
        )

    def _parse_actions(self, generation: Any) -> List[ParsedAction]:
        if isinstance(generation, list):
            items = generation
        else:
            text = str(generation)
            loaded = None
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, list):
                items = loaded
            elif isinstance(loaded, dict) and isinstance(loaded.get("actions"), list):
                items = loaded["actions"]
            else:
                items = [generation]
        return self.parser.parse_many(items)

    def validate_action(self, action: ParsedAction, kg_context: Optional[Dict[str, Any]] = None) -> List[ValidationFinding]:
        text = "\n".join([action.source_action, action.source_explanation, " ".join(action.command)])
        findings: List[ValidationFinding] = []
        findings.extend(self._validate_cves(text, kg_context))
        findings.extend(self._validate_commands(action.command))
        findings.extend(self._validate_policy(action))
        findings.extend(self._validate_attack_path(text, kg_context))
        return findings

    def _validate_cves(self, text: str, kg_context: Optional[Dict[str, Any]]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        referenced = {f"CVE-{year}-{number}".upper() for year, number in _CVE_PATTERN.findall(text)}
        malformed = {token for token in re.findall(r"\bCVE-\d{4}-\d{1,3}\b|\bCVE-\d{2,3}-\d{4,7}\b", text, flags=re.IGNORECASE)}
        for token in sorted(malformed):
            findings.append(ValidationFinding("cve_authenticity", "error", "Malformed CVE identifier.", token.upper()))
        current_year = datetime.now(timezone.utc).year
        trusted_cves = self.allowed_cves | _context_cves(kg_context)
        for cve in sorted(referenced):
            year = int(cve.split("-")[1])
            if year < 1999 or year > current_year + 1:
                findings.append(ValidationFinding("cve_authenticity", "error", "CVE year is outside the plausible public CVE range.", cve))
            elif trusted_cves and cve not in trusted_cves and self.require_context_cve_match:
                findings.append(ValidationFinding("cve_authenticity", "warning", "CVE was not present in KG-RAG context or allow-list; verify before using it as evidence.", cve))
        return findings

    def _validate_commands(self, commands: Sequence[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for command in commands:
            try:
                tokens = shlex.split(command, posix=True)
            except ValueError as exc:
                findings.append(ValidationFinding("command_syntax", "error", f"Command cannot be parsed safely: {exc}.", command))
                continue
            if not tokens:
                findings.append(ValidationFinding("command_syntax", "error", "Command is empty after parsing.", command))
                continue
            executable = tokens[1] if tokens[0] == "sudo" and len(tokens) > 1 else tokens[0]
            if executable not in _VALID_SHELL_STARTERS and not executable.lower().startswith(("get-", "set-", "new-", "remove-")):
                findings.append(ValidationFinding("command_syntax", "warning", "Command executable is not in the known incident-response command allow-list.", command))
        return findings

    def _validate_policy(self, action: ParsedAction) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        command_text = "\n".join(action.command)
        for pattern, reason in _HIGH_RISK_PATTERNS:
            match = pattern.search(command_text)
            if match:
                findings.append(ValidationFinding("policy_constraint", "error", f"Blocked high-risk operation: {reason}.", match.group(0)))
        if action.action_type in _DESTRUCTIVE_ACTION_TYPES and not action.rollback:
            findings.append(ValidationFinding("policy_constraint", "error", "Destructive or recovery actions must include an explicit rollback/backout plan."))
        if action.action_type in _DESTRUCTIVE_ACTION_TYPES and not action.evidence:
            findings.append(ValidationFinding("policy_constraint", "warning", "Destructive or recovery actions should cite evidence before execution."))
        return findings

    def _validate_attack_path(self, text: str, kg_context: Optional[Dict[str, Any]]) -> List[ValidationFinding]:
        if not kg_context:
            return []
        context_attack_ids = _context_attack_ids(kg_context)
        context_stages = _context_stages(kg_context)
        findings: List[ValidationFinding] = []
        referenced_attack_ids = {match.upper() for match in _MITRE_PATTERN.findall(text)}
        for technique in sorted(referenced_attack_ids - context_attack_ids):
            findings.append(ValidationFinding("attack_path", "warning", "Referenced ATT&CK technique is not linked to the incident KG path.", technique))
        lowered = text.lower()
        stage_mentions = {stage for stage in context_stages if stage and stage.replace("_", " ") in lowered}
        if context_stages and not referenced_attack_ids and not stage_mentions:
            findings.append(ValidationFinding("attack_path", "warning", "Generated action does not reference the KG-RAG attack stage or ATT&CK path."))
        return findings

    @staticmethod
    def _summarize(ok: bool, total: int, blocked: int, findings: Sequence[ValidationFinding]) -> str:
        errors = sum(1 for finding in findings if finding.severity == "error")
        warnings = sum(1 for finding in findings if finding.severity == "warning")
        status = "accepted" if ok else "blocked"
        return f"Post-processing {status}: {total} action(s), {blocked} blocked, {errors} error(s), {warnings} warning(s)."


def load_context(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated incident-response actions after KG-RAG fine-tuning.")
    parser.add_argument("--generation", help="Generated model response text or JSON actions.")
    parser.add_argument("--input", type=Path, help="File containing generated response text or JSON actions.")
    parser.add_argument("--kg-context", type=Path, help="SecurityContext JSON produced by kg_rag.py or enriched_training_dataset.py metadata.")
    parser.add_argument("--allowed-cve", action="append", default=[], help="Trusted CVE identifier; can be repeated.")
    parser.add_argument("--allow-external-cves", action="store_true", help="Do not warn when a syntactically valid CVE is absent from KG-RAG context.")
    args = parser.parse_args()

    if args.input:
        generation = args.input.read_text(encoding="utf-8")
    elif args.generation:
        generation = args.generation
    else:
        parser.error("Provide --generation or --input.")
    processor = GenerationPostProcessor(
        allowed_cves=args.allowed_cve,
        require_context_cve_match=not args.allow_external_cves,
    )
    result = processor.process(generation, kg_context=load_context(args.kg_context))
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
