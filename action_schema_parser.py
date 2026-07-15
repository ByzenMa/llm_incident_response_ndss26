"""Parse incident-response actions into a normalized action schema.

The parser is designed for outputs produced from the
``kimhammar/CSLE-IncidentResponse-V1`` dataset and compatible LLM generations.
It accepts free text, JSON-like answers with ``Action``/``Explanation`` fields,
or already-structured records, and emits a stable schema:

``action_type``, ``target``, ``command``, ``precondition``, ``risk``,
``rollback``, and ``evidence``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for CLI dataset loading
    load_dataset = None


ACTION_TYPES: Dict[str, Sequence[str]] = {
    "containment": ("isolate", "block", "disable", "quarantine", "contain", "disconnect", "deny", "sinkhole"),
    "evidence_acquisition": ("image", "capture", "collect", "preserve", "export", "snapshot", "forensic", "memory", "disk"),
    "eradication": ("remove", "delete", "eradicate", "clean", "patch", "rotate", "revoke", "terminate", "kill"),
    "recovery": ("restore", "rebuild", "recover", "reimage", "redeploy", "restart", "resynchronize"),
    "investigation": ("analyze", "inspect", "review", "query", "hunt", "scan", "triage", "correlate", "verify"),
    "notification": ("notify", "escalate", "report", "contact", "open ticket", "inform"),
    "monitoring": ("monitor", "watch", "alert", "log", "baseline", "detect"),
}

_TARGET_PATTERNS = (
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"\b(?:host|server|workstation|endpoint|vm|container|pod|node|account|user|service|database|bucket|subnet|vlan|firewall|router|switch|domain|url|process|file)[:= ]+[A-Za-z0-9._:/@-]+",
    r"\b[A-Za-z]:\\[^\s,;]+",
    r"/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+",
    r"\b[a-zA-Z0-9][a-zA-Z0-9._-]{1,}\.(?:exe|dll|sh|py|conf|log|service)\b",
)

_COMMAND_PATTERNS = (
    r"`([^`]+)`",
    r"\b(?:sudo\s+)?(?:iptables|ufw|netsh|systemctl|service|kubectl|docker|aws|az|gcloud|ssh|scp|rsync|dd|volatility|tcpdump|tshark|yara|grep|find|chmod|chown|rm|mv|cp|kill|pkill|passwd|usermod|auditctl|journalctl|wevtutil|powershell|Get-[A-Za-z]+|Set-[A-Za-z]+|New-[A-Za-z]+|Remove-[A-Za-z]+)\b[^.;\n]*",
)


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    for idx in range(start, len(text)):
        if text[idx] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_normalize_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_field(record: Dict[str, Any], names: Iterable[str]) -> str:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        if name.lower() in lowered:
            return _normalize_text(lowered[name.lower()])
    return ""


def _infer_action_type(text: str) -> str:
    lowered = text.lower()
    scores = {
        action_type: sum(1 for keyword in keywords if keyword in lowered)
        for action_type, keywords in ACTION_TYPES.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score else "other"


def _extract_targets(text: str) -> List[str]:
    targets: List[str] = []
    for pattern in _TARGET_PATTERNS:
        for match in re.findall(pattern, text):
            value = match if isinstance(match, str) else " ".join(match)
            value = value.strip().rstrip(".,;)")
            if value and value not in targets:
                targets.append(value)
    return targets


def _extract_commands(text: str) -> List[str]:
    commands: List[str] = []
    for pattern in _COMMAND_PATTERNS:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            command = (match if isinstance(match, str) else " ".join(match)).strip().rstrip(".,;")
            if command and command not in commands:
                commands.append(command)
    return commands


def _sentence_matching(text: str, keywords: Sequence[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            return sentence.strip()
    return ""


@dataclass
class ParsedAction:
    action_type: str
    target: List[str]
    command: List[str]
    precondition: str
    risk: str
    rollback: str
    evidence: List[str]
    source_action: str
    source_explanation: str


class ActionSchemaParser:
    """Rule-based parser for incident-response action-schema records."""

    def parse(self, item: Any) -> ParsedAction:
        record = item if isinstance(item, dict) else _first_json_object(str(item)) or {"text": item}
        source_action = _extract_field(record, ("action", "response", "plan", "command", "text"))
        source_explanation = _extract_field(record, ("explanation", "rationale", "reason"))
        text = "\n".join(part for part in (source_action, source_explanation) if part) or _normalize_text(item)

        explicit_type = _extract_field(record, ("action_type", "type", "category"))
        explicit_target = _extract_field(record, ("target", "targets", "asset", "assets", "host", "hosts"))
        explicit_command = _extract_field(record, ("command", "commands"))
        explicit_evidence = _extract_field(record, ("evidence", "artifacts", "logs"))

        targets = [t.strip() for t in re.split(r"[,;]\s*", explicit_target) if t.strip()] if explicit_target else _extract_targets(text)
        commands = [c.strip() for c in re.split(r"\s*;\s*", explicit_command) if c.strip()] if explicit_command else _extract_commands(text)
        evidence = [e.strip() for e in re.split(r"[,;]\s*", explicit_evidence) if e.strip()] if explicit_evidence else []
        if not evidence:
            evidence_text = _sentence_matching(text, ("evidence", "log", "image", "snapshot", "pcap", "netflow", "audit", "forensic"))
            evidence = [evidence_text] if evidence_text else []

        return ParsedAction(
            action_type=explicit_type or _infer_action_type(text),
            target=targets,
            command=commands,
            precondition=_extract_field(record, ("precondition", "preconditions"))
            or _sentence_matching(text, ("before", "after confirming", "if ", "provided", "only when", "ensure")),
            risk=_extract_field(record, ("risk", "risks"))
            or _sentence_matching(text, ("risk", "impact", "downtime", "disrupt", "data loss", "false positive")),
            rollback=_extract_field(record, ("rollback", "backout", "revert"))
            or _sentence_matching(text, ("rollback", "revert", "restore", "undo", "back out")),
            evidence=evidence,
            source_action=source_action,
            source_explanation=source_explanation,
        )

    def parse_many(self, items: Iterable[Any]) -> List[ParsedAction]:
        return [self.parse(item) for item in items]


def load_csle_answers(data_file: str = "examples_16_june.json", limit: Optional[int] = None) -> List[str]:
    """Load answer texts from kimhammar/CSLE-IncidentResponse-V1."""
    if load_dataset is None:
        raise RuntimeError("Install the optional 'datasets' package to load Hugging Face datasets.")
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files=data_file)
    answers = list(dataset["train"]["answers"][0])
    return answers[:limit] if limit is not None else answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize incident-response actions into action-schema JSON.")
    parser.add_argument("--text", help="Single action or model response to parse.")
    parser.add_argument("--input", type=Path, help="JSON/JSONL/TXT file containing actions or responses.")
    parser.add_argument("--from-dataset", action="store_true", help="Load answers from kimhammar/CSLE-IncidentResponse-V1.")
    parser.add_argument("--data-file", default="examples_16_june.json", help="Dataset file to load when --from-dataset is set.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to parse.")
    args = parser.parse_args()

    items: List[Any]
    if args.from_dataset:
        items = load_csle_answers(args.data_file, args.limit)
    elif args.input:
        raw = args.input.read_text(encoding="utf-8")
        try:
            loaded = json.loads(raw)
            items = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            items = [line for line in raw.splitlines() if line.strip()]
        if args.limit is not None:
            items = items[: args.limit]
    elif args.text:
        items = [args.text]
    else:
        parser.error("Provide --text, --input, or --from-dataset.")

    schema_parser = ActionSchemaParser()
    print(json.dumps([asdict(action) for action in schema_parser.parse_many(items)], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
