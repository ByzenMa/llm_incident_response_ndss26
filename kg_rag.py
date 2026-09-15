"""Small security knowledge-graph RAG reference implementation.

This module implements part 3 of the incident-response pipeline. It builds a
structured security context graph connecting CVEs, MITRE ATT&CK techniques,
assets, services, mitigations, and detection/security rules, then retrieves
incident-specific context for downstream LLM prompts or response planning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from incident_log_parser import IncidentJSON, incident_to_dict, parse_logs


@dataclass(frozen=True)
class KGNode:
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGEdge:
    source: str
    relation: str
    target: str


@dataclass
class SecurityContext:
    incident: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    recommendations: List[str]
    prompt_context: str


_DEFAULT_TECHNIQUES: Dict[str, Dict[str, Any]] = {
    "initial_access": {"id": "T1190", "name": "Exploit Public-Facing Application"},
    "execution": {"id": "T1059", "name": "Command and Scripting Interpreter"},
    "persistence": {"id": "T1053", "name": "Scheduled Task/Job"},
    "privilege_escalation": {"id": "T1068", "name": "Exploitation for Privilege Escalation"},
    "defense_evasion": {"id": "T1070", "name": "Indicator Removal"},
    "credential_access": {"id": "T1003", "name": "OS Credential Dumping"},
    "discovery": {"id": "T1087", "name": "Account Discovery"},
    "lateral_movement": {"id": "T1021", "name": "Remote Services"},
    "command_and_control": {"id": "T1071", "name": "Application Layer Protocol"},
    "exfiltration": {"id": "T1041", "name": "Exfiltration Over C2 Channel"},
    "impact": {"id": "T1486", "name": "Data Encrypted for Impact"},
    "reconnaissance": {"id": "T1595", "name": "Active Scanning"},
}

_DEFAULT_MITIGATIONS: Dict[str, Sequence[str]] = {
    "initial_access": ("Patch exposed services", "Restrict ingress to trusted networks"),
    "execution": ("Constrain script interpreters", "Enable command-line logging"),
    "credential_access": ("Reset exposed credentials", "Enable LSASS protection and MFA"),
    "lateral_movement": ("Disable unnecessary remote services", "Segment affected networks"),
    "command_and_control": ("Block known C2 indicators", "Add DNS/HTTP egress monitoring"),
    "exfiltration": ("Throttle and inspect outbound transfers", "Preserve proxy and NetFlow evidence"),
    "impact": ("Isolate impacted hosts", "Restore from known-good immutable backups"),
}

_DEFAULT_RULES: Dict[str, Sequence[str]] = {
    "ssh": ("Alert on repeated SSH failures from one source",),
    "rdp": ("Alert on anomalous RDP source geolocation",),
    "http": ("Detect exploit strings and webshell uploads in web logs",),
    "https": ("Inspect rare JA3/SNI combinations for egress beacons",),
    "dns": ("Detect high-entropy DNS queries and tunneling volume",),
    "smb": ("Alert on PsExec-like service creation over SMB",),
    "aws": ("Alert on unusual IAM policy changes and S3 bulk reads",),
}


class SecurityKnowledgeGraph:
    """In-memory KG with deterministic retrieval for incident context."""

    def __init__(self) -> None:
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Set[KGEdge] = set()

    def add_node(self, node_id: str, node_type: str, name: Optional[str] = None, **properties: Any) -> str:
        self.nodes[node_id] = KGNode(id=node_id, type=node_type, name=name or node_id, properties=properties)
        return node_id

    def add_edge(self, source: str, relation: str, target: str) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.add(KGEdge(source=source, relation=relation, target=target))

    def ingest_incident(self, incident: IncidentJSON) -> str:
        incident_id = self.add_node(incident.incident_id, "incident", incident.summary, severity=incident.severity)
        for cve in incident.cves:
            cve_id = self.add_node(cve, "cve", cve)
            self.add_edge(incident_id, "mentions_cve", cve_id)
        for asset in incident.assets:
            asset_id = self.add_node(f"asset:{asset}", "asset", asset)
            self.add_edge(incident_id, "affects_asset", asset_id)
        for service in incident.services:
            service_id = self.add_node(f"service:{service}", "service", service)
            self.add_edge(incident_id, "involves_service", service_id)
        for ioc in incident.iocs:
            ioc_id = self.add_node(f"ioc:{ioc.type}:{ioc.value}", "ioc", ioc.value, ioc_type=ioc.type)
            self.add_edge(incident_id, "has_ioc", ioc_id)
        for stage in incident.attack_stages:
            stage_id = self.add_node(f"stage:{stage}", "attack_stage", stage)
            self.add_edge(incident_id, "has_attack_stage", stage_id)
            technique = _DEFAULT_TECHNIQUES.get(stage)
            if technique:
                technique_id = self.add_node(f"attack:{technique['id']}", "mitre_attack", technique["name"], technique_id=technique["id"])
                self.add_edge(stage_id, "maps_to_attack", technique_id)
            for mitigation in _DEFAULT_MITIGATIONS.get(stage, ()): 
                mitigation_id = self.add_node(f"mitigation:{mitigation}", "mitigation", mitigation)
                self.add_edge(stage_id, "has_mitigation", mitigation_id)
        for service in incident.services:
            for rule in _DEFAULT_RULES.get(service, ()): 
                rule_id = self.add_node(f"rule:{rule}", "security_rule", rule)
                self.add_edge(f"service:{service}", "monitored_by", rule_id)
        return incident_id

    def neighbors(self, node_ids: Iterable[str], depth: int = 2) -> Tuple[List[KGNode], List[KGEdge]]:
        frontier = set(node_ids)
        visited = set(frontier)
        selected_edges: Set[KGEdge] = set()
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for edge in self.edges:
                if edge.source in frontier or edge.target in frontier:
                    selected_edges.add(edge)
                    for node_id in (edge.source, edge.target):
                        if node_id not in visited:
                            visited.add(node_id)
                            next_frontier.add(node_id)
            frontier = next_frontier
            if not frontier:
                break
        return [self.nodes[node_id] for node_id in sorted(visited) if node_id in self.nodes], sorted(selected_edges, key=lambda e: (e.source, e.relation, e.target))

    def retrieve_context(self, incident: IncidentJSON, depth: int = 2) -> SecurityContext:
        incident_id = self.ingest_incident(incident)
        nodes, edges = self.neighbors([incident_id], depth=depth)
        recommendations = _recommendations_from_nodes(nodes)
        prompt_context = _format_prompt_context(incident, nodes, edges, recommendations)
        return SecurityContext(
            incident=incident_to_dict(incident),
            nodes=[asdict(node) for node in nodes],
            edges=[asdict(edge) for edge in edges],
            recommendations=recommendations,
            prompt_context=prompt_context,
        )


def _recommendations_from_nodes(nodes: Sequence[KGNode]) -> List[str]:
    recommendations = [node.name for node in nodes if node.type in {"mitigation", "security_rule"}]
    seen = set()
    result = []
    for item in recommendations:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _format_prompt_context(incident: IncidentJSON, nodes: Sequence[KGNode], edges: Sequence[KGEdge], recommendations: Sequence[str]) -> str:
    attack = ", ".join(node.name for node in nodes if node.type == "mitre_attack") or "unknown"
    services = ", ".join(incident.services) or "unknown"
    assets = ", ".join(incident.assets) or "unknown"
    cves = ", ".join(incident.cves) or "none"
    return (
        "Structured security context for incident response:\n"
        f"- Incident: {incident.incident_id} severity={incident.severity}\n"
        f"- Assets: {assets}\n"
        f"- Services: {services}\n"
        f"- CVEs: {cves}\n"
        f"- MITRE ATT&CK context: {attack}\n"
        f"- Recommended mitigations/rules: {'; '.join(recommendations) or 'none'}\n"
        f"- KG facts: {len(nodes)} nodes, {len(edges)} edges"
    )


def build_context_from_logs(logs: Iterable[str], incident_id: Optional[str] = None, depth: int = 2) -> SecurityContext:
    incident = parse_logs(logs, incident_id=incident_id)
    return SecurityKnowledgeGraph().retrieve_context(incident, depth=depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KG-RAG security context from incident logs.")
    parser.add_argument("--text", help="Single log/alert string to parse and enrich.")
    parser.add_argument("--input", type=Path, help="Text file containing log/alert lines.")
    parser.add_argument("--incident-json", type=Path, help="Existing incident JSON from incident_log_parser.py.")
    parser.add_argument("--incident-id", help="Optional stable incident identifier for raw logs.")
    parser.add_argument("--depth", type=int, default=2, help="Neighborhood depth for graph retrieval.")
    args = parser.parse_args()

    if args.incident_json:
        data = json.loads(args.incident_json.read_text(encoding="utf-8"))
        incident = IncidentJSON(
            incident_id=data["incident_id"],
            observed_at=data.get("observed_at", ""),
            summary=data.get("summary", ""),
            cves=data.get("cves", []),
            assets=data.get("assets", []),
            services=data.get("services", []),
            attack_stages=data.get("attack_stages", []),
            severity=data.get("severity", "unknown"),
            source=data.get("source", "log_parser"),
            raw_events=data.get("raw_events", []),
        )
        # Re-parse raw logs to recover IOC dataclasses if the JSON came from this parser.
        if data.get("iocs"):
            from incident_log_parser import IOC

            incident.iocs = [IOC(type=item["type"], value=item["value"]) for item in data["iocs"]]
        context = SecurityKnowledgeGraph().retrieve_context(incident, depth=args.depth)
    else:
        if args.input:
            logs = args.input.read_text(encoding="utf-8").splitlines()
        elif args.text:
            logs = [args.text]
        else:
            parser.error("Provide --text, --input, or --incident-json.")
        context = build_context_from_logs(logs, incident_id=args.incident_id, depth=args.depth)
    print(json.dumps(asdict(context), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
