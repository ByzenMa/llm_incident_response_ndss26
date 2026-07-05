from kg_rag import build_context_from_logs


def test_build_context_from_logs_links_attack_and_mitigations():
    context = build_context_from_logs(
        ["CVE-2023-34362 exploit against host=moveit-01 http service followed by data exfil upload"],
        incident_id="inc-kg",
    )

    node_types = {node["type"] for node in context.nodes}
    edge_relations = {edge["relation"] for edge in context.edges}

    assert context.incident["incident_id"] == "inc-kg"
    assert "cve" in node_types
    assert "mitre_attack" in node_types
    assert "mitigation" in node_types
    assert "security_rule" in node_types
    assert "mentions_cve" in edge_relations
    assert "maps_to_attack" in edge_relations
    assert any("Patch exposed services" == item for item in context.recommendations)
    assert "Structured security context" in context.prompt_context
