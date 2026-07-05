import json
from pathlib import Path

import pytest

from enriched_training_dataset import (
    KG_RAG_MODE,
    build_enriched_examples,
    load_preprocessed_kg_rag_examples,
    load_training_examples,
    preprocess_kg_rag_dataset,
)


def _write_source(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "instructions": [["critical ssh brute force host=web-01 src=203.0.113.10 CVE-2024-12345"]],
                "answers": [["Investigate and contain SSH brute force."]],
            }
        ),
        encoding="utf-8",
    )


def test_build_enriched_examples_contains_incident_and_kg_context():
    instructions, answers, metadata = build_enriched_examples(
        ["critical ssh brute force host=web-01 src=203.0.113.10 CVE-2024-12345"],
        ["Contain the incident."],
    )

    assert answers == ["Contain the incident."]
    assert "<incident_json>" in instructions[0]
    assert "<kg_rag_context>" in instructions[0]
    assert metadata[0]["validation"]["ok"] is True
    assert metadata[0]["incident"]["cves"] == ["CVE-2024-12345"]
    assert "mentions_cve" in {edge["relation"] for edge in metadata[0]["kg_rag"]["edges"]}


def test_preprocess_saves_local_file_then_training_loader_reads_it(tmp_path):
    source = tmp_path / "examples_16_june.json"
    output = tmp_path / "examples_16_june_kg_rag.json"
    _write_source(source)

    summary = preprocess_kg_rag_dataset(data_file=str(source), output_path=output)
    instructions, answers, metadata = load_training_examples(mode=KG_RAG_MODE, processed_data_file=str(output))

    assert summary["output_path"] == str(output)
    assert output.exists()
    assert len(instructions) == len(answers) == len(metadata) == 1
    assert "Structured security context" in instructions[0]
    assert answers == ["Investigate and contain SSH brute force."]
    assert metadata[0]["validation"]["ok"] is True


def test_kg_rag_training_mode_requires_preprocessed_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preprocessed_kg_rag_examples(str(tmp_path / "missing.json"))
