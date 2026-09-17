import json

import pytest

from batch_generation_rag import process_prediction_records, save_batch_rag_output
from generation_rag import KG_RAG, TEXT_RAG, PostGenerationRAG, TextDocument, TextRAGRetriever


def test_batch_text_rag_preserves_predictions_and_adds_rag_outputs():
    records = [
        {"id": "r0", "instruction": "SSH brute force host=web-01", "generation": "Contain the source."},
        {"id": "r1", "instruction": "DNS anomaly host=dns-01", "generation": "Review DNS logs."},
    ]
    original = json.loads(json.dumps(records))
    augmenter = PostGenerationRAG(
        TEXT_RAG,
        text_retriever=TextRAGRetriever(
            [
                TextDocument("ssh", "Investigate SSH logs and isolate the host."),
                TextDocument("dns", "Review DNS queries and resolver logs."),
            ],
            top_k=1,
        ),
    )

    output = process_prediction_records(records, augmenter, show_progress=False)

    assert records == original
    assert [record["generation"] for record in output] == [record["generation"] for record in original]
    assert all(record["rag_mode"] == TEXT_RAG for record in output)
    assert output[0]["rag_output"]["retrieved_items"][0]["document_id"] == "ssh"
    assert output[1]["rag_output"]["retrieved_items"][0]["document_id"] == "dns"
    assert "revision_prompt" in output[0]["rag_output"]


def test_batch_kg_rag_extracts_incident_context():
    records = [
        {
            "id": "kg-1",
            "instruction": "CVE-2023-34362 exploit host=moveit-01 http service",
            "generation": "Investigate initial access and collect web logs.",
        }
    ]

    output = process_prediction_records(
        records,
        PostGenerationRAG(KG_RAG, kg_depth=2),
        show_progress=False,
    )

    rag_output = output[0]["rag_output"]
    assert rag_output["metadata"]["incident"]["cves"] == ["CVE-2023-34362"]
    assert rag_output["retrieved_items"]
    assert "<kg_rag_context>" in rag_output["revision_prompt"]


def test_batch_rag_supports_custom_fields_and_progress(capsys):
    records = [{"id": "custom", "prompt": "Inspect host=web-01", "prediction": "Collect logs."}]
    augmenter = PostGenerationRAG(
        TEXT_RAG,
        text_retriever=TextRAGRetriever([TextDocument("logs", "Collect host logs.")]),
    )

    process_prediction_records(
        records,
        augmenter,
        instruction_field="prompt",
        generation_field="prediction",
    )

    output = capsys.readouterr().out
    assert "Processing record 1/1; id=custom" in output
    assert "retrieved_items=1" in output


def test_batch_rag_rejects_missing_required_fields():
    augmenter = PostGenerationRAG(
        TEXT_RAG,
        text_retriever=TextRAGRetriever([TextDocument("one", "reference")]),
    )

    with pytest.raises(ValueError, match="missing generation field"):
        process_prediction_records([{"instruction": "prompt"}], augmenter, show_progress=False)


def test_save_batch_rag_output_writes_jsonl(tmp_path):
    output = tmp_path / "rag" / "batch.jsonl"
    records = [{"id": "one", "rag_output": {"mode": TEXT_RAG}}]

    save_batch_rag_output(output, records)

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records
