import json

from generation_rag import (
    KG_RAG,
    TEXT_RAG,
    PostGenerationRAG,
    TextDocument,
    TextRAGRetriever,
    load_text_corpus,
    save_rag_output,
)


def test_text_rag_retrieves_relevant_documents_and_builds_revision_prompt():
    retriever = TextRAGRetriever(
        [
            TextDocument("ssh", "Investigate SSH authentication logs and isolate the affected host."),
            TextDocument("dns", "Review DNS cache entries."),
        ],
        top_k=1,
    )

    result = PostGenerationRAG(TEXT_RAG, text_retriever=retriever).prepare_revision(
        "SSH brute force host=web-01", "Block the source address."
    )

    assert result.mode == TEXT_RAG
    assert result.retrieved_items[0]["document_id"] == "ssh"
    assert "<draft_generation>" in result.revision_prompt
    assert "<text_rag_context>" in result.revision_prompt


def test_kg_rag_builds_structured_context_from_draft():
    result = PostGenerationRAG(KG_RAG, kg_depth=2).prepare_revision(
        "CVE-2023-34362 exploit host=moveit-01 http service",
        "Investigate initial access and collect web logs.",
    )

    assert result.mode == KG_RAG
    assert result.metadata["incident"]["cves"] == ["CVE-2023-34362"]
    assert result.retrieved_items
    assert "<kg_rag_context>" in result.revision_prompt


def test_load_text_corpus_supports_training_dataset_shape(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps({"instructions": [["SSH alert", "DNS alert"]], "answers": [["Contain SSH", "Inspect DNS"]]}),
        encoding="utf-8",
    )

    documents = load_text_corpus(path)

    assert [document.document_id for document in documents] == ["0", "1"]
    assert documents[0].text == "SSH alert\nContain SSH"


def test_save_rag_output_persists_complete_result(tmp_path):
    output = tmp_path / "rag" / "result.json"
    augmentation = PostGenerationRAG(
        TEXT_RAG,
        text_retriever=TextRAGRetriever([TextDocument("reference-1", "Collect firewall evidence.")]),
    ).prepare_revision("Investigate host=web-01", "Contain the host.")

    save_rag_output(output, augmentation)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["mode"] == TEXT_RAG
    assert saved["query"] == augmentation.query
    assert saved["context_text"] == augmentation.context_text
    assert saved["revision_prompt"] == augmentation.revision_prompt
    assert saved["retrieved_items"][0]["document_id"] == "reference-1"
