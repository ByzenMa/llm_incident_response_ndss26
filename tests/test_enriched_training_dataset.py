import json
from pathlib import Path

import pytest

from enriched_training_dataset import (
    KG_RAG_MODE,
    build_enriched_examples,
    load_preprocessed_kg_rag_examples,
    load_training_examples,
    preprocess_kg_rag_dataset,
    split_and_save_training_examples,
    split_training_examples,
    save_paired_original_and_kg_rag_splits,
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


def test_preprocess_prints_progress(tmp_path, capsys):
    source = tmp_path / "examples_16_june.json"
    output = tmp_path / "examples_16_june_kg_rag.json"
    _write_source(source)

    preprocess_kg_rag_dataset(data_file=str(source), output_path=output, progress_interval=1)

    captured = capsys.readouterr()
    assert "Loading source examples" in captured.out
    assert "Enriched 1/1 examples" in captured.out
    assert "Preprocessing finished successfully" in captured.out


def test_split_training_examples_is_reproducible_and_aligned():
    instructions = [f"instruction-{index}" for index in range(10)]
    answers = [f"answer-{index}" for index in range(10)]
    metadata = [{"source_index": index} for index in range(10)]

    first_train, first_test = split_training_examples(instructions, answers, metadata, test_ratio=0.3, seed=7)
    second_train, second_test = split_training_examples(instructions, answers, metadata, test_ratio=0.3, seed=7)

    assert (first_train, first_test) == (second_train, second_test)
    assert len(first_train[0]) == 7
    assert len(first_test[0]) == 3
    for subset in (first_train, first_test):
        for instruction, answer, item_metadata in zip(*subset):
            index = item_metadata["source_index"]
            assert instruction == f"instruction-{index}"
            assert answer == f"answer-{index}"


def test_split_and_save_writes_separate_local_files(tmp_path):
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"

    summary = split_and_save_training_examples(
        ["i0", "i1", "i2", "i3"],
        ["a0", "a1", "a2", "a3"],
        [{"source_index": index} for index in range(4)],
        train_path,
        test_path,
        test_ratio=0.25,
        seed=3,
    )

    train = load_preprocessed_kg_rag_examples(str(train_path))
    test = load_preprocessed_kg_rag_examples(str(test_path))
    assert summary["train_examples"] == len(train[0]) == 3
    assert summary["test_examples"] == len(test[0]) == 1


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_split_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="test_ratio"):
        split_training_examples(["i0", "i1"], ["a0", "a1"], [], test_ratio=ratio)


def test_original_and_kg_rag_splits_have_matching_source_indices(tmp_path):
    paths = [tmp_path / name for name in ("original_train.json", "original_test.json", "kg_train.json", "kg_test.json")]
    summary = save_paired_original_and_kg_rag_splits(
        [f"original-{index}" for index in range(8)],
        [f"enriched-{index}" for index in range(8)],
        [f"answer-{index}" for index in range(8)],
        [{"source_index": index, "kg_rag": {"index": index}} for index in range(8)],
        *paths,
        test_ratio=0.25,
        seed=11,
    )

    original_train = load_preprocessed_kg_rag_examples(str(paths[0]))
    original_test = load_preprocessed_kg_rag_examples(str(paths[1]))
    kg_train = load_preprocessed_kg_rag_examples(str(paths[2]))
    kg_test = load_preprocessed_kg_rag_examples(str(paths[3]))
    assert [item["source_index"] for item in original_train[2]] == [item["source_index"] for item in kg_train[2]]
    assert [item["source_index"] for item in original_test[2]] == [item["source_index"] for item in kg_test[2]]
    assert original_train[1] == kg_train[1]
    assert original_test[1] == kg_test[1]
    assert set(summary["train_source_indices"]).isdisjoint(summary["test_source_indices"])
    assert set(summary["train_source_indices"] + summary["test_source_indices"]) == set(range(8))
