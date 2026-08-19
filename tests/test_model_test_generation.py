import json

from model_test_generation import build_prediction_records, parse_args, save_prediction_records


class _FakeProcessor:
    def process(self, generation, kg_context=None):
        from response_post_processor import PostProcessResult

        return PostProcessResult(True, [], [], [], f"checked {generation} with {bool(kg_context)}")


def test_build_predictions_enables_post_processing_by_default():
    records = build_prediction_records(
        ["prompt"],
        ["expected"],
        [{"source_index": 4, "kg_rag": {"incident": {"cves": []}}}],
        generation_fn=lambda instruction: f"prediction for {instruction}",
        model_name_or_path="local-model",
        processor=_FakeProcessor(),
    )

    assert records[0]["id"] == "4"
    assert records[0]["generation"] == "prediction for prompt"
    assert records[0]["expected_answer"] == "expected"
    assert records[0]["post_processing_enabled"] is True
    assert records[0]["post_processing"]["accepted"] is True


def test_build_predictions_can_disable_post_processing():
    records = build_prediction_records(
        ["prompt"],
        ["expected"],
        [],
        generation_fn=lambda _: "prediction",
        model_name_or_path="base-model",
        enable_post_processing=False,
    )

    assert records[0]["post_processing_enabled"] is False
    assert records[0]["post_processing"] is None


def test_prediction_records_are_saved_as_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    records = [{"id": "0", "generation": "one"}, {"id": "1", "generation": "two"}]

    save_prediction_records(path, records)

    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == records


def test_cli_defaults_to_post_processing(monkeypatch):
    monkeypatch.setattr("sys.argv", ["model_test_generation.py", "--model-name-or-path", "model"])
    assert parse_args().enable_post_processing is True

    monkeypatch.setattr(
        "sys.argv", ["model_test_generation.py", "--model-name-or-path", "model", "--no-post-processing"]
    )
    assert parse_args().enable_post_processing is False
