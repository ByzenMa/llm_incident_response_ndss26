import json

from fine_tune_llm import DEFAULT_MODEL_OUTPUT_DIR, parse_args, save_fine_tuned_artifacts


class _FakeModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


def test_save_fine_tuned_artifacts_saves_model_tokenizer_and_metadata(tmp_path):
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    metadata = {"dataset_mode": "kg_rag", "num_training_examples": 2}

    save_fine_tuned_artifacts(model, tokenizer, tmp_path, save_tokenizer=True, metadata=metadata)

    assert model.saved_to == str(tmp_path)
    assert tokenizer.saved_to == str(tmp_path)
    assert json.loads((tmp_path / "training_metadata.json").read_text(encoding="utf-8")) == metadata


def test_save_fine_tuned_artifacts_can_skip_tokenizer(tmp_path):
    model = _FakeModel()
    tokenizer = _FakeTokenizer()

    save_fine_tuned_artifacts(model, tokenizer, tmp_path, save_tokenizer=False, metadata={})

    assert model.saved_to == str(tmp_path)
    assert tokenizer.saved_to is None


def test_model_output_arguments_default_to_local_saving(monkeypatch, tmp_path):
    output_dir = tmp_path / "trained-model"
    monkeypatch.setattr("sys.argv", ["fine_tune_llm.py", "--model-output-dir", str(output_dir)])

    args = parse_args()

    assert DEFAULT_MODEL_OUTPUT_DIR.name == "deepseek-r1-distill-qwen-14b-lora"
    assert args.model_output_dir == output_dir
    assert args.save_local_model is True
    assert args.save_tokenizer is True
    assert args.test_ratio == 0.2
    assert args.split_seed == 99125
    assert args.processed_data_file.endswith("_train.json")
    assert args.test_data_file.endswith("_test.json")
