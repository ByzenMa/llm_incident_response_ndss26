"""Run a saved model on a held-out dataset and persist prediction records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from enriched_training_dataset import DEFAULT_KG_RAG_TEST_FILE, load_examples_from_local_json
from response_post_processor import GenerationPostProcessor


DEFAULT_PREDICTIONS_FILE = Path("model_test_predictions.jsonl")


def _print_progress(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[model_test_generation] {message}", flush=True)


def build_prediction_records(
    instructions: Sequence[str],
    answers: Sequence[str],
    metadata: Sequence[Dict[str, Any]],
    generation_fn: Callable[[str], str],
    model_name_or_path: str,
    enable_post_processing: bool = True,
    processor: Optional[GenerationPostProcessor] = None,
    show_progress: bool = True,
    progress_interval: int = 1,
) -> List[Dict[str, Any]]:
    """Generate predictions and optionally apply the default safety gate."""
    if len(instructions) != len(answers):
        raise ValueError("Test instructions and answers must have equal lengths.")
    if metadata and len(metadata) != len(instructions):
        raise ValueError("Test metadata must align with instructions.")
    safety_processor = processor or GenerationPostProcessor()
    records: List[Dict[str, Any]] = []
    total = len(instructions)
    interval = max(1, progress_interval)
    post_processing_mode = "enabled" if enable_post_processing else "disabled"
    _print_progress(
        f"Starting test generation for {total} examples with model={model_name_or_path}; "
        f"post-processing={post_processing_mode}.",
        show_progress,
    )
    for index, (instruction, expected_answer) in enumerate(zip(instructions, answers)):
        item_metadata = metadata[index] if metadata else {}
        source_id = str(item_metadata.get("source_index", index))
        completed = index + 1
        should_report = completed == 1 or completed == total or completed % interval == 0
        if should_report:
            _print_progress(f"Generating prediction {completed}/{total}; source_index={source_id}.", show_progress)
        kg_context = item_metadata.get("kg_rag")
        generation = generation_fn(instruction)
        post_processing = None
        if enable_post_processing:
            post_processing = asdict(safety_processor.process(generation, kg_context=kg_context))
        records.append(
            {
                "id": source_id,
                "model_name_or_path": model_name_or_path,
                "instruction": instruction,
                "expected_answer": expected_answer,
                "generation": generation,
                "kg_context": kg_context,
                "post_processing_enabled": enable_post_processing,
                "post_processing": post_processing,
            }
        )
        if should_report:
            status = "not-run"
            if post_processing is not None:
                status = "accepted" if post_processing["accepted"] else "blocked"
            _print_progress(
                f"Completed prediction {completed}/{total}; source_index={source_id}; "
                f"post-processing={status}.",
                show_progress,
            )
    _print_progress(f"Completed test generation for {total} examples.", show_progress)
    return records


def save_prediction_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save one complete prediction record per JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _model_generation_function(model: Any, tokenizer: Any, device: str, max_new_tokens: int, temperature: float, do_sample: bool):
    def generate(instruction: str) -> str:
        inputs = tokenizer(instruction, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

    return generate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save predictions for a held-out incident-response test set.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model ID or local fine-tuned model directory.")
    parser.add_argument("--test-data-file", type=Path, default=Path(DEFAULT_KG_RAG_TEST_FILE))
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument(
        "--no-post-processing",
        dest="enable_post_processing",
        action="store_false",
        default=True,
        help="Save raw predictions without running the safety gate (enabled by default).",
    )
    parser.add_argument("--allowed-cve", action="append", default=[], help="Trusted CVE; may be repeated.")
    parser.add_argument("--allow-external-cves", action="store_true", help="Allow CVEs absent from the test KG context.")
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--progress-interval", type=int, default=1, help="Print progress every N test examples.")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    _print_progress(f"Loading model and tokenizer from {args.model_name_or_path} on {device}.", args.show_progress)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=dtype, device_map="auto")
    model.eval()
    _print_progress(f"Loading held-out test data from {args.test_data_file}.", args.show_progress)
    instructions, answers, metadata = load_examples_from_local_json(args.test_data_file)
    processor = GenerationPostProcessor(
        allowed_cves=args.allowed_cve,
        require_context_cve_match=not args.allow_external_cves,
    )
    records = build_prediction_records(
        instructions,
        answers,
        metadata,
        _model_generation_function(model, tokenizer, device, args.max_new_tokens, args.temperature, args.do_sample),
        model_name_or_path=args.model_name_or_path,
        enable_post_processing=args.enable_post_processing,
        processor=processor,
        show_progress=args.show_progress,
        progress_interval=args.progress_interval,
    )
    _print_progress(f"Saving {len(records)} predictions to {args.output}.", args.show_progress)
    save_prediction_records(args.output, records)
    print(f"Saved {len(records)} test predictions to {args.output}.")


if __name__ == "__main__":
    main()
