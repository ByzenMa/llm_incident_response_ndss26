"""Batch-build text-RAG or KG-RAG outputs for saved model predictions."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from generation_rag import KG_RAG, TEXT_RAG, PostGenerationRAG, TextRAGRetriever, load_text_corpus
from response_evaluation import load_evaluation_records


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _print_progress(message: str, enabled: bool) -> None:
    if enabled:
        print(f"[batch_generation_rag] {message}", flush=True)


def process_prediction_records(
    records: Sequence[Dict[str, Any]],
    augmenter: PostGenerationRAG,
    instruction_field: str = "instruction",
    generation_field: str = "generation",
    show_progress: bool = True,
    progress_interval: int = 1,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Add a complete RAG output to copied records, optionally limiting the prefix."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    selected_records = records[:limit] if limit is not None else records
    output_records = copy.deepcopy(list(selected_records))
    total = len(output_records)
    interval = max(1, progress_interval)
    _print_progress(
        f"Starting {augmenter.mode} processing for {total} prediction records.",
        show_progress,
    )
    for index, record in enumerate(output_records):
        if instruction_field not in record:
            raise ValueError(f"Record {index} is missing instruction field {instruction_field!r}.")
        if generation_field not in record:
            raise ValueError(f"Record {index} is missing generation field {generation_field!r}.")
        instruction = str(record[instruction_field])
        generation = record[generation_field]
        generation_text = generation if isinstance(generation, str) else json.dumps(generation, ensure_ascii=False)
        completed = index + 1
        should_report = completed == 1 or completed == total or completed % interval == 0
        record_id = str(record.get("id", record.get("source_index", index)))
        if should_report:
            _print_progress(
                f"Processing record {completed}/{total}; id={record_id}.",
                show_progress,
            )
        augmentation = augmenter.prepare_revision(instruction, generation_text)
        record["rag_mode"] = augmenter.mode
        record["rag_output"] = asdict(augmentation)
        if should_report:
            _print_progress(
                f"Completed record {completed}/{total}; id={record_id}; "
                f"retrieved_items={len(augmentation.retrieved_items)}.",
                show_progress,
            )
    _print_progress(
        f"Completed {augmenter.mode} processing for {total} prediction records.",
        show_progress,
    )
    return output_records


def save_batch_rag_output(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save batch RAG records as JSONL without changing the input file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process a model prediction file with text-RAG or KG-RAG.")
    parser.add_argument("--input", type=Path, required=True, help="Model prediction JSON/JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file containing per-record RAG outputs.")
    parser.add_argument("--mode", choices=(TEXT_RAG, KG_RAG), required=True)
    parser.add_argument("--text-corpus", type=Path, help="Text corpus required by text_rag mode.")
    parser.add_argument("--top-k", type=_positive_int, default=3)
    parser.add_argument("--kg-depth", type=_positive_int, default=2)
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--generation-field", default="generation")
    parser.add_argument("--progress-interval", type=_positive_int, default=1)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Only process and save the first N prediction records.",
    )
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    args = parser.parse_args()

    text_retriever = None
    if args.mode == TEXT_RAG:
        if not args.text_corpus:
            parser.error("--text-corpus is required for text_rag mode.")
        text_retriever = TextRAGRetriever(load_text_corpus(args.text_corpus), top_k=args.top_k)
    augmenter = PostGenerationRAG(args.mode, text_retriever=text_retriever, kg_depth=args.kg_depth)
    input_records = load_evaluation_records(args.input)
    output_records = process_prediction_records(
        input_records,
        augmenter,
        instruction_field=args.instruction_field,
        generation_field=args.generation_field,
        show_progress=args.show_progress,
        progress_interval=args.progress_interval,
        limit=args.limit,
    )
    save_batch_rag_output(args.output, output_records)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "input": str(args.input),
                "output": str(args.output),
                "record_count": len(output_records),
                "limit": args.limit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
