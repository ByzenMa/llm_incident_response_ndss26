"""Preprocess examples_16_june.json into KG-RAG-enriched fine-tuning data.

This module connects the stage-2 log parser and stage-3 KG-RAG builder.  It is
kept separate from model training so enriched examples are first saved to a
local JSON file and only then loaded by ``fine_tune_llm.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from incident_log_parser import IncidentJSON, parse_logs
from kg_rag import SecurityKnowledgeGraph

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for Hugging Face loading
    load_dataset = None


DATASET_NAME = "kimhammar/CSLE-IncidentResponse-V1"
DEFAULT_DATA_FILE = "examples_16_june.json"
DEFAULT_KG_RAG_DATA_FILE = "examples_16_june_kg_rag.json"
ORIGINAL_MODE = "original"
KG_RAG_MODE = "kg_rag"


def _print_progress(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[enriched_training_dataset] {message}", flush=True)


def _extract_pairs_from_mapping(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    instructions = data.get("instructions")
    answers = data.get("answers")
    if instructions is None or answers is None:
        raise ValueError("Training JSON must contain 'instructions' and 'answers' fields.")
    if instructions and isinstance(instructions[0], list):
        instructions = instructions[0]
    if answers and isinstance(answers[0], list):
        answers = answers[0]
    metadata = data.get("metadata", [])
    if metadata and isinstance(metadata[0], list):
        metadata = metadata[0]
    return list(instructions), list(answers), list(metadata or [])


def load_examples_from_local_json(path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load original or preprocessed examples from a local JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return _extract_pairs_from_mapping(data)
    if isinstance(data, list):
        instructions = [item["instruction"] for item in data]
        answers = [item["answer"] for item in data]
        metadata = [item.get("metadata", {}) for item in data]
        return instructions, answers, metadata
    raise ValueError(f"Unsupported training JSON shape in {path}.")


def load_original_examples(data_file: str = DEFAULT_DATA_FILE, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load instruction/answer pairs from local JSON or the Hugging Face dataset."""
    local_path = Path(data_file)
    if local_path.exists():
        instructions, answers, _ = load_examples_from_local_json(local_path)
    else:
        if load_dataset is None:
            raise RuntimeError("Install the optional 'datasets' package or provide a local examples_16_june.json file.")
        dataset = load_dataset(DATASET_NAME, data_files=data_file)
        instructions = list(dataset["train"]["instructions"][0])
        answers = list(dataset["train"]["answers"][0])
    return _limit_pairs(instructions, answers, limit)


def _limit_pairs(instructions: Sequence[str], answers: Sequence[str], limit: Optional[int]) -> Tuple[List[str], List[str]]:
    if len(instructions) != len(answers):
        raise ValueError(f"Instruction/answer length mismatch: {len(instructions)} != {len(answers)}")
    upper = len(instructions) if limit is None else min(limit, len(instructions))
    return list(instructions[:upper]), list(answers[:upper])


def build_enriched_instruction(original_instruction: str, incident: IncidentJSON, context: Any) -> str:
    """Attach structured incident JSON and KG-RAG context to an instruction."""
    return (
        f"{original_instruction.strip()}\n\n"
        "Use the following preprocessed structured incident JSON and KG-RAG security context when producing the response plan.\n"
        "<incident_json>\n"
        f"{json.dumps(asdict(incident), ensure_ascii=False, indent=2)}\n"
        "</incident_json>\n"
        "<kg_rag_context>\n"
        f"{context.prompt_context}\n"
        "</kg_rag_context>"
    )


def enrich_instruction(original_instruction: str, index: int, kg_depth: int = 2) -> Tuple[str, Dict[str, Any]]:
    """Run stage 2 and stage 3 for one original instruction."""
    incident = parse_logs([original_instruction], incident_id=f"csle-example-{index:06d}")
    context = SecurityKnowledgeGraph().retrieve_context(incident, depth=kg_depth)
    context_dict = asdict(context)
    metadata = {
        "source_index": index,
        "incident": asdict(incident),
        "kg_rag": context_dict,
        "validation": validate_enrichment(incident, context_dict),
    }
    return build_enriched_instruction(original_instruction, incident, context), metadata


def build_enriched_examples(
    instructions: Sequence[str],
    answers: Sequence[str],
    limit: Optional[int] = None,
    kg_depth: int = 2,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Convert original examples into directly trainable KG-RAG-enriched examples."""
    limited_instructions, limited_answers = _limit_pairs(instructions, answers, limit)
    enriched_instructions: List[str] = []
    metadata: List[Dict[str, Any]] = []
    total = len(limited_instructions)
    _print_progress(f"Starting KG-RAG enrichment for {total} examples.", show_progress)
    interval = max(1, progress_interval)
    for index, instruction in enumerate(limited_instructions):
        enriched, item_metadata = enrich_instruction(instruction, index=index, kg_depth=kg_depth)
        enriched_instructions.append(enriched)
        metadata.append(item_metadata)
        completed = index + 1
        if completed == total or completed == 1 or completed % interval == 0:
            validation_status = "ok" if item_metadata["validation"].get("ok") else "failed"
            _print_progress(f"Enriched {completed}/{total} examples; latest validation={validation_status}.", show_progress)
    _print_progress(f"Completed KG-RAG enrichment for {total} examples.", show_progress)
    return enriched_instructions, limited_answers, metadata


def validate_enrichment(incident: IncidentJSON, context: Dict[str, Any]) -> Dict[str, Any]:
    """Check that stage-2 incident fields and stage-3 KG links are well formed."""
    required_incident_fields = {"incident_id", "observed_at", "summary", "iocs", "cves", "assets", "services", "attack_stages", "severity", "raw_events"}
    incident_dict = asdict(incident)
    missing_incident_fields = sorted(required_incident_fields - set(incident_dict))
    context_nodes = context.get("nodes", [])
    context_edges = context.get("edges", [])
    node_types = {node.get("type") for node in context_nodes}
    edge_relations = {edge.get("relation") for edge in context_edges}
    expected_edges = set()
    if incident.cves:
        expected_edges.add("mentions_cve")
    if incident.assets:
        expected_edges.add("affects_asset")
    if incident.services:
        expected_edges.add("involves_service")
    if incident.iocs:
        expected_edges.add("has_ioc")
    if incident.attack_stages:
        expected_edges.add("has_attack_stage")
    errors = []
    if missing_incident_fields:
        errors.append(f"missing incident fields: {missing_incident_fields}")
    missing_edges = sorted(expected_edges - edge_relations)
    if missing_edges:
        errors.append(f"missing KG relations: {missing_edges}")
    if "incident" not in node_types:
        errors.append("KG context is missing incident node")
    if not context.get("prompt_context"):
        errors.append("KG context is missing prompt_context")
    return {
        "ok": not errors,
        "errors": errors,
        "incident_counts": {
            "iocs": len(incident.iocs),
            "cves": len(incident.cves),
            "assets": len(incident.assets),
            "services": len(incident.services),
            "attack_stages": len(incident.attack_stages),
        },
        "kg_counts": {"nodes": len(context_nodes), "edges": len(context_edges)},
    }


def save_training_examples(output_path: Path, instructions: Sequence[str], answers: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> None:
    """Persist examples in the same top-level shape used by the HF dataset."""
    output = {"instructions": [list(instructions)], "answers": [list(answers)], "metadata": [list(metadata)]}
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def preprocess_kg_rag_dataset(
    data_file: str = DEFAULT_DATA_FILE,
    output_path: Path = Path(DEFAULT_KG_RAG_DATA_FILE),
    limit: Optional[int] = None,
    kg_depth: int = 2,
    show_progress: bool = True,
    progress_interval: int = 25,
) -> Dict[str, Any]:
    """Preprocess examples_16_june.json into a local KG-RAG training file."""
    _print_progress(f"Loading source examples from {data_file}.", show_progress)
    instructions, answers = load_original_examples(data_file=data_file, limit=limit)
    _print_progress(f"Loaded {len(instructions)} instructions and {len(answers)} answers.", show_progress)
    enriched_instructions, enriched_answers, metadata = build_enriched_examples(
        instructions,
        answers,
        kg_depth=kg_depth,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    _print_progress("Validating enriched incident JSON and KG-RAG context.", show_progress)
    validation_errors = [item["validation"] for item in metadata if not item["validation"].get("ok")]
    if validation_errors:
        raise ValueError(f"KG-RAG enrichment validation failed: {validation_errors}")
    _print_progress(f"Saving preprocessed KG-RAG training data to {output_path}.", show_progress)
    save_training_examples(output_path, enriched_instructions, enriched_answers, metadata)
    _print_progress("Preprocessing finished successfully.", show_progress)
    return {
        "source_data_file": data_file,
        "output_path": str(output_path),
        "num_instructions": len(enriched_instructions),
        "num_answers": len(enriched_answers),
        "validation_errors": validation_errors,
    }


def load_preprocessed_kg_rag_examples(processed_data_file: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load already-preprocessed local KG-RAG examples for model training."""
    path = Path(processed_data_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed KG-RAG data file not found: {path}. Run `python enriched_training_dataset.py --mode kg_rag --output {path}` first."
        )
    instructions, answers, metadata = load_examples_from_local_json(path)
    instructions, answers = _limit_pairs(instructions, answers, limit)
    metadata = metadata[: len(instructions)]
    failed = [item.get("validation") for item in metadata if item.get("validation") and not item["validation"].get("ok")]
    if failed:
        raise ValueError(f"Preprocessed KG-RAG data contains validation failures: {failed}")
    return instructions, answers, metadata


def load_training_examples(
    mode: str = ORIGINAL_MODE,
    data_file: str = DEFAULT_DATA_FILE,
    processed_data_file: str = DEFAULT_KG_RAG_DATA_FILE,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load examples for fine-tuning without doing KG-RAG work in the training path."""
    if mode == ORIGINAL_MODE:
        instructions, answers = load_original_examples(data_file=data_file, limit=limit)
        return instructions, answers, []
    if mode == KG_RAG_MODE:
        return load_preprocessed_kg_rag_examples(processed_data_file=processed_data_file, limit=limit)
    raise ValueError(f"Unsupported dataset mode: {mode}. Expected '{ORIGINAL_MODE}' or '{KG_RAG_MODE}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess examples_16_june.json into KG-RAG fine-tuning examples.")
    parser.add_argument("--mode", choices=(ORIGINAL_MODE, KG_RAG_MODE), default=KG_RAG_MODE)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Original examples_16_june.json source, local or HF data file name.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kg-depth", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_KG_RAG_DATA_FILE), help="Local file to save preprocessed KG-RAG training data.")
    parser.add_argument("--progress-interval", type=int, default=25, help="Print enrichment progress every N examples.")
    args = parser.parse_args()

    if args.mode == ORIGINAL_MODE:
        _print_progress(f"Loading original examples from {args.data_file}.")
        instructions, answers = load_original_examples(args.data_file, limit=args.limit)
        _print_progress(f"Saving {len(instructions)} original examples to {args.output}.")
        save_training_examples(args.output, instructions, answers, [])
        summary = {"mode": args.mode, "output_path": str(args.output), "num_instructions": len(instructions), "num_answers": len(answers)}
    else:
        summary = preprocess_kg_rag_dataset(
            args.data_file,
            args.output,
            limit=args.limit,
            kg_depth=args.kg_depth,
            progress_interval=args.progress_interval,
        )
        summary["mode"] = args.mode
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
