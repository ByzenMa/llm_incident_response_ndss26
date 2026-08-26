"""Rank paired base and KG-RAG generations with action-first comparison rules."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from response_evaluation import (
    LabelSimilarityEvaluator,
    SentenceTransformerSimilarity,
    load_evaluation_records,
)
from response_model_comparison import validate_paired_records


BASE_WINNER = "base"
KG_RAG_WINNER = "kg_rag_finetuned"
TIE_WINNER = "tie"


def _percentage(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("percentage must be between 0 and 100")
    return parsed


@dataclass
class GenerationPreference:
    index: int
    record_id: str
    winner: str
    reason: str
    base_action_accuracy: float
    kg_rag_action_accuracy: float
    base_semantic_similarity: float
    kg_rag_semantic_similarity: float


@dataclass
class GenerationComparisonReport:
    record_count: int
    base_better_count: int
    kg_rag_better_count: int
    tie_count: int
    base_better_indices: List[int]
    kg_rag_better_indices: List[int]
    tie_indices: List[int]
    base_better_record_ids: List[str]
    kg_rag_better_record_ids: List[str]
    tie_record_ids: List[str]
    comparisons: List[GenerationPreference]
    swap_percentage: float = 0.0
    swap_seed: Optional[int] = None
    swapped_indices: List[int] = field(default_factory=list)


def _record_id(record: dict, index: int) -> str:
    for key in ("id", "record_id", "source_index", "incident_id", "prompt_id"):
        if record.get(key) is not None:
            return str(record[key])
    return str(index)


def _greater(left: float, right: float, tolerance: float) -> bool:
    return left > right + tolerance


def swap_generation_percentage(
    base_records: Sequence[dict],
    kg_rag_records: Sequence[dict],
    percentage: float,
    seed: int = 99125,
) -> tuple[List[dict], List[dict], List[int]]:
    """Swap only paired ``generation`` values in deterministic copied records."""
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("swap percentage must be between 0 and 100.")
    validate_paired_records(base_records, kg_rag_records)
    for index, (base_record, kg_record) in enumerate(zip(base_records, kg_rag_records)):
        if "generation" not in base_record or "generation" not in kg_record:
            raise ValueError(f"Record {index} must contain generation in both output files.")

    base_copies = copy.deepcopy(list(base_records))
    kg_copies = copy.deepcopy(list(kg_rag_records))
    swap_count = min(
        len(base_copies),
        math.floor(len(base_copies) * percentage / 100.0 + 0.5),
    )
    swapped_indices = sorted(random.Random(seed).sample(range(len(base_copies)), swap_count))
    for index in swapped_indices:
        base_generation = base_copies[index]["generation"]
        base_copies[index]["generation"] = kg_copies[index]["generation"]
        kg_copies[index]["generation"] = base_generation
    return base_copies, kg_copies, swapped_indices


def save_output_records(path: Path, records: Sequence[dict]) -> None:
    """Write copied prediction records as JSONL without modifying source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def choose_better_generation(
    base_action: float,
    kg_action: float,
    base_semantic: float,
    kg_semantic: float,
    tolerance: float = 1e-9,
) -> tuple[str, str]:
    """Apply both-higher, action-first, then semantic tie-break rules."""
    base_action_higher = _greater(base_action, kg_action, tolerance)
    kg_action_higher = _greater(kg_action, base_action, tolerance)
    base_semantic_higher = _greater(base_semantic, kg_semantic, tolerance)
    kg_semantic_higher = _greater(kg_semantic, base_semantic, tolerance)

    if base_action_higher and base_semantic_higher:
        return BASE_WINNER, "both_action_accuracy_and_semantic_similarity_higher"
    if kg_action_higher and kg_semantic_higher:
        return KG_RAG_WINNER, "both_action_accuracy_and_semantic_similarity_higher"
    if base_action_higher:
        return BASE_WINNER, "action_accuracy_higher"
    if kg_action_higher:
        return KG_RAG_WINNER, "action_accuracy_higher"
    if base_semantic_higher:
        return BASE_WINNER, "semantic_similarity_higher"
    if kg_semantic_higher:
        return KG_RAG_WINNER, "semantic_similarity_higher"
    return TIE_WINNER, "equal_within_tolerance"


class GenerationResultComparator:
    """Compare paired outputs and list indexes where each model is better."""

    def __init__(
        self,
        evaluator: Optional[LabelSimilarityEvaluator] = None,
        tolerance: float = 1e-9,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        self.evaluator = evaluator or LabelSimilarityEvaluator()
        self.tolerance = tolerance

    def compare(self, base_records: Sequence[dict], kg_rag_records: Sequence[dict]) -> GenerationComparisonReport:
        validate_paired_records(base_records, kg_rag_records)
        base_scores = self.evaluator.evaluate(base_records, progress_label="base generations")
        kg_scores = self.evaluator.evaluate(kg_rag_records, progress_label="KG-RAG fine-tuned generations")
        comparisons: List[GenerationPreference] = []
        for index, (base_score, kg_score) in enumerate(
            zip(base_scores.record_evaluations, kg_scores.record_evaluations)
        ):
            winner, reason = choose_better_generation(
                base_score.action_accuracy,
                kg_score.action_accuracy,
                base_score.semantic_similarity,
                kg_score.semantic_similarity,
                self.tolerance,
            )
            comparisons.append(
                GenerationPreference(
                    index=index,
                    record_id=_record_id(base_records[index], index),
                    winner=winner,
                    reason=reason,
                    base_action_accuracy=base_score.action_accuracy,
                    kg_rag_action_accuracy=kg_score.action_accuracy,
                    base_semantic_similarity=base_score.semantic_similarity,
                    kg_rag_semantic_similarity=kg_score.semantic_similarity,
                )
            )

        def indices(winner: str) -> List[int]:
            return [item.index for item in comparisons if item.winner == winner]

        def record_ids(winner: str) -> List[str]:
            return [item.record_id for item in comparisons if item.winner == winner]

        base_indices = indices(BASE_WINNER)
        kg_indices = indices(KG_RAG_WINNER)
        tie_indices = indices(TIE_WINNER)
        return GenerationComparisonReport(
            record_count=len(comparisons),
            base_better_count=len(base_indices),
            kg_rag_better_count=len(kg_indices),
            tie_count=len(tie_indices),
            base_better_indices=base_indices,
            kg_rag_better_indices=kg_indices,
            tie_indices=tie_indices,
            base_better_record_ids=record_ids(BASE_WINNER),
            kg_rag_better_record_ids=record_ids(KG_RAG_WINNER),
            tie_record_ids=record_ids(TIE_WINNER),
            comparisons=comparisons,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and KG-RAG generation text with action-first ranking.")
    parser.add_argument("--base-output", type=Path, required=True, help="Base-model JSON/JSONL prediction output.")
    parser.add_argument("--kg-rag-output", type=Path, required=True, help="KG-RAG fine-tuned JSON/JSONL prediction output.")
    parser.add_argument("--semantic-model", help="Optional SentenceTransformer model for semantic similarity.")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Score equality tolerance.")
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    parser.add_argument("--swap-percentage", type=_percentage, default=0.0, help="Percentage of paired generation values to swap.")
    parser.add_argument("--swap-seed", type=int, default=99125, help="Seed used to choose swapped indexes.")
    parser.add_argument("--swapped-base-output", type=Path, help="New JSONL file for the optionally swapped base records.")
    parser.add_argument("--swapped-kg-rag-output", type=Path, help="New JSONL file for the optionally swapped KG-RAG records.")
    parser.add_argument("--output", type=Path, required=True, help="Local JSON comparison report.")
    args = parser.parse_args()
    swap_requested = args.swap_percentage > 0 or args.swapped_base_output or args.swapped_kg_rag_output
    if swap_requested and (not args.swapped_base_output or not args.swapped_kg_rag_output):
        parser.error("Provide both --swapped-base-output and --swapped-kg-rag-output when swapping or copying outputs.")
    base_records = load_evaluation_records(args.base_output)
    kg_rag_records = load_evaluation_records(args.kg_rag_output)
    swapped_indices: List[int] = []
    if swap_requested:
        base_records, kg_rag_records, swapped_indices = swap_generation_percentage(
            base_records, kg_rag_records, args.swap_percentage, args.swap_seed
        )
        save_output_records(args.swapped_base_output, base_records)
        save_output_records(args.swapped_kg_rag_output, kg_rag_records)
    semantic_scorer = SentenceTransformerSimilarity(args.semantic_model) if args.semantic_model else None
    evaluator = LabelSimilarityEvaluator(
        semantic_scorer=semantic_scorer,
        show_progress=args.show_progress,
        progress_interval=args.progress_interval,
    )
    report = GenerationResultComparator(evaluator=evaluator, tolerance=args.tolerance).compare(
        base_records,
        kg_rag_records,
    )
    report.swap_percentage = args.swap_percentage
    report.swap_seed = args.swap_seed if swap_requested else None
    report.swapped_indices = swapped_indices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
