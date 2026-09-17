"""Compare text-RAG and KG-RAG final generations on paired test records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from generation_rag import KG_RAG, TEXT_RAG
from response_evaluation import (
    LabelSimilarityEvaluator,
    ResponseEvaluator,
    SafetyEvaluationReport,
    SentenceTransformerSimilarity,
    SimilarityEvaluationReport,
    load_evaluation_records,
)
from response_model_comparison import validate_paired_records


@dataclass
class ScoreGap:
    text_rag: float
    kg_rag: float
    kg_rag_minus_text_rag: float


@dataclass
class ErrorRateGap:
    text_rag: float
    kg_rag: float
    reduction_from_text_to_kg: float


@dataclass
class RAGComparisonReport:
    paired_record_count: int
    text_rag_similarity: SimilarityEvaluationReport
    kg_rag_similarity: SimilarityEvaluationReport
    text_rag_safety: SafetyEvaluationReport
    kg_rag_safety: SafetyEvaluationReport
    response_action_accuracy_gap: ScoreGap
    evidence_accuracy_gap: ScoreGap
    mean_semantic_similarity_gap: ScoreGap
    incorrect_command_rate_gap: ErrorRateGap
    unsafe_action_rate_gap: ErrorRateGap
    incomplete_action_rate_gap: ErrorRateGap


def _score_gap(text_score: float, kg_score: float) -> ScoreGap:
    return ScoreGap(text_score, kg_score, kg_score - text_score)


def _error_gap(text_rate: float, kg_rate: float) -> ErrorRateGap:
    return ErrorRateGap(text_rate, kg_rate, text_rate - kg_rate)


def validate_rag_modes(text_records: Sequence[dict], kg_records: Sequence[dict]) -> None:
    if any(record.get("rag_mode") != TEXT_RAG for record in text_records):
        raise ValueError("Every text-RAG record must have rag_mode='text_rag'.")
    if any(record.get("rag_mode") != KG_RAG for record in kg_records):
        raise ValueError("Every KG-RAG record must have rag_mode='kg_rag'.")
    for name, records in ((TEXT_RAG, text_records), (KG_RAG, kg_records)):
        if any(not isinstance(record.get("post_processing"), dict) for record in records):
            raise ValueError(f"Every {name} record must contain a recorded post_processing report.")


class RAGGenerationComparator:
    def __init__(
        self,
        similarity_evaluator: Optional[LabelSimilarityEvaluator] = None,
        safety_evaluator: Optional[ResponseEvaluator] = None,
    ) -> None:
        self.similarity_evaluator = similarity_evaluator or LabelSimilarityEvaluator()
        self.safety_evaluator = safety_evaluator or ResponseEvaluator()

    def compare(self, text_records: Sequence[dict], kg_records: Sequence[dict]) -> RAGComparisonReport:
        validate_paired_records(text_records, kg_records)
        validate_rag_modes(text_records, kg_records)
        text_similarity = self.similarity_evaluator.evaluate(text_records, progress_label="text-RAG")
        kg_similarity = self.similarity_evaluator.evaluate(kg_records, progress_label="KG-RAG")
        text_safety = self.safety_evaluator.evaluate_recorded_post_processing(text_records)
        kg_safety = self.safety_evaluator.evaluate_recorded_post_processing(kg_records)
        return RAGComparisonReport(
            len(text_records),
            text_similarity,
            kg_similarity,
            text_safety,
            kg_safety,
            _score_gap(
                text_similarity.response_action_accuracy.average,
                kg_similarity.response_action_accuracy.average,
            ),
            _score_gap(text_similarity.evidence_accuracy.average, kg_similarity.evidence_accuracy.average),
            _score_gap(
                text_similarity.mean_semantic_similarity.average,
                kg_similarity.mean_semantic_similarity.average,
            ),
            _error_gap(text_safety.incorrect_command_rate.rate, kg_safety.incorrect_command_rate.rate),
            _error_gap(text_safety.unsafe_action_rate.rate, kg_safety.unsafe_action_rate.rate),
            _error_gap(text_safety.incomplete_action_rate.rate, kg_safety.incomplete_action_rate.rate),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired text-RAG and KG-RAG prediction outputs.")
    parser.add_argument("--text-rag-output", type=Path, required=True)
    parser.add_argument("--kg-rag-output", type=Path, required=True)
    parser.add_argument("--semantic-model", help="Optional SentenceTransformer model.")
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorer = SentenceTransformerSimilarity(args.semantic_model) if args.semantic_model else None
    report = RAGGenerationComparator(
        similarity_evaluator=LabelSimilarityEvaluator(
            semantic_scorer=scorer,
            show_progress=args.show_progress,
            progress_interval=args.progress_interval,
        )
    ).compare(
        load_evaluation_records(args.text_rag_output),
        load_evaluation_records(args.kg_rag_output),
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
