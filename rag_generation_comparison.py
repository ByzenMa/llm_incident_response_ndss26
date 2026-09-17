"""Compare retrieval quality in paired text-RAG and KG-RAG output files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from generation_rag import KG_RAG, TEXT_RAG
from response_evaluation import load_evaluation_records
from response_model_comparison import validate_paired_records


@dataclass
class TopKMetrics:
    """Cumulative retrieval metrics for the first k ranked results."""

    k: int
    record_count: int
    available_item_count: int
    expected_item_count: int
    matched_item_count: int
    match_rate: float
    average_score: float


@dataclass
class TopKComparison:
    k: int
    text_rag: TopKMetrics
    kg_rag: TopKMetrics
    match_rate_gap: float
    average_score_gap: float
    winner_by_average_score: str


@dataclass
class RAGComparisonReport:
    paired_record_count: int
    match_threshold: float
    top1: TopKComparison
    top2: TopKComparison
    top3: TopKComparison


def _retrieved_items(record: dict, index: int, mode: str) -> list[dict]:
    rag_output = record.get("rag_output")
    if not isinstance(rag_output, dict):
        raise ValueError(f"{mode} record {index} must contain a rag_output object.")
    items = rag_output.get("retrieved_items")
    if not isinstance(items, list):
        raise ValueError(f"{mode} record {index} rag_output must contain retrieved_items.")
    return items


def validate_rag_outputs(text_records: Sequence[dict], kg_records: Sequence[dict]) -> None:
    """Validate paired modes and score-bearing retrieval outputs."""
    validate_paired_records(text_records, kg_records)
    for mode, records in ((TEXT_RAG, text_records), (KG_RAG, kg_records)):
        for index, record in enumerate(records):
            if record.get("rag_mode") != mode:
                raise ValueError(f"Every {mode} record must have rag_mode={mode!r}.")
            for rank, item in enumerate(_retrieved_items(record, index, mode), start=1):
                if not isinstance(item, dict) or not isinstance(item.get("score"), (int, float)):
                    raise ValueError(
                        f"{mode} record {index} retrieved item {rank} must contain a numeric score."
                    )


def calculate_top_k_metrics(
    records: Sequence[dict],
    k: int,
    match_threshold: float = 0.0,
) -> TopKMetrics:
    """Calculate cumulative match rate and mean score over the first k slots.

    Missing retrieval slots count as zero-score, non-matches. This keeps files
    with fewer retrieved candidates directly comparable.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: list[float] = []
    for index, record in enumerate(records):
        items = _retrieved_items(record, index, str(record.get("rag_mode", "RAG")))[:k]
        scores.extend(float(item["score"]) for item in items)
    expected = len(records) * k
    matched = sum(score > match_threshold for score in scores)
    score_sum = sum(scores)
    return TopKMetrics(
        k=k,
        record_count=len(records),
        available_item_count=len(scores),
        expected_item_count=expected,
        matched_item_count=matched,
        match_rate=matched / expected if expected else 0.0,
        average_score=score_sum / expected if expected else 0.0,
    )


def _winner(text_score: float, kg_score: float) -> str:
    if kg_score > text_score:
        return KG_RAG
    if text_score > kg_score:
        return TEXT_RAG
    return "tie"


class RAGGenerationComparator:
    def __init__(self, match_threshold: float = 0.0) -> None:
        self.match_threshold = match_threshold

    def compare(self, text_records: Sequence[dict], kg_records: Sequence[dict]) -> RAGComparisonReport:
        validate_rag_outputs(text_records, kg_records)
        comparisons: list[TopKComparison] = []
        for k in (1, 2, 3):
            text_metrics = calculate_top_k_metrics(text_records, k, self.match_threshold)
            kg_metrics = calculate_top_k_metrics(kg_records, k, self.match_threshold)
            comparisons.append(
                TopKComparison(
                    k=k,
                    text_rag=text_metrics,
                    kg_rag=kg_metrics,
                    match_rate_gap=kg_metrics.match_rate - text_metrics.match_rate,
                    average_score_gap=kg_metrics.average_score - text_metrics.average_score,
                    winner_by_average_score=_winner(text_metrics.average_score, kg_metrics.average_score),
                )
            )
        return RAGComparisonReport(len(text_records), self.match_threshold, *comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare top-1/2/3 retrieval scores in text-RAG and KG-RAG outputs.")
    parser.add_argument("--text-rag-output", type=Path, required=True)
    parser.add_argument("--kg-rag-output", type=Path, required=True)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.0,
        help="A retrieved item matches when its score is greater than this value.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = RAGGenerationComparator(args.match_threshold).compare(
        load_evaluation_records(args.text_rag_output),
        load_evaluation_records(args.kg_rag_output),
    )
    report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
