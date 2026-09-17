import pytest

from generation_rag import KG_RAG, TEXT_RAG
from rag_generation_comparison import RAGGenerationComparator, calculate_top_k_metrics


def _record(record_id, mode, scores):
    return {
        "id": record_id,
        "rag_mode": mode,
        "rag_output": {
            "retrieved_items": [
                {"document_id": f"{mode}-{rank}", "score": score}
                for rank, score in enumerate(scores, start=1)
            ]
        },
    }


def test_rag_comparison_reports_top1_top2_and_top3_score_metrics():
    text_records = [_record("r0", TEXT_RAG, [0.8, 0.4, 0.1]), _record("r1", TEXT_RAG, [0.6, 0.2])]
    kg_records = [_record("r0", KG_RAG, [0.9, 0.7, 0.5]), _record("r1", KG_RAG, [0.7, 0.3, 0.1])]

    report = RAGGenerationComparator(match_threshold=0.25).compare(text_records, kg_records)

    assert report.top1.text_rag.average_score == pytest.approx(0.7)
    assert report.top1.kg_rag.average_score == pytest.approx(0.8)
    assert report.top2.text_rag.match_rate == pytest.approx(0.75)
    assert report.top2.kg_rag.match_rate == pytest.approx(1.0)
    assert report.top3.text_rag.average_score == pytest.approx(2.1 / 6)
    assert report.top3.kg_rag.average_score == pytest.approx(3.2 / 6)
    assert report.top3.winner_by_average_score == KG_RAG


def test_top_k_missing_slots_count_as_zero_score_non_matches():
    metrics = calculate_top_k_metrics([_record("r0", TEXT_RAG, [0.9])], k=3, match_threshold=0.5)

    assert metrics.available_item_count == 1
    assert metrics.expected_item_count == 3
    assert metrics.match_rate == pytest.approx(1 / 3)
    assert metrics.average_score == pytest.approx(0.3)


def test_rag_comparison_requires_numeric_scores_in_rag_output():
    text_records = [_record("r0", TEXT_RAG, [0.8])]
    kg_records = [_record("r0", KG_RAG, [0.9])]
    del kg_records[0]["rag_output"]["retrieved_items"][0]["score"]

    with pytest.raises(ValueError, match="numeric score"):
        RAGGenerationComparator().compare(text_records, kg_records)
