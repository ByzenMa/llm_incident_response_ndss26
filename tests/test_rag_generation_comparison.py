from dataclasses import asdict

from generation_rag import KG_RAG, TEXT_RAG
from rag_generation_comparison import RAGGenerationComparator
from response_post_processor import GenerationPostProcessor


def _record(record_id, mode, generation, expected):
    return {
        "id": record_id,
        "rag_mode": mode,
        "generation": generation,
        "expected_answer": expected,
        "post_processing": asdict(GenerationPostProcessor().process(generation)),
    }


def test_rag_comparison_reports_text_to_kg_effect_gaps():
    expected = {"action_type": "containment", "target": "host=web-01", "evidence": "firewall log"}
    text_records = [
        _record("r0", TEXT_RAG, {"action_type": "investigation", "command": "unknown-tool scan"}, expected)
    ]
    kg_records = [_record("r0", KG_RAG, expected, expected)]

    report = RAGGenerationComparator().compare(text_records, kg_records)

    assert report.response_action_accuracy_gap.kg_rag_minus_text_rag == 1.0
    assert report.evidence_accuracy_gap.kg_rag_minus_text_rag == 1.0
    assert report.incorrect_command_rate_gap.reduction_from_text_to_kg == 1.0
    assert report.incomplete_action_rate_gap.reduction_from_text_to_kg == 1.0
