import json

from response_evaluation import LabelSimilarityEvaluator
from response_generation_comparison import GenerationResultComparator, choose_better_generation


def _semantic_score(left, _right):
    return float(json.loads(left)["semantic_score"])


def _record(index, action_type, semantic_score, expected_action="containment"):
    return {
        "id": f"record-{index}",
        "generation": {"action_type": action_type, "semantic_score": semantic_score},
        "expected_answer": {"action_type": expected_action},
    }


def test_action_accuracy_has_priority_over_semantic_similarity():
    winner, reason = choose_better_generation(1.0, 0.0, 0.1, 0.9)

    assert winner == "base"
    assert reason == "action_accuracy_higher"


def test_generation_comparator_reports_base_better_indices_by_all_rules():
    base = [
        _record(0, "containment", 0.9),
        _record(1, "containment", 0.2),
        _record(2, "containment", 0.8),
        _record(3, "investigation", 0.9),
        _record(4, "containment", 0.5),
    ]
    kg_rag = [
        _record(0, "investigation", 0.2),
        _record(1, "investigation", 0.9),
        _record(2, "containment", 0.3),
        _record(3, "containment", 0.2),
        _record(4, "containment", 0.5),
    ]
    evaluator = LabelSimilarityEvaluator(semantic_scorer=_semantic_score, show_progress=False)

    report = GenerationResultComparator(evaluator=evaluator).compare(base, kg_rag)

    assert report.base_better_indices == [0, 1, 2]
    assert report.base_better_record_ids == ["record-0", "record-1", "record-2"]
    assert report.kg_rag_better_indices == [3]
    assert report.tie_indices == [4]
    assert report.comparisons[0].reason == "both_action_accuracy_and_semantic_similarity_higher"
    assert report.comparisons[1].reason == "action_accuracy_higher"
    assert report.comparisons[2].reason == "semantic_similarity_higher"


def test_comparator_honors_score_tolerance():
    assert choose_better_generation(1.0, 1.0, 0.50001, 0.5, tolerance=0.001) == (
        "tie",
        "equal_within_tolerance",
    )
