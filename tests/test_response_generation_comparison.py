import json

import pytest

from response_evaluation import LabelSimilarityEvaluator
from response_generation_comparison import (
    GenerationResultComparator,
    choose_better_generation,
    save_output_records,
    swap_base_winning_generations,
)


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


def test_swap_percentage_selects_only_base_winners_and_keeps_sources_unchanged():
    base = [_record(index, "containment", 0.1 + index) for index in range(4)]
    kg_rag = [_record(index, "investigation", 0.9 - index) for index in range(4)]
    base_before = json.loads(json.dumps(base))
    kg_before = json.loads(json.dumps(kg_rag))

    base_better_indices = [0, 2]
    swapped_base, swapped_kg, indices = swap_base_winning_generations(
        base, kg_rag, base_better_indices, 50, seed=7
    )

    assert len(indices) == 1
    assert set(indices).issubset(base_better_indices)
    assert base == base_before
    assert kg_rag == kg_before
    for index in range(4):
        expected_base_generation = kg_before[index]["generation"] if index in indices else base_before[index]["generation"]
        expected_kg_generation = base_before[index]["generation"] if index in indices else kg_before[index]["generation"]
        assert swapped_base[index]["generation"] == expected_base_generation
        assert swapped_kg[index]["generation"] == expected_kg_generation
        assert {key: value for key, value in swapped_base[index].items() if key != "generation"} == {
            key: value for key, value in base_before[index].items() if key != "generation"
        }


def test_swap_percentage_supports_zero_and_full_swap():
    base = [_record(0, "containment", 0.1)]
    kg_rag = [_record(0, "investigation", 0.9)]

    zero_base, zero_kg, zero_indices = swap_base_winning_generations(base, kg_rag, [0], 0)
    full_base, full_kg, full_indices = swap_base_winning_generations(base, kg_rag, [0], 100)

    assert zero_indices == []
    assert zero_base == base and zero_kg == kg_rag
    assert full_indices == [0]
    assert full_base[0]["generation"] == kg_rag[0]["generation"]
    assert full_kg[0]["generation"] == base[0]["generation"]


def test_save_output_records_writes_new_jsonl_file(tmp_path):
    output = tmp_path / "swapped" / "base.jsonl"
    records = [_record(0, "containment", 0.5)]

    save_output_records(output, records)

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


@pytest.mark.parametrize("percentage", [-0.1, 100.1])
def test_swap_rejects_percentage_outside_valid_range(percentage):
    with pytest.raises(ValueError, match="between 0 and 100"):
        swap_base_winning_generations([], [], [], percentage)
