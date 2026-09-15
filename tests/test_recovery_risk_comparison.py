from dataclasses import asdict

import pytest

from model_test_generation import RECOVERY_ONLY_STRATEGY, RISK_AWARE_STRATEGY
from recovery_risk_comparison import RecoveryRiskComparator, validate_strategy_modes
from response_post_processor import GenerationPostProcessor


def _record(record_id, strategy, generation, expected):
    return {
        "id": record_id,
        "response_strategy": strategy,
        "generation": generation,
        "expected_answer": expected,
        "post_processing": asdict(GenerationPostProcessor().process(generation)),
    }


def test_recovery_only_vs_risk_aware_effect_gaps_and_coverage():
    expected = {
        "action_type": "containment",
        "target": "host=web-01",
        "evidence": "firewall log",
    }
    recovery = _record(
        "r0",
        RECOVERY_ONLY_STRATEGY,
        {"action_type": "recovery", "target": "host=web-01", "command": "reboot"},
        expected,
    )
    risk_aware = _record(
        "r0",
        RISK_AWARE_STRATEGY,
        {
            "action_type": "containment",
            "target": "host=web-01",
            "command": "tcpdump -i eth0",
            "evidence": "firewall log",
            "precondition": "confirm alert",
            "risk": "capture disk usage",
            "rollback": "remove capture",
        },
        expected,
    )

    report = RecoveryRiskComparator().compare([recovery], [risk_aware])

    assert report.response_action_accuracy_gap.risk_aware_minus_recovery_only == 1.0
    assert report.unsafe_action_rate_gap.reduction_from_recovery_to_risk_aware == 1.0
    assert report.incomplete_action_rate_gap.reduction_from_recovery_to_risk_aware == 1.0
    assert report.recovery_only_coverage.recovery_action_rate == 1.0
    assert report.risk_aware_coverage.risk_coverage == 1.0
    assert report.risk_aware_coverage.rollback_coverage == 1.0


def test_strategy_comparison_rejects_wrong_arm_metadata():
    with pytest.raises(ValueError, match="recovery arm"):
        validate_strategy_modes(
            [{"response_strategy": RISK_AWARE_STRATEGY}],
            [{"response_strategy": RISK_AWARE_STRATEGY}],
        )
