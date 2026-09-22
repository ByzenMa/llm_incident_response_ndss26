import json

from recovery_risk_postprocess_experiment import BOTH, run_saved_output_experiment


def test_saved_output_experiment_writes_both_arms_and_comparison_without_generation(tmp_path):
    generation = {
        "action_type": "containment",
        "target": "host=web-01",
        "evidence": "firewall alert",
    }
    records = [{"id": "r0", "generation": generation, "expected_answer": generation}]
    recovery_path = tmp_path / "recovery.jsonl"
    risk_path = tmp_path / "risk.jsonl"
    report_path = tmp_path / "report.json"

    summary = run_saved_output_experiment(
        records,
        mode=BOTH,
        recovery_output=recovery_path,
        risk_aware_output=risk_path,
        comparison_output=report_path,
    )

    recovery = json.loads(recovery_path.read_text(encoding="utf-8").splitlines()[0])
    risk_aware = json.loads(risk_path.read_text(encoding="utf-8").splitlines()[0])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert recovery["generation"] == risk_aware["generation"] == generation
    assert recovery["response_strategy"] == "recovery_only"
    assert risk_aware["response_strategy"] == "risk_aware"
    assert report["paired_record_count"] == 1
    assert summary["model_generation_run"] is False
    assert records[0] == {"id": "r0", "generation": generation, "expected_answer": generation}
