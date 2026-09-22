"""Run recovery-only vs risk-aware post-processing on saved model outputs.

This driver never loads or invokes a language model. It reuses each saved
``generation`` verbatim, creates policy-specific JSONL files, and compares the
two arms when ``--mode both`` is selected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from recovery_risk_comparison import (
    RecoveryRiskComparator,
    build_post_processed_strategy_arm,
    save_strategy_records,
)
from response_evaluation import load_evaluation_records
from response_post_processor import RECOVERY_ONLY_STRATEGY, RISK_AWARE_STRATEGY

BOTH = "both"


def run_saved_output_experiment(
    records: Sequence[Dict[str, Any]],
    mode: str = BOTH,
    recovery_output: Optional[Path] = None,
    risk_aware_output: Optional[Path] = None,
    comparison_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Post-process saved generations without performing model inference."""
    if mode not in {BOTH, RECOVERY_ONLY_STRATEGY, RISK_AWARE_STRATEGY}:
        raise ValueError(f"Unsupported mode: {mode}.")
    summary: Dict[str, Any] = {"mode": mode, "record_count": len(records), "model_generation_run": False}
    recovery_records = None
    risk_records = None
    if mode in {BOTH, RECOVERY_ONLY_STRATEGY}:
        recovery_records = build_post_processed_strategy_arm(records, RECOVERY_ONLY_STRATEGY)
        if recovery_output is None:
            raise ValueError("recovery_output is required for recovery-only processing.")
        save_strategy_records(recovery_output, recovery_records)
        summary["recovery_only_output"] = str(recovery_output)
    if mode in {BOTH, RISK_AWARE_STRATEGY}:
        risk_records = build_post_processed_strategy_arm(records, RISK_AWARE_STRATEGY)
        if risk_aware_output is None:
            raise ValueError("risk_aware_output is required for risk-aware processing.")
        save_strategy_records(risk_aware_output, risk_records)
        summary["risk_aware_output"] = str(risk_aware_output)
    if mode == BOTH:
        if comparison_output is None:
            raise ValueError("comparison_output is required when mode is 'both'.")
        report = RecoveryRiskComparator().compare(recovery_records, risk_records)
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_output.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["comparison_output"] = str(comparison_output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process an existing model output; no model generation is run."
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing model prediction JSON/JSONL file.")
    parser.add_argument(
        "--mode",
        choices=(BOTH, RECOVERY_ONLY_STRATEGY, RISK_AWARE_STRATEGY),
        default=BOTH,
    )
    parser.add_argument("--recovery-only-output", type=Path, default=Path("recovery_only_postprocessed.jsonl"))
    parser.add_argument("--risk-aware-output", type=Path, default=Path("risk_aware_postprocessed.jsonl"))
    parser.add_argument("--comparison-output", type=Path, default=Path("recovery_vs_risk_aware_report.json"))
    args = parser.parse_args()
    summary = run_saved_output_experiment(
        load_evaluation_records(args.input),
        mode=args.mode,
        recovery_output=args.recovery_only_output,
        risk_aware_output=args.risk_aware_output,
        comparison_output=args.comparison_output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
