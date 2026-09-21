#!/usr/bin/env python3
"""Lint the literature-to-idea evidence chain for missing proof obligations."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOS = {
    "nicklashansen/tdmpc2",
    "danijar/dreamerv3",
    "thuml/TrajWorld",
    "siddarth-c/KDD23-ADAC",
    "CMACH508/CausalExploration",
    "RL-DLMU/PRLight-and-PLight",
    "zyr17/UniLight",
    "XingshuaiHuang/ModelLight",
}
WEIGHTS = {
    "diagnosis_fit": 0.25,
    "formal_empirical_evidence": 0.15,
    "direct_tsc_evidence": 0.15,
    "cityflow_compatibility": 0.15,
    "novelty_delta_potential": 0.15,
    "code_and_license": 0.10,
    "compute_feasibility": 0.05,
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def lint(stage: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    master = read_csv("papers_master.csv")
    deep = read_csv("paper_evidence.csv")
    claims = read_csv("claims.csv")
    audits = read_csv("code_audit_matrix.csv")
    baselines = read_csv("baseline_boundary_matrix.csv")
    risks = read_csv("risk_matrix.csv")
    translations = read_csv("translation_matrix.csv")
    gaps = read_csv("gap_matrix.csv")
    result_audit = read_csv("paper_result_audit.csv")

    require(len(master) >= 100, f"candidate pool has {len(master)} papers; need >=100", errors)
    require(20 <= len(deep) <= 40, f"deep-read set has {len(deep)} papers; need 20-40", errors)
    statuses = Counter(row["publication_status"] for row in deep)
    require("formal_main_track" in statuses and "formal_journal" in statuses and "preprint" in statuses,
            "publication statuses are not explicitly separated", errors)
    claim_counts = Counter(row["paper_id"] for row in claims)
    for paper in deep:
        require(claim_counts[paper["paper_id"]] >= 3, f"{paper['paper_id']} has fewer than 3 atomic claims", errors)
        for field in ("motivation", "state_action", "prediction_targets", "limitation", "evidence_pointer", "traffic_inspiration"):
            require(bool(paper[field].strip()), f"{paper['paper_id']} missing {field}", errors)
    for claim in claims:
        require(claim["source_level"] in {"direct", "indirect", "inference", "model_generated"},
                f"{claim['claim_id']} has invalid source level", errors)
        require(bool(claim["evidence_quote_or_pointer"].strip()), f"{claim['claim_id']} lacks evidence pointer", errors)
        require(not (claim["source_level"] == "model_generated" and claim["related_gap"]),
                f"{claim['claim_id']} uses model-generated evidence for a gap", errors)
    require(len(result_audit) == len(deep),
            f"result audit has {len(result_audit)} rows; expected {len(deep)}", errors)
    require({row["paper_id"] for row in result_audit} == {row["paper_id"] for row in deep},
            "result audit paper set does not match deep-read set", errors)
    for row in result_audit:
        for field in ("dataset_protocol", "seed_variance_status", "evidence_pointer", "numeric_admission", "followup"):
            require(bool(row[field].strip()), f"{row['paper_id']} result audit missing {field}", errors)

    require({row["repository"] for row in audits} == EXPECTED_REPOS,
            "static audit repository set is incomplete or contains unexpected repositories", errors)
    for audit in audits:
        require(len(audit["audited_commit"]) == 40, f"{audit['repository']} lacks immutable commit SHA", errors)
        require(audit["audit_mode"] == "static_only_no_install_no_execution", f"{audit['repository']} audit boundary missing", errors)
        require(bool(audit["key_files"].strip()), f"{audit['repository']} lacks code pointers", errors)
        require(bool(audit["required_interface_change"].strip()), f"{audit['repository']} lacks interface mapping", errors)
    require(len(baselines) >= 7, "baseline boundary is too small", errors)
    require(any(row["name"] == "MaxPressure" for row in baselines), "MaxPressure baseline missing", errors)
    require(any("direct_tsc" in row["category"] for row in baselines), "direct TSC model-based baseline missing", errors)
    require(len(risks) >= 10, "risk matrix is too small", errors)
    for risk in risks:
        for field in ("detection", "mitigation", "kill_condition", "source"):
            require(bool(risk[field].strip()), f"{risk['risk_id']} missing {field}", errors)
    require(len(translations) >= 8, "translation matrix is too small", errors)
    for row in translations:
        for field in ("source_papers", "current_code", "interface_change", "expected_signal", "non_transferable"):
            require(bool(row[field].strip()), f"{row['translation_id']} missing {field}", errors)

    require(len(gaps) >= 3, "need at least three falsifiable gaps", errors)
    for gap in gaps:
        for field in ("nearest_methods", "not_a_gap", "evidence_boundary", "linked_risks", "falsification", "death_condition", "fallback"):
            require(bool(gap[field].strip()), f"{gap['gap_id']} missing {field}", errors)
        calculated = sum(float(gap[key]) * weight for key, weight in WEIGHTS.items())
        require(abs(calculated - float(gap["weighted_score"])) <= 0.011,
                f"{gap['gap_id']} weighted score mismatch: stored={gap['weighted_score']} calculated={calculated:.3f}", errors)
        require("not a global novelty claim" in gap["evidence_boundary"].lower() or "not an algorithmic novelty claim" in gap["evidence_boundary"].lower() or "currently a deferred" in gap["evidence_boundary"].lower() or "not inventing" in gap["evidence_boundary"].lower(),
                f"{gap['gap_id']} lacks a conservative novelty boundary", errors)

    if stage == "final":
        ideas = read_csv("idea_pool.csv")
        require(len(ideas) >= 3, "final idea pool needs at least three candidates", errors)
        for idea in ideas:
            for field in ("source_gaps", "nearest_methods", "method_delta", "why_it_may_beat_maxpressure", "minimum_experiment", "ablation", "death_condition", "fallback"):
                require(bool(idea[field].strip()), f"{idea['idea_id']} missing {field}", errors)
        for name in ("research_synthesis.md", "minimum_experiment_plan.md", "revision_tasks.md"):
            require((ROOT / name).is_file() and (ROOT / name).stat().st_size > 500, f"{name} missing or too small", errors)
        require(any(row.get("recommendation_status") == "recommended_main_route" for row in ideas),
                "idea pool has no recommended main route", errors)
        for name in ("audits/checkpoint_b_evidence_coverage.md", "audits/checkpoint_d_idea_review.md"):
            require((ROOT / name).is_file() and (ROOT / name).stat().st_size > 500,
                    f"{name} missing or too small", errors)

    notes.extend([
        f"candidate_records={len(master)}",
        f"deep_read_records={len(deep)}",
        f"atomic_claims={len(claims)}",
        f"result_audit_rows={len(result_audit)}",
        f"static_code_audits={len(audits)}",
        f"baseline_rows={len(baselines)}",
        f"risk_rows={len(risks)}",
        f"translation_rows={len(translations)}",
        f"gap_rows={len(gaps)}",
        "GPT Pro was unavailable; internal audit checkpoints are review comments, not evidence.",
    ])
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gaps", "final"), default="gaps")
    args = parser.parse_args()
    errors, notes = lint(args.stage)
    status = "PASS" if not errors else "FAIL"
    report = [f"# Research lint report ({args.stage})", "", f"Status: {status}", "", "## Counts and notes", ""]
    report.extend(f"- {note}" for note in notes)
    report.extend(["", "## Errors", ""])
    report.extend(f"- {error}" for error in errors)
    if not errors:
        report.append("- None")
    output = ROOT / ("lint_report.md" if args.stage == "final" else "audits/checkpoint_c_gap_lint.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(status)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
