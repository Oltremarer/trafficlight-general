#!/usr/bin/env python3
"""Create idea artifacts only after the evidence/gap lint has passed."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


IDEAS = [
    {
        "idea_id": "I01",
        "working_name": "MaxPressure-Guarded Action-Semantic Graph World Model",
        "recommendation_status": "recommended_main_route",
        "source_gaps": "G01;G02;G03",
        "nearest_methods": "ModelLight (phase vector), UniLight (predicted neighbor impact), MaCAR (synchronous-action traffic prediction), TD-MPC2 (multi-step/value), PETS/MOPO/ADAC (uncertainty/pessimism).",
        "method_delta": "Use the selected phase_movement_mask to gate movement updates; send selected neighbor actions along directed road/movement edges; train 1-H step state/reward targets; score candidates by ensemble lower-confidence advantage over the identical MaxPressure proposal; execute MaxPressure unless the calibrated advantage exceeds a threshold.",
        "why_it_may_beat_maxpressure": "MaxPressure reacts to current queues. The proposed model can anticipate an upstream phase creating downstream inflow, but is allowed to deviate only when its predicted benefit is supported and sufficiently certain.",
        "minimum_experiment": "On flow-disjoint Jinan and Hangzhou states, compare current model, capacity-matched control, phase-mask model, phase-mask+neighbor-action model, H-step model, and guarded ensemble using one fixed candidate set; then run paired controller evaluation.",
        "ablation": "phase_movement_mask; neighbor actions; H-step loss; terminal value; ensemble penalty; support score; advantage threshold; fallback; equal-capacity and equal-search controls.",
        "death_condition": "Reject as the main route if action semantics do not improve candidate ranking in both cities, or if the complete guarded planner cannot achieve >=2% travel-time improvement with the preregistered CI/guardrails.",
        "fallback": "Retain only the smallest module that improves held-out candidate ranking; if none do, return to MaxPressure and abandon learned rollout planning.",
        "novelty_boundary": "This combination is a working hypothesis inside the audited set, not a confirmed global novelty claim. Every individual mechanism has prior art.",
        "implementation_order": "phase semantics -> neighbor actions -> H-step loss -> terminal value -> ensemble calibration -> conservative decision gate",
    },
    {
        "idea_id": "I02",
        "working_name": "Direct H-Step Candidate Ranker",
        "recommendation_status": "fallback_if_recursive_dynamics_fails",
        "source_gaps": "G02",
        "nearest_methods": "GMAN direct future sequence prediction, action-conditioned World Models, COMBO conservative value learning.",
        "method_delta": "Condition on current structured traffic state and a complete discrete joint-action sequence, then directly predict H-step cumulative return or pairwise advantage relative to MaxPressure instead of recursively predicting every next state.",
        "why_it_may_beat_maxpressure": "It targets the planner's actual decision—candidate ordering—and avoids compounding unused state dimensions, while MaxPressure remains the proposal and fallback.",
        "minimum_experiment": "Use the same counterfactual candidate-ranking dataset and compare rank correlation/regret to the recursive World Model at matched parameters and candidates.",
        "ablation": "return regression versus pairwise ranking loss; with/without movement mask; horizon 1/3/5; terminal continuation target.",
        "death_condition": "Reject if it does not improve held-out rank correlation/regret on both cities or if it overfits candidate generator patterns.",
        "fallback": "Use a one-step advantage estimator or MaxPressure only.",
        "novelty_boundary": "A direct ranker is not a full generative World Model and weakens the World Model paper story; use only if recursive dynamics is empirically the wrong tool.",
        "implementation_order": "counterfactual ranking dataset -> matched ranker -> ranking evaluation -> controller gate",
    },
    {
        "idea_id": "I03",
        "working_name": "Support-Calibrated Ensemble Gate",
        "recommendation_status": "conditional_safety_addon",
        "source_gaps": "G03",
        "nearest_methods": "PETS, MOPO, COMBO, MOReL, ADAC.",
        "method_delta": "Calibrate ensemble disagreement and trajectory support against real CityFlow candidate error, combine them into a lower-confidence advantage, and expose a deterministic MaxPressure fallback with a worst-flow degradation budget.",
        "why_it_may_beat_maxpressure": "It cannot create predictive skill by itself, but it can preserve rare reliable anticipatory improvements while filtering model exploitation that made the unguarded planner worse.",
        "minimum_experiment": "Measure reliability curves and high-regret detection AUROC on flow-disjoint calibration/test sets; sweep one frozen threshold and report coverage-risk curves before control results.",
        "ablation": "ensemble only; kNN/support only; combined score; advantage threshold only; fallback only; COMBO-style value alternative if variance is uncalibrated.",
        "death_condition": "Reject ensemble-based gating if disagreement/support is not monotonic with error on both cities or if safe coverage is too low to yield control gains.",
        "fallback": "Calibrated deterministic advantage threshold; ultimately always MaxPressure.",
        "novelty_boundary": "Conservative uncertainty and fallback are established mechanisms; any contribution must be traffic-specific calibration and outcome evidence.",
        "implementation_order": "bootstrap ensemble -> offline calibration -> frozen threshold -> coverage-risk audit -> paired control",
    },
    {
        "idea_id": "I04",
        "working_name": "Heterogeneous Traffic-Dynamics Pretraining",
        "recommendation_status": "deferred_extension",
        "source_gaps": "G04",
        "nearest_methods": "TrajWorld, UniST, X-Light, CrossLight, PLight/PRLight.",
        "method_delta": "Pretrain a masked movement/action dynamics backbone across independent flows, roadnets and cities, then adapt only small schema-specific components.",
        "why_it_may_beat_maxpressure": "Broader dynamics data may improve rare-demand and unseen-city prediction, but only if the learned action semantics already rank candidates correctly.",
        "minimum_experiment": "After I01 passes, compare per-city training versus cross-city pretraining at identical target-city data and compute; hold out one city/flow family.",
        "ablation": "flow diversity; roadnet diversity; city tokens; schema masks; local adapter; equal-data and equal-compute controls.",
        "death_condition": "Reject if equal-budget target-city ranking/control does not improve or if either city suffers negative transfer.",
        "fallback": "Per-city models with a shared data schema but no shared weights.",
        "novelty_boundary": "Pretraining and cross-city TSC transfer are established; this cannot be the first experiment or a generic novelty claim.",
        "implementation_order": "defer until within-city mechanism and safety gate pass all death conditions",
    },
]


def main() -> int:
    checkpoint = ROOT / "audits/checkpoint_c_gap_lint.md"
    if not checkpoint.is_file() or "Status: PASS" not in checkpoint.read_text(encoding="utf-8"):
        raise SystemExit("gap lint has not passed; idea creation is blocked")
    fields = list(IDEAS[0])
    with (ROOT / "idea_pool.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(IDEAS)
    lines = ["# Idea cards", "", "These are testable research candidates, not established innovations.", ""]
    for idea in IDEAS:
        lines.extend([
            f"## {idea['idea_id']} — {idea['working_name']}", "",
            f"Status: `{idea['recommendation_status']}`", "",
            f"- Evidence gaps: {idea['source_gaps']}",
            f"- Nearest methods: {idea['nearest_methods']}",
            f"- Method delta: {idea['method_delta']}",
            f"- Why it may beat MaxPressure: {idea['why_it_may_beat_maxpressure']}",
            f"- Minimum experiment: {idea['minimum_experiment']}",
            f"- Ablation: {idea['ablation']}",
            f"- Death condition: {idea['death_condition']}",
            f"- Fallback: {idea['fallback']}",
            f"- Novelty boundary: {idea['novelty_boundary']}", "",
        ])
    (ROOT / "idea_cards.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(IDEAS)} evidence-gated ideas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
