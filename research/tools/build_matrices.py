#!/usr/bin/env python3
"""Build evidence-backed baseline, risk, translation, and gap matrices."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_csv(name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    with (ROOT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


BASELINES = [
    {
        "baseline_id": "B01", "name": "FixedTime", "category": "rule_floor",
        "model_based": "no", "coordination": "schedule_only", "transfer": "none",
        "covers": "A non-adaptive lower-bound controller and simulator sanity check.",
        "does_not_cover": "Reactive pressure, learning, counterfactual prediction, or cross-demand robustness.",
        "why_required": "Detects broken environments and establishes whether learning beats a trivial schedule.",
        "implementation_source": "src/cityflow_tsc/policies.py; existing project implementation",
        "protocol_contract": "Same roadnet, exact flow file, signal timing, horizon and metric definitions as every compared controller.",
        "decision_role": "floor_only",
    },
    {
        "baseline_id": "B02", "name": "MaxPressure", "category": "theory_grounded_primary",
        "model_based": "no", "coordination": "implicit_through_pressure", "transfer": "zero_tuning",
        "covers": "Strong reactive movement/queue semantics and a controller that works without learned dynamics.",
        "does_not_cover": "Anticipatory multi-step effects, learned uncertainty, or candidate support.",
        "why_required": "The proposed planner starts from and deviates from MaxPressure; it is the primary causal comparator and fallback.",
        "implementation_source": "src/cityflow_tsc/policies.py:51-82; Varaiya 2013 and PressLight context",
        "protocol_contract": "Use identical observation semantics and phase_movement_mask; log every proposed and overridden action.",
        "decision_role": "primary_baseline_and_safety_anchor",
    },
    {
        "baseline_id": "B03", "name": "SharedDQN", "category": "model_free_learned",
        "model_based": "no", "coordination": "parameter_sharing", "transfer": "not_intrinsic",
        "covers": "Whether a learned model-free controller can exploit the same structured state and training data budget.",
        "does_not_cover": "Explicit graph communication, world-model planning, or theory-grounded pressure priors unless added.",
        "why_required": "Separates benefit from model-based prediction versus merely using a neural controller.",
        "implementation_source": "existing project DQN command and src/cityflow_tsc/dqn.py",
        "protocol_contract": "Match environment steps, replay data access, seeds/flows, action timing and evaluation checkpoints.",
        "decision_role": "required_learned_baseline",
    },
    {
        "baseline_id": "B04", "name": "PressLight or maintained MPLight-equivalent", "category": "strong_pressure_rl",
        "model_based": "no", "coordination": "pressure_reward_or_parameter_sharing", "transfer": "not_intrinsic",
        "covers": "Strong learned pressure/movement representation; tests whether a World Model adds more than good traffic inductive bias.",
        "does_not_cover": "Explicit next-state dynamics or calibrated model rollout.",
        "why_required": "LibSignal reports PressLight/IDQN as stable and sample-efficient; MaxPressure alone is not the full learned baseline boundary.",
        "implementation_source": "PressLight KDD 2019; LibSignal Machine Learning 2024",
        "protocol_contract": "Prefer a maintained reimplementation; verify license, phase mapping, CityFlow fork and metric semantics before using numbers.",
        "decision_role": "required_for_paper_claim_not_minimum_architecture_test",
    },
    {
        "baseline_id": "B05", "name": "CoLight or UniLight", "category": "graph_coordination_rl",
        "model_based": "no_or_auxiliary_prediction", "coordination": "explicit_graph_messages", "transfer": "limited",
        "covers": "Benefits from neighbor communication and graph modeling without using online world-model candidate rollouts.",
        "does_not_cover": "Joint neighbor-action-conditioned future scoring or conservative fallback.",
        "why_required": "Separates a generic graph-network gain from action-conditioned dynamics.",
        "implementation_source": "CoLight CIKM 2019; UniLight IJCAI 2022",
        "protocol_contract": "Do not compare raw UniLight numbers without reconciling its forked CityFlow travel-time calculation.",
        "decision_role": "required_if_neighbor_action_module_survives",
    },
    {
        "baseline_id": "B06", "name": "GPLight", "category": "large_scale_grouped_rl",
        "model_based": "no", "coordination": "dynamic_agent_grouping", "transfer": "within_protocol",
        "covers": "Intersection heterogeneity and scaling to large networks.",
        "does_not_cover": "Learned environment dynamics, uncertainty, or MaxPressure-anchored planning.",
        "why_required": "A paper claiming large-network graph benefits must confront a recent grouped MARL baseline.",
        "implementation_source": "GPLight IJCAI 2023",
        "protocol_contract": "Only required for a large-network claim; the 1,089-intersection protocol is not comparable to Jinan/Hangzhou by raw score.",
        "decision_role": "conditional_large_scale_baseline",
    },
    {
        "baseline_id": "B07", "name": "ModelLight and PLight/PRLight", "category": "direct_tsc_model_based",
        "model_based": "yes", "coordination": "local_or_graph_encoder", "transfer": "meta_or_policy_reuse",
        "covers": "Direct prior art for action-conditioned traffic transition prediction, imaginary data, and cross-flow/roadnet reuse.",
        "does_not_cover": "Calibrated conservative online MPC against MaxPressure.",
        "why_required": "Prevents claiming generic traffic World Model, phase-conditioned dynamics, or transfer as new.",
        "implementation_source": "ModelLight arXiv 2021; PLight/PRLight ESWA 2026",
        "protocol_contract": "Design-level comparison is mandatory; executable baseline only after license/protocol issues are resolved independently.",
        "decision_role": "nearest_method_boundary",
    },
    {
        "baseline_id": "B08", "name": "ADAC", "category": "direct_tsc_pessimistic_model_based",
        "model_based": "yes", "coordination": "finite_mdp_state", "transfer": "offline_data",
        "covers": "Traffic-specific offline model learning and distance-based pessimistic reward shaping.",
        "does_not_cover": "Neural graph rollout, phase-movement semantics, ensemble calibration, or explicit fallback.",
        "why_required": "Nearest direct evidence for pessimism in traffic model-based learning.",
        "implementation_source": "ADAC KDD 2023 and static-audited repository",
        "protocol_contract": "Do not copy unlicensed code; compare mechanisms and, if implemented, reproduce independently on our CityFlow contract.",
        "decision_role": "nearest_conservative_tsc_boundary",
    },
    {
        "baseline_id": "B09", "name": "X-Light and CrossLight", "category": "cross_city_transfer",
        "model_based": "auxiliary_or_no", "coordination": "trajectory_transformer_or_pattern_structure", "transfer": "cross_city",
        "covers": "Recent cross-city zero-shot and offline-to-online TSC transfer.",
        "does_not_cover": "A deployed world-model planner with calibrated rollout uncertainty.",
        "why_required": "Any cross-city/pretraining claim must improve over recent transfer methods, not only single-city controllers.",
        "implementation_source": "X-Light IJCAI 2024; CrossLight KDD 2024",
        "protocol_contract": "Match source/target city partition and adaptation budget; never mix zero-shot with online-adaptation results.",
        "decision_role": "required_only_for_pretraining_or_cross_city_claim",
    },
]


RISKS = [
    {"risk_id":"R01","risk":"Model exploitation","mechanism":"Planner chooses candidates where model error creates falsely high reward.","affected_stage":"planning","severity":"critical","likelihood":"high","detection":"Predicted-versus-CityFlow candidate ranking; per-step regret; visualize selected high-error candidates.","mitigation":"Uncertainty penalty, support gate, advantage threshold, MaxPressure fallback.","kill_condition":"Selected candidate error is positively associated with predicted advantage or ranking correlation is non-positive.","source":"MOPO D07; MOReL D09; ADAC D10"},
    {"risk_id":"R02","risk":"Compounding rollout error","mechanism":"One-step prediction errors accumulate over the planner horizon.","affected_stage":"model_and_planning","severity":"critical","likelihood":"high","detection":"Held-out 1-5 step state/reward error with teacher-forced and free-running curves.","mitigation":"H-step loss, direct multi-step head, shorter adaptive horizon, terminal value.","kill_condition":"Free-running error grows faster than candidate score separation or horizon>1 reduces rank correlation.","source":"MBPO D06; GMAN D22; current training.py/model.py"},
    {"risk_id":"R03","risk":"OOD action sequence","mechanism":"Local mutations or their combinations are rare/absent in the training trajectories.","affected_stage":"candidate_generation","severity":"critical","likelihood":"high","detection":"Action n-gram counts, state-action kNN/density, ensemble error calibration by support bins.","mitigation":"Policy prior/proposal restriction, support threshold, MP fallback.","kill_condition":"No support metric predicts real ranking error better than chance.","source":"MOPO D07; MOReL D09; ADAC D10"},
    {"risk_id":"R04","risk":"Missing phase-movement semantics","mechanism":"An arbitrary action embedding does not encode which turns receive green.","affected_stage":"transition_model","severity":"critical","likelihood":"confirmed_in_current_code","detection":"Movement-level error split by selected permitted versus blocked movements; phase-mask ablation.","mitigation":"Gate action effect with phase_movement_mask and movement/turn embeddings.","kill_condition":"Mask-aware model does not improve held-out movement error or candidate ranking at equal capacity.","source":"ModelLight D13; PressLight D20; current model.py:145-153"},
    {"risk_id":"R05","risk":"Missing neighbor joint actions","mechanism":"Neighbor states are aggregated before actions, so inflow cannot respond to neighbors' selected phases.","affected_stage":"graph_dynamics","severity":"critical","likelihood":"confirmed_in_current_code","detection":"Counterfactual pairs differing only in upstream action; downstream movement error and rank correlation.","mitigation":"Action-conditioned graph messages along road/movement edges.","kill_condition":"Explicit neighbor actions add no predictive/ranking value over neighbor states at matched capacity.","source":"MaCAR D18; UniLight D17; current model.py:118-149"},
    {"risk_id":"R06","risk":"Uncertainty miscalibration","mechanism":"Ensemble variance may not track counterfactual prediction or ranking error.","affected_stage":"safety_gate","severity":"critical","likelihood":"medium_high","detection":"Reliability curve, AUROC for high-regret candidates, Spearman between disagreement and absolute error.","mitigation":"Calibrate on flow-disjoint validation; compare ensemble, kNN support and combined score; COMBO-style alternative if needed.","kill_condition":"Uncertainty fails monotonic calibration on both cities.","source":"PETS D05; MOPO D07; COMBO D08"},
    {"risk_id":"R07","risk":"Validation/test leakage","mechanism":"Flows used for normalization, model selection, uncertainty calibration or policy reuse enter test evaluation.","affected_stage":"data_protocol","severity":"critical","likelihood":"medium","detection":"Manifest hash audit for train/calibration/test roadnet and flow files; nested selection log.","mitigation":"Flow-disjoint train, validation/calibration and final test partitions; freeze checkpoint and thresholds.","kill_condition":"Any test flow or derived statistic influences checkpoint/threshold selection.","source":"research protocol; PLight/PRLight audit"},
    {"risk_id":"R08","risk":"Seed-only pseudo-replication","mechanism":"Changing RNG on one deterministic flow does not create independent traffic demand.","affected_stage":"evaluation","severity":"critical","likelihood":"confirmed_in_first_run","detection":"Count unique flow hashes and demand profiles per paired unit.","mitigation":"Use independent held-out flow files in Jinan and Hangzhou; pair controllers within each flow.","kill_condition":"Fewer than the preregistered independent flows per city are available.","source":"current 5090 run and locked protocol"},
    {"risk_id":"R09","risk":"Switching/yellow-time cost mismatch","mechanism":"Predicted action value ignores lost green time or minimum-green constraints.","affected_stage":"model_and_environment_contract","severity":"high","likelihood":"medium","detection":"Error/regret by change-versus-hold action; audit phase_elapsed, yellow and all-red transitions.","mitigation":"Keep timing in environment contract; include elapsed/stage and explicit change cost in targets.","kill_condition":"Planner improves predicted reward by excessive switching while real queue/wait degrades.","source":"current model.py:183-203; policy.py:134-138"},
    {"risk_id":"R10","risk":"Search-budget confound","mechanism":"More candidate evaluations, not a better model, create apparent gains.","affected_stage":"experiment","severity":"high","likelihood":"medium","detection":"Equal candidate count/horizon/wall-clock variants; oracle and random-score controls.","mitigation":"Hold proposal set fixed across ablations and report compute.","kill_condition":"Gain disappears at matched candidate set/search budget.","source":"minimum experiment requirement"},
    {"risk_id":"R11","risk":"Cross-city schema and dynamics shift","mechanism":"Road topology, phase count, demand and driver dynamics differ across cities.","affected_stage":"pretraining_transfer","severity":"high","likelihood":"high","detection":"Per-city error/ranking breakdown, schema-mask coverage, negative-transfer tests.","mitigation":"Padded schema masks, city-held-out validation, local calibration or adapters.","kill_condition":"Pretraining hurts either city or gains vanish after equal within-city data budget.","source":"TrajWorld D03; X-Light D15; CrossLight D16; UniST D24"},
    {"risk_id":"R12","risk":"Metric/simulator semantic mismatch","mechanism":"CityFlow forks or different travel-time/phase conventions make numbers incomparable.","affected_stage":"baseline_comparison","severity":"critical","likelihood":"high_for_external_numbers","detection":"Source SHA, simulator SHA, roadnet/flow hashes, metric definition and signal-timing manifest.","mitigation":"Run all executable baselines in one environment contract; use literature numbers only as context.","kill_condition":"Baseline cannot be brought under the same simulator and metric contract.","source":"UniLight D17 warning; LibSignal D25"},
]


TRANSLATIONS = [
    {"translation_id":"T01","source_papers":"D13;D20","source_mechanism":"Phase vector / pressure-grounded movement semantics","current_limitation":"Action id embedding is broadcast to movements; phase_movement_mask is not consumed.","current_code":"model.py:65-77,145-153; data.py:106-108","proposed_location":"GraphWorldModel action-to-movement conditioning","interface_change":"Add phase_movement_mask buffer/tensor to model construction/checkpoint and select per-action movement gates.","dependency":"Existing NetworkSpec.padded_phase_movement_mask","expected_signal":"Lower permitted/blocked movement error and better candidate rank correlation.","non_transferable":"ModelLight single-intersection LSTM code and PressLight controller are not copied.","priority":"1"},
    {"translation_id":"T02","source_papers":"D17;D18","source_mechanism":"Route predicted outflow to neighbors and condition on synchronous joint actions","current_limitation":"Neighbor node state is aggregated before local action injection; neighbor actions cannot affect downstream predictions.","current_code":"model.py:118-149","proposed_location":"Action-conditioned edge/message block before transition head","interface_change":"Gather neighbor action embeddings and edge movement mappings; keep [B,N] discrete action contract.","dependency":"neighbor_index, neighbor_mask, road/movement topology","expected_signal":"Counterfactual upstream-action changes improve downstream prediction/ranking.","non_transferable":"UniLight custom CityFlow metric implementation and MaCAR full RL controller.","priority":"1"},
    {"translation_id":"T03","source_papers":"D01;D02;D06","source_mechanism":"H-step consistency/reward learning aligned with deployment rollout","current_limitation":"Training is one-step MSE while planning recursively rolls H=3.","current_code":"training.py:42-67; model.py:159-205","proposed_location":"TrajectoryTransitionDataset sequence mode and multi-step loss","interface_change":"Dataset yields H-step observations/actions/rewards/masks; maintain one-step compatibility only in a later implementation plan.","dependency":"Contiguous trajectories and episode boundaries","expected_signal":"Lower free-running 2-5 step error and improved candidate ranking.","non_transferable":"Dreamer latent actor-critic stack.","priority":"1"},
    {"translation_id":"T04","source_papers":"D01;D02","source_mechanism":"Terminal value and continuation heads","current_limitation":"Planner truncates reward at horizon without estimating value beyond it.","current_code":"model.py:73-77,159-205","proposed_location":"Value/continuation heads and rollout terminal score","interface_change":"Checkpoint schema and training targets add value/continuation; planner score adds discounted terminal V.","dependency":"Stable value target or pressure-derived bootstrap","expected_signal":"Horizon-robust candidate ranking without increasing rollout length.","non_transferable":"Continuous policy prior until a discrete prior is justified.","priority":"2"},
    {"translation_id":"T05","source_papers":"D05;D07","source_mechanism":"Bootstrapped dynamics ensemble and lower-confidence reward","current_limitation":"Single deterministic model emits no epistemic signal.","current_code":"model.py and checkpoint.py","proposed_location":"Ensemble wrapper over independently initialized GraphWorldModel members","interface_change":"Checkpoint stores member states and planner receives mean/std per candidate.","dependency":"Bootstrap resampling and flow-disjoint calibration set","expected_signal":"Disagreement monotonically predicts real candidate error/regret.","non_transferable":"Gaussian continuous-state assumptions and continuous CEM.","priority":"2"},
    {"translation_id":"T06","source_papers":"D09;D10","source_mechanism":"Unknown/support detection and pessimistic penalty","current_limitation":"Any valid phase mutation can be selected regardless of data support.","current_code":"policy.py:76-155; data.py","proposed_location":"Candidate diagnostics and conservative gate","interface_change":"Planner gets support score per state-action sequence and calibrated penalty/threshold.","dependency":"Training-trajectory index or density estimator","expected_signal":"High support corresponds to lower ranking error; gate reduces worst-flow regret.","non_transferable":"ADAC unlicensed finite-MDP code.","priority":"2"},
    {"translation_id":"T07","source_papers":"D08","source_mechanism":"Value conservatism when uncertainty is unreliable","current_limitation":"No learned value function and no alternative if ensemble calibration fails.","current_code":"future only","proposed_location":"Deferred conservative value baseline","interface_change":"Would require synthetic replay and Q/value training; outside minimum skeleton.","dependency":"Failure of simpler calibrated ensemble/support gate","expected_signal":"Out-of-support candidates receive lower value without explicit variance.","non_transferable":"Full COMBO implementation.","priority":"defer"},
    {"translation_id":"T08","source_papers":"D22","source_mechanism":"Direct multi-step sequence prediction","current_limitation":"Recursive rollout may propagate error.","current_code":"model.py:159-205","proposed_location":"Optional direct H-step dynamics head for diagnostic comparison","interface_change":"Model outputs [B,H,N,M,F] conditioned on the whole action sequence.","dependency":"Sequence dataset","expected_signal":"Direct head improves H-step error/ranking over recursive model at matched capacity.","non_transferable":"GMAN lacks action conditioning and cannot be used as controller.","priority":"2"},
    {"translation_id":"T09","source_papers":"D23","source_mechanism":"Physics-guided dynamics","current_limitation":"No explicit conservation constraint despite movement counts.","current_code":"model.py transition_head","proposed_location":"Movement inflow/outflow residual or conservation regularizer","interface_change":"Need incoming/outgoing movement-edge incidence and source/sink masks.","dependency":"Reliable topology mapping from roadnet","expected_signal":"Lower physically impossible predictions and better cross-demand generalization.","non_transferable":"STDEN latent-potential differential equation is not action-conditioned.","priority":"defer_after_T01_T02"},
    {"translation_id":"T10","source_papers":"D03;D24","source_mechanism":"Heterogeneous schema masks and multi-environment pretraining","current_limitation":"Checkpoint is tied to one network schema hash.","current_code":"checkpoint.py; data.py; policy.py:58-66","proposed_location":"Future shared backbone plus schema-specific masks/adapters","interface_change":"Observation/trajectory/checkpoint schemas must carry feature, movement, phase, roadnet and city tokens.","dependency":"Within-city mechanism already passes performance gate","expected_signal":"Positive transfer at equal target-city data budget.","non_transferable":"Unlicensed TrajWorld/UniST implementation and action-free UniST objective.","priority":"defer"},
    {"translation_id":"T11","source_papers":"D15;D16","source_mechanism":"Cross-city trajectory meta-training","current_limitation":"Only one city/flow was used in the diagnostic run.","current_code":"train_world_model.py orchestration and future dataset registry","proposed_location":"Experiment protocol, not first model class","interface_change":"Explicit source-city, calibration-flow and held-out-city manifests.","dependency":"Multiple independent flows and cities","expected_signal":"Zero-shot/adaptation gains relative to X-Light/CrossLight class baselines.","non_transferable":"Their controller architectures do not demonstrate world-model MPC gains.","priority":"defer"},
    {"translation_id":"T12","source_papers":"D25;D26","source_mechanism":"Unified evaluation and theory-grounded simple-state controls","current_limitation":"One deterministic flow and no strong learned pressure baseline in the initial run.","current_code":"experiment scripts and metrics, no public model API change","proposed_location":"Evaluation harness and preregistration","interface_change":"Flow manifests, paired metrics, bootstrap CI, source SHA, compute accounting.","dependency":"Independent Jinan/Hangzhou flows","expected_signal":"Results can be attributed to model mechanism rather than protocol or complexity.","non_transferable":"GPL library code cannot enter MIT core; raw external metrics are contextual only.","priority":"1"},
]


GAPS = [
    {"gap_id":"G01","candidate_gap":"Within the audited set, no evaluated method establishes that a structured TSC MPC model combining explicit phase-to-movement gating and synchronous neighbor-action messages improves candidate ranking over the same MaxPressure proposals.","nearest_methods":"ModelLight D13 uses a local phase vector; UniLight D17 predicts neighbor impact; MaCAR D18 predicts traffic under synchronous actions; PressLight D20 supplies movement/pressure semantics.","not_a_gap":"Using action-conditioned traffic prediction, graph communication, or phase representations individually is prior art.","evidence_boundary":"Audited 26-paper set and 8 repositories only; this is not a global novelty claim.","linked_risks":"R04;R05;R12","falsification":"At matched parameter count and candidate set, phase mask plus neighbor action must improve held-out movement error and CityFlow candidate Spearman/Kendall in both cities.","death_condition":"No ranking improvement on either city, or gains disappear when generic extra-capacity controls are matched.","fallback":"Keep only phase_movement_mask gating if it improves movement predictions; otherwise abandon action-semantic architecture.","diagnosis_fit":"5.0","formal_empirical_evidence":"4.0","direct_tsc_evidence":"4.5","cityflow_compatibility":"4.5","novelty_delta_potential":"3.0","code_and_license":"3.5","compute_feasibility":"4.5","weighted_score":"4.23","lint_status":"pass_checkpoint_c_2026-08-13"},
    {"gap_id":"G02","candidate_gap":"Current one-step training is misaligned with recursive multi-step ranking; the audited TSC methods do not establish a MaxPressure-anchored comparison of H-step consistency plus terminal value under identical candidates.","nearest_methods":"TD-MPC2 D01 and DreamerV3 D02 train multi-step/value components; MBPO D06 limits model horizon; image TSC World Model D14 uses multi-step imagination; GMAN D22 predicts future sequences directly.","not_a_gap":"Multi-step world-model training and terminal value are established general mechanisms, and multi-step TSC imagination already exists.","evidence_boundary":"Application/mechanism-integration question, not an algorithmic novelty claim by itself.","linked_risks":"R01;R02;R09;R10","falsification":"H-step loss and terminal value must improve free-running error and real candidate ranking at equal horizon/search budget, with value removed in ablation.","death_condition":"Validation loss improves but rank correlation/regret does not, or longer horizons worsen both cities.","fallback":"Use horizon 1 with a calibrated one-step advantage model; do not claim long-horizon planning.","diagnosis_fit":"4.5","formal_empirical_evidence":"5.0","direct_tsc_evidence":"3.0","cityflow_compatibility":"4.0","novelty_delta_potential":"2.5","code_and_license":"4.0","compute_feasibility":"3.5","weighted_score":"3.88","lint_status":"pass_checkpoint_c_2026-08-13"},
    {"gap_id":"G03","candidate_gap":"The audited direct TSC model-based methods do not establish a calibrated lower-confidence candidate advantage with an explicit MaxPressure fallback and worst-flow degradation bound.","nearest_methods":"PETS D05 supplies ensemble dynamics; MOPO D07 penalizes uncertainty; COMBO D08 questions explicit uncertainty; MOReL D09 and ADAC D10 provide support/pessimistic alternatives.","not_a_gap":"Ensemble uncertainty, pessimism, support gates, and fallback/shield concepts are established outside or adjacent to TSC.","evidence_boundary":"The possible contribution is a traffic-specific calibrated decision contract and evidence, not inventing conservative MBRL.","linked_risks":"R01;R03;R06;R08","falsification":"On flow-disjoint calibration/test, uncertainty or support must monotonically predict error/regret and fallback must cap the worst-flow travel-time degradation at 1%.","death_condition":"Calibration fails on either city, fallback rejects nearly all improvements, or worst-flow bound is violated.","fallback":"Use deterministic advantage threshold calibrated on real counterfactual rollouts; if still unsafe, always use MaxPressure.","diagnosis_fit":"4.0","formal_empirical_evidence":"5.0","direct_tsc_evidence":"3.5","cityflow_compatibility":"4.0","novelty_delta_potential":"3.0","code_and_license":"3.5","compute_feasibility":"3.5","weighted_score":"3.85","lint_status":"pass_checkpoint_c_2026-08-13"},
    {"gap_id":"G04","candidate_gap":"It is unestablished whether a shared action-semantic traffic dynamics model pretrained across independent flows, roadnets and cities improves MaxPressure-anchored planning at equal target-city data budget.","nearest_methods":"TrajWorld D03 and UniST D24 pretrain heterogeneous predictors; X-Light D15 and CrossLight D16 transfer TSC policies; PLight/PRLight D12 transfer via environmental similarity.","not_a_gap":"Cross-environment pretraining, cross-city TSC transfer, and policy reuse are established.","evidence_boundary":"Only relevant after within-city dynamics and ranking are validated; currently a deferred high-risk extension.","linked_risks":"R07;R08;R11;R12","falsification":"Pretraining must improve held-out target-city ranking/control at equal target data and compute, with no negative transfer in either city.","death_condition":"No equal-budget gain, city-specific harm, or benefit disappears after local calibration/data matching.","fallback":"Use per-city models with shared feature definitions only.","diagnosis_fit":"2.5","formal_empirical_evidence":"4.5","direct_tsc_evidence":"4.0","cityflow_compatibility":"2.5","novelty_delta_potential":"3.0","code_and_license":"2.0","compute_feasibility":"2.0","weighted_score":"3.03","lint_status":"pass_checkpoint_c_2026-08-13"},
]


def build_markdown() -> None:
    (ROOT / "saturation_report.md").write_text(
        "# Evidence saturation report\n\n"
        "The discovery pool contains 464 deduplicated OpenAlex candidates plus targeted primary-source follow-up. "
        "The 26-paper deep set covers all locked layers: general World Models, direct TSC model-based methods, graph traffic dynamics, "
        "multi-agent action coordination, cross-city transfer, uncertainty/pessimism, strong baselines, and negative/protocol evidence.\n\n"
        "Saturation was judged at the mechanism level, not by citation count. Citation chasing repeatedly returned the same ancestors: "
        "PETS/MBPO for model use, MOPO/MOReL/COMBO for pessimism, PressLight/CoLight for TSC baselines, and ModelLight/ADAC/UniLight/MaCAR "
        "for direct traffic mechanisms. Three missed papers remain explicit in `missed_papers.csv`; they can narrow a future claim but do not "
        "leave a required current layer empty.\n\n"
        "This does not prove global novelty. A venue submission requires a final 2026 search refresh around the surviving mechanism and its exact wording.\n",
        encoding="utf-8",
    )
    lines = ["# Gap cards", ""]
    for gap in GAPS:
        lines.extend([
            f"## {gap['gap_id']}", "", gap["candidate_gap"], "",
            f"- Nearest methods: {gap['nearest_methods']}",
            f"- Not a gap: {gap['not_a_gap']}",
            f"- Evidence boundary: {gap['evidence_boundary']}",
            f"- Falsification: {gap['falsification']}",
            f"- Death condition: {gap['death_condition']}",
            f"- Fallback: {gap['fallback']}",
            f"- Weighted score: {gap['weighted_score']}/5", "",
        ])
    (ROOT / "gap_cards.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "risk_detection_plan.md").write_text(
        "# Risk detection plan\n\n"
        "Every critical risk in `risk_matrix.csv` is tied to an observable diagnostic and a kill condition. "
        "The order is: validate flow-disjoint data manifests; measure 1-5 step errors; measure counterfactual candidate ranking; "
        "calibrate support/uncertainty; then run controller comparisons. Control metrics without these diagnostics cannot establish "
        "a World Model mechanism. External-paper numbers are contextual only until rerun under one simulator contract.\n",
        encoding="utf-8",
    )


def main() -> None:
    baseline_fields = list(BASELINES[0])
    write_csv("prelim_baseline_map.csv", baseline_fields, BASELINES)
    write_csv("baseline_boundary_matrix.csv", baseline_fields, BASELINES)
    risk_fields = list(RISKS[0])
    write_csv("prelim_risk_map.csv", risk_fields, RISKS)
    write_csv("risk_matrix.csv", risk_fields, RISKS)
    write_csv("translation_matrix.csv", list(TRANSLATIONS[0]), TRANSLATIONS)
    write_csv("gap_matrix.csv", list(GAPS[0]), GAPS)
    build_markdown()
    print(f"wrote {len(BASELINES)} baselines, {len(RISKS)} risks, {len(TRANSLATIONS)} translations, {len(GAPS)} gaps")


if __name__ == "__main__":
    main()
