# Idea cards

These are testable research candidates, not established innovations.

## I01 — MaxPressure-Guarded Action-Semantic Graph World Model

Status: `recommended_main_route`

- Evidence gaps: G01;G02;G03
- Nearest methods: ModelLight (phase vector), UniLight (predicted neighbor impact), MaCAR (synchronous-action traffic prediction), TD-MPC2 (multi-step/value), PETS/MOPO/ADAC (uncertainty/pessimism).
- Method delta: Use the selected phase_movement_mask to gate movement updates; send selected neighbor actions along directed road/movement edges; train 1-H step state/reward targets; score candidates by ensemble lower-confidence advantage over the identical MaxPressure proposal; execute MaxPressure unless the calibrated advantage exceeds a threshold.
- Why it may beat MaxPressure: MaxPressure reacts to current queues. The proposed model can anticipate an upstream phase creating downstream inflow, but is allowed to deviate only when its predicted benefit is supported and sufficiently certain.
- Minimum experiment: On flow-disjoint Jinan and Hangzhou states, compare current model, capacity-matched control, phase-mask model, phase-mask+neighbor-action model, H-step model, and guarded ensemble using one fixed candidate set; then run paired controller evaluation.
- Ablation: phase_movement_mask; neighbor actions; H-step loss; terminal value; ensemble penalty; support score; advantage threshold; fallback; equal-capacity and equal-search controls.
- Death condition: Reject as the main route if action semantics do not improve candidate ranking in both cities, or if the complete guarded planner cannot achieve >=2% travel-time improvement with the preregistered CI/guardrails.
- Fallback: Retain only the smallest module that improves held-out candidate ranking; if none do, return to MaxPressure and abandon learned rollout planning.
- Novelty boundary: This combination is a working hypothesis inside the audited set, not a confirmed global novelty claim. Every individual mechanism has prior art.

## I02 — Direct H-Step Candidate Ranker

Status: `fallback_if_recursive_dynamics_fails`

- Evidence gaps: G02
- Nearest methods: GMAN direct future sequence prediction, action-conditioned World Models, COMBO conservative value learning.
- Method delta: Condition on current structured traffic state and a complete discrete joint-action sequence, then directly predict H-step cumulative return or pairwise advantage relative to MaxPressure instead of recursively predicting every next state.
- Why it may beat MaxPressure: It targets the planner's actual decision—candidate ordering—and avoids compounding unused state dimensions, while MaxPressure remains the proposal and fallback.
- Minimum experiment: Use the same counterfactual candidate-ranking dataset and compare rank correlation/regret to the recursive World Model at matched parameters and candidates.
- Ablation: return regression versus pairwise ranking loss; with/without movement mask; horizon 1/3/5; terminal continuation target.
- Death condition: Reject if it does not improve held-out rank correlation/regret on both cities or if it overfits candidate generator patterns.
- Fallback: Use a one-step advantage estimator or MaxPressure only.
- Novelty boundary: A direct ranker is not a full generative World Model and weakens the World Model paper story; use only if recursive dynamics is empirically the wrong tool.

## I03 — Support-Calibrated Ensemble Gate

Status: `conditional_safety_addon`

- Evidence gaps: G03
- Nearest methods: PETS, MOPO, COMBO, MOReL, ADAC.
- Method delta: Calibrate ensemble disagreement and trajectory support against real CityFlow candidate error, combine them into a lower-confidence advantage, and expose a deterministic MaxPressure fallback with a worst-flow degradation budget.
- Why it may beat MaxPressure: It cannot create predictive skill by itself, but it can preserve rare reliable anticipatory improvements while filtering model exploitation that made the unguarded planner worse.
- Minimum experiment: Measure reliability curves and high-regret detection AUROC on flow-disjoint calibration/test sets; sweep one frozen threshold and report coverage-risk curves before control results.
- Ablation: ensemble only; kNN/support only; combined score; advantage threshold only; fallback only; COMBO-style value alternative if variance is uncalibrated.
- Death condition: Reject ensemble-based gating if disagreement/support is not monotonic with error on both cities or if safe coverage is too low to yield control gains.
- Fallback: Calibrated deterministic advantage threshold; ultimately always MaxPressure.
- Novelty boundary: Conservative uncertainty and fallback are established mechanisms; any contribution must be traffic-specific calibration and outcome evidence.

## I04 — Heterogeneous Traffic-Dynamics Pretraining

Status: `deferred_extension`

- Evidence gaps: G04
- Nearest methods: TrajWorld, UniST, X-Light, CrossLight, PLight/PRLight.
- Method delta: Pretrain a masked movement/action dynamics backbone across independent flows, roadnets and cities, then adapt only small schema-specific components.
- Why it may beat MaxPressure: Broader dynamics data may improve rare-demand and unseen-city prediction, but only if the learned action semantics already rank candidates correctly.
- Minimum experiment: After I01 passes, compare per-city training versus cross-city pretraining at identical target-city data and compute; hold out one city/flow family.
- Ablation: flow diversity; roadnet diversity; city tokens; schema masks; local adapter; equal-data and equal-compute controls.
- Death condition: Reject if equal-budget target-city ranking/control does not improve or if either city suffers negative transfer.
- Fallback: Per-city models with a shared data schema but no shared weights.
- Novelty boundary: Pretraining and cross-city TSC transfer are established; this cannot be the first experiment or a generic novelty claim.
