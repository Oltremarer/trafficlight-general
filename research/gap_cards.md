# Gap cards

## G01

Within the audited set, no evaluated method establishes that a structured TSC MPC model combining explicit phase-to-movement gating and synchronous neighbor-action messages improves candidate ranking over the same MaxPressure proposals.

- Nearest methods: ModelLight D13 uses a local phase vector; UniLight D17 predicts neighbor impact; MaCAR D18 predicts traffic under synchronous actions; PressLight D20 supplies movement/pressure semantics.
- Not a gap: Using action-conditioned traffic prediction, graph communication, or phase representations individually is prior art.
- Evidence boundary: Audited 26-paper set and 8 repositories only; this is not a global novelty claim.
- Falsification: At matched parameter count and candidate set, phase mask plus neighbor action must improve held-out movement error and CityFlow candidate Spearman/Kendall in both cities.
- Death condition: No ranking improvement on either city, or gains disappear when generic extra-capacity controls are matched.
- Fallback: Keep only phase_movement_mask gating if it improves movement predictions; otherwise abandon action-semantic architecture.
- Weighted score: 4.23/5

## G02

Current one-step training is misaligned with recursive multi-step ranking; the audited TSC methods do not establish a MaxPressure-anchored comparison of H-step consistency plus terminal value under identical candidates.

- Nearest methods: TD-MPC2 D01 and DreamerV3 D02 train multi-step/value components; MBPO D06 limits model horizon; image TSC World Model D14 uses multi-step imagination; GMAN D22 predicts future sequences directly.
- Not a gap: Multi-step world-model training and terminal value are established general mechanisms, and multi-step TSC imagination already exists.
- Evidence boundary: Application/mechanism-integration question, not an algorithmic novelty claim by itself.
- Falsification: H-step loss and terminal value must improve free-running error and real candidate ranking at equal horizon/search budget, with value removed in ablation.
- Death condition: Validation loss improves but rank correlation/regret does not, or longer horizons worsen both cities.
- Fallback: Use horizon 1 with a calibrated one-step advantage model; do not claim long-horizon planning.
- Weighted score: 3.88/5

## G03

The audited direct TSC model-based methods do not establish a calibrated lower-confidence candidate advantage with an explicit MaxPressure fallback and worst-flow degradation bound.

- Nearest methods: PETS D05 supplies ensemble dynamics; MOPO D07 penalizes uncertainty; COMBO D08 questions explicit uncertainty; MOReL D09 and ADAC D10 provide support/pessimistic alternatives.
- Not a gap: Ensemble uncertainty, pessimism, support gates, and fallback/shield concepts are established outside or adjacent to TSC.
- Evidence boundary: The possible contribution is a traffic-specific calibrated decision contract and evidence, not inventing conservative MBRL.
- Falsification: On flow-disjoint calibration/test, uncertainty or support must monotonically predict error/regret and fallback must cap the worst-flow travel-time degradation at 1%.
- Death condition: Calibration fails on either city, fallback rejects nearly all improvements, or worst-flow bound is violated.
- Fallback: Use deterministic advantage threshold calibrated on real counterfactual rollouts; if still unsafe, always use MaxPressure.
- Weighted score: 3.85/5

## G04

It is unestablished whether a shared action-semantic traffic dynamics model pretrained across independent flows, roadnets and cities improves MaxPressure-anchored planning at equal target-city data budget.

- Nearest methods: TrajWorld D03 and UniST D24 pretrain heterogeneous predictors; X-Light D15 and CrossLight D16 transfer TSC policies; PLight/PRLight D12 transfer via environmental similarity.
- Not a gap: Cross-environment pretraining, cross-city TSC transfer, and policy reuse are established.
- Evidence boundary: Only relevant after within-city dynamics and ranking are validated; currently a deferred high-risk extension.
- Falsification: Pretraining must improve held-out target-city ranking/control at equal target data and compute, with no negative transfer in either city.
- Death condition: No equal-budget gain, city-specific harm, or benefit disappears after local calibration/data matching.
- Fallback: Use per-city models with shared feature definitions only.
- Weighted score: 3.03/5
