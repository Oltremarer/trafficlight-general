# Checkpoint D: adversarial idea review

Status: CONDITIONAL PASS FOR MINIMUM EXPERIMENT ONLY

## Strongest surviving candidate

I01 survives because it directly targets four confirmed implementation limitations, can reuse the present CityFlow trajectory/planner chain, and has a staged experiment that can kill each mechanism independently. It is not admitted as a novelty claim: phase vectors, graph communication, synchronous action prediction, multi-step/value learning and pessimistic planning all have prior art.

## Reviewer attacks that must be answered experimentally

1. A larger network, not action semantics, caused the gain. Required answer: M0-cap parameter-matched control.
2. Lower state MSE does not imply better decisions. Required answer: CityFlow candidate Spearman/Kendall and true regret.
3. More search caused the gain. Required answer: byte-identical candidate tensors and equal 64×3 budget.
4. Ensemble variance is uncalibrated. Required answer: coverage-risk and true-error/regret calibration on a flow-disjoint set.
5. Fallback hides a weak learned controller. Required answer: acceptance coverage, accepted-only regret, P0/P1/P2/P3 ablation and worst-flow behavior.
6. Results are demand overfitting. Required answer: independent Jinan and Hangzhou flows, with flow-blocked paired statistics.
7. The contribution is a bundle of known parts. Required answer: demonstrate a traffic-specific mechanism result that nearest direct baselines and generic capacity controls do not reproduce, then refresh novelty evidence before writing.

## Decision

Authorize only E0–E4 in `minimum_experiment_plan.md`. Do not authorize heterogeneous pretraining, video models, unconstrained joint-action search or a DreamerV3 rewrite until I01 passes both-city ranking and control thresholds.
