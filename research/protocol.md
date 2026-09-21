# Research protocol

## Decision question

Which evidence-backed mechanism can make a structured-state CityFlow World
Model rank discrete multi-intersection signal plans better than MaxPressure on
held-out demand, without introducing unacceptable worst-flow degradation?

## Locked current diagnosis

The first RTX 5090 run proves execution feasibility but not a performance gain.
It used one deterministic Jinan flow, so repeated evaluation seeds are not
independent traffic conditions. The World Model changed 909 of 1440 actions
relative to MaxPressure but increased travel time by 0.56% and average queue by
2.41%.

Static inspection establishes four implementation limitations:

1. `phase_movement_mask` is persisted and checked but is not consumed by the
   transition or reward model.
2. Neighbor state aggregation happens before action injection; a node cannot
   explicitly condition on its neighbors' selected signal actions.
3. Training is one-step state/reward MSE while deployment recursively rolls the
   model for three steps.
4. The planner has no epistemic uncertainty, candidate-support test, terminal
   value, or conservative fallback to MaxPressure.

These observations define search axes; they do not establish novelty or prove
causality for the performance result.

## Evidence admission

- Formal publication claims require an official proceedings/journal page or
  paper PDF. Search-result snippets are discovery evidence only.
- Code claims require an exact repository, branch/default branch, file path,
  and static code pointer.
- A repository is "official" only when linked by the paper/project/author or
  explicitly identifies itself as official with matching authorship.
- GitHub Stars are dated attention metadata, never reproducibility evidence.
- Workshop, preprint, thesis, and formal main-track papers are separate statuses.
- Numeric result claims retain dataset, metric, protocol, table/figure pointer,
  seeds/variance, and negative results.

## Stage gates

- L0-L1: scope/query protocol must cover method, baseline, risk, negative,
  benchmark, survey, and implementation searches.
- L2-L4: deep-read set must cover all six research layers and include strong
  baselines and risk papers, not only World Model papers.
- L5-L6: every deep paper needs mechanism, baseline, limitation, at least three
  atomic claims, and citation-neighbor checks.
- L7-L10: no gap passes without a nearest strong baseline, linked risks, and a
  falsification route.
- L11: `idea_pool.csv` is created only after the gap lint passes.
- L12: the final recommendation must include a killable minimum experiment and
  explicit stop conditions.

## Mechanism scoring

Each mechanism is scored from 0 to 5 and weighted as follows: direct diagnosis
fit 25%, formal empirical evidence 15%, direct TSC evidence 15%, CityFlow
compatibility 15%, novelty delta 15%, code/license readiness 10%, and compute
risk 5%. Missing nearest baseline, code/evidence pointer, or death condition is
an automatic rejection regardless of score.

## Static-code boundary

External repositories are read through GitHub APIs or raw file views only. No
external repository is installed or executed. Unlicensed code can inform an
abstract mechanism but cannot be copied. Current project code is not changed
during research.
