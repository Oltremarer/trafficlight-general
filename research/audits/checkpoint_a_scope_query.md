# Checkpoint A — scope and query audit

reviewer: Codex internal adversarial audit
evidence_status: review_comment_not_evidence
date: 2026-08-13
verdict: pass

## Fatal objections

None. The protocol keeps one primary operator axis: action-conditioned dynamics
and conservative planning relative to MaxPressure.

## Major objections checked

- Trend bias: queries include classic baselines, risk, negative, benchmark, and
  implementation searches in addition to recent World Models.
- Fake-gap risk: traffic forecasting, video generation, policy reuse, and full
  World Models are explicitly separated.
- Evidence leakage: snippets and model-generated statements cannot support
  gaps; formal status and code identity are separate fields.
- Scope inflation: implementation and external reproduction are out of scope.

## Minor objections

- Cross-city and foundation-model queries may produce many low-value preprints;
  formal status and deep-read thresholds must be enforced.
- The 2% future performance threshold is a project decision, not a literature
  fact, and must remain labeled as an acceptance rule.

## Missing evidence

- Candidate and code evidence will be populated at L2-L6.
- Current action-ranking quality is unknown; this becomes a required kill test.

## Required revision tasks

None before L2. Do not create `idea_pool.csv` until L10 passes.

## Do not proceed until

The query table continues to include negative-result, strong-baseline,
benchmark, and implementation searches. This condition is satisfied.
