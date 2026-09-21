# Source provenance

## Independent RL TrafficLight copy (2026-09-21)

This repository was initialized from the working files in
`/Users/azure/Documents/ChatGPT/world model traffic`, including its uncommitted
source/test changes. Source, tests, documentation, research, and scripts were
copied; `.git`, virtual environments, caches, and experiment output were excluded.
The initial copy is committed locally as `c02a285`. This repository has independent
Git metadata and an independent Python environment.

The baseline implementations are local PyTorch architecture/learning-rule ports,
informed by these audited immutable revisions:

- LLMTSCS: `d5d4180f34edb843e1d1b462d5846c75d6d4533a`.
- LibSignal: `127af9f93902778e556de2eedb2b606c4c9447e6`.
- cMALC-D: `93a9d9f60f77153c25fbac739ad7615e194ad7cb`.

No complete upstream simulator/training framework or pretrained weights were
vendored. Source-specific feature and reward choices, graph construction, local
training adaptations, and limitations are documented in `docs/baselines.md` and
serialized in every baseline run's profile/config. Same algorithm names do not
assert numerical equivalence to upstream training or published results.

## Foundation provenance retained from the source workspace

The CityFlow experiment foundation in this repository was imported from
`Oltremarer/Cityflow` branch `codex/cityflow-foundation` at commit:

```text
d39d6d0252fca9a4b00663b342b7bded7a7f25aa
```

The original workspace retained that import's commit history; this independent
repository starts from a new working-file snapshot. Historical LLMLight
reproduction tools and patches were deliberately excluded because they are not
part of the World Model traffic-signal implementation.
