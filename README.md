# RL TrafficLight

This is an independent copy of the traffic World Model workspace, extended with
P1–P3 RL baseline interfaces. The original workspace is not modified. The Python
package stays `cityflow_tsc` for compatibility; the distribution and new command
are `rl-trafficlight`.

The new baseline path supports PressLight/E-PressLight, MPLight/E/A-MPLight,
CoLight/E/A-CoLight, IDQN/shared DQN, FRAP, additional LibSignal-style MPLight/CoLight
profiles, IPPO/MAPPO, and discrete-phase MADDPG. These are **local PyTorch ports**
with explicit observation/reward profiles and implementation differences, not
upstream-runtime or paper-result reproductions. There are 16 selectable profiles.

```bash
python -m pip install -e '.[rl,dev]'
rl-trafficlight list
```

Use an environment with a working CityFlow installation for actual traffic runs.
CityFlow is not installed automatically. See [baseline usage and implementation
boundaries](docs/baselines.md) for training, checkpoint evaluation, resume, data
schemas, provenance, and the local validation command. Dynamic durations, offline
DiffLight and cross-city adaptation are P4/P5 and are not implemented here.

The original World Model, rule-policy and shared-DQN entry points remain below.

## Original World Model foundation

This project adds an action-conditioned graph World Model to a CityFlow-first traffic signal control foundation. It keeps simulator lifecycle, traffic semantics, model input, policy decisions, trajectory persistence, and evaluation metrics separate so that the World Model and every baseline use the same execution contract.

The project does **not** depend on LLMLight or an LLM runtime.

## Core data flow

```text
roadnet + flow + timing configuration
                |
                v
       stable NetworkSpec
                |
                v
 CityFlowBackend -> NetworkSnapshot -> ObservationBuilder
                                            |
                                            v
                                          Policy
                                            |
                                            v
 CityFlowBackend <- PhaseController <- ActionBatch
        |
        +-> reward, metrics, and mask-preserving trajectory
```

The World Model path reuses that trajectory and policy boundary:

```text
complete CityFlow episodes
          |
          v
episode-level train/validation split -> graph World Model training
                                              |
                                              v
current observation -> MaxPressure proposal -> valid local candidates
                                              |
                                              v
                         imagined multi-step returns -> first joint action
                                              |
                                              v
                                  unchanged TrafficEnv and CityFlowBackend
```

The model predicts only four primitive quantities: incoming/outgoing vehicle and queue counts. Vehicle pressure and queue pressure remain derived quantities, so an imagined state cannot contain counts and pressures that contradict each other.

The first public contract uses a whole-network observation and a whole-network action batch. Policies never receive the raw CityFlow engine and do not implement yellow-light timing themselves.

Episode duration, decision interval, yellow time, and all-red time must align with the configured simulator step. Invalid timing is rejected before an episode starts, and every backend step is checked for monotonic simulator time.

## Implemented foundation

- Stable roadnet parsing for intersections, lanes, road-link movements, phases, and directional neighbors.
- Dense movement observations with explicit movement, validity, neighbor, and action masks.
- Explicit signal stage (`green`, `yellow`, or `all-red`) so partial episodes do not corrupt phase semantics.
- CityFlow lifecycle isolated behind a narrow backend.
- Environment-owned green/yellow/all-red timing and action validation.
- Random, fixed-time, and queue-based MaxPressure policies.
- A minimal shared-parameter DQN baseline with replay, target network, checkpointing, and deterministic evaluation.
- Per-intersection queue reward, simulator-tick metrics, and full-episode runner.
- NPZ trajectory storage with a versioned JSON manifest and SHA-256 provenance.
- Deterministic action-conditioned graph World Model with masked movement encoding, neighbor message passing, next-state and reward heads.
- Short-horizon candidate planner that uses MaxPressure as the proposal and changes only valid intersection actions.
- Episode-level offline data splitting and checkpoint binding to the roadnet, topology, feature schema, control timing, normalization statistics, and source trajectory hashes.
- One experiment command that collects data, trains and restores the model, then evaluates fixed-time, MaxPressure, and the World Model under the same CityFlow contract.

The first model is deliberately compact and deterministic. It establishes the real model-based control chain before adding uncertainty, latent stochastic dynamics, or wider trajectory-search algorithms.

## Installation

CityFlow 0.1 should be installed separately in a supported Linux environment. Then install this package:

```bash
python -m pip install -e .
```

Install the optional PyTorch dependency for the trainable DQN baseline:

```bash
python -m pip install -e '.[rl]'
```

For local contract and integration tests that do not launch CityFlow:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Run a CityFlow episode

```bash
cityflow-tsc \
  --roadnet /path/to/roadnet_3_4.json \
  --flow /path/to/anon_3_4_jinan_real.json \
  --output /path/to/new/run_directory \
  --policy max-pressure \
  --green-phases 1,2,3,4 \
  --decision-interval 30 \
  --yellow-time 5 \
  --duration 3600 \
  --seed 0
```

The output directory must be empty. A completed run contains:

```text
cityflow.config.json
run.config.json
metrics.json
trajectory.npz
trajectory.manifest.json
```

Optional CityFlow replay files are emitted with `--save-replay`.

## Train the shared DQN baseline

```bash
cityflow-tsc-train-dqn \
  --roadnet /path/to/roadnet_3_4.json \
  --flow /path/to/anon_3_4_jinan_real.json \
  --output /path/to/new/dqn_experiment \
  --episodes 20 \
  --duration 3600 \
  --eval-seeds 100,101,102 \
  --seed 0
```

Training and evaluation use the same environment, observation, action, reward, metric, and trajectory contracts as the rule-based policies. The experiment root records every episode trajectory, the final checkpoint, raw per-seed evaluation metrics, and mean/std aggregates. Checkpoint v2 binds the weights to the roadnet hash, observation feature schema, topology, movements, and action-to-CityFlow phase mappings. Evaluation trajectory manifests record the exact checkpoint hash. Checkpoint v1 is intentionally rejected because it cannot prove these semantic identities.

The DQN vectorizer includes both feature values and per-feature validity masks, so a genuinely zero-valued traffic feature remains distinguishable from a missing observation.

Only load checkpoints produced by a trusted run; PyTorch checkpoint files use its standard serialization format.

## Train and evaluate the World Model

The end-to-end command can collect its own offline dataset. Random episodes provide action coverage, while MaxPressure and fixed-time episodes keep the dataset connected to meaningful controllers:

```bash
cityflow-tsc-world-model \
  --roadnet /path/to/roadnet_3_4.json \
  --flow /path/to/anon_3_4_jinan_real.json \
  --output /path/to/new/world_model_experiment \
  --collect-episodes 12 \
  --collect-policies random,max-pressure,fixed-time \
  --duration 3600 \
  --epochs 50 \
  --batch-size 128 \
  --device cuda \
  --eval-seeds 100,101,102
```

The output directory must be new or empty. It contains:

```text
dataset/                         collected complete episodes
checkpoints/graph_world_model.pt
evaluation/fixed_time/          raw per-seed metrics and trajectories
evaluation/max_pressure/
evaluation/world_model/
experiment_summary.json         training losses, hashes, raw runs, mean/std
```

Existing trajectories can be supplied by repeating `--trajectory-manifest`; built-in collection is then skipped. All manifests must share the same roadnet, topology, observation schema, action mapping, and control timing. Training and validation are split by whole episodes rather than individual transitions.

The primary controlled comparison is MaxPressure against “the same MaxPressure proposal plus World Model scoring.” Fixed-time is a floor baseline. The existing shared DQN command remains the model-free learned baseline for larger experiments.

Only load World Model checkpoints produced by a trusted run; they use PyTorch serialization. Loading rejects mismatched roadnets, topology/action mappings, feature schemas, and control timing.

## Metric semantics

- `average_travel_time_s`: CityFlow's episode average travel time.
- `average_queue_vehicles`: mean total waiting-vehicle count over simulator ticks.
- `average_waiting_time_s`: mean accumulated stopped time over all observed vehicles.
- `throughput_vehicles`: vehicles observed active and later no longer active before the episode ends.

Reward is intentionally separate from evaluation. The initial reward is the negative incoming-lane queue for each intersection.

## Reference-code boundary

The architecture was informed by LLMLight, CoLLMLight, DiffLight, and CoordLight, but the core package is a clean implementation:

- LLMLight: CityFlow experiment organization and traffic features.
- CoLLMLight: roadnet topology and network-level context.
- DiffLight: temporal trajectories and explicit missing-data masks.
- CoordLight: fixed directional neighbors, masks, and shared-policy MARL structure.

DiffLight's GPL implementation is not copied into the core. LLMLight and CoLLMLight repositories did not expose a clear license during the architecture audit, so their code is not vendored.

The public repository contains only the CityFlow experiment foundation and the algorithms built on its contracts; unrelated historical reproduction utilities are excluded.
