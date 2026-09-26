# RL baseline usage and implementation boundaries

For the dedicated four-phase CoLight experiment with round-based fitted Q
learning and published-paper reference tables, use [CoLight training](colight.md).
Its `python -m cityflow_tsc.colight_experiment` entry point has separate defaults
from the generic `--baseline colight` learner described below.

P1–P3 use the existing `TrafficEnv.reset/step`, `Policy.reset/act`, and
`EpisodeRunner.run` contracts. A profile selects the observation view, reward and
learner. `NetworkObservation` retains its original movement fields and adds an
optional `baseline_view`; `PolicyOutput` adds optional recorded behavior. Rule
policies and the original shared-DQN/World Model commands keep their defaults.

## Implemented profiles

| CLI name | Network / learner | Observation, reward and sharing |
|---|---|---|
| `presslight`, `e-presslight` | Current-phase-specific Q branches | phase8 plus general/efficient queue pressure; absolute queue pressure reward; shared parameters |
| `mplight`, `e-mplight` | Lane embeddings, phase demands, relation-gated phase competition | General vehicle pressure / efficient queue pressure; shared parameters |
| `a-mplight` | Phase competition with separate running-vehicle embeddings | Efficient queue pressure plus moving vehicles within 167m of lane end |
| `colight`, `e-colight`, `a-colight` | Shared Q with multi-head neighbor attention | Counts / efficient queue pressure / pressure+running; geometry kNN including self; negative queue reward |
| `idqn`, `shared-dqn` | Per-intersection / shared dense Q | Padded incoming lane counts and current phase; road-averaged negative queue |
| `frap`, `libsignal-mplight` | Independent / shared phase competition | Two dedicated L/T demand lanes per phase, excluding right turns; road-averaged negative queue |
| `libsignal-colight` | Shared neighbor-attention Q | Directed incoming-road neighbors plus self; parallel edges deduplicated |
| `ippo`, `mappo` | Shared GRU actor; local / centralized V critic | Lane counts+queues and phase; agent identity; common mean of local negative queue rewards |
| `maddpg` | Independent simplex actors and joint centralized Q critics | Whole-network replay, recorded action vectors, argmax phase execution, soft targets |

LLMTSCS-derived profiles validate four incoming/outgoing approaches with three
dedicated L/T/R lanes each and recognizable phase meanings. Lane ordering is
derived from topology and turn types, not assumed from physical lane indices.
The source eight-bit movement-permission representation is not a phase one-hot.
Unsupported shared-turn lanes or unrecognized phases fail explicitly. The source
phase order may differ between intersections and from configured CityFlow engine
IDs. PressLight/CoLight heads and PressLight's current-phase branches use the
eight canonical source phases; their Q values are mapped into each node's local
action order before action selection or TD updates. Unavailable source phases
are excluded. Exploration, replay and environment actions remain local indices;
the source-to-local mapping is stored in observations, trajectories and checkpoints.
Generic profiles use the existing padded intersection representation. FRAP and
LibSignal-MPLight require exactly two distinct dedicated left/through demand lanes
per green phase; shared-turn lanes, duplicate pairs or other demand counts fail
explicitly. Their competition relation is true only for pairs sharing exactly
one demand. Right turns remain part of the simulator's physical phases.
This does not remove the foundation loader's
four-direction-neighbor limitation for arbitrary roadnets.

Profiles are ports, not exact upstream replications:

- LLMTSCS models are re-expressed in PyTorch. Widths are configurable. Round-based
  collection is retained, but bounded replay minibatch updates replace Keras fit,
  validation split, and round checkpoint selection. Target updates and epsilon
  schedules are controlled by the saved TrainConfig. Original pretrained Keras
  weights are not accepted.
- Source-inspired phase-competition and graph networks remain distinct from the
  plain DQN. Canonical MPLight uses its eight left/through movements; LibSignal
  FRAP/MPLight derive explicit two-demand phase pairs from roadLinks in local
  lane order. Their embedding order is phase then demand, as in LibSignal.
  Configurable network widths and the local Q learner still differ from the
  upstream training setup. Three CoLight variants share one network class with
  different input profiles.
- `pre_mean` uses pre-tick snapshots, while `post_mean` uses post-tick snapshots.
  LLMTSCS reward factors are represented in profiles; learner reward scaling is a
  separate saved setting. The source's final T−1 logged next-state workaround is
  not reproduced: these ports use the actual boundary observation and explicit
  terminated/truncated bootstrap rules. LibSignal DQN/CoLight's additional historical
  ×12 reward multiplier is not applied by `idqn`, `shared-dqn` or `libsignal-colight`;
  this difference is visible in their profile factor and configurable reward scale.
  Local `libsignal-colight` also includes phase observations and uses the shared
  local Adam Q learner.
- IPPO/MAPPO reuse the recurrent actor/local-versus-global critic idea from
  cMALC-D, with fresh complete episodes and actual behavior logprob/value/hidden
  state. They use this project's lane observations and queue reward rather than
  cMALC-D's TSflow representation/reward. `nstep` defaults to five-step returns;
  `gae` is an explicit alternative. Neither implements the source target critic
  or LLM curriculum. These profiles do not provide LibSignal's separate IPPO
  adapter; SOTL/E-MP/A-MP profiles are also not implemented.
- PPO processes full recurrent episodes rather than shuffled independent time
  steps; memory use grows with episode length and network size. Q replay capacity
  counts whole-network transitions, not individual intersections. MADDPG has one
  actor/critic pair per intersection and scales accordingly.
- Discrete-phase MADDPG differentiates through legal simplex action vectors and
  executes their argmax. It is explicitly a local discrete adaptation, not an
  assertion of equivalence to every MADDPG or LibSignal implementation.

## Installation and commands

From this repository, install the Python package and optional RL/test dependencies:

```bash
python -m pip install -e '.[rl,dev]'
rl-trafficlight list
```

Actual traffic execution additionally requires a working CityFlow Python module
and matching roadnet/flow files. The current development verification used the
deterministic test backend and real PyTorch gradients; it did not install CityFlow
or produce traffic benchmark results. No TensorFlow or LLM service is required.

Train one baseline (paths and budgets are supplied by the user):

```bash
rl-trafficlight train \
  --baseline a-colight \
  --roadnet /absolute/path/roadnet.json \
  --flow /absolute/path/flow.json \
  --output /absolute/path/runs/a-colight \
  --episodes 20 --seed 42 --eval-seeds 100,101,102 \
  --duration 3600 --decision-interval 30 --yellow-time 5 \
  --green-phases 1,2,3,4
```

For `ippo`/`mappo`, use the same command with that baseline ID. Optional training
settings include `--return-estimator nstep|gae`, `--n-steps`, `--ppo-epochs`,
`--ppo-clip`, `--gamma`, and `--reward-scale`. `--no-bootstrap-truncated` treats the
time limit as an episodic learning boundary; the default bootstraps time limits
but never true termination. Changing that rule is a protocol change.

CLI and Python use `make_train_config(profile, **overrides)` for the same defaults.
IPPO/MAPPO/MADDPG default to gamma=0.99, reward_scale=1.0 and learning_rate=0.0003;
Q profiles default to 0.8, 0.05 and 0.001 respectively. Explicit configurations
are used unchanged. These are local defaults, not source experiment presets.

The interval includes signal transition time: a 30s interval with 5s yellow gives
25s of new green after a switch, and 30s of continued green without a switch.
All-red is independently configurable. The manifest records actual timing.

By default, `--checkpoint-every 100` saves a checkpoint and matching protocol
sidecar every 100 completed episodes, plus the final episode of each invocation.
A 100-episode run therefore saves only `episode_0099.pt`; a shorter 20-episode run
still saves `checkpoints/a-colight.episode_0019.pt` at its end. The cadence uses
the cumulative completed-episode count across resumes and may be changed when
resuming. `--curve-every 10 --curve-seed 9000` independently evaluates each tenth
completed episode in memory, retaining curve metrics and trajectories without
saving intermediate model weights. Curves are disabled by default (`0`) and do
not select the final model. Their summaries contain a model-state hash; a
checkpoint path/hash is present only when that round actually saved one.
Use `--checkpoint-every 10` only when intermediate weights are needed for later
reevaluation. Existing checkpoint files are never deleted.
Only after
both files are written is `checkpoints/latest.json` atomically updated. A failed
later save leaves the preceding checkpoint pair usable. The summary prints the
final checkpoint path; copy that path into evaluation:

```bash
rl-trafficlight evaluate \
  --checkpoint /absolute/path/runs/a-colight/checkpoints/a-colight.episode_0019.pt \
  --roadnet /absolute/path/roadnet.json \
  --flow /absolute/path/test-flow.json \
  --output /absolute/path/runs/a-colight-eval --seed 200
```

Evaluation restores training timing/profile and checks the roadnet hash. P1–P3
checkpoints are topology-specific; cross-topology migration is not silently
allowed. Evaluation creates no training updates/replay entries. The default
duration is 3600s; `--duration` explicitly selects another evaluation horizon.

Resume at an episode boundary by adding `--resume /absolute/path/model.pt` to the
same train command with a **new empty output directory**, matching training
configuration/seed/flow/timing and `--episodes` set to the number of additional
episodes. The previous completed-episode count determines subsequent episode
indices and simulator seeds. Checkpoints include optimizer, targets, replay where
applicable, RNG, schema and counters. Mid-episode simulator-state restoration is
not supported. Versioned full-replay checkpoints can consume substantial disk;
retention is a user choice, not automatic deletion.

The phase fixes use Q checkpoint version 2. Version 1 Q checkpoints are rejected;
restart those runs rather than resuming with changed action/demand semantics.
PPO/MADDPG checkpoint formats are unchanged.

Training uses the final training checkpoint for evaluation, not evaluation-based
checkpoint selection. `--eval-flow` selects a separate evaluation flow; without
it, evaluation uses the training flow with different simulator seeds. Multiple
evaluation seeds of one trained model are not independent training repetitions.

Training, ordinary evaluation and rule-controller runs now collect the same
`core-lifecycle-id-ledger-v1` metrics used by A/B: scheduled, generated, entered,
finished, active-unfinished and not-entered vehicle counts, plus completion rate.
`lifecycle.manifest.json` records the vehicle-ID partition and collector hash;
the existing ATT/AWT/queue definitions are unchanged. This exact v1 accounting
supports one-second simulation steps and one-shot integer departures within the
episode horizon. Other flow protocols or backends without complete vehicle IDs
report `lifecycle_metrics_available=0` and an explicit reason in that manifest;
missing lifecycle metrics are omitted, never replaced by zero.

The A/B `summarize.py` and `finalize.py` scripts use the shared
`cityflow_tsc.checkpoint_artifacts` module (install this checkout or set
`PYTHONPATH=src` when running them). Completed cleanup manifests and deletion
journals can supply the original identities of removed intermediate checkpoints.
Final model weights remain mandatory. Unknown missing files still fail validation.
For read-only checks use `summarize(root, write_outputs=False)` and
`validate(root, write_audit=False)`; these preserve the previously delivered files.

## Python integration

```python
from cityflow_tsc.baselines.registry import create_learner, make_train_config
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc.runtime import build_environment

# network/control/scenario use the existing project types.
env = build_environment(control, network, profile="a-colight")
initial = env.reset(scenario)
config = make_train_config("a-colight")  # overrides, e.g. gamma=0.9, go here
learner = create_learner("a-colight", network, initial[0], config, seed=42)
result = TrainingRunner().run_episode(env, learner, scenario, initial=initial)
learner.save("checkpoint.pt")
```

Direct Python checkpoints can be restored with the same learner's `load()`.
The CLI additionally requires its matching `.protocol.json` sidecar for simulator
timing and provenance; an arbitrary Python checkpoint alone is not a CLI run.

## Data and validation

The default runner still writes v1 trajectories. Baseline mode explicitly selects
`BaselineTrajectoryWriter`, which creates `baseline.trajectory.npz` plus its v2
manifest. It stores legacy movement observations, dedicated baseline views,
graph/phase mapping, local rewards and their unscaled time-aggregated components,
actual elapsed time/executed phase, and
available behavior probabilities, values, hidden states or actor vectors. The
matching reader verifies checksums and time/node dimensions. The World Model v1
reader rejects v2 rather than treating missing or changed data as equivalent.

Run the relevant checks with:

```bash
python -m pytest tests/test_baseline_observations.py tests/test_baseline_q.py \
  tests/test_baseline_actor_critic.py tests/test_baseline_integration.py \
  tests/test_core_chain.py tests/test_dqn_chain.py tests/test_world_model_chain.py \
  tests/test_rich_world_model_prediction_chain.py tests/test_counterfactual.py \
  tests/test_bounded_counterfactual.py
```

Tests cover exact lane ordering and reward windows, masks, distinct architectures,
real parameter updates, recurrent on-policy probabilities, checkpoint restoration,
all-profile environment integration, CLI training/evaluation/resume, v1 rejection
of v2, failed-save preservation, heterogeneous physical phase permutations,
right-turn-free FRAP demands, CLI/Python default parity, and existing World
Model/control compatibility. The RL CI job runs the full suite, including all
baseline tests. The NumPy-only CI job runs core contracts and baseline observation
tests, with Torch-dependent baseline tests skipped.
Their deterministic backend is not evidence of traffic performance or paper
reproduction. See [source provenance](../SOURCE_PROVENANCE.md) for audited commits.

## W&B experiment records

Install the optional SDK with `python -m pip install -e '.[tracking]'`.
The baseline `train`/`evaluate` commands and the rule-controller CLI accept
`--wandb-mode offline|online|disabled`, `--wandb-project`, `--wandb-entity`,
`--wandb-group`, and `--wandb-name`. The default is offline (or `WANDB_MODE`);
online mode uses the existing W&B login. A run is named by baseline, flow and seed.

Append `--wandb-mode online --wandb-project rl-trafficlight` to an existing command
to upload. For learning curves, use the existing `--curve-every 1` only when the
experiment protocol calls for evaluation after each round. Logging does not change
the evaluation cadence, learner or checkpoint selection. Rule evaluations produce
one point rather than a fabricated training curve.

`train/*`, `env/*`, and `eval/*` use the explicit completed `round` axis.
Final-checkpoint evaluation seeds use `final_eval/*` and `evaluation_index`.
Summary statistics distinguish those evaluation seeds from independent training
seeds. The last-ten-evaluation summary records the actual round IDs and whether
they are exactly the final ten training rounds; sparse or shorter runs are labeled.

Every invocation also writes `tracking/config.json`, append-only `history.jsonl`,
`summary.json` on success, and `status.json`. W&B files live below
`<output>/tracking/wandb/`; on 5090 set `--output` to an absolute directory under
`/mnt/pan`. Missing SDK or SDK failures leave the independent JSON records intact;
an online initialization failure attempts offline mode. Offline W&B directories
can later be uploaded using `wandb sync <offline-run-directory>`.

Checkpoint resume creates a new W&B run segment, records `resume_checkpoint`, and
continues the completed-round axis; it does not overwrite or pretend to merge
previous offline history. Existing completed experiments are not uploaded by this
change, and multi-training-seed statistics remain the responsibility of the
experiment aggregator.
