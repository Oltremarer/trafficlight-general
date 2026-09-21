"""Train/evaluate local PyTorch ports through the existing CityFlow interfaces."""
from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from dataclasses import fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .baselines.contracts import TrainConfig
from .baselines.profiles import BaselineProfile, get_profile, list_profiles
from .baselines.registry import create_learner, make_train_config
from .baselines.training import TrainingRunner
from .baselines.trajectory import BaselineTrajectoryWriter, write_json
from .config import ControlConfig, ScenarioConfig
from .runner import EpisodeRunner
from .runtime import build_environment
from .topology import load_network_spec
from .trajectory import sha256_file


def _integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(v.strip()) for v in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _new_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scenario(args, output: Path, seed: int, *, flow=None) -> ScenarioConfig:
    return ScenarioConfig(args.roadnet, flow or args.flow, output, args.duration,
                          seed, args.thread_num)


def _aggregate(runs: list[dict]) -> dict:
    if not runs:
        return {}
    keys = set.intersection(*(set(run['metrics']) for run in runs))
    return {key: {"mean": float(np.mean([r['metrics'][key] for r in runs])),
                  "std": float(np.std([r['metrics'][key] for r in runs]))}
            for key in sorted(keys)}


def _runtime_versions() -> dict:
    versions = {"python": platform.python_version(), "system": platform.system(),
                "architecture": platform.machine()}
    for package in ('numpy', 'torch', 'cityflow'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _control(args) -> ControlConfig:
    return ControlConfig(decision_interval_s=args.decision_interval,
                         simulator_step_s=args.simulator_step,
                         yellow_time_s=args.yellow_time,
                         all_red_time_s=args.all_red_time,
                         yellow_phase_id=args.yellow_phase_id,
                         all_red_phase_id=args.all_red_phase_id,
                         green_phase_ids=args.green_phases)


def _training_config(args, profile) -> TrainConfig:
    settings = {field.name: getattr(args, field.name) for field in fields(TrainConfig)
                if hasattr(args, field.name) and getattr(args, field.name) is not None}
    return make_train_config(profile, **settings)


def _save(learner, checkpoint: Path, protocol: dict) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or checkpoint.with_suffix('.protocol.json').exists():
        raise FileExistsError(f"refusing to overwrite a versioned checkpoint: {checkpoint}")
    learner.save(checkpoint)
    write_json(checkpoint.with_suffix('.protocol.json'), {
        **protocol, "checkpoint_sha256": sha256_file(checkpoint),
        "completed_episodes": learner.completed_episodes,
    })
    write_json(checkpoint.parent / 'latest.json', {
        "checkpoint_file": checkpoint.name,
        "protocol_file": checkpoint.with_suffix('.protocol.json').name,
        "checkpoint_sha256": sha256_file(checkpoint),
    })


def _read_protocol(checkpoint: Path) -> dict:
    metadata = checkpoint.with_suffix('.protocol.json')
    if not metadata.is_file():
        raise ValueError(f"CLI checkpoint requires its protocol sidecar: {metadata}")
    protocol = json.loads(metadata.read_text())
    if sha256_file(checkpoint) != protocol.get('checkpoint_sha256'):
        raise ValueError("checkpoint does not match its protocol sidecar")
    return protocol


def _evaluate(env, learner, scenario, *, initial=None) -> dict:
    result = EpisodeRunner().run(
        env, learner.policy, scenario, deterministic=True,
        policy_metadata={"mode": "frozen_evaluation", "profile": learner.profile.to_dict(),
                         "train_config": learner.config.to_dict()},
        writer_factory=BaselineTrajectoryWriter, initial=initial,
    )
    return {"seed": scenario.seed, "metrics": result.metrics,
            "trajectory_path": result.trajectory_path, "manifest_path": result.manifest_path}


def train(args) -> dict:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    profile = get_profile(args.baseline)
    config = _training_config(args, profile)
    control = _control(args)
    output = _new_output(args.output)
    initial_scenario = _scenario(args, output / 'train' / 'episode_0000', args.seed)
    network = load_network_spec(initial_scenario.roadnet_path, control)
    protocol = {
        "profile": profile.to_dict(), "train_config": config.to_dict(),
        "control": control.to_dict(), "duration_s": args.duration,
        "roadnet_sha256": sha256_file(initial_scenario.roadnet_path),
        "flow_sha256": sha256_file(initial_scenario.flow_path),
        "training_seed": args.seed, "evaluation_seeds": list(args.eval_seeds),
        "waiting_speed_threshold": args.waiting_speed_threshold,
        "runtime_versions": _runtime_versions(), "device": args.device,
        "checkpoint_selection": "final_training_episode; no evaluation-based selection",
        "implementation": "local PyTorch ports; no claim of upstream or paper result reproduction",
    }
    env = build_environment(control, network, args.waiting_speed_threshold, profile=profile)
    protocol['backend_class'] = f'{type(env.backend).__module__}.{type(env.backend).__qualname__}'
    try:
        write_json(output / 'experiment.config.json', protocol)
        initial = env.reset(initial_scenario)
        learner = create_learner(profile, network, initial[0], config, args.seed, args.device)
        if args.resume:
            previous = _read_protocol(args.resume)
            for key in ('profile', 'train_config', 'control', 'roadnet_sha256', 'flow_sha256',
                        'training_seed', 'duration_s', 'waiting_speed_threshold'):
                if json.dumps(previous[key], sort_keys=True) != json.dumps(protocol[key], sort_keys=True):
                    raise ValueError(f"resume protocol mismatch: {key}")
            learner.load(args.resume)
            env.close()
            initial = None
        start_episode = learner.completed_episodes
        runner = TrainingRunner()
        training_runs = []
        for offset in range(args.episodes):
            episode = start_episode + offset
            scenario = _scenario(args, output / 'train' / f'episode_{episode:04d}', args.seed + episode)
            if offset or initial is None:
                env = build_environment(control, network, args.waiting_speed_threshold, profile=profile)
            result = runner.run_episode(env, learner, scenario, initial=initial if offset == 0 else None)
            initial = None
            checkpoint = output / 'checkpoints' / f'{profile.baseline_id}.episode_{episode:04d}.pt'
            _save(learner, checkpoint, protocol)
            training_runs.append({"episode": episode, "seed": scenario.seed, "metrics": result.metrics})
        evaluation_runs = []
        for seed in args.eval_seeds:
            scenario = _scenario(args, output / 'evaluation' / f'seed_{seed}', seed, flow=args.eval_flow)
            evaluation_runs.append(_evaluate(
                build_environment(control, network, args.waiting_speed_threshold, profile=profile),
                learner, scenario,
            ))
        summary = {
            "baseline": profile.baseline_id, "checkpoint": str(checkpoint),
            "environment_steps": learner.environment_steps, "gradient_steps": learner.gradient_steps,
            "completed_episodes": learner.completed_episodes, "training": training_runs,
            "evaluation": {"runs": evaluation_runs, "aggregate": _aggregate(evaluation_runs),
                           "replication_unit": "evaluation seeds of one trained model"},
        }
        write_json(output / 'training_summary.json', summary)
        return summary
    finally:
        env.close()


def evaluate(args) -> dict:
    protocol = _read_protocol(args.checkpoint)
    profile = BaselineProfile(**protocol['profile'])
    config = TrainConfig(**protocol['train_config'])
    timing = dict(protocol['control'])
    timing['green_phase_ids'] = tuple(timing['green_phase_ids'])
    control = ControlConfig(**timing)
    output = _new_output(args.output)
    scenario = _scenario(args, output, args.seed)
    if sha256_file(scenario.roadnet_path) != protocol['roadnet_sha256']:
        raise ValueError("this P1-P3 checkpoint is topology-specific; evaluation roadnet differs")
    network = load_network_spec(scenario.roadnet_path, control)
    threshold = protocol['waiting_speed_threshold']
    env = build_environment(control, network, threshold, profile=profile)
    try:
        initial = env.reset(scenario)
        learner = create_learner(profile, network, initial[0], config, args.seed, args.device)
        learner.load(args.checkpoint)
        result = _evaluate(env, learner, scenario, initial=initial)
        summary = {"baseline": profile.baseline_id, "checkpoint": str(args.checkpoint),
                   "checkpoint_sha256": sha256_file(args.checkpoint), **result}
        write_json(output / 'evaluation_summary.json', summary)
        return summary
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rl-trafficlight')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List implemented local baseline profiles and provenance')
    train_parser = sub.add_parser('train', help='Train a baseline and evaluate its final checkpoint')
    eval_parser = sub.add_parser('evaluate', help='Load and evaluate a saved checkpoint')
    for child in (train_parser, eval_parser):
        child.add_argument('--roadnet', type=Path, required=True)
        child.add_argument('--flow', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--duration', type=int, default=3600)
        child.add_argument('--seed', type=int, default=0)
        child.add_argument('--thread-num', type=int, default=1)
        child.add_argument('--device', default='cpu')
    train_parser.add_argument('--baseline', choices=[p.baseline_id for p in list_profiles()], required=True)
    train_parser.add_argument('--episodes', type=int, default=20)
    train_parser.add_argument('--eval-seeds', type=_integers, default=(100, 101, 102))
    train_parser.add_argument('--eval-flow', type=Path)
    train_parser.add_argument('--resume', type=Path)
    train_parser.add_argument('--decision-interval', type=int, default=30)
    train_parser.add_argument('--simulator-step', type=float, default=1.)
    train_parser.add_argument('--yellow-time', type=int, default=5)
    train_parser.add_argument('--all-red-time', type=int, default=0)
    train_parser.add_argument('--yellow-phase-id', type=int, default=0)
    train_parser.add_argument('--all-red-phase-id', type=int)
    train_parser.add_argument('--green-phases', type=_integers, default=(1, 2, 3, 4))
    train_parser.add_argument('--waiting-speed-threshold', type=float, default=.1)
    for name in ('hidden_dim', 'batch_size', 'replay_capacity', 'warmup_transitions', 'updates_per_round',
                 'target_update_steps', 'epsilon_decay_steps', 'ppo_epochs', 'n_steps'):
        train_parser.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('learning_rate', 'gamma', 'reward_scale', 'epsilon_start', 'epsilon_end',
                 'gradient_clip', 'ppo_clip', 'entropy_coef', 'value_coef', 'gae_lambda', 'tau'):
        train_parser.add_argument('--' + name.replace('_', '-'), type=float)
    train_parser.add_argument('--return-estimator', choices=('nstep', 'gae'))
    train_parser.add_argument('--bootstrap-truncated', action=argparse.BooleanOptionalAction, default=None)
    eval_parser.add_argument('--checkpoint', type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'list':
        result = [p.to_dict() for p in list_profiles()]
    elif args.command == 'train':
        result = train(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
