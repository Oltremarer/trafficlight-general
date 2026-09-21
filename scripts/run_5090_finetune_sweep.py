#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cityflow_tsc.world_model.selection import selection_from_summary


CITY_ORDER = ("jinan", "hangzhou")


@dataclass(frozen=True)
class Job:
    stage: str
    city: str
    candidate_id: str
    output: Path
    command: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.stage}:{self.city}:{self.candidate_id}"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty ranking")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_plan(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["candidate_id"] for row in rows]
    if not rows or len(set(ids)) != len(rows):
        raise ValueError("fine-tune plan must contain non-empty unique candidates")
    return rows


def _gpu_memory_used_mib() -> Optional[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _candidate_args(row: Mapping[str, str]) -> List[str]:
    return [
        "--encoder-hidden-dim",
        row["encoder_hidden_dim"],
        "--movement-latent-dim",
        row["movement_latent_dim"],
        "--adapter-learning-rate",
        row["adapter_learning_rate"],
        "--pretrained-learning-rate",
        row["pretrained_learning_rate"],
        "--frozen-pretrained-epochs",
        row["frozen_pretrained_epochs"],
        "--reward-loss-weight",
        row["reward_loss_weight"],
        "--checkpoint-reward-selection-weight",
        row.get("checkpoint_reward_loss_weight", "0.0"),
        "--consistency-loss-weight",
        row["node_consistency_loss_weight"],
        "--movement-consistency-loss-weight",
        row["movement_consistency_loss_weight"],
        "--reconstruction-loss-weight",
        row["reconstruction_loss_weight"],
        "--temporal-decay",
        row["temporal_decay"],
        "--batch-size",
        row["batch_size"],
        "--weight-decay",
        row.get("weight_decay", "0.00001"),
        "--gradient-clip-norm",
        row.get("gradient_clip_norm", "10.0"),
        "--seed",
        row["seed"],
    ]


def _train_command(
    python: Path,
    dataset_index: Path,
    output: Path,
    checkpoint: Optional[Path],
    row: Mapping[str, str],
    epochs: int,
    max_train_windows: Optional[int],
    max_train_trajectories: Optional[int],
    evaluate_test: bool,
    model: str = "latent",
) -> Tuple[str, ...]:
    command = [
        str(python),
        "-m",
        "cityflow_tsc.train_world_model_prediction",
        "train",
        "--dataset-index",
        str(dataset_index),
        "--output",
        str(output),
        "--model",
        model,
        "--device",
        "cuda",
        "--history-length",
        row.get("history_length", "3"),
        "--rollout-horizon",
        "5",
        "--epochs",
        str(epochs),
        "--sampling-strategy",
        row.get("sampling_strategy", "flow_policy_balanced"),
        "--window-seed",
        "7301",
        "--num-workers",
        "1",
    ]
    if model == "latent":
        if checkpoint is None:
            raise ValueError("latent training requires a TD-MPC2 checkpoint")
        command.extend(["--tdmpc2-checkpoint", str(checkpoint)])
        command.extend(_candidate_args(row))
    else:
        command.extend(
            [
                "--encoder-hidden-dim",
                "128",
                "--movement-latent-dim",
                "64",
                "--adapter-learning-rate",
                "0.0003",
                "--pretrained-learning-rate",
                "0.00003",
                "--frozen-pretrained-epochs",
                "5",
                "--reward-loss-weight",
                "0.1",
                "--consistency-loss-weight",
                "2.0",
                "--movement-consistency-loss-weight",
                "2.0",
                "--reconstruction-loss-weight",
                "0.5",
                "--temporal-decay",
                "0.8",
                "--batch-size",
                "128",
                "--seed",
                "73",
            ]
        )
    if max_train_windows is not None:
        command.extend(["--max-train-windows", str(max_train_windows)])
    if max_train_trajectories is not None:
        command.extend(["--max-train-trajectories", str(max_train_trajectories)])
    if evaluate_test:
        command.append("--evaluate-test")
    return tuple(command)


def _summary_path(job: Job) -> Path:
    return job.output / "prediction_summary.json"


def _complete(job: Job) -> bool:
    path = _summary_path(job)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if job.stage == "final" and payload.get("test_evaluation") is None:
        return False
    return payload.get("model_kind") in {"pretrained_latent", "direct_observation"}


def _run_jobs(
    jobs: Sequence[Job],
    run_dir: Path,
    max_parallel: int,
    launch_interval_s: int,
    gpu_launch_ceiling_mib: int,
    deadline_unix_s: Optional[float] = None,
) -> bool:
    pending = [job for job in jobs if not _complete(job)]
    active: Dict[str, Tuple[Job, subprocess.Popen, object, int]] = {}
    attempts: Dict[str, int] = {}
    last_launch = 0.0
    state_path = run_dir / "scheduler_state.json"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    deadline_reached = False
    while pending or active:
        for key, (job, process, handle, attempt) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del active[key]
            if return_code == 0 and _complete(job):
                continue
            if attempt >= 2:
                _write_json(
                    state_path,
                    {
                        "status": "failed",
                        "failed_job": job.key,
                        "return_code": return_code,
                        "attempt": attempt,
                    },
                )
                raise RuntimeError(f"job failed twice: {job.key}")
            if job.output.exists():
                failed = job.output.with_name(
                    f"{job.output.name}.failed_attempt{attempt}_{int(time.time())}"
                )
                shutil.move(str(job.output), str(failed))
            pending.insert(0, job)

        now = time.time()
        while pending and len(active) < max_parallel:
            if deadline_unix_s is not None and now >= deadline_unix_s:
                deadline_reached = True
                break
            if now - last_launch < launch_interval_s:
                break
            used = _gpu_memory_used_mib()
            if used is not None and used > gpu_launch_ceiling_mib:
                break
            job = pending.pop(0)
            attempt = attempts.get(job.key, 0) + 1
            attempts[job.key] = attempt
            log_path = logs / f"{job.stage}_{job.city}_{job.candidate_id}.attempt{attempt}.log"
            handle = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment.setdefault("OMP_NUM_THREADS", "2")
            environment.setdefault("MKL_NUM_THREADS", "2")
            process = subprocess.Popen(
                job.command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            active[job.key] = (job, process, handle, attempt)
            last_launch = time.time()
            now = last_launch

        _write_json(
            state_path,
            {
                "status": "running",
                "pending": [job.key for job in pending],
                "active": {
                    key: {"pid": value[1].pid, "attempt": value[3]}
                    for key, value in active.items()
                },
                "completed": sum(_complete(job) for job in jobs),
                "total": len(jobs),
                "gpu_memory_used_mib": _gpu_memory_used_mib(),
                "deadline_unix_s": deadline_unix_s,
                "deadline_reached": deadline_reached,
            },
        )
        if deadline_reached and not active:
            _write_json(
                state_path,
                {
                    "status": "time_budget_reached",
                    "pending": [job.key for job in pending],
                    "active": {},
                    "completed": sum(_complete(job) for job in jobs),
                    "total": len(jobs),
                    "deadline_unix_s": deadline_unix_s,
                },
            )
            return False
        if pending or active:
            time.sleep(10)
    return True


def _selection_score(summary_path: Path) -> Dict[str, float]:
    """Read the validation-only score without relying on JSON key order.

    ``features`` is stored with ``sort_keys=True``.  It is therefore invalid
    to use the first four JSON fields as the four vehicle/queue quantities.
    Reward prediction remains reported, but it is not a model-selection target
    for this state-prediction experiment.
    """
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return selection_from_summary(payload)


def _rank(
    jobs: Iterable[Job],
    run_dir: Path,
    stage: str,
    city: str,
) -> List[Dict[str, object]]:
    rows = []
    for job in jobs:
        if job.city != city:
            continue
        metrics = _selection_score(_summary_path(job))
        rows.append(
            {
                "rank": 0,
                "candidate_id": job.candidate_id,
                **metrics,
                "summary_path": str(_summary_path(job)),
            }
        )
    rows.sort(key=lambda item: (item["selection_score"], item["candidate_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    _write_csv(run_dir / "rankings" / f"{stage}_{city}.csv", rows)
    return rows


def _interleave(per_city: Mapping[str, Sequence[Job]]) -> List[Job]:
    result = []
    maximum = max(len(items) for items in per_city.values())
    for index in range(maximum):
        for city in CITY_ORDER:
            if index < len(per_city[city]):
                result.append(per_city[city][index])
    return result


def _job_set(
    stage: str,
    candidates: Mapping[str, Sequence[Mapping[str, str]]],
    args: argparse.Namespace,
    epochs: int,
    max_windows: Optional[int],
    max_trajectories: Optional[int],
    evaluate_test: bool,
) -> List[Job]:
    per_city: Dict[str, List[Job]] = {city: [] for city in CITY_ORDER}
    for city in CITY_ORDER:
        for row in candidates[city]:
            candidate_id = row["candidate_id"]
            output = args.run_dir / stage / city / candidate_id
            per_city[city].append(
                Job(
                    stage=stage,
                    city=city,
                    candidate_id=candidate_id,
                    output=output,
                    command=_train_command(
                        args.python,
                        args.dataset_index[city],
                        output,
                        args.tdmpc2_checkpoint,
                        row,
                        epochs,
                        max_windows,
                        max_trajectories,
                        evaluate_test,
                    ),
                )
            )
    return _interleave(per_city)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--jinan-index", type=Path, required=True)
    parser.add_argument("--hangzhou-index", type=Path, required=True)
    parser.add_argument("--tdmpc2-checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--full-data-max-parallel", type=int, default=1)
    parser.add_argument("--launch-interval-s", type=int, default=20)
    parser.add_argument("--gpu-launch-ceiling-mib", type=int, default=26000)
    parser.add_argument("--screen-epochs", type=int, default=10)
    parser.add_argument("--screen-max-train-windows", type=int, default=60000)
    parser.add_argument("--screen-max-train-trajectories", type=int, default=550)
    parser.add_argument("--promotion-top-k", type=int, default=5)
    parser.add_argument("--promotion-epochs", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Stop after the fixed-seed screen and write validation rankings.",
    )
    parser.add_argument(
        "--skip-direct-references",
        action="store_true",
        help="Run only latent candidates when an already-frozen direct control is reused.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        help="Stop launching new jobs after this total wall-clock budget.",
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    # Preserve the venv launcher path. Resolving this symlink would execute the
    # base Conda interpreter and silently drop packages installed in the run venv.
    args.python = args.python.expanduser().absolute()
    args.plan = args.plan.expanduser().resolve()
    args.tdmpc2_checkpoint = args.tdmpc2_checkpoint.expanduser().resolve()
    args.dataset_index = {
        "jinan": args.jinan_index.expanduser().resolve(),
        "hangzhou": args.hangzhou_index.expanduser().resolve(),
    }
    if not str(args.run_dir).startswith("/mnt/pan/"):
        raise ValueError("run directory must be under /mnt/pan")
    if args.max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    if args.full_data_max_parallel <= 0:
        raise ValueError("full_data_max_parallel must be positive")
    if any(
        value <= 0
        for value in (
            args.screen_epochs,
            args.screen_max_train_windows,
            args.screen_max_train_trajectories,
            args.promotion_top_k,
            args.promotion_epochs,
            args.final_epochs,
        )
    ):
        raise ValueError("stage budgets must be positive")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0:
        raise ValueError("max_runtime_hours must be positive when provided")
    for path in (
        args.python,
        args.plan,
        args.tdmpc2_checkpoint,
        *args.dataset_index.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return args


def main() -> int:
    args = _parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started_unix_s = time.time()
    deadline_unix_s = (
        started_unix_s + args.max_runtime_hours * 3600.0
        if args.max_runtime_hours is not None
        else None
    )
    rows = _load_plan(args.plan)
    if args.promotion_top_k > len(rows):
        raise ValueError("promotion_top_k cannot exceed the planned candidates")
    by_id = {row["candidate_id"]: row for row in rows}
    common = {city: rows for city in CITY_ORDER}
    _write_json(
        args.run_dir / "run_manifest.json",
        {
            "candidate_count_per_city": len(rows),
            "cities": list(CITY_ORDER),
            "dataset_indices": {key: str(value) for key, value in args.dataset_index.items()},
            "plan": str(args.plan),
            "tdmpc2_checkpoint": str(args.tdmpc2_checkpoint),
            "source_commit": args.source_commit,
            "started_unix_s": started_unix_s,
            "deadline_unix_s": deadline_unix_s,
            "max_runtime_hours": args.max_runtime_hours,
            "max_parallel": args.max_parallel,
            "full_data_max_parallel": args.full_data_max_parallel,
            "screen_protocol": {
                "epochs": args.screen_epochs,
                "max_train_trajectories": args.screen_max_train_trajectories,
                "max_train_windows": args.screen_max_train_windows,
            },
            "promotion_protocol": {
                "top_k": args.promotion_top_k,
                "epochs": args.promotion_epochs,
                "full_train_data": True,
            },
            "final_protocol": {
                "top_k": 1,
                "epochs": args.final_epochs,
                "test_once": True,
                "direct_references": not args.skip_direct_references,
            },
            "selection_score": (
                "0.40*weighted named vehicle/queue normalized RMSE + "
                "0.20*weighted named arrival/departure-rate normalized RMSE + "
                "0.20*weighted all-feature normalized RMSE + "
                "0.20*last-two-step named vehicle/queue normalized RMSE"
            ),
        },
    )

    (args.run_dir / "stage.txt").write_text(
        f"screen_{len(rows)}_per_city\n", encoding="utf-8"
    )
    screen_jobs = _job_set(
        "screen",
        common,
        args,
        args.screen_epochs,
        args.screen_max_train_windows,
        args.screen_max_train_trajectories,
        False,
    )
    if not _run_jobs(
        screen_jobs,
        args.run_dir,
        args.max_parallel,
        args.launch_interval_s,
        args.gpu_launch_ceiling_mib,
        deadline_unix_s,
    ):
        (args.run_dir / "stage.txt").write_text(
            "time_budget_reached_screen\n", encoding="utf-8"
        )
        return 0
    screen_ranking = {
        city: _rank(screen_jobs, args.run_dir, "screen", city)
        for city in CITY_ORDER
    }
    if args.screen_only:
        _write_json(
            args.run_dir / "screen_selection.json",
            {
                "ranking_paths": {
                    city: str(args.run_dir / "rankings" / f"screen_{city}.csv")
                    for city in CITY_ORDER
                },
                "selection_status": (
                    "screen_complete_single_seed; do not use this ranking as the "
                    "final model choice before multi-seed full-data confirmation"
                ),
            },
        )
        (args.run_dir / "stage.txt").write_text(
            "screen_complete_pending_multiseed_promotion\n", encoding="utf-8"
        )
        return 0

    promoted = {
        city: [
            by_id[item["candidate_id"]]
            for item in screen_ranking[city][: args.promotion_top_k]
        ]
        for city in CITY_ORDER
    }
    (args.run_dir / "stage.txt").write_text(
        f"promote_top{args.promotion_top_k}_full_data\n", encoding="utf-8"
    )
    promote_jobs = _job_set(
        "promote", promoted, args, args.promotion_epochs, None, None, False
    )
    if not _run_jobs(
        promote_jobs,
        args.run_dir,
        args.full_data_max_parallel,
        args.launch_interval_s,
        args.gpu_launch_ceiling_mib,
        deadline_unix_s,
    ):
        (args.run_dir / "stage.txt").write_text(
            "time_budget_reached_promotion\n", encoding="utf-8"
        )
        return 0
    promote_ranking = {
        city: _rank(promote_jobs, args.run_dir, "promote", city)
        for city in CITY_ORDER
    }

    selected = {
        city: [by_id[promote_ranking[city][0]["candidate_id"]]] for city in CITY_ORDER
    }
    (args.run_dir / "stage.txt").write_text(
        "final_two_city_models\n" if args.skip_direct_references else "final_two_city_models_and_direct_controls\n",
        encoding="utf-8",
    )
    final_latent = _job_set(
        "final", selected, args, args.final_epochs, None, None, True
    )
    direct_jobs = []
    if not args.skip_direct_references:
        for city in CITY_ORDER:
            output = args.run_dir / "final" / city / "direct_reference"
            direct_jobs.append(
                Job(
                    stage="final",
                    city=city,
                    candidate_id="direct_reference",
                    output=output,
                    command=_train_command(
                        args.python,
                        args.dataset_index[city],
                        output,
                        None,
                        rows[0],
                        args.final_epochs,
                        None,
                        None,
                        True,
                        model="direct",
                    ),
                )
            )
    if not _run_jobs(
        _interleave(
            {
                city: [job for job in final_latent + direct_jobs if job.city == city]
                for city in CITY_ORDER
            }
        ),
        args.run_dir,
        args.full_data_max_parallel,
        args.launch_interval_s,
        args.gpu_launch_ceiling_mib,
        deadline_unix_s,
    ):
        (args.run_dir / "stage.txt").write_text(
            "time_budget_reached_final\n", encoding="utf-8"
        )
        return 0

    comparisons = {}
    if not args.skip_direct_references:
        for city in CITY_ORDER:
            latent_job = next(job for job in final_latent if job.city == city)
            direct_job = next(job for job in direct_jobs if job.city == city)
            output = args.run_dir / "final" / city / "comparison_test"
            comparison_path = output / "prediction_comparison.json"
            if not comparison_path.is_file():
                if output.exists():
                    failed = output.with_name(
                        f"{output.name}.incomplete_{int(time.time())}"
                    )
                    shutil.move(str(output), str(failed))
                subprocess.run(
                    [
                        str(args.python),
                        "-m",
                        "cityflow_tsc.train_world_model_prediction",
                        "compare",
                        "--direct-summary",
                        str(_summary_path(direct_job)),
                        "--latent-summary",
                        str(_summary_path(latent_job)),
                        "--output",
                        str(output),
                        "--split",
                        "test",
                    ],
                    check=True,
                )
            comparisons[city] = str(output / "prediction_comparison.json")
    _write_json(
        args.run_dir / "final_selection.json",
        {
            "selected_candidates": {
                city: promote_ranking[city][0] for city in CITY_ORDER
            },
            "test_comparisons": comparisons,
            "direct_references_run": not args.skip_direct_references,
        },
    )
    (args.run_dir / "stage.txt").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
