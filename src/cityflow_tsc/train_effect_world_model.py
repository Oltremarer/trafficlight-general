from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model import SCHEMA, GROUPS
from .effect_model.data import prepare_data
from .effect_model.model import EffectModel, effect_loss
from .effect_model.evaluation import validation_score, evaluate, oracle_metrics, aggregate_records, sync


def configuration(dataset):
    return {
        "schema": SCHEMA, "dataset": str(dataset), "seeds": [42, 43, 44],
        "model": {"hidden": 64, "gru_layers": 1, "graph_layers": 2, "relation_types": 8,
                  "heads": [128, 64, 36], "dropout": .1, "objects": 272, "windows": 36,
                  "input_dims_by_type": [20, 11, 5], "action_input_dim": 218, "relative_dim": 13,
                  "pair_symmetry": "average both ordered evaluations; dropout disabled in inference",
                  "relative_position_scale_m": 1000, "relative_hop_scale": 8,
                  "no_effect_adjacency_or_travel_time_mask": True, "no_trainable_node_identity": True},
        "training": {"A": {"batch": 32, "max_epochs": 300, "queries": 384},
                     "optimizer": "AdamW", "lr": .0003, "minimum_lr": .00003, "weight_decay": .0001,
                     "betas": [.9, .999], "epsilon": 1e-8, "clip_norm": 1., "warmup_epochs": 5,
                     "validation_every_epochs": 5, "patience_validation_checks": 10,
                     "precision": "FP32, AMP disabled, TF32 disabled", "cpu_threads": 4,
                     "B": {"batch": 64, "max_epochs": 100, "queries": 8640,
                           "freeze_encoder_actions_single": True, "zero_output_initialization": True,
                           "epoch_zero_is_checkpoint_candidate": True}},
        "loss": {"pointwise": "SmoothL1(beta=1)", "groups": "road/intersection/boundary equally weighted",
                 "formula": "mean_groups(0.5*all+0.5*nonzero; all if group has no nonzero target in batch)+0.1*total",
                 "scales": "training nonzero absolute effect P95 >=1, separate factor type and object category",
                 "total_scale": "training nonzero absolute signed total-effect P95 >=1, separate factor type",
                 "target_mask_used_only_in_loss": True},
        "checkpoint_metric": "mean over 4 validation roots of mean absolute joint-delta error across that root's 3+ candidates,272 locations,36 windows, in raw vehicle-seconds",
        "checkpoint_min_delta": 0., "joint_labels_in_training_loss": False,
        "development_split": {"train": "train_036/train_012, t600/t1800, both policies (8 roots)",
                              "validation": "same flows, t3000, both policies (4 roots)",
                              "development_holdout": "train_069, t600/t1800/t3000, both policies (6 roots)"},
        "evaluation": {"objective": "sum road+intersection+boundary waiting vehicle-seconds",
                       "selection": "predicted delta only; common future baseline never read for online scoring",
                       "candidate_set": "all sampled 3+ interventions with complete references at each root",
                       "tie_break": "first in frozen branch plan; absolute 1e-6 tolerance for reporting ties",
                       "horizons_seconds": [30, 60, 120, 180], "factor_prediction_batch": 128,
                       "online_timing": "history aggregation+normalization+prescribed action features+transfer+one encoding+factor prediction+composition+argmin+readback; no disk load",
                       "offline_data_preparation_reported_separately": True},
        "limitations": ["Development holdout, not unseen-source blind testing; all 18 roots were explored before protocol fixing",
                        "No neural performance inference from oracle factor decomposition",
                        "Fixed 180-second response model, not recurrent state rollout or arbitrary signal durations",
                        "No new collection, LLM, RL closed loop, cross-network transfer, or parameter search",
                        "Stored simulator timings are CPU restore+rollout+physical recording under original collection concurrency, not a new matched hardware benchmark"],
    }


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), **metadata}, temporary)
    os.replace(temporary, path)


def learning_rate(epoch, maximum):
    if epoch <= 5:
        return .0003 * epoch / 5
    fraction = (epoch - 5) / (maximum - 5)
    return .00003 + .5 * (.0003 - .00003) * (1 + math.cos(math.pi * fraction))


def train_stage(model, train, validation, scales, totals, device, output, seed, stage, progress):
    maximum, batch, key, kind = (300, 32, "single", 0) if stage == "A" else (100, 64, "pair", 1)
    is_pair = stage == "B"
    history = torch.as_tensor(np.stack([r["normalized_history"] for r in train]), device=device)
    actions = torch.as_tensor(np.stack([r["action_bank"] for r in train]), device=device)
    base = torch.as_tensor(np.stack([r["base_phase"] for r in train]), device=device)
    queries = len(train[0][key])
    root_ids = torch.arange(len(train), device=device).repeat_interleave(queries)
    nodes = torch.as_tensor(np.concatenate([r[key + "_nodes"] for r in train]), device=device)
    targets = torch.as_tensor(np.concatenate([r[key + "_actions"] for r in train]), device=device)
    labels = torch.as_tensor(np.concatenate([r[key] for r in train]), device=device)
    if len(labels) != (384 if stage == "A" else 8640):
        raise ValueError("Wrong training query count")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=.0003, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    frozen = None
    if is_pair:
        model.eval()
        with torch.no_grad():
            frozen = model.encode(history, actions)
    best = math.inf
    selected_epoch = None
    bad_checks = 0
    checkpoint = output / f"{stage}_best.pt"
    log_path = output / f"{stage}_epochs.jsonl"
    records = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    if is_pair:
        best, per_root = validation_score(model, validation, scales, device, True)
        selected_epoch = 0
        save_checkpoint(checkpoint, model, {"seed": seed, "stage": stage, "epoch": 0, "validation_joint_mae": best})
        records.append({"epoch": 0, "validation_joint_mae": best, "validation_per_root": per_root})
        with log_path.open("a") as stream:
            stream.write(json.dumps(records[-1]) + "\n")
        progress(stage="training", seed=seed, model=stage, epoch=0, validation_joint_mae=best)
    for epoch in range(1, maximum + 1):
        model.train(True)
        lr = learning_rate(epoch, maximum)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(labels), device=device)
        losses = []
        for offset in range(0, len(order), batch):
            ix = order[offset:offset + batch]
            optimizer.zero_grad(set_to_none=True)
            if is_pair:
                prediction = model.pair(frozen, root_ids[ix], nodes[ix], targets[ix], base)
            else:
                unique, inverse = torch.unique(root_ids[ix], return_inverse=True)
                encoded = model.encode(history[unique], actions[unique])
                prediction = model.single(encoded, inverse, nodes[ix, 0], targets[ix, 0], base[unique])
            loss = effect_loss(prediction, labels[ix], scales[kind], totals[kind])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": lr, "updates": len(losses),
                  "elapsed_s": time.perf_counter() - start}
        if epoch % 5 == 0:
            score, per_root = validation_score(model, validation, scales, device, is_pair)
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            record.update(validation_joint_mae=score, validation_per_root=per_root)
            if score < best:
                best, selected_epoch, bad_checks = score, epoch, 0
                save_checkpoint(checkpoint, model, {"seed": seed, "stage": stage, "epoch": epoch,
                                                     "validation_joint_mae": score})
            else:
                bad_checks += 1
        records.append(record)
        with log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        progress(stage="training", seed=seed, model=stage, epoch=epoch, loss=record["loss"],
                 best_validation_joint_mae=best if math.isfinite(best) else None,
                 selected_epoch=selected_epoch, elapsed_s=record["elapsed_s"])
        if bad_checks >= 10:
            break
    sync(device)
    result = {"seed": seed, "stage": stage, "epochs_run": epoch, "selected_epoch": selected_epoch,
              "best_validation_joint_mae": best, "training_wall_s": time.perf_counter() - start,
              "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(device),
              "updates": epoch * (len(labels) // batch), "checkpoint": str(checkpoint),
              "stop_reason": "patience" if bad_checks >= 10 else "max_epochs", "records": records}
    atomic_json(output / f"{stage}_training.json", result)
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    return result


def execute(dataset, run):
    def progress(**values):
        atomic_json(run / "status.json", {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "pid": os.getpid(), **values})

    start = time.perf_counter()
    config = configuration(dataset)
    source_dir = Path(__file__).parent
    config["source_sha256"] = {str(p.relative_to(source_dir)): sha256(p) for p in
                               [Path(__file__), *sorted((source_dir / "effect_model").glob("*.py")),
                                source_dir / "counterfactual/writer.py"]}
    config["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(run / "config.json", config)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    roots, static, raw_scales, raw_totals, data_manifest = prepare_data(dataset, run, progress)
    scales = torch.as_tensor(raw_scales, device=device)
    totals = torch.as_tensor(raw_totals, device=device)
    train = [r for r in roots if r["split"] == "train"]
    validation = [r for r in roots if r["split"] == "validation"]
    progress(stage="oracle_evaluation")
    oracle = oracle_metrics(roots, device)
    atomic_json(run / "oracle_evaluation.json", {"records": oracle, "label": "CityFlow true factors, not neural predictions",
                "aggregate": {k: aggregate_records([r for r in oracle if r["model"] == k])
                              for k in ("oracle_single", "oracle_pair")}})
    seed_results = []
    for seed in config["seeds"]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        directory = run / f"seed_{seed}";directory.mkdir()
        predictions = directory / "predictions";predictions.mkdir()
        a = EffectModel(static).to(device)
        for parameter in a.pair_head.parameters():
            parameter.requires_grad_(False)
        a_result = train_stage(a, train, validation, scales, totals, device, directory, seed, "A", progress)
        b = copy.deepcopy(a)
        b.start_pair_stage()
        b_result = train_stage(b, train, validation, scales, totals, device, directory, seed, "B", progress)
        for name, value in a.state_dict().items():
            if not name.startswith("pair_head.") and not torch.equal(value, b.state_dict()[name]):
                raise RuntimeError(f"Frozen A parameter changed during B: {name}")
        if abs(b_result["records"][0]["validation_joint_mae"] - a_result["best_validation_joint_mae"]) > 1e-6:
            raise RuntimeError("Epoch-zero B must reproduce selected A")
        progress(stage="evaluating", seed=seed)
        records_a = evaluate(a, roots, scales, device, False, predictions, seed, "A")
        records_b = evaluate(b, roots, scales, device, True, predictions, seed, "B")
        result = {"seed": seed, "A_training": {k: v for k, v in a_result.items() if k != "records"},
                  "B_training": {k: v for k, v in b_result.items() if k != "records"},
                  "A": aggregate_records(records_a), "B": aggregate_records(records_b),
                  "records_A": records_a, "records_B": records_b,
                  "model_parameters": sum(p.numel() for p in a.parameters())}
        atomic_json(directory / "evaluation.json", result)
        seed_results.append(result)
        atomic_json(run / "completed_seeds.json", {"seeds": [r["seed"] for r in seed_results],
                    "results": [{k: v for k, v in r.items() if not k.startswith("records_")} for r in seed_results]})
        del a, b
        torch.cuda.empty_cache()
    final = {"stage": "complete", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "total_wall_s": time.perf_counter() - start, "data_preparation_s": data_manifest["data_preparation_s"],
             "results": [{k: v for k, v in r.items() if not k.startswith("records_")} for r in seed_results],
             "limitations": config["limitations"], "errors": []}
    atomic_json(run / "summary.json", final)
    progress(**final)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the fixed three-seed action-effect A/B experiment")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    run, dataset = args.run_dir.resolve(), args.dataset.resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("All experiment outputs must use mounted /mnt/pan")
    if run == dataset or run.is_relative_to(dataset):
        raise ValueError("Training cannot write inside the immutable source collection")
    if shutil.disk_usage(run).free < 120_000_000_000:
        raise OSError("Insufficient disk reserve")
    (run / "logs").mkdir(exist_ok=True)
    if args.detach:
        argv = [x for x in (sys.argv[1:] if argv is None else argv) if x != "--detach"]
        env = dict(os.environ, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        with (run / "logs/launcher.log").open("ab") as stream:
            process = subprocess.Popen([sys.executable, "-u", "-m", "cityflow_tsc.train_effect_world_model", *argv],
                                       env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        atomic_json(run / "launcher.json", {"pid": process.pid, "run_dir": str(run),
                                             "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(json.dumps({"pid": process.pid, "run_dir": str(run), "status": "launched"}))
        return 0
    import fcntl
    with (run / "training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "config.json").exists():
            raise ValueError("Use a fresh training run; implicit mixed-state resume is not supported")
        try:
            execute(dataset, run)
        except Exception as exc:
            atomic_json(run / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(run / "status.json", {"stage": "failed", "error": repr(exc), "pid": os.getpid()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
