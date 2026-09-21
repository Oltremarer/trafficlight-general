#!/usr/bin/env python3
"""Matched validation-only latency benchmark for CityFlow, World Models, and LLM judges.

The timed regions are deliberately narrow:

* CityFlow: five future decision intervals plus one compact state readout.
* World Model: a loaded model's five-step ``rollout``.
* LLM: a loaded model's ``generate`` call over a prepared compact prompt.

Context replay, model/tokenizer loading, tokenization, and warm-up are recorded but
excluded from the corresponding primary latency columns.  This is a latency study,
not a prediction-accuracy or traffic-control evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cityflow_tsc.config import ControlConfig, ScenarioConfig
from cityflow_tsc.runtime import build_rich_environment
from cityflow_tsc.topology import load_network_spec
from cityflow_tsc.trajectory import TrajectoryReader, sha256_file
from cityflow_tsc.world_model.latent_prediction import (
    PretrainedLatentTrafficModel,
    TrafficPredictionConfig,
)
from cityflow_tsc.world_model.rich_data import (
    RichFeatureStatistics,
    RichTrajectorySequenceDataset,
)


HORIZON = 5
WARMUP_REPEATS = 3
MEASURE_REPEATS = 10
CITY_ORDER = ("jinan", "hangzhou")
REFERENCE_IDS = {"jinan": "T04", "hangzhou": "T28"}
REFERENCE_HISTORY_LENGTHS = {"jinan": 3, "hangzhou": 5}
CORE_FEATURES = ("incoming_vehicle_count", "incoming_queue_count")
DEFAULT_LLM_MODEL_ROOT = Path(
    os.environ.get("CITYFLOW_TSC_LLM_MODEL_ROOT", "/mnt/pan/world-model/models")
).expanduser()


@dataclass(frozen=True)
class LoadedWorldModel:
    label: str
    city: str
    checkpoint_path: Path
    checkpoint_sha256: str
    model: PretrainedLatentTrafficModel
    statistics: RichFeatureStatistics
    history_length: int


@dataclass(frozen=True)
class LoadedLlm:
    label: str
    model_path: Path
    model: Any
    tokenizer: Any | None
    input_device: torch.device | None
    backend: str
    load_mode: str


@dataclass(frozen=True)
class LlmSpec:
    """One judge model, loaded only for its own measurement pass.

    ``transformers`` supports the ordinary Hugging Face checkpoints.  ``llama_cpp``
    is deliberately limited to a local GGUF file, which is needed for the official
    Gemma 3 QAT Q4 release.
    """

    label: str
    model_path: Path
    backend: str
    load_mode: str


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty collection")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.percentile(ordered, percentile))


def _summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "count": int(len(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": _percentile(values, 90.0),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _duration_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_validation_manifests(index_path: Path) -> List[Path]:
    payload = _load_json(index_path)
    selected: List[Path] = []
    for item in payload.get("trajectories", []):
        if str(item.get("split", "")).lower() == "validation":
            selected.append(Path(item["manifest_path"]).expanduser().resolve())
    if not selected:
        raise ValueError(f"dataset index has no validation trajectories: {index_path}")
    return sorted(dict.fromkeys(selected))


def _evenly_spaced(items: Sequence[Path], count: int) -> List[Path]:
    if len(items) < count:
        raise ValueError(f"need {count} validation trajectories, found {len(items)}")
    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(position)] for position in positions]


def _case_rows(city: str, index_path: Path, cases_per_city: int) -> List[Dict[str, Any]]:
    manifests = _evenly_spaced(_resolve_validation_manifests(index_path), cases_per_city)
    reader = TrajectoryReader()
    rows = []
    steps = (12, 36, 60, 84)
    for case_number, manifest_path in enumerate(manifests, start=1):
        record = reader.load(manifest_path)
        manifest = record["manifest"]
        if manifest.get("policy_metadata", {}).get("split") != "validation":
            raise ValueError(f"non-validation manifest selected: {manifest_path}")
        step = steps[(case_number - 1) % len(steps)]
        if step + HORIZON > int(manifest["steps"]):
            raise ValueError(f"trajectory is too short for latency case: {manifest_path}")
        rows.append(
            {
                "case_id": f"{city}_V{case_number:02d}",
                "city": city,
                "split": "validation",
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "trajectory_sha256": manifest["trajectory_sha256"],
                "flow_sha256": manifest["flow_sha256"],
                "policy": manifest["policy"],
                "flow_id": manifest.get("policy_metadata", {}).get("flow_id"),
                "decision_step": step,
                "horizon": HORIZON,
            }
        )
    return rows


def _load_world_model(
    label: str, city: str, summary_path: Path, history_length: int, device: str
) -> LoadedWorldModel:
    summary = _load_json(summary_path)
    checkpoint_path = Path(summary["checkpoint_path"]).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != "cityflow-rich-prediction-v2":
        raise ValueError(f"unsupported prediction checkpoint: {checkpoint_path}")
    if payload.get("model_kind") != "pretrained_latent":
        raise ValueError(f"latency benchmark expects a latent model: {checkpoint_path}")
    model_config = TrafficPredictionConfig(**payload["model_config"])
    model = PretrainedLatentTrafficModel(model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return LoadedWorldModel(
        label=label,
        city=city,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        model=model,
        statistics=RichFeatureStatistics.from_dict(payload["statistics"]),
        history_length=history_length,
    )


def _screen_history_length(plan_path: Path, candidate_id: str) -> int:
    with plan_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate_id") == candidate_id:
                return int(row["history_length"])
    raise ValueError(f"candidate {candidate_id} is absent from {plan_path}")


def _load_llm_specs(path: Path, model_root: Path) -> List[LlmSpec]:
    payload = _load_json(path)
    entries = payload.get("llms")
    if not isinstance(entries, list) or not entries:
        raise ValueError("llm config must contain a non-empty 'llms' list")
    specs: List[LlmSpec] = []
    labels = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each LLM configuration entry must be an object")
        label = str(entry.get("label", "")).strip()
        configured_path = Path(str(entry.get("model_path", ""))).expanduser()
        model_path = (
            configured_path
            if configured_path.is_absolute()
            else (model_root / configured_path)
        ).resolve()
        backend = str(entry.get("backend", "transformers")).strip().lower()
        load_mode = str(entry.get("load_mode", "native")).strip().lower()
        if not label or label in labels:
            raise ValueError(f"LLM labels must be present and unique: {label!r}")
        if backend not in {"transformers", "llama_cpp"}:
            raise ValueError(f"unsupported LLM backend for {label}: {backend}")
        if backend == "transformers" and load_mode not in {"native", "bnb4"}:
            raise ValueError(f"unsupported transformers load mode for {label}: {load_mode}")
        if backend == "llama_cpp" and load_mode != "gguf_q4":
            raise ValueError(
                f"llama_cpp requires load_mode=gguf_q4 for {label}, got {load_mode}"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"LLM path does not exist for {label}: {model_path}")
        labels.add(label)
        specs.append(LlmSpec(label, model_path, backend, load_mode))
    return specs


def _load_transformers_llm(spec: LlmSpec) -> LoadedLlm:
    from transformers import (  # Imported lazily: CityFlow-only pilot errors stay clear.
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if spec.load_mode == "bnb4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # DeepSeek-V2-Lite's bundled remote code predates transformers 4.57 and
        # reads ``DynamicCache.seen_tokens``.  The current cache exposes the
        # identical value through ``get_seq_length``.  Preserve KV-cache
        # generation (rather than disabling it and changing latency semantics).
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(  # type: ignore[attr-defined]
                lambda cache: cache.get_seq_length()
            )
        if not hasattr(DynamicCache, "get_max_length"):
            def _get_max_length(cache: Any) -> Optional[int]:
                # The current dynamic cache represents an unbounded maximum as
                # ``-1``; the legacy DeepSeek code expects ``None`` instead.
                maximum_length = cache.get_max_cache_shape()
                return (
                    None
                    if maximum_length is None or maximum_length < 0
                    else int(maximum_length)
                )

            DynamicCache.get_max_length = _get_max_length  # type: ignore[attr-defined]
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(
                cache: Any, new_sequence_length: int, layer_idx: int = 0
            ) -> int:
                previous_length = int(cache.get_seq_length(layer_idx))
                maximum_length = cache.get_max_length()
                if (
                    maximum_length is not None
                    and previous_length + new_sequence_length > maximum_length
                ):
                    return int(maximum_length - new_sequence_length)
                return previous_length

            DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]
    model = AutoModelForCausalLM.from_pretrained(spec.model_path, **options).eval()
    input_device = next(model.parameters()).device
    return LoadedLlm(
        spec.label,
        spec.model_path,
        model,
        tokenizer,
        input_device,
        "transformers",
        spec.load_mode,
    )


def _load_llama_cpp_llm(spec: LlmSpec) -> LoadedLlm:
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise RuntimeError(
            "llama_cpp is required for GGUF judges; install a CUDA-enabled llama-cpp-python "
            "build before running this specification"
        ) from error
    model = Llama(
        model_path=str(spec.model_path),
        n_gpu_layers=-1,
        n_ctx=1024,
        verbose=False,
    )
    return LoadedLlm(
        spec.label,
        spec.model_path,
        model,
        None,
        None,
        "llama_cpp",
        spec.load_mode,
    )


def _load_llm(spec: LlmSpec) -> LoadedLlm:
    if spec.backend == "transformers":
        return _load_transformers_llm(spec)
    if spec.backend == "llama_cpp":
        return _load_llama_cpp_llm(spec)
    raise AssertionError(f"unreachable backend: {spec.backend}")


def _make_dataset_batches(
    case_rows: Sequence[Mapping[str, Any]], model: LoadedWorldModel, device: str
) -> Dict[str, Dict[str, torch.Tensor]]:
    manifests = [Path(row["manifest_path"]) for row in case_rows]
    dataset = RichTrajectorySequenceDataset(
        manifests,
        history_length=model.history_length,
        rollout_horizon=HORIZON,
        statistics=model.statistics,
    )
    index_lookup = {entry: index for index, entry in enumerate(dataset._index)}
    batches: Dict[str, Dict[str, torch.Tensor]] = {}
    for record_index, row in enumerate(case_rows):
        key = (record_index, int(row["decision_step"]))
        if key not in index_lookup:
            raise ValueError(f"missing window {key} for {row['case_id']}")
        item = dataset[index_lookup[key]]
        batches[str(row["case_id"])] = {
            name: value.unsqueeze(0).to(device) for name, value in item.items()
        }
    return batches


def _compact_readout(features: np.ndarray, feature_names: Sequence[str]) -> Dict[str, float]:
    indices = {name: feature_names.index(name) for name in CORE_FEATURES}
    return {
        "mean_incoming_vehicles": float(features[..., indices["incoming_vehicle_count"]].mean()),
        "mean_incoming_queue": float(features[..., indices["incoming_queue_count"]].mean()),
    }


def _world_model_readout(
    model: LoadedWorldModel, batch: Mapping[str, torch.Tensor]
) -> Dict[str, float]:
    with torch.inference_mode():
        output = model.model.rollout(dict(batch))
        features = model.statistics.denormalize_features_tensor(output["features"])
    values = features.detach().float().cpu().numpy()
    return _compact_readout(values, tuple(batch_feature_names(model)))


def batch_feature_names(model: LoadedWorldModel) -> Tuple[str, ...]:
    # The model configuration deliberately contains the feature count rather than
    # labels; rich-data's public schema is the contract used by its checkpoint.
    from cityflow_tsc.observations import RichTrafficObservationBuilder

    names = tuple(RichTrafficObservationBuilder.feature_names)
    if len(names) != model.model.config.feature_count:
        raise ValueError("checkpoint feature count disagrees with rich observation schema")
    return names


def _timed_world_model(
    model: LoadedWorldModel, batch: Mapping[str, torch.Tensor]
) -> Tuple[List[float], Dict[str, float]]:
    with torch.inference_mode():
        for _ in range(WARMUP_REPEATS):
            model.model.rollout(dict(batch))
        _cuda_sync()
        timings = []
        output = None
        for _ in range(MEASURE_REPEATS):
            _cuda_sync()
            start = time.perf_counter_ns()
            output = model.model.rollout(dict(batch))
            _cuda_sync()
            timings.append(_duration_ms(start))
    assert output is not None
    features = model.statistics.denormalize_features_tensor(output["features"])
    return timings, _compact_readout(
        features.detach().float().cpu().numpy(), batch_feature_names(model)
    )


def _new_cityflow_environment(
    row: Mapping[str, Any], output_dir: Path,
) -> Tuple[Any, ScenarioConfig, Dict[str, Any], Mapping[str, np.ndarray]]:
    record = TrajectoryReader().load(Path(row["manifest_path"]))
    manifest = record["manifest"]
    control = ControlConfig(**manifest["control"])
    network = load_network_spec(Path(manifest["scenario"]["roadnet_path"]), control)
    scenario_data = dict(manifest["scenario"])
    scenario_data["output_dir"] = str(output_dir)
    scenario_data["save_replay"] = False
    scenario = ScenarioConfig(**scenario_data)
    env = build_rich_environment(control, network)
    return env, scenario, manifest, record["arrays"]


def _timed_cityflow(
    row: Mapping[str, Any], runtime_root: Path
) -> Tuple[List[float], List[float], Dict[str, float]]:
    rollout_timings: List[float] = []
    context_timings: List[float] = []
    readout: Dict[str, float] | None = None
    for repeat in range(MEASURE_REPEATS):
        env, scenario, manifest, arrays = _new_cityflow_environment(
            row, runtime_root / str(row["case_id"]) / f"repeat_{repeat:02d}"
        )
        try:
            context_start = time.perf_counter_ns()
            observation, _ = env.reset(scenario)
            for action in arrays["actions"][: int(row["decision_step"])]:
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("context replay terminated before measured window")
            context_timings.append(_duration_ms(context_start))
            start = time.perf_counter_ns()
            for action in arrays["actions"][
                int(row["decision_step"]) : int(row["decision_step"]) + HORIZON
            ]:
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("measured CityFlow window terminated early")
            readout = _compact_readout(
                observation.features[None, None], manifest["feature_names"]
            )
            rollout_timings.append(_duration_ms(start))
        finally:
            env.close()
    assert readout is not None
    return rollout_timings, context_timings, readout


def _prompt(source_label: str, row: Mapping[str, Any], readout: Mapping[str, float]) -> str:
    payload = {
        "source": source_label,
        "city": row["city"],
        "horizon": HORIZON,
        "mean_incoming_vehicles": round(float(readout["mean_incoming_vehicles"]), 2),
        "mean_incoming_queue": round(float(readout["mean_incoming_queue"]), 2),
    }
    return (
        "You are a traffic-signal assistant. Given this five-step forecast, "
        "return exactly one JSON object with a short advisory field equal to "
        '"hold" or "relieve". Forecast: '
        + json.dumps(payload, separators=(",", ":"))
    )


def _prepared_inputs(llm: LoadedLlm, prompt: str) -> Dict[str, Any]:
    if llm.backend == "llama_cpp":
        # Tokenization is intentionally outside the measured region, matching the
        # Transformers path below.
        return {"tokens": llm.model.tokenize(prompt.encode("utf-8"), add_bos=True)}
    assert llm.tokenizer is not None and llm.input_device is not None
    encoded = llm.tokenizer(prompt, return_tensors="pt")
    return {name: value.to(llm.input_device) for name, value in encoded.items()}


def _llama_cpp_generate(model: Any, tokens: Sequence[int]) -> List[int]:
    from itertools import islice

    model.reset()
    return list(
        islice(
            model.generate(
                list(tokens), temp=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0
            ),
            24,
        )
    )


def _timed_llm(llm: LoadedLlm, prepared: Mapping[str, Any]) -> Tuple[List[float], str]:
    if llm.backend == "llama_cpp":
        tokens = prepared["tokens"]
        if not isinstance(tokens, list):
            raise TypeError("llama_cpp prepared input must contain token ids")
        for _ in range(WARMUP_REPEATS):
            _llama_cpp_generate(llm.model, tokens)
        _cuda_sync()
        timings = []
        output: List[int] = []
        for _ in range(MEASURE_REPEATS):
            _cuda_sync()
            start = time.perf_counter_ns()
            output = _llama_cpp_generate(llm.model, tokens)
            _cuda_sync()
            timings.append(_duration_ms(start))
        return timings, llm.model.detokenize(output).decode("utf-8", errors="replace")

    assert llm.tokenizer is not None
    generate_options = {
        "do_sample": False,
        "max_new_tokens": 24,
        "use_cache": True,
        "pad_token_id": llm.tokenizer.pad_token_id,
    }
    with torch.inference_mode():
        for _ in range(WARMUP_REPEATS):
            llm.model.generate(**prepared, **generate_options)
        _cuda_sync()
        timings = []
        output = None
        for _ in range(MEASURE_REPEATS):
            _cuda_sync()
            start = time.perf_counter_ns()
            output = llm.model.generate(**prepared, **generate_options)
            _cuda_sync()
            timings.append(_duration_ms(start))
    assert output is not None
    generated = output[0, prepared["input_ids"].shape[1] :]
    return timings, llm.tokenizer.decode(generated, skip_special_tokens=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _append_measurements(
    rows: List[Dict[str, Any]],
    case: Mapping[str, Any],
    source_label: str,
    source_kind: str,
    source_timings: Sequence[float],
    llm: LoadedLlm,
    llm_timings: Sequence[float],
) -> None:
    if len(source_timings) != len(llm_timings):
        raise ValueError("source and LLM repetitions differ")
    for repeat, (source_ms, judge_ms) in enumerate(zip(source_timings, llm_timings), start=1):
        rows.append(
            {
                "case_id": case["case_id"],
                "city": case["city"],
                "split": "validation",
                "source": source_label,
                "source_kind": source_kind,
                "llm": llm.label,
                "repeat": repeat,
                "source_ms": source_ms,
                "llm_judge_only_ms": judge_ms,
                "source_plus_llm_ms": source_ms + judge_ms,
            }
        )


def _full_summary(
    cases: Sequence[Mapping[str, Any]],
    cityflow_rows: Sequence[Mapping[str, Any]],
    world_rows: Sequence[Mapping[str, Any]],
    llm_rows: Sequence[Mapping[str, Any]],
    models: Sequence[LoadedWorldModel],
    llms: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for row in cityflow_rows:
        grouped.setdefault(("cityflow_rollout", str(row["city"])), []).append(
            float(row["cityflow_rollout_ms"])
        )
    for row in world_rows:
        grouped.setdefault(("world_model_rollout", str(row["source"])), []).append(
            float(row["world_model_rollout_ms"])
        )
    for row in llm_rows:
        for metric in ("llm_judge_only_ms", "source_plus_llm_ms"):
            grouped.setdefault(
                (metric, str(row["source"]), str(row["llm"])) , []
            ).append(float(row[metric]))
    return {
        "stage": "complete",
        "scope": "latency_only; validation_only; not prediction or controller evidence",
        "protocol": {
            "cities": list(CITY_ORDER),
            "cases": len(cases),
            "cases_per_city": len(cases) // len(CITY_ORDER),
            "split": "validation",
            "horizon": HORIZON,
            "warmup_repeats": WARMUP_REPEATS,
            "measured_repeats": MEASURE_REPEATS,
            "cityflow_timed_scope": "five-step rollout plus structured result readout",
            "world_model_timed_scope": "loaded model five-step rollout",
            "llm_timed_scope": "loaded model prepared-prompt generate only",
            "excluded_from_primary_latency": [
                "model loading",
                "tokenizer loading",
                "prompt construction",
                "tokenization",
                "CityFlow context replay",
                "warmup",
            ],
        },
        "models": [
            {
                "label": model.label,
                "city": model.city,
                "checkpoint_path": str(model.checkpoint_path),
                "checkpoint_sha256": model.checkpoint_sha256,
                "history_length": model.history_length,
            }
            for model in models
        ],
        "llms": [
            dict(llm)
            for llm in llms
        ],
        "aggregates": [
            {"metric": key[0], "group": list(key[1:]), **_summary(values)}
            for key, values in sorted(grouped.items())
        ],
        "expected_measurements": {
            "cityflow": len(cases) * MEASURE_REPEATS,
            "world_model": len(cases) * 2 * MEASURE_REPEATS,
            "llm_judges": len(cases) * 3 * len(llms) * MEASURE_REPEATS,
        },
        "observed_measurements": {
            "cityflow": len(cityflow_rows),
            "world_model": len(world_rows),
            "llm_judges": len(llm_rows),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--llm-config",
        required=True,
        type=Path,
        help="JSON object with a non-empty 'llms' list; models are run sequentially.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_LLM_MODEL_ROOT,
        help=(
            "Root used for relative model_path entries; defaults to "
            "/mnt/pan/world-model/models."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases-per-city", type=int, default=24)
    parser.add_argument("--pilot-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cases_per_city <= 0:
        raise ValueError("cases-per-city must be positive")
    model_root = args.model_root.expanduser().resolve()
    llm_specs = _load_llm_specs(args.llm_config.expanduser().resolve(), model_root)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    _atomic_json(
        output / "run_manifest.json",
        {
            "stage": "setup",
            "screen_root": str(args.screen_root.resolve()),
            "reference_root": str(args.reference_root.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "llm_config_path": str(args.llm_config.expanduser().resolve()),
            "llm_config_sha256": sha256_file(args.llm_config.expanduser().resolve()),
            "llm_model_root": str(model_root),
            "llms": [
                {
                    "label": spec.label,
                    "model_path": str(spec.model_path),
                    "backend": spec.backend,
                    "load_mode": spec.load_mode,
                }
                for spec in llm_specs
            ],
            "split": "validation",
            "horizon": HORIZON,
            "pilot_only": bool(args.pilot_only),
        },
    )

    all_cases: List[Dict[str, Any]] = []
    for city in CITY_ORDER:
        all_cases.extend(
            _case_rows(
                city,
                args.dataset_root / city / f"{city}_dataset.index.json",
                args.cases_per_city,
            )
        )
    _atomic_json(output / "cases.json", {"split": "validation", "cases": all_cases})

    models: List[LoadedWorldModel] = []
    for city in CITY_ORDER:
        ranking_path = args.screen_root / "rankings" / f"screen_{city}.csv"
        with ranking_path.open("r", encoding="utf-8") as handle:
            leader = next(csv.DictReader(handle))
        leader_summary = Path(leader["summary_path"])
        reference_summary = (
            args.reference_root / "final" / city / REFERENCE_IDS[city] / "prediction_summary.json"
        )
        for label, summary_path, history_length in (
            (
                f"{city}_{REFERENCE_IDS[city]}_reference",
                reference_summary,
                REFERENCE_HISTORY_LENGTHS[city],
            ),
            (
                f"{city}_{leader['candidate_id']}_single_seed_latency_candidate",
                leader_summary,
                _screen_history_length(
                    args.screen_root / "expanded_finetune_plan.csv", leader["candidate_id"]
                ),
            ),
        ):
            models.append(
                _load_world_model(label, city, summary_path, history_length, args.device)
            )

    per_city_models = {
        city: [model for model in models if model.city == city] for city in CITY_ORDER
    }
    batches = {
        model.label: _make_dataset_batches(
            [case for case in all_cases if case["city"] == model.city], model, args.device
        )
        for model in models
    }
    _atomic_json(
        output / "stage.json",
        {"stage": "models_loaded", "models": [model.label for model in models]},
    )

    active_cases = (
        [next(case for case in all_cases if case["city"] == city) for city in CITY_ORDER]
        if args.pilot_only
        else all_cases
    )
    cityflow_rows: List[Dict[str, Any]] = []
    world_rows: List[Dict[str, Any]] = []
    source_readouts: Dict[Tuple[str, str], Dict[str, float]] = {}
    for case in active_cases:
        cityflow_timings, context_timings, cityflow_readout = _timed_cityflow(
            case, output / "cityflow_runtime"
        )
        source_readouts[(case["case_id"], "cityflow")] = cityflow_readout
        for repeat, (rollout_ms, context_ms) in enumerate(
            zip(cityflow_timings, context_timings), start=1
        ):
            cityflow_rows.append(
                {
                    "case_id": case["case_id"],
                    "city": case["city"],
                    "split": "validation",
                    "repeat": repeat,
                    "cityflow_rollout_ms": rollout_ms,
                    "context_replay_ms_excluded": context_ms,
                }
            )
        for model in per_city_models[case["city"]]:
            timings, readout = _timed_world_model(model, batches[model.label][case["case_id"]])
            source_readouts[(case["case_id"], model.label)] = readout
            for repeat, rollout_ms in enumerate(timings, start=1):
                world_rows.append(
                    {
                        "case_id": case["case_id"],
                        "city": case["city"],
                        "split": "validation",
                        "source": model.label,
                        "repeat": repeat,
                        "world_model_rollout_ms": rollout_ms,
                    }
                )
        _atomic_json(
            output / "progress.json",
            {
                "stage": "source_measurement",
                "completed_cases": len({row["case_id"] for row in cityflow_rows}),
                "total_cases": len(active_cases),
                "cityflow_measurements": len(cityflow_rows),
                "world_model_measurements": len(world_rows),
            },
        )
    _write_csv(output / "cityflow_measurements.csv", cityflow_rows)
    _write_csv(output / "world_model_measurements.csv", world_rows)
    _atomic_json(output / "stage.json", {"stage": "sources_complete", "cases": len(active_cases)})

    if args.pilot_only:
        _atomic_json(
            output / "pilot_complete.json",
            {
                "stage": "pilot_complete",
                "cases": active_cases,
                "cityflow_measurements": len(cityflow_rows),
                "world_model_measurements": len(world_rows),
                "note": "LLM loading and full benchmark intentionally deferred until pilot source validation.",
            },
        )
        return 0

    source_timings: Dict[Tuple[str, str], List[float]] = {}
    for row in cityflow_rows:
        source_timings.setdefault((str(row["case_id"]), "cityflow"), []).append(
            float(row["cityflow_rollout_ms"])
        )
    for row in world_rows:
        source_timings.setdefault((str(row["case_id"]), str(row["source"])), []).append(
            float(row["world_model_rollout_ms"])
        )

    llm_rows: List[Dict[str, Any]] = []
    successful_llms: List[Dict[str, str]] = []
    failed_llms: List[Dict[str, str]] = []
    for spec in llm_specs:
        llm: LoadedLlm | None = None
        rows_before = len(llm_rows)
        try:
            llm = _load_llm(spec)
            successful_llms.append(
                {
                    "label": llm.label,
                    "model_path": str(llm.model_path),
                    "backend": llm.backend,
                    "load_mode": llm.load_mode,
                }
            )
            for case in active_cases:
                labels = ["cityflow"] + [model.label for model in per_city_models[case["city"]]]
                for source_label in labels:
                    prepared = _prepared_inputs(
                        llm,
                        _prompt(
                            source_label,
                            case,
                            source_readouts[(case["case_id"], source_label)],
                        ),
                    )
                    # Prompt construction and tokenization happen before the timer.
                    llm_timings, response = _timed_llm(llm, prepared)
                    _append_measurements(
                        llm_rows,
                        case,
                        source_label,
                        "cityflow" if source_label == "cityflow" else "world_model",
                        source_timings[(case["case_id"], source_label)],
                        llm,
                        llm_timings,
                    )
                    (output / "logs" / f"{case['case_id']}_{source_label}_{llm.label}.txt").write_text(
                        response, encoding="utf-8"
                    )
                _atomic_json(
                    output / "progress.json",
                    {
                        "stage": "llm_measurement",
                        "active_llm": llm.label,
                        "completed_cases_for_active_llm": case["case_id"],
                        "total_cases": len(active_cases),
                        "llm_measurements": len(llm_rows),
                    },
                )
        except Exception as error:
            # A partial model pass must never leak into the aggregate comparison.
            del llm_rows[rows_before:]
            failed_llms.append(
                {
                    "label": spec.label,
                    "model_path": str(spec.model_path),
                    "backend": spec.backend,
                    "load_mode": spec.load_mode,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            (output / "logs" / f"{spec.label}_failure.txt").write_text(
                failed_llms[-1]["error"], encoding="utf-8"
            )
        finally:
            if llm is not None:
                del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _write_csv(output / "llm_measurements.csv", llm_rows)
    summary = _full_summary(
        active_cases,
        cityflow_rows,
        world_rows,
        llm_rows,
        models,
        successful_llms,
    )
    summary["model_failures"] = failed_llms
    summary["stage"] = "complete" if not failed_llms else "incomplete_models"
    _atomic_json(output / "latency_summary.json", summary)
    _atomic_json(
        output / "stage.json",
        {
            "stage": summary["stage"],
            "cases": len(active_cases),
            "successful_llms": len(successful_llms),
            "failed_llms": len(failed_llms),
        },
    )
    return 0 if not failed_llms else 2


if __name__ == "__main__":
    raise SystemExit(main())
