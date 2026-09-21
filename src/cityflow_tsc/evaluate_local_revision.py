"""Locked v8 comparison on common 272 x 4 fields, with honest two-encoder timing."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, tie_argmin
from .effect_model.local_data import load_roots, prepare_labels
from .effect_model.local_metrics import make_model, predict_single4, predict_pair4, field_record, summarize
from .effect_model.local_revision import FAMILIES, SCHEDULES, compose
from .train_coarse_effects import read
from .train_formal_effects import initialize, now
from .train_temporal_effects import SEEDS

LOCKED_FILES = ('protocol.json', 'static.npz', 'normalization.npz', 'pair_normalization.npz',
                'contrast_scale.json', 'a_roots.json', 'b_roots.json', 'source_hashes.json')


def lock_checkpoints(run, families=FAMILIES, schedules=SCHEDULES, locked_files=LOCKED_FILES):
    run = Path(run)
    if (run / 'checkpoints_locked.json').exists():
        raise RuntimeError('Existing checkpoint lock is retained')
    checkpoints = {}
    for family in families:
        for seed in SEEDS:
            directory = run / family / f'seed_{seed}'
            done = read(directory / 'complete.json')
            digest = sha256(directory / 'best.pt')
            if (done['stage'] != 'complete' or done['family'] != family or done['seed'] != seed or
                    done['updates'] != schedules[family]['updates'] or done['test_used'] or
                    done['checkpoint_sha256'] != digest):
                raise ValueError('Incomplete or changed training artifact')
            checkpoints[f'{family}/seed_{seed}/best.pt'] = digest
    atomic_json(run / 'checkpoints_locked.json', {'checkpoints': checkpoints,
        'files': {f: sha256(run / f) for f in locked_files}, 'locked_at': now(), 'test_used': False})


def verify_lock(run, families=FAMILIES, locked_files=LOCKED_FILES):
    lock = read(run / 'checkpoints_locked.json')
    expected = {f'{f}/seed_{s}/best.pt' for f in families for s in SEEDS}
    if set(lock['checkpoints']) != expected or set(lock['files']) != set(locked_files) or lock['test_used']:
        raise ValueError('All twelve new checkpoints must be locked together')
    for name, digest in {**lock['checkpoints'], **lock['files']}.items():
        if sha256(run / name) != digest:
            raise ValueError('Locked artifact changed: ' + name)


def infer_joint(a, b, root, single_scales, pair_scales, device, family, budget):
    single = predict_single4(a, root, single_scales, device, family == 'A_L4')
    joint = compose(root['s_incidence'], single)
    if budget:
        indices, pair = predict_pair4(b, root, pair_scales, device, budget)
        joint += compose(root['p_incidence'][:, indices], pair)
    selected = tie_argmin(joint[:, :, -1].sum(1))
    return joint, single, selected


def evaluate(run):
    run = Path(run)
    verify_lock(run)
    prepare_labels(run, test=True)
    roots = load_roots(run, 'A_L4', ('validation', 'test'))
    if {s: sum(r['split'] == s for r in roots) for s in ('validation', 'test')} != {'validation': 78, 'test': 12}:
        raise ValueError('Evaluation population mismatch')
    initialize(42)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for final comparison')
    device = torch.device('cuda:0')
    single_scales = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    pair_scales = tensor_scales(_read_npz(run / 'pair_normalization.npz'), 'pair', device)
    directory = run / 'evaluation'
    directory.mkdir(exist_ok=True)
    results, comparisons, components = {}, {}, {}
    for seed in SEEDS:
        b = make_model(run, 'B_L4', seed, device, selected=True).eval()
        components[str(seed)] = read(run / 'B_L4' / f'seed_{seed}/best_validation.json')
        for family in ('A_ref', 'A_C', 'A_D', 'A_L4'):
            a = make_model(run, family, seed, device, selected=True).eval()
            for budget in (0, 24, 120):
                name = family + (f'+B_L4_{budget}' if budget else '')
                # Warm-up is input-only; the timed path repeats BOTH encoders when B is used.
                infer_joint(a, b, roots[0], single_scales, pair_scales, device, family, budget)
                torch.cuda.synchronize()
                rows = []
                for root in roots:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    joint, single, selected = infer_joint(a, b, root, single_scales, pair_scales, device, family, budget)
                    torch.cuda.synchronize()
                    elapsed = 1000 * (time.perf_counter() - started)
                    record = field_record(root, joint, single)
                    if record['selected'] != selected:
                        raise ValueError('Timed and reported decisions differ')
                    rows.append({**record, 'inference_ms': elapsed, 'pair_budget': budget,
                                 'pair_queries': budget * 16, 'encoder_passes': 2 if budget else 1})
                split_metrics = {s: summarize([r for r in rows if r['split'] == s]) for s in ('validation', 'test')}
                results.setdefault(name, {})[str(seed)] = split_metrics
                atomic_json(directory / f'{name}_seed_{seed}.json', split_metrics)
                atomic_json(directory / 'progress.json', {'stage': 'evaluating', 'method': name, 'seed': seed,
                    'completed_method_seeds': sum(len(v) for v in results.values()), 'total_method_seeds': 36,
                    'updated_at': now()})
            del a
        del b
    for name, seeds in results.items():
        comparisons[name] = {}
        for split in ('validation', 'test'):
            metrics = [seeds[str(s)][split] for s in SEEDS]
            comparisons[name][split] = {
                'regret_seeds': [m['regret'] for m in metrics],
                'regret_mean': float(np.mean([m['regret'] for m in metrics])),
                'joint4_mae_mean': float(np.mean([m['joint4_mae'] for m in metrics])),
                'inference_ms_mean': float(np.mean([m['inference_ms'] for m in metrics])),
                'worse_than_reference_count': sum(m['worse_than_reference_count'] for m in metrics),
                'root_seed_count': sum(m['root_count'] for m in metrics)}
    baselines = {}
    for name in ('Zero', 'True-S'):
        rows = [field_record(r, np.zeros((65, 272, 4)) if name == 'Zero'
                            else compose(r['s_incidence'], np.asarray(r['single4'], dtype=np.float64))) for r in roots]
        baselines[name] = {s: summarize([r for r in rows if r['split'] == s]) for s in ('validation', 'test')}
    atomic_json(directory / 'summary.json', {'stage': 'complete', 'results': results, 'comparison': comparisons,
        'baselines': baselines, 'B_component_validation_18': components,
        'test_scope': 'previously inspected diagnostic, not blind or new demand',
        'common_prediction_metric': 'natural mean MAE over 65 candidates x 272 positions x 4 cumulative horizons',
        'timing_scope': 'single pass per root after input-only warm-up; includes H2D, both independent encoders for A+B, selected decoders, D2H, FP64 full-field composition and choice; excludes loading checkpoints/files and truth metrics; not a dedicated latency benchmark',
        'selection_used_test': False, 'finished_at': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run_dir)


if __name__ == '__main__':
    main()
