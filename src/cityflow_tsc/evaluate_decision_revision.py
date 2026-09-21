"""v9 locked evaluation: fixed-v8-A main pair comparison and secondary new-A crosses."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, group_mean
from .effect_model.local_metrics import field_record, summarize
from .effect_model import decision_revision as v9
from .evaluate_local_revision import (LOCKED_FILES as V8_FILES, lock_checkpoints as base_lock,
    verify_lock as base_verify, infer_joint)
from .train_coarse_effects import read
from .train_formal_effects import initialize, now
from .train_temporal_effects import SEEDS

LOCKED_FILES = (*V8_FILES, 'pair_contrast_scale.json', 'training_cache.json', 'fixed_A_D.ready.json')
A_FAMILIES = ('A_D_v8', 'A_D_Top8', 'A_D_Local')
B_FAMILIES = ('B_Match', 'B_Contrast')


def lock_checkpoints(run):
    base_lock(run, v9.FAMILIES, v9.SCHEDULES, LOCKED_FILES)


def verify_lock(run):
    base_verify(run, v9.FAMILIES, LOCKED_FILES)
    ready = read(run / 'fixed_A_D.ready.json')
    for name, digest in ready['files'].items():
        if sha256(run / name) != digest:
            raise ValueError('Frozen selection prediction changed')


def summary(rows):
    result = summarize(rows)
    result['joint_total240_mae'] = group_mean(rows, 'joint_total240_mae')
    result['by_flow'] = {str(k): summarize([r for r in rows if r['flow_id'] == k])
                         for k in sorted({r['flow_id'] for r in rows})}
    result['by_cohort'] = {str(k): summarize([r for r in rows if r['cohort_id'] == k])
                           for k in sorted({r['cohort_id'] for r in rows})}
    result['independent_flow_count'] = len(result['by_flow'])
    return result


def split_results(rows, selection_ids):
    validation = [r for r in rows if r['split'] == 'validation']
    diagnostic = [r for r in rows if r['split'] == 'test']
    selected = [r for r in validation if r['root_id'] in selection_ids]
    remaining = [r for r in validation if r['root_id'] not in selection_ids]
    if tuple(map(len, (validation, diagnostic, selected, remaining))) != (78, 12, 18, 60):
        raise ValueError('Evaluation and selection populations changed')
    return {k: summary(v) for k, v in (('validation', validation), ('test', diagnostic),
        ('B_selection18', selected), ('B_remaining60', remaining))}


def evaluate(run):
    run = Path(run)
    verify_lock(run)
    v9.reuse_labels(run, test=True)
    roots = v9.load_roots(run, 'A_D_Top8', ('validation', 'test'))
    pair_roots = v9.load_roots(run, 'B_Match', ('validation',))
    selection_ids = {r['root_id'] for r in pair_roots}
    initialize(42)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for final evaluation')
    device = torch.device('cuda:0')
    ss = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    ps = tensor_scales(_read_npz(run / 'pair_normalization.npz'), 'pair', device)
    directory = run / 'evaluation'
    directory.mkdir(exist_ok=True)
    results, components = {}, {}
    for seed in SEEDS:
        v9.attach_fixed_predictions(run, pair_roots, seed)
        pairs = {f: v9.make_model(run, f, seed, device, selected=True).eval() for f in B_FAMILIES}
        for family, model in pairs.items():
            components.setdefault(family, {})[str(seed)] = v9.validate_model(model, pair_roots, ss, ps, device, family)
        for family in A_FAMILIES:
            a = v9.make_model(run, family, seed, device, selected=True).eval()
            combinations = [(None, 0)] + [(b, budget) for b in B_FAMILIES for budget in (24, 120)]
            for b_family, budget in combinations:
                name = family + (f'+{b_family}_{budget}' if budget else '')
                b = pairs[b_family] if budget else None
                infer_joint(a, b, roots[0], ss, ps, device, family, budget)
                torch.cuda.synchronize()
                rows = []
                for root in roots:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    joint, single, choice = infer_joint(a, b, root, ss, ps, device, family, budget)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    record = field_record(root, joint, single)
                    if record['selected'] != choice:
                        raise ValueError('Timed decision differs from reported choice')
                    target = np.asarray(root['joint4'], dtype=np.float64)
                    predicted_total, true_total = joint[:, :, -1].sum(1), target[:, :, -1].sum(1)
                    nonzero = np.abs(true_total) > 1e-6
                    record.update(joint_total240_mae=float(np.abs((joint - target)[:, :, -1].sum(1)).mean()),
                        joint_nonzero_total_count=int(nonzero.sum()),
                        joint_sign_error_count=int((np.sign(predicted_total[nonzero]) != np.sign(true_total[nonzero])).sum()),
                        inference_ms=elapsed, pair_budget=budget, pair_queries=budget * 16,
                        encoder_passes=2 if budget else 1)
                    rows.append(record)
                metrics = split_results(rows, selection_ids)
                results.setdefault(name, {})[str(seed)] = metrics
                atomic_json(directory / f'{name}_seed_{seed}.json', metrics)
                atomic_json(directory / 'progress.json', {'stage': 'evaluating', 'method': name, 'seed': seed,
                    'completed_method_seeds': sum(len(s) for s in results.values()), 'total_method_seeds': 45,
                    'updated_at': now()})
            del a
        del pairs
    comparison = {}
    for name, seeds in results.items():
        comparison[name] = {}
        for split in ('validation', 'test', 'B_selection18', 'B_remaining60'):
            metrics = [seeds[str(s)][split] for s in SEEDS]
            comparison[name][split] = {'regret_seeds': [m['regret'] for m in metrics],
                **{k + '_mean': float(np.mean([m[k] for m in metrics])) for k in
                   ('regret', 'joint4_mae', 'joint_total240_mae', 'single4_mae', 'single_total240_mae', 'inference_ms')},
                'worse_than_reference_count': sum(m['worse_than_reference_count'] for m in metrics),
                'root_seed_count': sum(m['root_count'] for m in metrics)}
    source = Path(read(run / 'protocol.json')['reuse_v8']) / 'evaluation/summary.json'
    prior = read(source)
    atomic_json(directory / 'summary.json', {'stage': 'complete', 'results': results, 'comparison': comparison,
        'B_component_validation18': components,
        'v8_comparison': {k: prior['comparison'][k] for k in ('A_D', 'A_D+B_L4_24', 'A_D+B_L4_120')},
        'v8_summary_source': {'path': str(source), 'sha256': sha256(source)},
        'primary_B_comparison': 'A_D_v8 with B_Match or B_Contrast vs frozen v8 A_D and A_D+B_L4',
        'secondary_comparison': 'predeclared new A_D_Top8 and A_D_Local crossed with new B; no diagnostic-based selection',
        'test_scope': 'previously inspected 12-root diagnostic, not blind and not new demand',
        'timing_scope': 'one warm pass; H2D, independent A/B encoders, selected decoders, D2H, FP64 composition and choice; no truth metrics/file loading; not a dedicated latency benchmark',
        'selection_used_test': False, 'finished_at': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    evaluate(parser.parse_args().run_dir)


if __name__ == '__main__':
    main()
