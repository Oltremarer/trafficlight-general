"""v11 locked A-only and fixed-A/new-B comparisons; no new-A/new-B cross search."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model import coverage_revision as v11, decision_revision as v9
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, group_mean
from .evaluate_connection_revision import complete_record
from .evaluate_decision_revision import split_results
from .evaluate_local_revision import LOCKED_FILES as BASE_FILES, lock_checkpoints as base_lock, verify_lock as base_verify, infer_joint
from .train_coarse_effects import read
from .train_formal_effects import initialize, now
from .train_temporal_effects import SEEDS

LOCKED_FILES = (*BASE_FILES, 'training_cache.json', 'fixed_A_D.ready.json',
                'collection/selection.json', 'collection/complete.json')


def lock_checkpoints(run):
    base_lock(run, v11.FAMILIES, v11.SCHEDULES, LOCKED_FILES)


def evaluate(run):
    run = Path(run)
    base_verify(run, v11.FAMILIES, LOCKED_FILES)
    # Only the old validation/test cache is reused here; test has no pair labels.
    v9.reuse_labels(run, test=True)
    roots = v11.load_roots(run, 'A_ActionMargin', ('validation', 'test'))
    pair_roots = v11.load_roots(run, 'B_Coverage', ('validation',))
    selection_ids = {r['root_id'] for r in pair_roots}
    initialize(42)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    ss = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    ps = tensor_scales(_read_npz(run / 'pair_normalization.npz'), 'pair', device)
    directory = run / 'evaluation'
    directory.mkdir(exist_ok=True)
    results, components = {}, {}
    for seed in SEEDS:
        b = v11.make_model(run, 'B_Coverage', seed, device, selected=True).eval()
        v11.attach_fixed_predictions(run, pair_roots, seed)
        components[str(seed)] = v11.validate_model(b, pair_roots, ss, ps, device, 'B_Coverage')
        for family, budgets in (('A_ActionMargin', (0,)), ('A_D_v8', (0, 24, 120))):
            a = v11.make_model(run, family, seed, device, selected=True).eval()
            for budget in budgets:
                name = family + (f'+B_Coverage_{budget}' if budget else '')
                infer_joint(a, b, roots[0], ss, ps, device, family, budget)
                torch.cuda.synchronize()
                rows = []
                for root in roots:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    joint, single, choice = infer_joint(a, b, root, ss, ps, device, family, budget)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    record = complete_record(root, joint, single, choice, elapsed,
                        pair_budget=budget, pair_queries=16 * budget, encoder_passes=2 if budget else 1)
                    record.update(v11.single_choice_metrics(root, single))
                    rows.append(record)
                metrics = split_results(rows, selection_ids)
                for value in metrics.values():
                    value['single_action_regret'] = group_mean(value['records'], 'single_action_regret')
                    for sub in value['by_flow'].values():
                        sub['single_action_regret'] = group_mean(sub['records'], 'single_action_regret')
                results.setdefault(name, {})[str(seed)] = metrics
                atomic_json(directory / f'{name}_seed_{seed}.json', metrics)
                atomic_json(directory / 'progress.json', {'stage': 'evaluating', 'method': name, 'seed': seed,
                    'completed_method_seeds': sum(len(s) for s in results.values()), 'total_method_seeds': 12,
                    'updated_at': now()})
            del a
        del b
    comparison = {}
    for name, seeds in results.items():
        comparison[name] = {}
        for split in ('validation', 'test', 'B_selection18', 'B_remaining60'):
            metrics = [seeds[str(s)][split] for s in SEEDS]
            comparison[name][split] = {'regret_seeds': [m['regret'] for m in metrics],
                **{k + '_mean': float(np.mean([m[k] for m in metrics])) for k in
                    ('regret', 'joint4_mae', 'joint_total240_mae', 'single4_mae', 'single_total240_mae',
                     'single_action_regret', 'inference_ms')},
                'worse_than_reference_count': sum(m['worse_than_reference_count'] for m in metrics),
                'root_seed_count': sum(m['root_count'] for m in metrics)}
    prior = Path(read(run / 'protocol.json')['reuse_v10']) / 'evaluation/summary.json'
    atomic_json(directory / 'summary.json', {'stage': 'complete', 'results': results, 'comparison': comparison,
        'B_component_validation18': components, 'v10_comparison': read(prior)['comparison'],
        'v10_summary_source': {'path': str(prior), 'sha256': sha256(prior)},
        'test_scope': 'Previously inspected diagnostic; not blind, new-demand, or closed-loop evidence',
        'timing_scope': 'warm end-to-end model scoring; may overlap collection, not dedicated latency benchmark',
        'finished_at': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    evaluate(parser.parse_args().run_dir)
