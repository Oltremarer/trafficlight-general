"""v10 old-model ensembles, locked new-B evaluation, and factor error attribution."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model import connection_revision as v10, decision_revision as v9
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, tie_argmin, decision_metrics
from .effect_model.local_metrics import predict_single4, predict_pair4, field_record
from .effect_model.local_revision import compose
from .evaluate_decision_revision import summary, split_results
from .evaluate_local_revision import (LOCKED_FILES as OLD_FILES, lock_checkpoints as base_lock,
                                     verify_lock as base_verify, infer_joint)
from .train_coarse_effects import read
from .train_formal_effects import initialize, now
from .train_temporal_effects import SEEDS

LOCKED_FILES = (*OLD_FILES, 'training_cache.json', 'fixed_A_D.ready.json')
ENSEMBLES = ('A_Mean3', 'A_Mean3+B_Match_Mean3_24', 'A_Mean3+B_Match_Consensus3_24')


def lock_checkpoints(run):
    base_lock(run, v10.FAMILIES, v10.SCHEDULES, LOCKED_FILES)


def verify_lock(run):
    base_verify(run, v10.FAMILIES, LOCKED_FILES)
    ready = read(run / 'fixed_A_D.ready.json')
    for name, digest in ready['files'].items():
        if sha256(run / name) != digest:
            raise ValueError('Fixed A validation predictions changed')


def chosen_metrics(scores, truth, selected):
    """Report the policy's actual choice without falsifying its predicted field."""
    scores, truth = np.asarray(scores, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if not 0 <= selected < len(scores) or scores.shape != truth.shape:
        raise ValueError('Invalid chosen candidate')
    result = decision_metrics(scores, truth)
    optimum, value = float(truth.min()), float(truth[selected])
    benefit = float(truth[0] - value)
    available = float(truth[0] - optimum)
    result.update(selected=int(selected), regret=value - optimum,
        optimal_choice=bool(value <= optimum + 1e-6), worse_than_reference=bool(value > truth[0] + 1e-6),
        excess_wait_vs_reference=max(0., value - float(truth[0])), benefit_vs_reference=benefit,
        benefit_capture=benefit / available if available else None,
        predicted_delta_J=float(scores[selected]), true_delta_J=value)
    return result


def complete_record(root, joint, single, selected, elapsed_ms, **extra):
    result = field_record(root, joint, single)
    truth = np.asarray(root['joint4'], dtype=np.float64)
    scores, true_scores = joint[:, :, -1].sum(1), truth[:, :, -1].sum(1)
    nonzero = np.abs(true_scores) > 1e-6
    result.update(chosen_metrics(scores, true_scores, selected))
    result.update(joint_total240_mae=float(np.abs(scores - true_scores).mean()),
        joint_nonzero_total_count=int(nonzero.sum()),
        joint_sign_error_count=int((np.sign(scores[nonzero]) != np.sign(true_scores[nonzero])).sum()),
        inference_ms=elapsed_ms, **extra)
    return result


def ensemble_infer(a_models, b_models, root, ss, ps, device, method):
    singles, joints = [], []
    for seed in SEEDS:
        single = predict_single4(a_models[seed], root, ss, device)
        joint = compose(root['s_incidence'], single)
        singles.append(single)
        if method != 'A_Mean3':
            indices, pair = predict_pair4(b_models[seed], root, ps, device, 24)
            joint += compose(root['p_incidence'][:, indices], pair)
        joints.append(joint)
    mean_single, mean_joint = np.mean(singles, axis=0), np.mean(joints, axis=0)
    details = {}
    if method == ENSEMBLES[2]:
        s_scores = np.stack([compose(root['s_incidence'], s)[:, :, -1].sum(1) for s in singles])
        j_scores = np.stack([j[:, :, -1].sum(1) for j in joints])
        details = v10.consensus_choice(s_scores, j_scores)
        selected = details.pop('selected')
    else:
        selected = tie_argmin(mean_joint[:, :, -1].sum(1))
    return mean_joint, mean_single, selected, details


def initialize_evaluation(run):
    initialize(42)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for evaluation')
    device = torch.device('cuda:0')
    ss = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    ps = tensor_scales(_read_npz(run / 'pair_normalization.npz'), 'pair', device)
    return device, ss, ps


def ensemble_rows(run, roots, device, ss, ps):
    a = {s: v10.make_model(run, 'A_D_v8', s, device, True).eval() for s in SEEDS}
    b = {s: v10.make_model(run, 'B_Match', s, device, True).eval() for s in SEEDS}
    for method in ENSEMBLES:
        ensemble_infer(a, b, roots[0], ss, ps, device, method)
        rows = []
        for root in roots:
            torch.cuda.synchronize()
            started = time.perf_counter()
            joint, single, choice, details = ensemble_infer(a, b, root, ss, ps, device, method)
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - started)
            rows.append(complete_record(root, joint, single, choice, elapsed,
                pair_budget=0 if method == 'A_Mean3' else 24,
                encoder_passes=3 if method == 'A_Mean3' else 6, **details))
        yield method, rows


def evaluate_old_validation(run):
    """Independent of training; never opens diagnostic inputs or labels."""
    run = Path(run)
    device, ss, ps = initialize_evaluation(run)
    roots = v10.load_roots(run, 'A_D_v8', ('validation',))
    if len(roots) != 78:
        raise ValueError('Expected 78 validation roots')
    selection_ids = {r['root_id'] for r in read(run / 'b_roots.json') if r['split'] == 'validation'}
    directory = run / 'ensemble_validation'
    directory.mkdir(exist_ok=True)
    files = {}
    for method, rows in ensemble_rows(run, roots, device, ss, ps):
        metrics = {'validation': summary(rows),
            'B_selection18': summary([r for r in rows if r['root_id'] in selection_ids]),
            'B_remaining60': summary([r for r in rows if r['root_id'] not in selection_ids])}
        path = directory / (method + '.json')
        if path.exists():
            raise RuntimeError('Existing ensemble results retained; duplicate evaluation refused')
        atomic_json(path, {'method': method, 'metrics': metrics, 'protocol_sha256': sha256(run / 'protocol.json')})
        files[path.name] = sha256(path)
        atomic_json(directory / 'progress.json', {'stage': 'evaluating', 'methods': len(files),
            'total_methods': 3, 'updated_at': now()})
    atomic_json(directory / 'summary.json', {'stage': 'complete', 'files': files, 'finished_at': now(),
        'test_used': False, 'not_used_for_training_or_selection': True,
        'timing_scope': 'three or six encoder passes, actual concurrent workload; not an isolated benchmark'})
    atomic_json(directory / 'progress.json', {'stage': 'complete', 'methods': 3, 'total_methods': 3, 'updated_at': now()})


def factor_attribution(root, predicted_single, predicted_pair):
    """Signed score-gap identity, using all true pair labels available at this root."""
    s = compose(root['s_incidence'], np.asarray(root['single4'], dtype=np.float64))[:, :, -1].sum(1)
    p = compose(root['p_incidence'], np.asarray(root['pair4'], dtype=np.float64))[:, :, -1].sum(1)
    hs = compose(root['s_incidence'], predicted_single)[:, :, -1].sum(1)
    hp = compose(root['p_incidence'], predicted_pair)[:, :, -1].sum(1)
    y = np.asarray(root['joint4'], dtype=np.float64)[:, :, -1].sum(1)
    selected, optimum = tie_argmin(hs + hp), tie_argmin(y)
    gap = lambda values: float(values[selected] - values[optimum])
    single_error, pair_error, higher = gap(hs - s), gap(hp - p), gap(y - s - p)
    discrepancy = gap(hs + hp) - gap(y) - (single_error + pair_error - higher)
    if abs(discrepancy) > 1e-6:
        raise ValueError('Factor score-gap decomposition is inconsistent')
    return {**{k: root[k] for k in ('root_id', 'split', 'flow_id', 'cohort_id', 'policy', 'time_s')},
        'selected': selected, 'true_optimum': optimum, 'true_regret': gap(y),
        'predicted_gap': gap(hs + hp), 'single_gap_error': single_error,
        'pair_gap_error': pair_error, 'higher_order_gap': higher,
        'decomposition_residual_max_abs': float(np.max(np.abs(y - s - p))),
        'True_S': decision_metrics(s, y), 'True_SP': decision_metrics(s + p, y),
        'Pred_S_True_P': decision_metrics(hs + p, y),
        'True_S_Pred_P': decision_metrics(s + hp, y)}


def evaluate_final(run):
    run = Path(run)
    verify_lock(run)
    v9.reuse_labels(run, test=True)
    device, ss, ps = initialize_evaluation(run)
    roots = v10.load_roots(run, 'A_D_v8', ('validation', 'test'))
    pairs = v10.load_roots(run, 'B_Adapter', ('validation',))
    selection_ids = {r['root_id'] for r in pairs}
    directory = run / 'evaluation'
    directory.mkdir(exist_ok=True)
    ensemble_manifest = read(run / 'ensemble_validation/summary.json')
    if ensemble_manifest['stage'] != 'complete':
        raise ValueError('Old-model validation is incomplete')
    ensembles = {}
    for name, rows in ensemble_rows(run, [r for r in roots if r['split'] == 'test'], device, ss, ps):
        prior_path = run / 'ensemble_validation' / (name + '.json')
        if sha256(prior_path) != ensemble_manifest['files'][prior_path.name]:
            raise ValueError('Old-model validation artifact changed')
        prior = read(prior_path)
        if prior['protocol_sha256'] != sha256(run / 'protocol.json'):
            raise ValueError('Ensemble protocol changed')
        ensembles[name] = split_results(prior['metrics']['validation']['records'] + rows, selection_ids)
        atomic_json(directory / (name + '.json'), ensembles[name])
    results, components, attribution = {}, {}, {}
    for seed in SEEDS:
        a = v10.make_model(run, 'A_D_v8', seed, device, True).eval()
        b = v10.make_model(run, 'B_Adapter', seed, device, True).eval()
        v10.attach_fixed_predictions(run, pairs, seed)
        components[str(seed)] = v10.validate_model(b, pairs, ss, ps, device, 'B_Adapter')
        for budget in (24, 120):
            method = f'A_D_v8+B_Adapter_{budget}'
            infer_joint(a, b, roots[0], ss, ps, device, 'A_D_v8', budget)
            rows = []
            for root in roots:
                torch.cuda.synchronize()
                started = time.perf_counter()
                joint, single, choice = infer_joint(a, b, root, ss, ps, device, 'A_D_v8', budget)
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - started)
                rows.append(complete_record(root, joint, single, choice, elapsed, pair_budget=budget, encoder_passes=2))
            metrics = split_results(rows, selection_ids)
            results.setdefault(method, {})[str(seed)] = metrics
            atomic_json(directory / f'{method}_seed_{seed}.json', metrics)
            atomic_json(directory / 'progress.json', {'stage': 'evaluating',
                'completed_method_seeds': sum(map(len, results.values())), 'total_method_seeds': 6,
                'ensemble_methods': 3, 'updated_at': now()})
        old_b = v10.make_model(run, 'B_Match', seed, device, True).eval()
        for family, model in (('B_Match', old_b), ('B_Adapter', b)):
            records = []
            for root in pairs:
                _, prediction = predict_pair4(model, root, ps, device, 120)
                records.append(factor_attribution(root, np.asarray(root['fixed_A_D4'], dtype=np.float64), prediction))
            attribution.setdefault(family, {})[str(seed)] = records
        del a, b, old_b
    comparison = {}
    for method, seeds in results.items():
        comparison[method] = {}
        for split in ('validation', 'test', 'B_selection18', 'B_remaining60'):
            metrics = [seeds[str(seed)][split] for seed in SEEDS]
            comparison[method][split] = {'regret_seeds': [m['regret'] for m in metrics],
                **{key + '_mean': float(np.mean([m[key] for m in metrics])) for key in
                   ('regret', 'joint4_mae', 'joint_total240_mae', 'single4_mae', 'single_total240_mae', 'inference_ms')},
                'worse_than_reference_count': sum(m['worse_than_reference_count'] for m in metrics),
                'root_seed_count': sum(m['root_count'] for m in metrics)}
    prior_path = Path(read(run / 'protocol.json')['reuse_v9']) / 'evaluation/summary.json'
    prior = read(prior_path)
    atomic_json(directory / 'factor_attribution.json', {'scope': '18 B-selection validation roots; descriptive, not independent confirmation',
        'identity': 'predicted_gap-true_gap = single_gap_error+pair_gap_error-higher_order_gap', 'results': attribution})
    atomic_json(directory / 'summary.json', {'stage': 'complete', 'results': results,
        'comparison': comparison, 'ensembles': ensembles, 'B_component_validation18': components,
        'prior_comparison': {k: prior['comparison'][k] for k in
            ('A_D_v8', 'A_D_v8+B_Match_24', 'A_D_v8+B_Match_120')},
        'prior_summary_source': {'path': str(prior_path), 'sha256': sha256(prior_path)},
        'attribution': str(directory / 'factor_attribution.json'),
        'test_scope': 'previously inspected12-root diagnostic, not blind or new demand',
        'selection_used_test': False, 'ensemble_claim': 'one fixed ensemble per method, not three new seeds; unanimity is not calibrated confidence',
        'timing_scope': 'actual encoder passes and FP64 field composition; cached validation timing may overlap training; not a dedicated latency benchmark',
        'finished_at': now()})
    atomic_json(directory / 'progress.json', {'stage': 'complete', 'completed_method_seeds': 6,
        'total_method_seeds': 6, 'ensemble_methods': 3, 'updated_at': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('ensemble-validation', 'final'), required=True)
    args = parser.parse_args()
    (evaluate_old_validation if args.stage == 'ensemble-validation' else evaluate_final)(args.run_dir)


if __name__ == '__main__':
    main()
