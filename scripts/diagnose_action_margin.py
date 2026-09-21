"""Read-only validation and training-label diagnostics; never opens test roots."""
import argparse
from pathlib import Path

import numpy as np
import torch

from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from cityflow_tsc.effect_model import coverage_revision as v11
from cityflow_tsc.effect_model.formal_data import _read_npz
from cityflow_tsc.effect_model.formal_model import tensor_scales, group_mean, multiscale_loss, prepare_inputs
from cityflow_tsc.effect_model.local_metrics import predict_single4
from cityflow_tsc.effect_model.local_revision import composed_contrast_loss
from cityflow_tsc.train_coarse_effects import read
from cityflow_tsc.train_formal_effects import initialize, now
from cityflow_tsc.train_single_revision import action_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-v11', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.source_v11
    protocol = read(run / 'protocol.json')
    initialize(42)
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    ss = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    validation = v11.load_roots(run, 'A_ActionMargin', ('validation',))
    training = [r for r in read(run / 'a_roots.json') if r['split'] == 'train']
    chosen = np.random.default_rng(20260921).choice(len(training), 24, replace=False)
    records, gradients = [], []
    for seed in (42, 43, 44):
        done = read(run / 'A_ActionMargin' / f'seed_{seed}/complete.json')
        if done['checkpoint_sha256'] != sha256(run / 'A_ActionMargin' / f'seed_{seed}/best.pt'):
            raise ValueError('Completed A checkpoint changed')
        for family in ('A_D_v8', 'A_ActionMargin'):
            model = v11.make_model(run, family, seed, device, selected=True).eval()
            for root in validation:
                pred = predict_single4(model, root, ss, device)
                p, y = pred[:, :, -1].sum(1), np.asarray(root['single4'], dtype=np.float64)[:, :, -1].sum(1)
                records.append({**{k: root[k] for k in ('root_id', 'cohort_id', 'flow_id', 'policy', 'time_s')},
                    'family': family, 'seed': seed, **v11.single_choice_metrics(root, pred),
                    'pred_abs_total': float(abs(p).mean()), 'true_abs_total': float(abs(y).mean()),
                    'pred_mean_total': float(p.mean()), 'true_mean_total': float(y.mean()),
                    'positive_bias': float((p-y).mean()),
                    'origin_slope': float(p @ y / (y @ y)) if y @ y else None})
            if family == 'A_D_v8':
                for index in chosen:
                    record = training[int(index)]
                    rid = record['root_id']
                    with np.load(Path(protocol['coarse_source']) / 'data' / (rid + '.npz')) as data:
                        root = {**record, **{k: data[k] for k in ('normalized_history', 'action_bank', 'base_phase',
                            'single_nodes', 'single_actions', 's_incidence')}}
                    history, actions, base = prepare_inputs(root, None, device)
                    nodes = torch.as_tensor(root['single_nodes'].reshape(-1), device=device, dtype=torch.long)
                    plans = torch.as_tensor(root['single_actions'].reshape(-1), device=device, dtype=torch.long)
                    with torch.no_grad():
                        encoded = model.encode(history, actions)
                        prediction, gate = model.single(encoded, torch.zeros_like(nodes), nodes, plans, base, return_gate=True)
                    p = prediction.detach().requires_grad_(True)
                    y = torch.as_tensor(np.load(Path(protocol['single_source']) / 'derived/roots' / rid / 'single.npy'), device=device)
                    incidence = torch.as_tensor(root['s_incidence'], device=device, dtype=torch.float32)
                    groups = torch.as_tensor(action_groups([root]), device=device)
                    baseline = multiscale_loss(p, y, ss, 'A_ref', gate) + .1 * composed_contrast_loss(
                        p * ss['s5'], y, incidence, float(read(run / 'contrast_scale.json')['scale']))
                    margin = .05 * v11.action_margin_loss(p * ss['s5'], y, groups, ss['sj'][-1])
                    g0 = torch.autograd.grad(baseline, p, retain_graph=True)[0]
                    g1 = torch.autograd.grad(margin, p)[0]
                    gradients.append({**record, 'seed': seed, 'baseline_loss': float(baseline.detach()),
                        'weighted_margin_loss': float(margin.detach()), 'gradient_norm_ratio': float(g1.norm() / g0.norm().clamp_min(1e-12)),
                        'gradient_cosine': float((g0*g1).sum() / (g0.norm()*g1.norm()).clamp_min(1e-12))})
            del model
    summary = {}
    for family in ('A_D_v8', 'A_ActionMargin'):
        rows = [r for r in records if r['family'] == family]
        summary[family] = {k: float(np.mean([group_mean([r for r in rows if r['seed']==s], k) for s in (42,43,44)]))
            for k in ('single_action_regret','pred_abs_total','true_abs_total','pred_mean_total','true_mean_total','positive_bias')}
    atomic_json(args.output, {'stage': 'complete', 'summary': summary, 'records': records, 'gradients': gradients,
        'gradient_scope': '24 seeded training roots x3 frozen old-A checkpoints; gradients wrt normalized output field, not model parameters',
        'scale_single_total240': float(ss['sj'][-1]), 'test_opened': False, 'finished_at': now()})
    print(summary)
    print('gradient_ratio_median/p90/max',np.percentile([r['gradient_norm_ratio'] for r in gradients],[50,90,100]))


if __name__ == '__main__':
    main()
