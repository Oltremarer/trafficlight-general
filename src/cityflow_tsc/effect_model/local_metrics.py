"""Physical cumulative fields, reference-first choices and frozen model loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..counterfactual.writer import sha256
from ..train_coarse_effects import read
from .formal_data import _read_npz
from .formal_model import FormalEffectModel, decision_metrics, group_mean, prepare_inputs, root_metadata
from .local_revision import LocalSingleModel, LocalPairModel, compose, cumulative4, pair_indices


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    static = _read_npz(run / 'static.npz')
    protocol = read(run / 'protocol.json')
    source = protocol['source_A'][str(seed)]
    if family in ('B_L4', 'A_ref'):
        if sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Frozen source A checkpoint changed')
        saved = torch.load(source['path'], map_location='cpu', weights_only=False)
        model = (LocalPairModel(static, saved['state_dict']) if family == 'B_L4'
                 else FormalEffectModel(static))
        if family == 'A_ref':
            model.load_state_dict(saved['state_dict'])
    elif family == 'A_L4':
        model = LocalSingleModel(static)
    elif family in ('A_C', 'A_D'):
        model = FormalEffectModel(static)
        for parameter in model.pair_head.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError('Unknown family')
    if selected and family != 'A_ref':
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if saved['family'] != family or saved['seed'] != seed or saved['test_used']:
            raise ValueError('Selected checkpoint identity mismatch')
        if family == 'B_L4' and saved['source_A_sha256'] != source['sha256']:
            raise ValueError('B encoder source changed')
        model.load_state_dict(saved['state_dict'])
    return model.to(device)


@torch.no_grad()
def predict_single4(model, root, scales, device, local=False):
    """Inputs contain only observed history, static context, and requested actions."""
    model.eval()
    history, actions, base = prepare_inputs(root, None, device)
    encoded = model.encode(history, actions)
    nodes = torch.as_tensor(root['single_nodes'], dtype=torch.long, device=device).reshape(-1)
    plans = torch.as_tensor(root['single_actions'], dtype=torch.long, device=device).reshape(-1)
    prediction = model.single(encoded, torch.zeros_like(nodes), nodes, plans, base)
    physical = prediction * scales['sp' if local else 's5']
    return (physical.to(torch.float64) if local else cumulative4(physical)).cpu().numpy()


@torch.no_grad()
def predict_pair4(model, root, scales, device, budget=120):
    # Selection precedes the encoder and detailed decoder. Budget zero is free.
    indices = pair_indices(root, model.rank_geometry.cpu().numpy(), budget)
    if not len(indices):
        return indices, np.empty((0, 272, 4), dtype=np.float64)
    model.eval()
    history, actions, base = prepare_inputs(root, None, device)
    encoded = model.encode(history, actions)
    nodes = torch.as_tensor(root['pair_nodes'][indices], dtype=torch.long, device=device)
    plans = torch.as_tensor(root['pair_actions'][indices], dtype=torch.long, device=device)
    chunks = []
    for start in range(0, len(indices), 128):
        n, a = nodes[start:start + 128], plans[start:start + 128]
        prediction = model.pair(encoded, torch.zeros(len(n), dtype=torch.long, device=device), n, a, base)
        chunks.append((prediction * scales['sp']).to(torch.float64).cpu().numpy())
    return indices, np.concatenate(chunks)


def field_record(root, joint4, single4=None):
    truth = np.asarray(root['joint4'], dtype=np.float64)
    if joint4.shape != truth.shape or not np.isfinite(joint4).all():
        raise ValueError('Invalid cumulative joint predictions')
    result = {**root_metadata(root), **decision_metrics(joint4[:, :, -1].sum(1), truth[:, :, -1].sum(1)),
        'joint4_mae': float(np.abs(joint4 - truth).mean())}
    if single4 is not None:
        target = np.asarray(root['single4'], dtype=np.float64)
        p, y = single4[:, :, -1].sum(1), target[:, :, -1].sum(1)
        nonzero = np.abs(y) > 1e-6
        result.update(single4_mae=float(np.abs(single4 - target).mean()),
            single_total240_mae=float(np.abs(p - y).mean()),
            single_nonzero_total_count=int(nonzero.sum()),
            single_sign_error_count=int((np.sign(p[nonzero]) != np.sign(y[nonzero])).sum()))
    return result


def summarize(rows):
    if not rows:
        raise ValueError('No evaluation roots')
    keys = ['regret', 'joint4_mae', 'optimal_choice', 'worse_than_reference',
            'excess_wait_vs_reference', 'benefit_vs_reference']
    keys += [k for k in ('single4_mae', 'single_total240_mae', 'pair4_mae', 'pair_total240_mae', 'joint_total240_mae',
                        'inference_ms') if all(k in r for r in rows)]

    def aggregate(selected):
        result = {k: group_mean(selected, k) for k in keys}
        result.update(root_count=len(selected),
            worse_than_reference_count=sum(int(r['worse_than_reference']) for r in selected),
            optimal_choice_count=sum(int(r['optimal_choice']) for r in selected))
        for kind in ('single', 'pair', 'joint'):
            if all(kind + '_sign_error_count' in r for r in selected):
                count = sum(r[kind + '_nonzero_total_count'] for r in selected)
                errors = sum(r[kind + '_sign_error_count'] for r in selected)
                result[kind + '_sign_errors'] = {'errors': errors, 'nonzero_totals': count,
                                                        'rate': errors / count if count else None}
        return result

    return {**aggregate(rows), 'by_time': {
        str(int(t)): aggregate([r for r in rows if float(r['time_s']) == t])
        for t in sorted({float(r['time_s']) for r in rows})},
        'by_policy': {p: aggregate([r for r in rows if r['policy'] == p]) for p in sorted({r['policy'] for r in rows})},
        'worst_roots': sorted(rows, key=lambda r: r['regret'], reverse=True)[:10], 'records': rows}


def validate_model(model, roots, single_scales, pair_scales, device, family, single_baseline_key='single4'):
    rows = []
    for root in roots:
        if family == 'B_L4':
            indices, pair = predict_pair4(model, root, pair_scales, device, 120)
            joint = compose(root['s_incidence'], np.asarray(root[single_baseline_key], dtype=np.float64))
            joint += compose(root['p_incidence'][:, indices], pair)
            record = field_record(root, joint)
            target = np.asarray(root['pair4'], dtype=np.float64)[indices]
            p, y = pair[:, :, -1].sum(1), target[:, :, -1].sum(1)
            nonzero = np.abs(y) > 1e-6
            record.update(pair4_mae=float(np.abs(pair - target).mean()),
                pair_total240_mae=float(np.abs(p - y).mean()), pair_nonzero_total_count=int(nonzero.sum()),
                pair_sign_error_count=int((np.sign(p[nonzero]) != np.sign(y[nonzero])).sum()))
        else:
            single = predict_single4(model, root, single_scales, device, family == 'A_L4')
            record = field_record(root, compose(root['s_incidence'], single), single)
        rows.append(record)
    return summarize(rows)
