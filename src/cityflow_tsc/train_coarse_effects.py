"""Twelve signed cumulative consequences per action, on frozen v5 data."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from .collect_formal_effects import _freeze_json
from .counterfactual.writer import atomic_json, sha256
from .effect_model import GROUPS
from .effect_model.data import normalize_history
from .effect_model.formal_data import _read_npz, _write_numpy, HORIZONS
from .effect_model.formal_model import FormalEffectModel, decision_metrics, group_mean, root_metadata
from .train_effect_world_model import save_checkpoint
from .train_formal_effects import initialize, now, optimizer_for, write_epoch
from .train_single_revision import action_groups
from .train_temporal_effects import grouped_batches, update_learning_rate, MAX_UPDATES, VALIDATE_EVERY, SEEDS


def read(path):
    return json.loads(Path(path).read_text())


def aggregate_effects(values):
    """Signed sums first; three disjoint locations, four cumulative horizons."""
    if values.shape[-2:] != (272, 48):
        raise ValueError('Expected physical effects over 272 objects and 48 windows')
    return np.stack([np.asarray(values[..., a:b, :], dtype=np.float64).sum(-2).cumsum(-1)[
        ..., np.asarray(HORIZONS) // 5 - 1] for a, b, _ in GROUPS], axis=-2)


def target_scales(training):
    if not training or any(r['split'] != 'train' for r in training):
        raise ValueError('Only training targets may fit scales')
    values = np.abs(np.concatenate([r['single_coarse'] for r in training]))
    scales = np.ones((3, 4), dtype=np.float64)
    for g in range(3):
        for h in range(4):
            nonzero = values[:, g, h][values[:, g, h] > 0]
            if len(nonzero):
                scales[g, h] = max(1., float(np.percentile(nonzero, 95)))
    return scales


class CoarseEffectModel(FormalEffectModel):
    def __init__(self, static, pooling='early'):
        super().__init__(static)
        if pooling not in ('early', 'late'):
            raise ValueError('Expected early or late location aggregation')
        self.pooling = pooling
        for module in (self.single_head, self.single_gate, self.pair_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # Both orders use exactly the same encoder and decoder parameters.
        self.coarse_head = nn.Sequential(nn.Linear(272, 128), nn.SiLU(), nn.Dropout(.1),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 4))
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)

    def coarse(self, encoded, roots, nodes, targets, base_phase):
        state, glob, action, relative, changed = self.query_parts(encoded, roots, nodes, targets, base_phase)
        features = torch.cat((state, glob, action, relative), -1)
        if self.pooling == 'late':
            identity = torch.cat([torch.eye(3, device=features.device, dtype=features.dtype)[i]
                .expand(b - a, -1) for i, (a, b, _) in enumerate(GROUPS)], 0)
            # Keep recipient traffic paired with its action-relative geometry through
            # the nonlinear decoder. These are latent contributions, not supervised
            # per-object physical predictions; only their 12 group means are targets.
            contributions = self.coarse_head(torch.cat((features,
                identity[None].expand(len(features), -1, -1)), -1))
            return torch.stack([contributions[:, a:b].mean(1) for a, b, _ in GROUPS], 1) * changed[:, None, None]
        pooled = torch.stack([features[:, a:b].mean(1) for a, b, _ in GROUPS], 1)
        identity = torch.eye(3, device=pooled.device, dtype=pooled.dtype)[None].expand(len(pooled), -1, -1)
        return self.coarse_head(torch.cat((pooled, identity), -1)) * changed[:, None, None]


def prepare(run, source, test=False):
    run, source = Path(run), Path(source).resolve()
    selection = read(source / 'selection.json')
    if selection.get('split_roots') != {'train': 660, 'validation': 78, 'test': 12}:
        raise ValueError('Expected the completed v5 multi arm')
    if read(source / 'training_temporal/summary.json')['stage'] != 'complete':
        raise ValueError('Completed source required')
    if test and not (run / 'checkpoints_locked.json').is_file():
        raise ValueError('All three new checkpoints must be locked before diagnostic labels')
    run.mkdir(parents=True, exist_ok=True)
    _freeze_json(run / 'protocol.json', {
        'schema': 'coarse-action-consequences-v6', 'source_run': str(source),
        'source_selection_sha256': sha256(source / 'selection.json'),
        'source_input_normalization_sha256': sha256(source / 'derived/normalization.npz'),
        'source_static_sha256': sha256(source / 'derived/static.npz'),
        'groups': [g[2] for g in GROUPS], 'horizons_s': list(HORIZONS),
        'loss': 'mean SmoothL1 beta=1 on 12 separately train-P95-normalized signed consequences',
        'encoder': 'unchanged FormalEffectModel encoder; no pretrained weights',
        'decoder': 'mean-pool query features by location group; shared 272-128-64-4 MLP with group one-hot',
        'seeds': list(SEEDS), 'updates': MAX_UPDATES, 'batch_queries': 32,
        'warmup_updates': 2200, 'validate_every_updates': VALIDATE_EVERY,
        'checkpoint_selection': 'validation regret, aggregate joint MAE, earlier update',
        'test_scope': 'previously inspected diagnostic, not blind generalization',
        'dense_field_MAE_comparable': False})
    if not (run / 'static.npz').exists():
        _write_numpy(run / 'static.npz', _read_npz(source / 'derived/static.npz'), compressed=True)
    stats = _read_npz(source / 'derived/normalization.npz')
    tasks = [t for t in selection['tasks'] if t['split'] in (('test',) if test else ('train', 'validation'))]
    records = []
    for index, task in enumerate(tasks):
        rid = task['root_id']
        target = run / 'data' / f'{rid}.npz'
        original = source / 'derived/roots' / rid
        info = {key: task[key] for key in ('root_id', 'split', 'cohort_id', 'flow_id', 'policy', 'time_s')}
        if not target.exists():
            inputs = _read_npz(original / 'inputs.npz')
            arrays = {key: inputs[key] for key in ('base_phase', 'action_bank', 'single_nodes', 'single_actions', 's_incidence')}
            arrays['normalized_history'] = normalize_history(inputs['history'], stats)
            for name in ('single', 'joint'):
                values = np.load(original / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                arrays[name + '_coarse'] = aggregate_effects(values)
                del values
            _write_numpy(target, arrays, compressed=True)
        records.append(info)
        atomic_json(run / 'prepare_progress.json', {'stage': 'test' if test else 'trainval',
            'prepared': index + 1, 'total': len(tasks), 'root_id': rid})
    _freeze_json(run / ('test_roots.json' if test else 'trainval_roots.json'), records)
    if not test and not (run / 'target_scales.npy').exists():
        training = [{**r, 'single_coarse': _read_npz(run / 'data' / f"{r['root_id']}.npz")['single_coarse']}
                    for r in records if r['split'] == 'train']
        _write_numpy(run / 'target_scales.npy', target_scales(training))
    atomic_json(run / ('test_ready.json' if test else 'trainval_ready.json'),
                {'stage': 'complete', 'roots': len(records), 'finished_at': now()})


def load_roots(run, test=False):
    run = Path(run)
    if test and not (run / 'checkpoints_locked.json').exists():
        raise ValueError('Diagnostic locked')
    records = read(run / ('test_roots.json' if test else 'trainval_roots.json'))
    return [{**r, **_read_npz(run / 'data' / f"{r['root_id']}.npz")} for r in records]


def summarize(rows):
    keys = ('regret', 'aggregate_joint_mae', 'optimal_choice', 'worse_than_reference',
            'excess_wait_vs_reference', 'benefit_vs_reference')
    summary = {k: group_mean(rows, k) for k in keys}
    summary['by_time'] = {str(int(t)): {k: group_mean([r for r in rows if r['time_s'] == t], k)
        for k in keys} for t in sorted({r['time_s'] for r in rows})}
    summary['records'] = rows
    return summary


@torch.no_grad()
def evaluate_model(model, roots, scales, device):
    model.eval()
    rows = []
    for root in roots:
        history = torch.as_tensor(root['normalized_history'][None], device=device)
        bank = torch.as_tensor(root['action_bank'][None], device=device)
        base = torch.as_tensor(root['base_phase'][None], device=device)
        encoded = model.encode(history, bank)
        single = model.coarse(encoded, torch.zeros(64, dtype=torch.long, device=device),
            torch.as_tensor(root['single_nodes'][:, 0], device=device),
            torch.as_tensor(root['single_actions'][:, 0], device=device), base) * scales
        predicted = (root['s_incidence'] @ single.cpu().numpy().astype(np.float64).reshape(64, -1)).reshape(65, 3, 4)
        truth = root['joint_coarse']
        rows.append({**root_metadata(root), **decision_metrics(predicted[:, :, -1].sum(1), truth[:, :, -1].sum(1)),
                     'aggregate_joint_mae': float(np.abs(predicted - truth).mean())})
    return summarize(rows)


def train_job(run, seed):
    import fcntl
    run = Path(run)
    directory = run / 'coarse' / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'job.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'started.json').exists():
            raise RuntimeError('Retained job exists; do not overwrite or silently restart')
        atomic_json(directory / 'started.json', {'seed': seed, 'pid': os.getpid(), 'started_at': now()})
        try:
            initialize(seed)
            protocol = read(run / 'protocol.json')
            pooling = protocol.get('pooling', 'early')
            torch.set_num_threads(int(protocol.get('cpu_threads_per_worker', 2)))
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required')
            device = torch.device('cuda:0')
            roots = load_roots(run)
            training = [r for r in roots if r['split'] == 'train']
            heldout = [r for r in roots if r['split'] == 'validation']
            if (len(training), len(heldout)) != (660, 78):
                raise ValueError('Frozen training/validation population changed')
            groups = action_groups(training)
            model = CoarseEffectModel(_read_npz(run / 'static.npz'), pooling=pooling).to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optimizer_for(params)
            scales = torch.as_tensor(np.load(run / 'target_scales.npy').astype(np.float32), device=device)
            histories = torch.as_tensor(np.stack([r['normalized_history'] for r in training]), device=device)
            actions = torch.as_tensor(np.stack([r['action_bank'] for r in training]), device=device)
            base = torch.as_tensor(np.stack([r['base_phase'] for r in training]), device=device)
            nodes = torch.as_tensor(np.concatenate([r['single_nodes'] for r in training])[:, 0], device=device)
            plans = torch.as_tensor(np.concatenate([r['single_actions'] for r in training])[:, 0], device=device)
            labels = torch.as_tensor(np.concatenate([r['single_coarse'] for r in training]).astype(np.float32), device=device)
            best = (float('inf'),) * 3
            started, losses = time.perf_counter(), []
            for update, cycle, indices in grouped_batches(groups, seed):
                model.train()
                rate = update_learning_rate(update)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                ix = torch.as_tensor(indices, device=device)
                unique, inverse = torch.unique(ix // 64, return_inverse=True)
                optimizer.zero_grad(set_to_none=True)
                encoded = model.encode(histories[unique], actions[unique])
                prediction = model.coarse(encoded, inverse, nodes[ix], plans[ix], base[unique])
                loss = nn.functional.smooth_l1_loss(prediction, labels[ix] / scales, beta=1.)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if update % 440 == 0 or update == MAX_UPDATES:
                    record = {'seed': seed, 'updates': update, 'sampling_cycle': cycle, 'lr': rate,
                              'loss': float(np.mean(losses)), 'elapsed_s': time.perf_counter() - started}
                    losses.clear()
                    if update % VALIDATE_EVERY == 0 or update == MAX_UPDATES:
                        metrics = evaluate_model(model, heldout, scales, device)
                        key = (metrics['regret'], metrics['aggregate_joint_mae'], update)
                        if not np.isfinite(key).all():
                            raise FloatingPointError('Nonfinite validation')
                        if key < best:
                            best = key
                            metadata = {'variant': 'coarse12' if pooling == 'early' else 'coarse12_latepool',
                                        'pooling': pooling, 'seed': seed, 'updates': update,
                                        'validation_regret': key[0], 'test_used': False}
                            save_checkpoint(directory / 'best.pt', model, metadata)
                            atomic_json(directory / 'best_validation.json', {**metadata, 'metrics': metrics})
                        record.update(validation_regret=metrics['regret'], validation_by_time=metrics['by_time'])
                    record['selected_update'] = int(best[2]) if np.isfinite(best[2]) else None
                    write_epoch(directory, record)
            result = {'stage': 'complete', 'seed': seed, 'updates': update, 'selected_update': int(best[2]),
                'validation_regret': best[0], 'checkpoint_sha256': sha256(directory / 'best.pt'),
                'training_wall_s': time.perf_counter() - started, 'finished_at': now(), 'test_used': False}
            atomic_json(directory / 'complete.json', result)
            atomic_json(directory / 'status.json', result)
            return result
        except BaseException as exc:
            atomic_json(directory / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
            raise


def train(run, source):
    run = Path(run)
    prepare(run, source)
    results, errors = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(train_job, str(run), seed): seed for seed in SEEDS}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'seed': futures[future], 'error': repr(exc)})
            atomic_json(run / 'training/queue.json', {'completed': len(results), 'total': 3, 'errors': errors})
    atomic_json(run / 'training/summary.json', {'stage': 'failed' if errors else 'complete', 'results': results, 'errors': errors})
    if errors:
        raise RuntimeError('Training failure retained')
    atomic_json(run / 'checkpoints_locked.json', {'checkpoints': {
        f'coarse/seed_{s}/best.pt': sha256(run / 'coarse' / f'seed_{s}' / 'best.pt') for s in SEEDS},
        'scales_sha256': sha256(run / 'target_scales.npy'), 'protocol_sha256': sha256(run / 'protocol.json'),
        'static_sha256': sha256(run / 'static.npz'), 'locked_at': now(), 'test_used': False})


def evaluate(run, source):
    run = Path(run)
    lock = read(run / 'checkpoints_locked.json')
    for name, digest in lock['checkpoints'].items():
        if sha256(run / name) != digest:
            raise ValueError('Locked checkpoint changed')
    for name, key in (('target_scales.npy', 'scales_sha256'), ('protocol.json', 'protocol_sha256'), ('static.npz', 'static_sha256')):
        if sha256(run / name) != lock[key]:
            raise ValueError('Locked training inputs changed')
    prepare(run, source, test=True)
    roots = load_roots(run, test=True)
    initialize(42)
    device = torch.device('cuda:0')
    scales = torch.as_tensor(np.load(run / 'target_scales.npy').astype(np.float32), device=device)
    results, validation_results = {}, {}
    for seed in SEEDS:
        model = CoarseEffectModel(_read_npz(run / 'static.npz')).to(device)
        saved = torch.load(run / 'coarse' / f'seed_{seed}' / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(saved['state_dict'])
        results[str(seed)] = evaluate_model(model, roots, scales, device)
        validation_results[str(seed)] = read(run / 'coarse' / f'seed_{seed}' / 'best_validation.json')
        del model, saved
    baseline = read(Path(source) / 'evaluation/summary.json')
    atomic_json(run / 'evaluation/summary.json', {'stage': 'complete', 'results': results,
        'validation_results': validation_results, 'baseline_A_ref': baseline,
        'test_scope': 'previously inspected diagnostic; not independent blind test',
        'aggregate_MAE_is_not_field_MAE': True, 'finished_at': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'evaluate'), required=True)
    args = parser.parse_args()
    run, source = args.run_dir.resolve(), args.source_run.resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')) or run == source:
        raise ValueError('Separate output directory on mounted /mnt/pan required')
    (train if args.stage == 'train' else evaluate)(run, source)


if __name__ == '__main__':
    main()
