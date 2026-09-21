"""Fixed-budget v8 jobs; dense A control, candidate contrasts, and local A/B heads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, multiscale_loss
from .effect_model.local_data import load_roots, prepare_labels
from .effect_model.local_metrics import make_model, validate_model
from .effect_model.local_revision import (FAMILIES, SCHEDULES, composed_contrast_loss, local_loss,
                                         pair_batches, pair_groups, root_batches)
from .effect_model import decision_revision as v9
from .effect_model import connection_revision as v10
from .effect_model import coverage_revision as v11
from .effect_model import balanced_revision as v12
from .effect_model import candidate_revision as v13
from .train_coarse_effects import read
from .train_effect_world_model import save_checkpoint
from .train_formal_effects import cached_encodings, initialize, now, optimizer_for, write_epoch
from .train_single_revision import action_groups
from .train_temporal_effects import SEEDS, grouped_batches, update_learning_rate


def select_checkpoint(run, directory, model, metrics, family, seed, update, cycle, best):
    secondary = 'pair_total240_mae' if family.startswith('B_') else 'joint4_mae'
    key = (float(metrics['regret']), float(metrics[secondary]), int(update))
    if not np.isfinite(key).all():
        raise FloatingPointError('Nonfinite validation selection')
    if key < best:
        protocol = read(run / 'protocol.json')
        metadata = {'family': family, 'seed': seed, 'updates': update, 'sampling_cycle': cycle,
            'validation_regret': key[0], 'validation_secondary': key[1], 'secondary': secondary,
            'test_used': False, 'protocol_sha256': sha256(run / 'protocol.json'),
            'source_A_sha256': protocol['source_A'][str(seed)]['sha256'] if family.startswith('B_') else None}
        save_checkpoint(directory / 'best.pt', model, metadata)
        atomic_json(directory / 'best_validation.json', {**metadata, 'metrics': metrics})
        return key
    return best


def train_job(run, family, seed):
    import fcntl
    run = Path(run)
    if family not in (*FAMILIES, *v9.FAMILIES, *v10.FAMILIES, *v11.FAMILIES, *v12.FAMILIES, *v13.FAMILIES) or seed not in SEEDS:
        raise ValueError('Unapproved job')
    revision = v11 if family in v11.FAMILIES else (v10 if family in v10.FAMILIES else (v9 if family in v9.FAMILIES else None))
    if family in v12.FAMILIES:
        revision = v12
    if family in v13.FAMILIES:
        revision = v13
    revised, is_pair = revision is not None, family.startswith('B_')
    loader, factory, validator = ((revision.load_roots, revision.make_model, revision.validate_model) if revised
                                  else (load_roots, make_model, validate_model))
    directory = run / family / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'job.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'started.json').exists():
            raise RuntimeError('Existing job retained; explicit recovery required')
        atomic_json(directory / 'started.json', {'family': family, 'seed': seed, 'pid': os.getpid(), 'started_at': now()})
        atomic_json(directory / 'status.json', {'stage': 'loading', 'pid': os.getpid(), 'family': family, 'seed': seed})
        try:
            initialize(seed)
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no CPU fallback')
            device = torch.device('cuda:0')
            schedule = (revision.SCHEDULES if revised else SCHEDULES)[family]
            roots = loader(run, family)
            training = [r for r in roots if r['split'] == 'train']
            heldout = [r for r in roots if r['split'] == 'validation']
            expected = (144, 18) if family == 'B_Coverage' else ((36, 18) if is_pair else (660, 78))
            if (len(training), len(heldout)) != expected:
                raise ValueError('Training population differs from frozen protocol')
            if revised and is_pair:
                revision.attach_fixed_predictions(run, heldout, seed)
            model = factory(run, family, seed, device)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = optimizer_for(parameters)
            single_scales = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
            pair_scales = tensor_scales(_read_npz(run / 'pair_normalization.npz'), 'pair', device)
            scales = pair_scales if is_pair else single_scales
            kind = 'pair' if is_pair else 'single'
            per_root = 1920 if is_pair else 64
            nodes = torch.as_tensor(np.concatenate([r[kind + '_nodes'] for r in training]), dtype=torch.long, device=device)
            plans = torch.as_tensor(np.concatenate([r[kind + '_actions'] for r in training]), dtype=torch.long, device=device)
            base = torch.as_tensor(np.stack([r['base_phase'] for r in training]), dtype=torch.long, device=device)
            if is_pair:
                cached = cached_encodings(model, training, device)
                batches = pair_batches(pair_groups(training), seed, schedule['updates'])
            else:
                histories = torch.as_tensor(np.stack([r['normalized_history'] for r in training]), device=device)
                banks = torch.as_tensor(np.stack([r['action_bank'] for r in training]), device=device)
                batches = (grouped_batches(action_groups(training), seed, schedule['updates']) if family == 'A_L4'
                           else root_batches(len(training), seed, schedule['updates']))
            contrast_scale = float(read(run / 'contrast_scale.json')['scale'])
            pair_contrast_scale = float(read(run / 'pair_contrast_scale.json')['scale']) if revision is v9 else None
            incidences = (torch.as_tensor(np.stack([r['s_incidence'] for r in training]), dtype=torch.float32, device=device)
                          if family == 'A_D' or (revised and not is_pair) else None)
            margin_groups = None
            if family in ('A_ActionMargin', 'A_MarginBalanced'):
                grouped = action_groups(training).reshape(len(training), 16, 4)
                grouped = grouped - np.arange(len(training))[:, None, None] * 64
                margin_groups = torch.as_tensor(grouped, dtype=torch.long, device=device)
            best = (float('inf'),) * 3
            if is_pair:
                metrics = validator(model, heldout, single_scales, pair_scales, device, family)
                best = select_checkpoint(run, directory, model, metrics, family, seed, 0, 0, best)
                atomic_json(directory / 'zero_validation.json', metrics)
            started = time.perf_counter()
            write_epoch(directory, {'updates': 0, 'family': family, 'seed': seed,
                'trainable_parameters': sum(p.numel() for p in parameters), 'elapsed_s': 0})
            losses, contrasts, balance_records = [], [], []
            label_key = 'pair4' if is_pair else ('single4' if family == 'A_L4' else 'single')
            for update, cycle, indices in batches:
                model.train()
                rate = update_learning_rate(update, schedule['updates'], schedule['warmup'])
                for group in optimizer.param_groups:
                    group['lr'] = rate
                ix = torch.as_tensor(indices, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                if is_pair:
                    prediction, logits = model.pair(cached, ix // per_root, nodes[ix], plans[ix], base, return_gate=True)
                else:
                    unique, inverse = torch.unique(ix // per_root, return_inverse=True)
                    encoded = model.encode(histories[unique], banks[unique])
                    prediction, logits = model.single(encoded, inverse, nodes[ix, 0], plans[ix, 0], base[unique], return_gate=True)
                labels = torch.as_tensor(np.stack([training[int(q // per_root)][label_key][int(q % per_root)]
                                                   for q in indices]), dtype=torch.float32, device=device)
                if revised:
                    incidence = incidences[int(indices[0] // 64)] if not is_pair else None
                    extras = {'groups': margin_groups[int(indices[0] // 64)]} if margin_groups is not None else {}
                    loss = revision.training_loss(family, prediction, labels, scales, logits, incidence,
                                            contrast_scale, pair_contrast_scale, **extras)
                    if revision in (v12, v13):
                        loss, balance_record = loss
                        balance_records.append(balance_record)
                elif family in ('A_C', 'A_D'):
                    loss = multiscale_loss(prediction, labels, scales, 'A_ref', logits)
                    if family == 'A_D':
                        contrast = composed_contrast_loss(prediction * scales['s5'], labels,
                                                          incidences[int(indices[0] // 64)], contrast_scale)
                        loss = loss + .1 * contrast
                        contrasts.append(float(contrast.detach()))
                else:
                    loss = local_loss(prediction, labels, scales['sp'], scales['sj'], logits)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if update == 1 or update % schedule['log'] == 0 or update % schedule['validate'] == 0 or update == schedule['updates']:
                    record = {'updates': update, 'sampling_cycle': cycle, 'loss': float(np.mean(losses)),
                        'lr': rate, 'family': family, 'seed': seed, 'elapsed_s': time.perf_counter() - started}
                    losses.clear()
                    if balance_records:
                        record.update({k: float(np.mean([r[k] for r in balance_records])) for k in balance_records[0]})
                        balance_records.clear()
                    if contrasts:
                        record['candidate_contrast_loss'] = float(np.mean(contrasts))
                        contrasts.clear()
                    if update % schedule['validate'] == 0 or update == schedule['updates']:
                        metrics = validator(model, heldout, single_scales, pair_scales, device, family)
                        best = select_checkpoint(run, directory, model, metrics, family, seed, update, cycle, best)
                        record.update(validation_regret=metrics['regret'], validation_joint4_mae=metrics['joint4_mae'],
                                      validation_by_time=metrics['by_time'])
                    record['selected_update'] = int(best[2]) if np.isfinite(best[2]) else None
                    write_epoch(directory, record)
            result = {'stage': 'complete', 'family': family, 'seed': seed, 'updates': update,
                'queries_processed': update * schedule['batch'], 'selected_update': int(best[2]),
                'validation_regret': best[0], 'validation_secondary': best[1],
                'checkpoint_sha256': sha256(directory / 'best.pt'),
                'training_wall_s': time.perf_counter() - started, 'test_used': False, 'finished_at': now()}
            atomic_json(directory / 'complete.json', result)
            atomic_json(directory / 'status.json', result)
            return result
        except Exception as exc:
            failure = {'stage': 'failed', 'family': family, 'seed': seed, 'error': repr(exc),
                       'traceback': traceback.format_exc(), 'finished_at': now()}
            atomic_json(directory / 'failure.json', failure)
            atomic_json(directory / 'status.json', failure)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'prepare-labels', 'prepare-decision'), required=True)
    parser.add_argument('--family', choices=(*FAMILIES, *v9.FAMILIES, *v10.FAMILIES, *v11.FAMILIES, *v12.FAMILIES, *v13.FAMILIES))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == 'prepare-decision':
        v9.prepare_frozen_predictions(args.run_dir)
    elif args.stage == 'prepare-labels':
        prepare_labels(args.run_dir)
    else:
        if args.family is None or args.seed is None:
            parser.error('Training requires --family and --seed')
        train_job(args.run_dir, args.family, args.seed)


if __name__ == '__main__':
    main()
