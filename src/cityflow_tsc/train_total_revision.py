"""Train only a small calibrated physical-total head; frozen v12 features are cached."""
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model import total_revision as v14
from .effect_model.local_revision import root_batches
from .train_coarse_effects import read
from .train_effect_world_model import save_checkpoint
from .train_formal_effects import initialize, now, optimizer_for, write_epoch
from .train_temporal_effects import update_learning_rate


def tensors(arrays, device):
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()
            if k in ('features', 'reference', 'changed', 'base', 'truth', 'incidence')}


@torch.no_grad()
def validate(head, data, arrays, metadata):
    head.eval()
    total = head(data['features'][660:], data['reference'][660:], data['changed'][660:],
                 data['base'][660:], metadata['total_scale']).cpu().numpy()
    return v14.summarize_records(v14.decision_records(total, arrays, metadata))


def train_job(run, family, seed):
    import fcntl
    if family != v14.FAMILIES[0] or seed not in (42, 43, 44):
        raise ValueError('Unknown total calibration job')
    run = Path(run)
    directory = run / family / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'job.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'started.json').exists():
            raise RuntimeError('Existing job retained')
        atomic_json(directory/'started.json', {'family':family,'seed':seed,'pid':os.getpid(),'started_at':now()})
        try:
            initialize(seed)
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no CPU fallback')
            device = torch.device('cuda:0')
            def progress(done, total):
                atomic_json(directory/'status.json', {'stage':'loading','family':family,'seed':seed,
                    'pid':os.getpid(),'prepared_roots':done,'total_roots':total,'updated_at':now()})
            progress(0, 738)
            cache_started = time.perf_counter()
            arrays, metadata = v14.prepare_cache(run, seed, device, progress)
            np.savez(directory/'frozen_cache.npz', **arrays)
            atomic_json(directory/'frozen_cache.json', metadata)
            cache_manifest = {'stage':'complete','source_checkpoint_sha256':metadata['source_checkpoint_sha256'],
                'files':{n:sha256(directory/n) for n in ('frozen_cache.npz','frozen_cache.json')},
                'preparation_wall_s':time.perf_counter()-cache_started}
            atomic_json(directory/'cache_complete.json', cache_manifest)
            data = tensors(arrays, device)
            torch.manual_seed(seed)
            head = v14.TotalCalibrator().to(device)
            optimizer = optimizer_for(head.parameters())
            schedule = v14.SCHEDULES[family]
            contrast_scale = float(read(run/'contrast_scale.json')['scale'])
            protocol_hash = sha256(run/'protocol.json')
            best = (float('inf'),)*3
            def select(metrics, update, cycle):
                nonlocal best
                key = (metrics['regret'],metrics['calibrated_single_total240_mae'],update)
                if not np.isfinite(key).all():
                    raise FloatingPointError('Nonfinite checkpoint selection')
                if key < best:
                    best = key
                    info = {'family':family,'seed':seed,'updates':update,'sampling_cycle':cycle,
                        'validation_regret':key[0],'validation_secondary':key[1],
                        'secondary':'calibrated_single_total240_mae','test_used':False,
                        'protocol_sha256':protocol_hash,'source_checkpoint_sha256':metadata['source_checkpoint_sha256']}
                    save_checkpoint(directory/'best.pt', head, info)
                    atomic_json(directory/'best_validation.json', {**info,'metrics':metrics})
            zero = validate(head,data,arrays,metadata)
            atomic_json(directory/'zero_validation.json',zero)
            select(zero,0,0)
            started = time.perf_counter()
            write_epoch(directory,{'updates':0,'family':family,'seed':seed,'elapsed_s':0,
                'trainable_parameters':sum(p.numel() for p in head.parameters()),
                'validation_regret':zero['regret'],'selected_update':0})
            losses=[]
            for update,cycle,indices in root_batches(660,seed,schedule['updates']):
                root=int(indices[0]//64)
                head.train()
                rate=update_learning_rate(update,schedule['updates'],schedule['warmup'])
                for group in optimizer.param_groups:
                    group['lr']=rate
                optimizer.zero_grad(set_to_none=True)
                predicted=head(data['features'][root],data['reference'][root],data['changed'][root],
                               data['base'][root],metadata['total_scale'])
                loss=v14.total_loss(predicted,data['truth'][root],data['incidence'][root],
                                    metadata['total_scale'],contrast_scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite total calibration loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if update==1 or update%schedule['log']==0 or update%schedule['validate']==0:
                    record={'family':family,'seed':seed,'updates':update,'sampling_cycle':cycle,
                        'loss':float(np.mean(losses)),'lr':rate,'elapsed_s':time.perf_counter()-started}
                    losses.clear()
                    if update%schedule['validate']==0:
                        metrics=validate(head,data,arrays,metadata)
                        select(metrics,update,cycle)
                        record.update(validation_regret=metrics['regret'],
                            validation_calibrated_single_total240_mae=metrics['calibrated_single_total240_mae'])
                    record['selected_update']=int(best[2])
                    write_epoch(directory,record)
            result={'stage':'complete','family':family,'seed':seed,'updates':update,
                'queries_processed':update*64,'selected_update':int(best[2]),'validation_regret':best[0],
                'validation_secondary':best[1],'checkpoint_sha256':sha256(directory/'best.pt'),
                'cache_manifest_sha256':sha256(directory/'cache_complete.json'),
                'source_checkpoint_sha256':metadata['source_checkpoint_sha256'],
                'training_wall_s':time.perf_counter()-started,'preparation_wall_s':cache_manifest['preparation_wall_s'],
                'test_used':False,'finished_at':now()}
            atomic_json(directory/'complete.json',result)
            atomic_json(directory/'status.json',result)
        except Exception as exc:
            failure={'stage':'failed','family':family,'seed':seed,'error':repr(exc),
                'traceback':traceback.format_exc(),'finished_at':now()}
            atomic_json(directory/'failure.json',failure)
            atomic_json(directory/'status.json',failure)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=('train',),required=True)
    parser.add_argument('--family',choices=v14.FAMILIES,required=True)
    parser.add_argument('--seed',type=int,choices=(42,43,44),required=True)
    args=parser.parse_args()
    train_job(args.run_dir,args.family,args.seed)
