"""v14 validation only: calibrated scalar decisions versus frozen v12 spatial fields."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model import total_revision as v14
from .evaluate_balanced_revision import LOCKED_FILES
from .evaluate_local_revision import verify_lock
from .train_coarse_effects import read
from .train_formal_effects import initialize, now
from .train_total_revision import tensors, validate


def evaluate(run):
    run=Path(run)
    verify_lock(run,v14.FAMILIES,LOCKED_FILES)
    initialize(42)
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    protocol=read(run/'protocol.json')
    results={'A_MarginBalanced':{},v14.FAMILIES[0]:{}}
    for seed in (42,43,44):
        directory=run/v14.FAMILIES[0]/f'seed_{seed}'
        done=read(directory/'complete.json')
        cache=read(directory/'cache_complete.json')
        if sha256(directory/'cache_complete.json')!=done['cache_manifest_sha256']:
            raise ValueError('Committed feature cache changed')
        for name,digest in cache['files'].items():
            if sha256(directory/name)!=digest:
                raise ValueError('Feature cache changed')
        source=protocol['source_balanced_A'][str(seed)]
        if sha256(Path(source['path']))!=source['sha256']:
            raise ValueError('Frozen v12 source changed')
        with np.load(directory/'frozen_cache.npz',allow_pickle=False) as loaded:
            arrays={k:loaded[k] for k in loaded.files}
        metadata=read(directory/'frozen_cache.json')
        data=tensors(arrays,device)
        head=v14.TotalCalibrator().to(device)
        old=validate(head,data,arrays,metadata)
        saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        if (saved['family']!=v14.FAMILIES[0] or saved['seed']!=seed or saved['test_used'] or
                saved['protocol_sha256']!=sha256(run/'protocol.json') or
                saved['source_checkpoint_sha256']!=source['sha256']):
            raise ValueError('Selected total-head identity changed')
        head.load_state_dict(saved['state_dict'])
        new=validate(head,data,arrays,metadata)
        for name,metrics in (('A_MarginBalanced',old),(v14.FAMILIES[0],new)):
            results[name][str(seed)]=metrics
            atomic_json(run/'evaluation'/f'{name}_seed_{seed}.json',metrics)
        atomic_json(run/'evaluation/progress.json',{'stage':'evaluating','completed':sum(map(len,results.values())),
            'total':6,'seed':seed,'updated_at':now()})
        del data,head
    keys=('regret','additive_regret','single_action_regret','calibrated_single_total240_mae',
          'calibrated_joint_total240_mae','single4_mae','joint4_mae','single_total240_mae')
    comparison={f:{k:float(np.mean([m[k] for m in seeds.values()])) for k in keys} for f,seeds in results.items()}
    atomic_json(run/'evaluation/summary.json',{'stage':'complete','comparison':comparison,'results':results,
        'scope':'validation78 also used for checkpoint selection; zero head eligible; not independent confirmation',
        'field_scope':'single4/joint4 and single_total240_mae are unchanged frozen v12 fields; calibrated_* metrics refer to separate physical-total output',
        'cost_scope':'incremental head training on pretrained v12; no end-to-end inference latency measured',
        'test_opened':False,'finished_at':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    evaluate(parser.parse_args().run_dir)
