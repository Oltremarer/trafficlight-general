"""Locked validation-only v12/v13 comparison; no diagnostic labels are opened."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual.writer import atomic_json
from .effect_model import balanced_revision as v12, coverage_revision as v11
from .effect_model import candidate_revision as v13
from .effect_model.formal_data import _read_npz
from .effect_model.formal_model import tensor_scales, group_mean
from .evaluate_connection_revision import complete_record
from .evaluate_decision_revision import summary
from .evaluate_local_revision import LOCKED_FILES as BASE_FILES, lock_checkpoints as base_lock, verify_lock, infer_joint
from .train_formal_effects import initialize, now
from .train_coarse_effects import read

LOCKED_FILES = (*BASE_FILES, 'training_cache.json')


def lock_checkpoints(run, revision=v12):
    base_lock(run, revision.FAMILIES, revision.SCHEDULES, LOCKED_FILES)


def evaluate(run):
    run = Path(run)
    revision = v13 if read(run / 'protocol.json')['schema'] == v13.SCHEMA else v12
    families = ('A_D_v8', 'A_MarginBalanced', *v13.FAMILIES) if revision is v13 else ('A_D_v8', *v12.FAMILIES)
    verify_lock(run, revision.FAMILIES, LOCKED_FILES)
    roots = revision.load_roots(run, revision.FAMILIES[0], ('validation',))
    if len(roots) != 78 or any(r['split'] != 'validation' for r in roots):
        raise ValueError('Only validation78 allowed')
    initialize(42)
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    ss = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    results = {}
    for family in families:
        for seed in (42,43,44):
            model = revision.make_model(run, family, seed, device, selected=True).eval()
            rows = []
            infer_joint(model, None, roots[0], ss, {}, device, family, 0)
            for root in roots:
                torch.cuda.synchronize()
                start = time.perf_counter()
                joint, single, choice = infer_joint(model, None, root, ss, {}, device, family, 0)
                torch.cuda.synchronize()
                record = complete_record(root, joint, single, choice, (time.perf_counter()-start)*1000,
                    pair_budget=0, encoder_passes=1)
                record.update(v11.single_choice_metrics(root, single))
                if revision is v13:
                    record.update(v13.additive_choice_metrics(root, choice))
                rows.append(record)
            metrics = summary(rows)
            metrics['single_action_regret'] = group_mean(rows,'single_action_regret')
            if revision is v13:
                for key in ('additive_regret', 'additive_optimal_choice', 'true_single_oracle_joint_regret'):
                    metrics[key] = group_mean(rows, key)
            for sub in metrics['by_flow'].values():
                sub['single_action_regret'] = group_mean(sub['records'],'single_action_regret')
            results.setdefault(family,{})[str(seed)] = metrics
            atomic_json(run / 'evaluation' / f'{family}_seed_{seed}.json', metrics)
            atomic_json(run / 'evaluation/progress.json', {'stage':'evaluating','completed':sum(len(s) for s in results.values()),
                'total':len(families)*3,'family':family,'seed':seed,'updated_at':now()})
            del model
    comparison = {f:{k:float(np.mean([m[k] for m in seeds.values()])) for k in
        ('regret','joint4_mae','single4_mae','single_total240_mae','single_action_regret','inference_ms')}
        for f,seeds in results.items()}
    if revision is v13:
        for family, seeds in results.items():
            for key in ('additive_regret', 'additive_optimal_choice', 'true_single_oracle_joint_regret'):
                comparison[family][key] = float(np.mean([m[key] for m in seeds.values()]))
    atomic_json(run / 'evaluation/summary.json', {'stage':'complete','results':results,'comparison':comparison,
        'test_opened':False,'scope':'validation78 only; also used for checkpoint selection, not independent confirmation',
        'finished_at':now()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    evaluate(parser.parse_args().run_dir)
