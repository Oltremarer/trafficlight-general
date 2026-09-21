"""v14: freeze v12 fields and calibrate the physical total of each single intervention."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen
from . import balanced_revision as v12, candidate_revision as v13, decision_revision as v9
from .formal_model import decision_metrics, group_mean, prepare_inputs, root_metadata, tensor_scales
from .formal_data import _read_npz
from .local_data import load_roots
from .local_metrics import field_record
from .local_revision import SCHEDULES as OLD_SCHEDULES, compose, composed_contrast_loss, cumulative4

FAMILIES = ('A_TotalCalibrated',)
SCHEDULES = {FAMILIES[0]: dict(OLD_SCHEDULES['A_D'])}
SCHEMA = 'frozen-single-total-calibration-v14'
TRAIN_MODULE = 'cityflow_tsc.train_total_revision'
EVALUATION_MODULE = 'cityflow_tsc.evaluate_total_revision'


class TotalCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(257, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, reference, changed, base, scale):
        delta = (self.head(features) - self.head(reference)).squeeze(-1)
        delta = delta * changed.to(delta.dtype)
        return base.to(torch.float64) + delta.to(torch.float64) * scale


def total_loss(predicted, truth, incidence, total_scale, contrast_scale):
    if predicted.shape != (64,) or truth.shape != (64,) or incidence.shape != (65, 64):
        raise ValueError('One complete root of true single totals required')
    physical = nn.functional.smooth_l1_loss(predicted / total_scale, truth / total_scale, beta=1.)
    contrast = composed_contrast_loss(predicted[:, None, None], truth[:, None, None], incidence, contrast_scale)
    return physical + .1 * contrast


def initialize_run(run, source_v13):
    from ..evaluate_balanced_revision import LOCKED_FILES
    from ..evaluate_local_revision import verify_lock
    run, source_v13 = Path(run).resolve(), Path(source_v13).resolve()
    if run.is_relative_to(source_v13) or source_v13.is_relative_to(run):
        raise ValueError('Independent run directory required')
    if (read(source_v13 / 'execution.json')['stage'] != 'complete' or
            read(source_v13 / 'evaluation/summary.json')['stage'] != 'complete'):
        raise ValueError('Completed v13 evidence required')
    verify_lock(source_v13, v13.FAMILIES, LOCKED_FILES)
    prior = read(source_v13 / 'protocol.json')
    source_v12 = Path(prior['source_v12'])
    verify_lock(source_v12, v12.FAMILIES, LOCKED_FILES)
    for name in v9.COPIED_FILES:
        copy_frozen(source_v13 / name, run / name)
    protocol = dict(prior)
    for key in ('gradient_fraction', 'max_margin_weight', 'gradient_scope', 'auxiliary_target', 'auxiliary_formula'):
        protocol.pop(key, None)
    protocol.update(schema=SCHEMA, source_v13=str(source_v13),
        source_v13_protocol_sha256=sha256(source_v13 / 'protocol.json'), schedules=SCHEDULES,
        A_loss='Huber(calibrated_single_total240/old_sj240,true_single_total240/old_sj240) + .1 original all-candidate contrast Huber; no margin',
        A_target='true single total240 only; never true joint or pair residual',
        A_selection='same78 validation true joint candidate regret, calibrated single total240 MAE, earlier update; zero calibrator eligible',
        A_frozen='seed-matched selected v12 encoder, action encoder, field head and gate; always eval and no gradients',
        A_head='257->64 SiLU->1; zero final layer; plan minus reference output; exact zero no-change correction',
        A_features='observed endpoint64 + observed global64 + requested action64 + reference action64 + frozen normalized single total1',
        initial_weights='frozen pretrained v12 plus fresh zero-output calibrator; incremental training, NOT matched from-scratch compute',
        precision='FP32 frozen backbone, cached features and trainable head; FP64 physical totals, losses and decision composition; no AMP or TF32',
        field_scope='all 272x48 spatial fields remain v12; decisions use a separately calibrated physical single total, not the field sum',
        inference_scope='one frozen encoding plus original field decoder and small total head; no future labels as inputs',
        normalization='reuse frozen v12 single_sj240 and contrast scale; no fitting',
        evaluation='validation78 only; compare zero calibrator v12 and selected total calibrator; report frozen-field and calibrated-total errors separately',
        test_scope='No diagnostic/test opened; repeated validation model selection, not independent confirmation')
    _freeze_json(run / 'protocol.json', protocol)
    v9.reuse_labels(run)


@torch.no_grad()
def inputs_for_root(backbone, root, scales, device):
    history, actions, base_phase = prepare_inputs(root, None, device)
    encoded = backbone.encode(history, actions)
    nodes = torch.as_tensor(root['single_nodes'], dtype=torch.long, device=device).reshape(-1)
    plans = torch.as_tensor(root['single_actions'], dtype=torch.long, device=device).reshape(-1)
    indices = torch.zeros_like(nodes)
    dense = backbone.single(encoded, indices, nodes, plans, base_phase) * scales['s5']
    fields = cumulative4(dense).cpu().numpy()
    base = fields[:, :, -1].sum(1)
    endpoint, glob = encoded['objects'][0, 240 + nodes], encoded['global'][0].expand(len(nodes), -1)
    requested, reference_action = encoded['actions'][0, nodes, plans], encoded['actions'][0, nodes, 0]
    normalized_total = torch.as_tensor(base / float(scales['sj'][-1]), dtype=torch.float32, device=device)[:, None]
    features = torch.cat((endpoint, glob, requested, reference_action, normalized_total), -1)
    reference = torch.cat((endpoint, glob, reference_action, reference_action, torch.zeros_like(normalized_total)), -1)
    changed = encoded['plan_changed'][0, nodes, plans]
    return features.cpu().numpy(), reference.cpu().numpy(), changed.cpu().numpy(), base, fields


def prepare_cache(run, seed, device, progress=None):
    """Only label-free features enter the head; labels are kept in separate arrays."""
    run = Path(run)
    protocol = read(run / 'protocol.json')
    source = Path(protocol['source_v12'])
    expected = protocol['source_balanced_A'][str(seed)]
    if sha256(Path(expected['path'])) != expected['sha256']:
        raise ValueError('Frozen v12 checkpoint changed')
    backbone = v12.make_model(source, 'A_MarginBalanced', seed, device, selected=True).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    scales = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    roots = load_roots(run, 'A_L4', ('train', 'validation'))
    roots = [r for split in ('train', 'validation') for r in roots if r['split'] == split]
    if tuple(sum(r['split'] == split for r in roots) for split in ('train', 'validation')) != (660, 78):
        raise ValueError('Expected660 training and78 validation roots')
    arrays = {k: [] for k in ('features', 'reference', 'changed', 'base', 'truth', 'incidence')}
    validation_rows, joint_truth, groups, metadata = [], [], [], []
    from ..train_single_revision import action_groups
    for i, root in enumerate(roots):
        features, reference, changed, base, fields = inputs_for_root(backbone, root, scales, device)
        truth = np.asarray(root['single4'], np.float64)[:, :, -1].sum(1)
        for key, value in zip(arrays, (features, reference, changed, base, truth, np.asarray(root['s_incidence'], np.float64))):
            arrays[key].append(value)
        metadata.append(root_metadata(root))
        if root['split'] == 'validation':
            joint = compose(root['s_incidence'], fields)
            true_joint = np.asarray(root['joint4'], np.float64)[:, :, -1].sum(1)
            record = field_record(root, joint, fields)
            record['joint_total240_mae'] = float(np.abs(joint[:, :, -1].sum(1)-true_joint).mean())
            validation_rows.append(record)
            joint_truth.append(true_joint)
            groups.append(action_groups([root]))
        if progress is not None and (i + 1) % 66 == 0:
            progress(i + 1, len(roots))
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    arrays.update(joint_truth=np.stack(joint_truth), groups=np.stack(groups))
    return arrays, {'roots': metadata, 'validation_fields': validation_rows, 'total_scale': float(scales['sj'][-1]),
                    'source_checkpoint_sha256': expected['sha256']}


def decision_records(totals, arrays, metadata):
    if totals.shape != (78, 64) or not np.isfinite(totals).all():
        raise ValueError('Expected78 validation roots x64 finite calibrated single totals')
    rows = []
    for i, p in enumerate(totals):
        index = 660 + i
        incidence, y = arrays['incidence'][index], arrays['truth'][index]
        scores, additive = incidence @ p, incidence @ y
        record = {**metadata['validation_fields'][i], **decision_metrics(scores, arrays['joint_truth'][i])}
        g = arrays['groups'][i]
        local_p, local_y = np.c_[np.zeros(16), p[g]], np.c_[np.zeros(16), y[g]]
        chosen = local_p.argmin(1)
        record.update(calibrated_single_total240_mae=float(np.abs(p-y).mean()),
            calibrated_joint_total240_mae=float(np.abs(scores-arrays['joint_truth'][i]).mean()),
            single_action_regret=float((local_y[np.arange(16), chosen]-local_y.min(1)).mean()),
            additive_regret=float(additive[record['selected']]-additive.min()),
            decision_source='calibrated_single_total240', field_source='frozen_v12')
        rows.append(record)
    return rows


def summarize_records(rows):
    from ..evaluate_decision_revision import summary
    result = summary(rows)
    for key in ('calibrated_single_total240_mae', 'calibrated_joint_total240_mae', 'single_action_regret', 'additive_regret'):
        result[key] = group_mean(rows, key)
        for subset in ('by_flow', 'by_cohort'):
            for sub in result[subset].values():
                sub[key] = group_mean(sub['records'], key)
    return result
