import importlib.util
import json
from pathlib import Path

import pytest

from cityflow_tsc.checkpoint_artifacts import CheckpointCatalog


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def archive(root, path, protocol):
    audit = root / 'control/checkpoint_retention_test'
    entry = {'checkpoint': str(path), 'sidecar': str(path.with_suffix('.protocol.json')),
             'completed_episodes': protocol['completed_episodes'], 'protocol': protocol}
    write(audit / 'manifest.json', {'run_root': str(root), 'delete': [entry]})
    write(audit / 'summary.json', {'status': 'complete', 'deleted_intermediate_checkpoints': 1})
    write(audit / 'deleted.jsonl', {k: entry[k] for k in ('checkpoint', 'sidecar')})
    return audit


def test_sparse_checkpoint_and_audited_cleanup_are_both_valid(tmp_path):
    output = tmp_path / 'train'
    final = output / 'checkpoints/mplight.episode_0001.pt'
    write(final.with_suffix('.protocol.json'), {'completed_episodes': 2, 'checkpoint_sha256': 'final'})
    final.write_bytes(b'weights')
    summary = {'completed_episodes': 2, 'checkpoint': str(final), 'training': [{'episode': 0}, {'episode': 1}]}
    config = {'profile': {'baseline_id': 'mplight'}, 'checkpoint_every': 100}
    assert CheckpointCatalog(tmp_path).validate_training(output, summary, config) == 1
    config.pop('checkpoint_every')  # Old runs saved every round, then were cleaned.
    missing = final.with_name('mplight.episode_0000.pt')
    with pytest.raises(FileNotFoundError):
        CheckpointCatalog(tmp_path).validate_training(output, summary, config)
    archive(tmp_path, missing, {'completed_episodes': 1, 'checkpoint_sha256': 'old'})
    assert CheckpointCatalog(tmp_path).validate_training(output, summary, config) == 1
    final.unlink()
    with pytest.raises(FileNotFoundError, match='required checkpoint weights'):
        CheckpointCatalog(tmp_path).validate_training(output, summary, config)


def test_incomplete_cleanup_and_orphan_weights_are_rejected(tmp_path):
    missing = tmp_path / 'model.pt'
    audit = archive(tmp_path, missing, {'completed_episodes': 10, 'checkpoint_sha256': 'hash'})
    (audit / 'deleted.jsonl').write_text('')
    with pytest.raises(ValueError, match='incomplete'):
        CheckpointCatalog(tmp_path).protocol(missing)
    missing.write_bytes(b'weights')
    with pytest.raises(ValueError, match='no protocol sidecar'):
        CheckpointCatalog(tmp_path).protocol(missing)


@pytest.mark.parametrize('owner', ['A', 'B'])
def test_summary_uses_archived_curve_identity_without_recreating_outputs(tmp_path, owner):
    script = Path(__file__).parents[1] / f'research/execution_{owner}_20260922/summarize.py'
    spec = importlib.util.spec_from_file_location('summary_' + owner, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write(tmp_path / 'control/binding.json', {'owner': owner, 'source_manifest_sha256': 'source', 'protocol_sha256': 'p'})
    write(tmp_path / 'control/environment.json', {'host': 'test'})
    write(tmp_path / 'handoff/datasets.json', {'scenarios': [{'scenario_id': 'S', 'roadnet_sha256': 'road', 'flow_sha256': 'flow'}]})
    write(tmp_path / 'handoff/result_record.template.json', {'metrics': {'ATT_engine_s': None}, 'missing_metric_reasons': {}})
    path = tmp_path / 'train/checkpoints/model.episode_0009.pt'
    archive(tmp_path, path, {'completed_episodes': 10, 'checkpoint_sha256': 'original'})
    write(tmp_path / 'control/queue.json', [{'task_id': 'curve', 'kind': 'curve_evaluate', 'evaluation_seed': 9000,
                                           'scenario_id': 'S', 'baseline': 'model', 'training_seed': 0}])
    write(tmp_path / 'curve/metrics.json', {'metrics': {'average_travel_time_s': 123.0}})
    write(tmp_path / 'state/curve.json', {'status': 'complete', 'output': str(tmp_path / 'curve'),
                                        'checkpoint': str(path), 'metric_schema_revision': 'v1',
                                        'collector_source_sha256': 'collector'})
    records, report = module.summarize(tmp_path, write_outputs=False)
    assert records[0]['checkpoint_sha256'] == 'original'
    assert records[0]['completed_episodes'] == 10
    assert records[0]['metrics']['ATT_engine_s'] == 123.0
    assert not (tmp_path / 'results').exists()
    assert not path.exists()
