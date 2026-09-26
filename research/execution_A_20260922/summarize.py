"""Export A's raw records and explicitly partial, model-level statistics."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

from cityflow_tsc.checkpoint_artifacts import CheckpointCatalog


def read(path):
    return json.loads(path.read_text())


def atomic(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def summarize(root, *, write_outputs=True):
    catalog = CheckpointCatalog(root)
    binding=read(root/'control/binding.json')
    queue=read(root/'control/queue.json')
    datasets={x['scenario_id']:x for x in read(root/'handoff/datasets.json')['scenarios']}
    template=read(root/'handoff/result_record.template.json')
    states={p.stem:read(p) for p in (root/'state').glob('*.json')}
    records=[]
    aliases={'ATT_engine_s':'average_travel_time_s','AWT_observed_s':'average_waiting_time_s',
             'average_network_queue_vehicles':'average_queue_vehicles'}
    for task in queue:
        state=states.get(task['task_id'],{})
        if state.get('status')!='complete':
            continue
        output=Path(state['output'])
        specs=[(seed,output/'evaluation'/('seed_%d'%seed)/'metrics.json','legacy_final')
               for seed in (10000,10001,10002)] if task['kind']=='train_then_evaluate' else [
                   (task['evaluation_seed'],output/'metrics.json',task['kind'])]
        for seed, path, role in specs:
            raw=read(path)
            record=json.loads(json.dumps(template))
            record.update(template_only=False,owner='A',status='complete',task_id=task['task_id'],
                          parent_task_id=task.get('parent_task_id',task['task_id']),evaluation_role=role,
                          scenario_id=task['scenario_id'],baseline_or_policy=task.get('baseline',task.get('policy')),
                          training_seed=task.get('training_seed'),evaluation_seed=seed,
                          attempt_id=output.name,raw_metric_path=str(path),log_path=state.get('log_path'),
                          command_argv=state.get('command_argv',[]),exit_code=state.get('exit_code'),
                          host_id=read(root/'control/environment.json')['host'],
                          metric_schema_revision=state['metric_schema_revision'],
                          collector_source_sha256=state['collector_source_sha256'],
                          protocol_sha256=binding['protocol_sha256'],
                          training_source_manifest_sha256=binding['source_manifest_sha256'],
                          evaluation_source_manifest_sha256=state.get('evaluation_source_manifest_sha256',binding['source_manifest_sha256']),
                          trajectory_manifest_path=raw.get('manifest_path'),
                          wall_seconds=state.get('wall_seconds'),
                          wall_seconds_scope='parent train+automatic evaluations' if role=='legacy_final' else 'single evaluation subprocess and verification')
            dataset=datasets[task['scenario_id']]
            for kind in ('roadnet','flow'):
                record[kind+'_sha256']=dataset[kind+'_sha256']
            checkpoint=state.get('checkpoint')
            if not checkpoint and '--checkpoint' in state.get('command_argv',[]):
                argv=state['command_argv'];checkpoint=argv[argv.index('--checkpoint')+1]
            if checkpoint:
                sidecar=catalog.protocol(checkpoint)
                record.update(checkpoint_path=checkpoint,checkpoint_sha256=sidecar['checkpoint_sha256'],
                              completed_episodes=sidecar['completed_episodes'])
            for name in record['metrics']:
                key=aliases.get(name,name)
                record['metrics'][name]=raw['metrics'].get(key)
                if key not in raw['metrics']:
                    record['missing_metric_reasons'][name]='Not collected by this metric revision'
            records.append(record)
    result_dir=root/'results'
    for filename,roles in [('legacy_final_records.jsonl',{'legacy_final'}),
                           ('lifecycle_final_records.jsonl',{'final_evaluate','rule_evaluate'}),
                           ('lifecycle_curve_records.jsonl',{'curve_evaluate'})]:
        selected=[r for r in records if r['evaluation_role'] in roles]
        if write_outputs:
            atomic(result_dir/filename,''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in selected))
    models=defaultdict(list)
    for row in records:
        if row['evaluation_role']=='final_evaluate':
            key=(row['scenario_id'],row['baseline_or_policy'],row['training_seed'],row['metric_schema_revision'],row['checkpoint_sha256'])
            models[key].append(row)
    grouped=defaultdict(list)
    for (scenario,baseline,seed,revision,checkpoint), rows in models.items():
        if sorted(r['evaluation_seed'] for r in rows) != [10000,10001,10002]:
            continue
        means={name:statistics.mean(r['metrics'][name] for r in rows)
               for name in rows[0]['metrics'] if all(r['metrics'][name] is not None for r in rows)}
        grouped[scenario,baseline,revision].append({'training_seed':seed,'checkpoint_sha256':checkpoint,'metrics':means})
    summaries=[]
    for (scenario,baseline,revision), rows in sorted(grouped.items()):
        expected=sum(t['kind']=='train_then_evaluate' and t['scenario_id']==scenario and t['baseline']==baseline for t in queue)
        summaries.append({'scenario':scenario,'baseline':baseline,'metric_revision':revision,
                          'scope':'A only; not the full five-training-seed result',
                          'n_independent_models':len(rows),'expected_A_models':expected,'models':rows,
                          'metrics':{key:{'mean':statistics.mean(r['metrics'][key] for r in rows),
                                          'sample_sd':statistics.stdev(r['metrics'][key] for r in rows) if len(rows)>1 else None}
                                     for key in rows[0]['metrics']}})
    status=Counter(states.get(t['task_id'],{}).get('status','pending') for t in queue)
    report={'updated_at':datetime.now(timezone.utc).isoformat(),'owner':'A','job_status':dict(status),
            'whole_five_seed_table_ready':False,'B_raw_results_received':False,
            'A_partial_model_summaries':summaries,
            'note':'Legacy and lifecycle metric revisions are separate. Final metrics average three eval seeds within each model first; sample SD across available A models, ddof=1. Await B raw records for n=5.'}
    if write_outputs:
        atomic(result_dir/'A_summary.json',json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if write_outputs:
        print(json.dumps({'records':len(records),'complete_A_model_groups':len(summaries),'status':dict(status)}))

    return records, report


if __name__=='__main__':
    summarize(Path(sys.argv[1]).resolve())
