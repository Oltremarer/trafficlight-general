"""Three independent v12/v13/v14 A jobs; never mutates the concurrently running v11 B."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .counterfactual.writer import atomic_json, sha256
from .effect_model import balanced_revision as v12
from .effect_model import candidate_revision as v13
from .effect_model import total_revision as v14
from .evaluate_balanced_revision import lock_checkpoints
from .run_formal_effects import command_environment, read
from .run_local_revision import resource_snapshot, resource_allows
from .train_formal_effects import now


def supervise(run, source_v11=None, source_v12=None, source_v13=None):
    import fcntl
    sources=[(revision,source) for revision,source in ((v12,source_v11),(v13,source_v12),(v14,source_v13)) if source is not None]
    if len(sources)!=1:
        raise ValueError('Exactly one source revision required')
    revision,source_run=sources[0]
    family = revision.FAMILIES[0]
    run = Path(run).resolve()
    if not Path('/mnt/pan').is_mount() or run == Path('/mnt/pan') or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required')
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'execution.json').exists():
            raise RuntimeError('Duplicate launch refused')
        for name in ('logs','tmp','cache'):
            (run / name).mkdir(exist_ok=True)
        state={'stage':'preparing','pid':os.getpid(),'run_dir':str(run),'started_at':now(),'jobs':{},'errors':[]}
        children={}

        def save():
            state['updated_at']=now()
            atomic_json(run / 'execution.json',state)

        def launch(name,module,args=()):
            logfile=run / 'logs' / (name.replace('/','_')+'.log')
            command=['nice','-n','15','ionice','-c','3',sys.executable,'-u','-m',module,'--run-dir',str(run),*args]
            with logfile.open('x') as stream:
                p=subprocess.Popen(command,env=command_environment(run,1),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            children[name]=p
            return {'stage':'running','pid':p.pid,'command':command,'log':str(logfile),'started_at':now()}

        save()
        try:
            revision.initialize_run(run, source_run)
            source=Path(__file__).parent
            atomic_json(run / 'source_hashes.json',{str(p.relative_to(source)):sha256(p) for p in sorted(source.rglob('*.py'))})
            for seed in (42,43,44):
                state['jobs'][f'{family}/seed_{seed}']={'stage':'pending','seed':seed}
            state['stage']='training'
            save()
            while True:
                for name,job in state['jobs'].items():
                    if job['stage']=='running' and children[name].poll() is not None:
                        code=children[name].returncode
                        done=run / name / 'complete.json'
                        success=code==0 and done.exists() and read(done)['stage']=='complete'
                        job.update(stage='complete' if success else 'failed',exit_code=code,finished_at=now())
                        if success:
                            job['result']=read(done)
                        else:
                            state['errors'].append(name+' failed; independent jobs continue')
                for name,job in state['jobs'].items():
                    if job['stage']!='pending':
                        continue
                    active=[n for n,j in state['jobs'].items() if j['stage']=='running']
                    loading=sum(not (run/n/'status.json').exists() or read(run/n/'status.json').get('stage')=='loading' for n in active)
                    resources=resource_snapshot()
                    state['resources']=resources
                    if len(active)>=3 or not resource_allows(resources,len(active),loading):
                        job['waiting_reason']='Reserved memory headroom for concurrent work'
                        continue
                    job.update(launch(name,getattr(revision,'TRAIN_MODULE','cityflow_tsc.train_local_revision'),['--stage','train','--family',family,'--seed',str(job['seed'])]))
                    job.pop('waiting_reason',None)
                    save()
                state['counts']={s:sum(j['stage']==s for j in state['jobs'].values()) for s in ('pending','running','complete','failed')}
                save()
                if not state['counts']['pending'] and not state['counts']['running']:
                    break
                time.sleep(15)
            if state['errors']:
                raise RuntimeError('Incomplete A training; no final evaluation')
            lock_checkpoints(run, revision)
            state['stage']='evaluating'
            state['evaluation']=launch('evaluation',getattr(revision,'EVALUATION_MODULE','cityflow_tsc.evaluate_balanced_revision'))
            save()
            code=children['evaluation'].wait()
            if code!=0 or read(run/'evaluation/summary.json')['stage']!='complete':
                raise RuntimeError('Evaluation failed; artifacts retained')
            state['evaluation'].update(stage='complete',exit_code=code,finished_at=now())
            state.update(stage='complete',finished_at=now())
            save()
        except Exception as exc:
            state.update(stage='failed',error=repr(exc),traceback=traceback.format_exc(),failed_at=now(),
                still_running={n:p.pid for n,p in children.items() if p.poll() is None})
            save()
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-v11',type=Path)
    sources.add_argument('--source-v12',type=Path)
    sources.add_argument('--source-v13',type=Path)
    args=parser.parse_args()
    supervise(args.run_dir,args.source_v11,args.source_v12,args.source_v13)
