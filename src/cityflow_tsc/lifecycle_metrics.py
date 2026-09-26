"""Read-only, complete vehicle-ID accounting for the pinned one-shot flows.

An evaluation is rejected if any scheduled vehicle is unobserved. This avoids
silently calling invalid routes or unobserved within-tick events completions.
Waiting-to-enter vehicles are read using get_vehicles(True), independently of
the speed dictionary used by the legacy throughput estimate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REVISION = 'core-lifecycle-id-ledger-v1'


class UnsupportedLifecycleFlow(ValueError):
    """The exact v1 denominator is not defined for this input protocol."""


class LifecycleLedger:
    def __init__(self, expected_ids, horizon_s):
        self.expected = set(expected_ids)
        self.horizon = float(horizon_s)
        self.seen = set()
        self.entered = set()
        self.completed = set()
        self.previous_pool = set()
        self.previous_active = set()
        self.time = None
        self.active_count_source = 'engine.get_vehicle_count()'

    @classmethod
    def from_scenario(cls, scenario, step_s):
        if step_s != 1:
            raise UnsupportedLifecycleFlow('Lifecycle v1 requires one-second snapshots')
        flows = json.loads(scenario.flow_path.read_text())
        for flow in flows:
            start, end = flow['startTime'], flow['endTime']
            if not (start == end and float(start).is_integer() and 0 <= start < scenario.duration_s):
                raise UnsupportedLifecycleFlow('Lifecycle v1 requires one-shot integer departures inside horizon')
            if flow['interval'] < 1 or not flow['route']:
                raise UnsupportedLifecycleFlow('Lifecycle v1 requires interval >= 1 and nonempty routes')
        return cls({'flow_%d_0'%i for i in range(len(flows))},scenario.duration_s)

    def observe(self, time_s, pool_ids, active_ids, active_count):
        pool, active = set(pool_ids), set(active_ids)
        if not active <= pool or len(active) != active_count:
            raise ValueError('Active IDs disagree with complete pool or engine active count')
        if not pool <= self.expected:
            raise ValueError('Unexpected vehicle IDs: %s'%sorted(pool-self.expected)[:5])
        if self.time == time_s:
            if pool != self.previous_pool or active != self.previous_active:
                raise ValueError('Vehicle membership changed without advancing time')
            return
        if self.time is not None and time_s != self.time+1:
            raise ValueError('Lifecycle snapshots must cover every simulator second')
        if pool & self.completed:
            raise ValueError('Completed vehicle ID reappeared')
        removed = self.previous_pool-pool
        if not removed <= self.previous_active:
            raise ValueError('A vehicle disappeared before entering the network')
        if (self.entered & pool)-active:
            raise ValueError('Entered vehicle unexpectedly returned to waiting buffer')
        self.completed.update(removed)
        self.seen.update(pool)
        self.entered.update(active)
        self.previous_pool, self.previous_active, self.time = pool,active,float(time_s)

    def summary(self):
        if self.time != self.horizon:
            raise ValueError('Lifecycle summary requires the complete episode horizon')
        missing = self.expected-self.seen
        if missing:
            raise ValueError('Scheduled vehicle generation not fully observed: %d missing, examples=%s'%(
                len(missing),sorted(missing)[:5]))
        waiting = self.previous_pool-self.previous_active
        if waiting != self.expected-self.entered:
            raise ValueError('Waiting-buffer membership disagrees with never-entered IDs')
        if self.expected != self.completed | self.previous_pool:
            raise ValueError('Lifecycle ledger is incomplete')
        scheduled, finished = len(self.expected),len(self.completed)
        return {'scheduled_vehicles':scheduled,'generated_vehicles':len(self.seen),
                'entered_vehicles':len(self.entered),'finished_vehicles':finished,
                'active_unfinished_vehicles':len(self.previous_active),
                'not_entered_vehicles':len(waiting),
                'completion_rate':finished/scheduled if scheduled else 0.0}

    def write_evidence(self, path):
        path = Path(path)
        metrics = self.summary()
        evidence = {'metric_schema_revision':REVISION,'metrics':metrics,'time_s':self.time,
                    'count_method':'Complete one-shot scheduled-ID ledger; real active and waiting pool observed each second; all scheduled generation accounted for.',
                    'active_count_source':self.active_count_source,
                    'collector_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'finished_vehicle_ids':sorted(self.completed),
                    'active_vehicle_ids':sorted(self.previous_active),
                    'waiting_to_enter_vehicle_ids':sorted(self.previous_pool-self.previous_active),
                    'unobserved_scheduled_vehicle_ids':sorted(self.expected-self.seen)}
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temp.replace(path)
        return metrics
