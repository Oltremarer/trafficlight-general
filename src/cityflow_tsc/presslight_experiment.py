"""Train/evaluate PressLight with its phase-branch network and pinned-author fitted-Q protocol."""
from __future__ import annotations

import json

from . import colight_experiment as common
from .baselines.presslight import PressLightConfig, PressLightLearner, PROFILE
from .colight_results import CoLightMetrics, DATASETS, write_comparison
from .runtime import build_environment
from .simulator import CityFlowBackend


EXPERIMENT_ID = 'presslight-round-fit-v1'


def make_environment(control, network, lane_change):
    env = build_environment(control, network, profile=PROFILE,
                            backend=CityFlowBackend(control, lane_change=lane_change))
    env.metrics = CoLightMetrics(network, lane_change=lane_change)
    return env


def references(dataset):
    if dataset not in DATASETS:
        return []
    rows = [dict(paper='LLMLight (arXiv v5)', table='2', paper_dataset=dataset,
                 paper_att_s=[291.57, 281.46, 275.85, 364.13, 417.01][DATASETS.index(dataset)],
                 url='https://arxiv.org/html/2312.16044v5#S4.T2',
                 setting='target-flow training; paper green/yellow/red=30/3/2; public code decision/yellow/red=30/5/0',
                 data_relation='official public roadnet/flow hashes match; independent PyTorch implementation and seeds')]
    for paper, table, values, url, setting in (
        ('Astra', '1', [455.35, 319.36, 311.42, 407.34, 463.12],
         'https://liuzhidan.github.io/files/2026-KDD-Astra.pdf', '8-phase control; yellow=2; differs from four-phase run'),
    ):
        rows.append(dict(paper=paper, table=table, paper_dataset=dataset,
                         paper_att_s=values[DATASETS.index(dataset)], url=url, setting=setting,
                         data_relation='same dataset label; files not cross-verified'))
    return rows


def report(output, protocol, aggregate):
    if protocol.get('baseline') != 'PressLight':
        raise ValueError('PressLight report requires PressLight experiment results')
    write_comparison(output, protocol, aggregate, reference_rows=references(protocol['dataset']))


def train(args):
    return common.train(args, profile=PROFILE, config_class=PressLightConfig,
                        learner_class=PressLightLearner, experiment_id=EXPERIMENT_ID,
                        baseline='PressLight', environment_factory=make_environment,
                        comparison_writer=report)


def evaluate(args):
    return common.evaluate(args, config_class=PressLightConfig, learner_class=PressLightLearner,
                           experiment_id=EXPERIMENT_ID, environment_factory=make_environment)


def compare(args):
    return common.compare(args, comparison_writer=report)


def build_parser():
    return common.build_parser(description=__doc__, name='PressLight', include_attention=False)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = {'train': train, 'evaluate': evaluate, 'compare': compare}[args.command](args)
    if args.command == 'train':
        result = {'output': str(args.output.resolve()), 'aggregate': result['aggregate']}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
