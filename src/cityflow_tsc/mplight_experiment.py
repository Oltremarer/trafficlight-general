"""Train/evaluate MPLight with its FRAP network and pinned-author fitted-Q protocol."""
from __future__ import annotations

import json

from . import colight_experiment as common
from .baselines.mplight import MPLightConfig, MPLightLearner, PROFILE
from .colight_results import CoLightMetrics, DATASETS, write_comparison
from .runtime import build_environment
from .simulator import CityFlowBackend


EXPERIMENT_ID = 'mplight-round-fit-v1'


def make_environment(control, network, lane_change):
    env = build_environment(control, network, profile=PROFILE,
                            backend=CityFlowBackend(control, lane_change=lane_change))
    env.metrics = CoLightMetrics(network, lane_change=lane_change)
    return env


def references(dataset):
    if dataset not in DATASETS:
        return []
    rows = [dict(paper='LLMLight (arXiv v5)', table='2', paper_dataset=dataset,
                 paper_att_s=[307.82, 304.51, 291.79, 345.60, 358.56][DATASETS.index(dataset)],
                 url='https://arxiv.org/html/2312.16044v5#S4.T2',
                 setting='target-flow training; paper green/yellow/red=30/3/2; public code decision/yellow/red=30/5/0',
                 data_relation='official public roadnet/flow hashes match; independent PyTorch implementation and seeds')]
    for paper, table, values, url, setting in (
        ('FuzzyLight (arXiv v2)', '2 (noise-free)', [300.25, 421.01, 507.53, 309.35, 346.32],
         'https://arxiv.org/html/2501.15820v2', '50 rounds; last ten tests; yellow/red=3/2'),
        ('Traffic-R1', '5 (appendix)', [310.54, 270.14, 272.37, 319.32, 365.42],
         'https://aclanthology.org/2026.acl-long.995.pdf', 'RL target-flow training; green/yellow/red=15/3/2'),
        ('Traffic-R1', '2', [455.34, 471.14, 427.37, 491.32, 425.42],
         'https://aclanthology.org/2026.acl-long.995.pdf', 'ZERO-SHOT transfer; green/yellow/red=15/3/2'),
        ('Astra', '1', [411.31, 298.84, 313.62, 301.34, 358.09],
         'https://liuzhidan.github.io/files/2026-KDD-Astra.pdf', '8-phase control; yellow=2; differs from four-phase run'),
    ):
        rows.append(dict(paper=paper, table=table, paper_dataset=dataset,
                         paper_att_s=values[DATASETS.index(dataset)], url=url, setting=setting,
                         data_relation='same dataset label; files not cross-verified'))
    return rows


def report(output, protocol, aggregate):
    if protocol.get('baseline') != 'MPLight':
        raise ValueError('MPLight report requires MPLight experiment results')
    write_comparison(output, protocol, aggregate, reference_rows=references(protocol['dataset']))


def train(args):
    return common.train(args, profile=PROFILE, config_class=MPLightConfig,
                        learner_class=MPLightLearner, experiment_id=EXPERIMENT_ID,
                        baseline='MPLight', environment_factory=make_environment,
                        comparison_writer=report)


def evaluate(args):
    return common.evaluate(args, config_class=MPLightConfig, learner_class=MPLightLearner,
                           experiment_id=EXPERIMENT_ID, environment_factory=make_environment)


def compare(args):
    return common.compare(args, comparison_writer=report)


def build_parser():
    return common.build_parser(description=__doc__, name='MPLight', include_attention=False)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = {'train': train, 'evaluate': evaluate, 'compare': compare}[args.command](args)
    if args.command == 'train':
        result = {'output': str(args.output.resolve()), 'aggregate': result['aggregate']}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
