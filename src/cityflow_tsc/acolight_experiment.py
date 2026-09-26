"""Train/evaluate Advanced-CoLight with efficient queue pressure and 167m running vehicles and graph attention."""
from __future__ import annotations

import json

from . import colight_experiment as common
from .baselines.acolight import ACoLightConfig, ACoLightLearner, PROFILE
from .colight_results import CoLightMetrics, DATASETS, write_comparison
from .runtime import build_environment
from .simulator import CityFlowBackend


EXPERIMENT_ID = 'acolight-round-fit-v1'


def make_environment(control, network, lane_change):
    env = build_environment(control, network, profile=PROFILE,
                            backend=CityFlowBackend(control, lane_change=lane_change))
    env.metrics = CoLightMetrics(network, lane_change=lane_change)
    return env


def references(dataset):
    if dataset not in DATASETS:
        return []
    rows = []
    for paper, table, values, url, setting in (
        ('LLMLight (arXiv v5)', '2', [274.67, 268.25, 260.66, 304.47, 329.16],
         'https://arxiv.org/html/2312.16044v5#S4.T2',
         'target-flow training; paper green/yellow/red=30/3/2; public code decision/yellow/red=30/5/0'),
        ('FuzzyLight (arXiv v2)', '2 (noise-free)', [255.30, 236.13, 232.76, 278.11, 319.39],
         'https://arxiv.org/html/2501.15820v2', '50 rounds; last ten tests; yellow/red=3/2'),
        ('Traffic-R1', '5 (appendix)', [247.31, 235.78, 242.56, 285.32, 323.19],
         'https://aclanthology.org/2026.acl-long.995.pdf', 'RL target-flow training; green/yellow/red=15/3/2'),
        ('Traffic-R1', '2', [347.31, 345.78, 342.56, 485.32, 523.19],
         'https://aclanthology.org/2026.acl-long.995.pdf', 'ZERO-SHOT transfer; green/yellow/red=15/3/2'),
    ):
        relation = ('official public roadnet/flow hashes match; independent implementation and seeds'
                    if paper.startswith('LLMLight') else 'same dataset label; files not cross-verified')
        rows.append(dict(paper=paper, table=table, paper_dataset=dataset,
                         paper_att_s=values[DATASETS.index(dataset)], url=url, setting=setting,
                         data_relation=relation))
    return rows


def report(output, protocol, aggregate):
    if protocol.get('baseline') != 'A-CoLight':
        raise ValueError('A-CoLight report requires A-CoLight experiment results')
    write_comparison(output, protocol, aggregate, reference_rows=references(protocol['dataset']))


def train(args):
    return common.train(args, profile=PROFILE, config_class=ACoLightConfig,
                        learner_class=ACoLightLearner, experiment_id=EXPERIMENT_ID,
                        baseline='A-CoLight', environment_factory=make_environment,
                        comparison_writer=report)


def evaluate(args):
    return common.evaluate(args, config_class=ACoLightConfig, learner_class=ACoLightLearner,
                           experiment_id=EXPERIMENT_ID, environment_factory=make_environment)


def compare(args):
    return common.compare(args, comparison_writer=report)


def build_parser():
    return common.build_parser(description=__doc__, name='A-CoLight')


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = {'train': train, 'evaluate': evaluate, 'compare': compare}[args.command](args)
    if args.command == 'train':
        result = {'output': str(args.output.resolve()), 'aggregate': result['aggregate']}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
