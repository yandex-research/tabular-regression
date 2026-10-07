# Local experiment integration for TabPack; see VENDOR.md.

"""Expose TabPack online-ensemble evaluations in the standard report schema."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import lib.experiment
import lib.util
from lib.types import JSONDict


def _compact_seed_experiment(experiment: JSONDict, *, ensemble: str) -> JSONDict:
    config = experiment['config']
    report = experiment['report']
    seed = config['seed']
    if not isinstance(seed, int):
        raise TypeError(f'Evaluation seed must be an int, got {seed!r}')
    if 'path' not in config['data']:
        raise KeyError('Evaluation config is missing data.path')

    try:
        metrics = report['online_ensembles'][ensemble]['report']['metrics']
    except KeyError as err:
        raise KeyError(
            f'Evaluation report is missing metrics for ensemble {ensemble!r}'
        ) from err
    if 'val' not in metrics or 'test' not in metrics:
        raise KeyError(f'Ensemble {ensemble!r} metrics must contain val and test')

    compact_report: JSONDict = {
        'function': report.get('function', 'vendor.tabpack.tabpack.main'),
        'metrics': deepcopy(metrics),
    }
    for key in ('prediction_type', 'time'):
        if key in report:
            compact_report[key] = report[key]

    return {
        'config': {
            'seed': seed,
            'data': deepcopy(config['data']),
        },
        'report': compact_report,
    }


def make_standard_report(source_report: JSONDict, *, ensemble: str) -> JSONDict:
    """Project an author-style TabPack evaluation to `bin.evaluate` schema."""
    compact_experiments = [
        _compact_seed_experiment(experiment, ensemble=ensemble)
        for experiment in source_report['experiments']
    ]
    compact_experiments.sort(key=lambda x: x['config']['seed'])

    seen_seeds: set[int] = set()
    for experiment in compact_experiments:
        seed = experiment['config']['seed']
        if seed in seen_seeds:
            raise ValueError(f'Duplicate evaluation seed: {seed}')
        seen_seeds.add(seed)

    report: JSONDict = {
        'function': 'bin.evaluate.main',
        'experiments': compact_experiments,
    }
    if 'time' in source_report:
        report['time'] = source_report['time']
    return report


def export_standard_evaluation(
    source_exp: str | Path,
    target_exp: str | Path,
    *,
    ensemble: str,
) -> Path:
    """Create or refresh a compact standard evaluation experiment."""
    source_exp = Path(source_exp)
    target_exp = Path(target_exp)
    if not (
        source_exp.joinpath('config.toml').exists()
        or source_exp.joinpath('config.json').exists()
    ):
        raise FileNotFoundError(f'TabPack source config is missing at {source_exp}')
    source_report_path = source_exp / 'report.json'
    if not source_report_path.exists():
        raise FileNotFoundError(f'TabPack source report is missing at {source_exp}')

    report = make_standard_report(
        json.loads(source_report_path.read_text()), ensemble=ensemble
    )
    config: JSONDict = {
        'source': os.path.relpath(source_exp, target_exp),
        'ensemble': ensemble,
        'n_seeds': len(report['experiments']),
        'format_version': 1,
    }

    if lib.experiment.is_experiment(target_exp):
        if lib.experiment.load_config(target_exp) != config:
            raise ValueError(f'Incompatible standard evaluation config at {target_exp}')
    else:
        lib.experiment.create(target_exp, config=config, parents=True, force=False)
    lib.experiment.dump_report(target_exp, report)
    return target_exp


if __name__ == '__main__':
    lib.util.init(torch_=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('source_exp')
    parser.add_argument('target_exp')
    parser.add_argument('--ensemble', default='greedy')
    export_standard_evaluation(**vars(parser.parse_args()))
