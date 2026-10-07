# Local experiment integration for TabPack; see VENDOR.md.

"""Run the original TabPack main/evaluation workflow from TOML experiments."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Literal

from loguru import logger

import bin.evaluate
import lib.env
import lib.experiment
import lib.util
import vendor.tabpack.tabpack
from vendor.tabpack.report_adapter import export_standard_evaluation

type OfflineEnsembleType = Literal['greedy']


def _check_source_exp(source_exp: str | Path) -> Path:
    if not lib.experiment.is_experiment(source_exp):
        raise RuntimeError(f'Not an experiment: {source_exp}')
    if not lib.experiment.is_done(source_exp):
        raise RuntimeError(f'Experiment is not done: {source_exp}')
    return Path(source_exp)


def _evaluate_ensemble(
    source_exp: str | Path,
    *,
    name: str,
    n_seeds: int,
    is_offline: bool,
    force: bool = False,
) -> Path:
    """Evaluate the unique members selected by an online/offline ensemble."""
    if n_seeds <= 0:
        raise ValueError(f'n_seeds must be positive, got {n_seeds}')

    source_exp = _check_source_exp(source_exp)
    source_config = lib.experiment.load_config(source_exp)
    source_report = lib.experiment.load_report(source_exp)

    ensemble_report = source_report['online_ensembles'][name]['report']
    ensemble_unique_ids = sorted(set(ensemble_report['ids']))
    pack_experiments = json.loads(source_exp.joinpath('experiments.json').read_text())

    base_config = deepcopy(source_config)
    if is_offline:
        assert not base_config['online_ensembles'][name][
            'include_current_ensemble_in_pool'
        ]
        assert base_config['online_ensembles'][name]['update_type'] != 'latest'

    base_config['n_models'] = len(ensemble_unique_ids)
    base_config['configs'] = [
        pack_experiments[model_id]['config'] for model_id in ensemble_unique_ids
    ]
    if base_config['optimizer']['type'] in ('AdamWPack', 'MuonAdamWPack'):
        base_config['optimizer']['shared_step'] = True

    for ensemble_name in list(source_config['online_ensembles']):
        if ensemble_name != name:
            del base_config['online_ensembles'][ensemble_name]
    for key in (
        'seed',
        'pack_size',
        'share_training_batches',
        'share_training_batch_sequence',
        'object_bagging',
        'sampler',
    ):
        base_config.pop(key, None)

    exp = (
        source_exp.parent
        / ('eval-offline-ensembles' if is_offline else 'eval-online-ensembles')
        / name
        / 'evaluation'
    )
    config: bin.evaluate.Config = {
        'function': source_report['function'],
        'n_seeds': n_seeds,
        'base_config': base_config,
    }
    if lib.experiment.is_experiment(exp):
        if lib.experiment.load_config(exp) != config:
            raise ValueError(f'Incompatible ensemble evaluation config at {exp}')
    else:
        lib.experiment.create(exp, config=config, parents=True)
    lib.experiment.run(bin.evaluate.main, None, exp, force=force, resume=True)
    return exp


def main(
    exp_prefix: str | Path,
    *,
    eval_online_ensembles: list[str],
    eval_online_ensembles_n_seeds: None | int,
    eval_offline_ensembles: list[OfflineEnsembleType],
    eval_offline_ensembles_n_seeds: None | int,
    clean: bool = False,
    force: bool = False,
    restart_crashed: bool = False,
) -> None:
    exp_prefix = Path(exp_prefix)
    main_exp = exp_prefix / 'main'
    if not lib.experiment.is_experiment(main_exp):
        raise RuntimeError(f'Missing main experiment: {main_exp}')

    # tabpack.main cannot resume mid-run, so an interrupted experiment
    # (the _RUNNING marker exists, but there is no report) can only be rerun.
    main_interrupted = not lib.experiment.is_fresh(
        main_exp
    ) and not lib.experiment.is_done(main_exp)
    if main_interrupted and not force:
        if not restart_crashed:
            raise RuntimeError(
                'The main experiment was interrupted (the _RUNNING marker exists,'
                f' but there is no report): {main_exp}\n'
                'Pass --restart-crashed to reset it and run from scratch.'
            )
        logger.warning(f'Resetting the interrupted experiment: {main_exp}')
        lib.experiment.reset(main_exp)

    if not lib.experiment.is_done(main_exp) or force:
        for path in exp_prefix.iterdir():
            if path.is_dir() and path != main_exp:
                logger.warning(f'Removing {path}')
                shutil.rmtree(path)

    lib.experiment.run(vendor.tabpack.tabpack.main, None, main_exp, force=force)

    for online_ensemble_name in eval_online_ensembles:
        if eval_online_ensembles_n_seeds is None:
            raise ValueError(
                '--eval-online-ensembles-n-seeds is required when evaluating'
                ' online ensembles'
            )
        source_evaluation = _evaluate_ensemble(
            main_exp,
            name=online_ensemble_name,
            n_seeds=eval_online_ensembles_n_seeds,
            is_offline=False,
            force=force,
        )
        if online_ensemble_name == 'greedy':
            export_standard_evaluation(
                source_evaluation,
                exp_prefix / 'evaluation',
                ensemble=online_ensemble_name,
            )

    for offline_ensemble_type in eval_offline_ensembles:
        if eval_offline_ensembles_n_seeds is None:
            raise ValueError(
                '--eval-offline-ensembles-n-seeds is required when evaluating'
                ' offline ensembles'
            )
        _evaluate_ensemble(
            main_exp,
            name=offline_ensemble_type,
            n_seeds=eval_offline_ensembles_n_seeds,
            is_offline=True,
            force=force,
        )

    if clean:
        for directory in (
            lib.env.get_project_dir(),
            lib.env.get_snapshot_dir(),
            lib.env.get_tmp_output_dir(),
        ):
            if directory is None:
                continue
            relative_exp = exp_prefix.resolve().relative_to(lib.env.get_project_dir())
            for path in directory.joinpath(relative_exp).rglob('*'):
                if not path.is_dir() and path.suffix in ('.pt', '.npz'):
                    path.unlink()


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_prefix')
    parser.add_argument('--eval-online-ensembles', nargs='*', default=[])
    parser.add_argument('--eval-online-ensembles-n-seeds', type=int)
    parser.add_argument('--eval-offline-ensembles', nargs='*', default=[])
    parser.add_argument('--eval-offline-ensembles-n-seeds', type=int)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--restart-crashed',
        action='store_true',
        help='reset and rerun the main experiment if it was interrupted'
        ' (has the _RUNNING marker, but no report)',
    )

    main(**vars(parser.parse_args()))
