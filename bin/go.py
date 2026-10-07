import argparse
import shutil
import sys
from pathlib import Path
from typing import cast

from loguru import logger

import bin.ensemble
import bin.evaluate
import bin.tune
import lib.experiment
import lib.util


def main(
    exp: str | Path,  # A tuning or evaluation experiment.
    *,
    n_seeds: int,
    ensemble_sizes: None | list[int],
    force: bool = False,
    resume: bool = False,
    clean: bool = False,
    full_clean: bool = False,
):
    if not resume:
        # Supporting resume=False for a sequence of experiments is tricky.
        raise ValueError('Currently, only resume=True is supported')
    if ensemble_sizes is None:
        ensemble_sizes = []

    exp = Path(exp).resolve()
    assert lib.experiment.is_experiment(exp)
    assert exp.name in ('tuning', 'evaluation')

    is_tuning = exp.name == 'tuning'
    tuning_exp = exp.with_name('tuning')
    evaluation_exp = exp.with_name('evaluation')
    ensemble_exps = [exp.with_name(f'ensemble-{x}') for x in ensemble_sizes]
    del exp

    if force:
        logger.warning('Resetting and removing existing experiments')
        # The tuning, evaluation and ensemble experiments are processed
        # in the reversed order.
        for exp in ensemble_exps:
            if exp.exists():
                shutil.rmtree(exp)
            del exp
        if is_tuning:
            if evaluation_exp.exists():
                shutil.rmtree(evaluation_exp)
            lib.experiment.reset(tuning_exp)
        else:
            lib.experiment.reset(evaluation_exp)

    if lib.experiment.is_experiment(evaluation_exp) and not lib.experiment.is_done(
        evaluation_exp
    ):
        assert not any(x.exists() for x in ensemble_exps)

    if is_tuning:
        if not lib.experiment.is_done(tuning_exp):
            assert not evaluation_exp.exists()
            assert not any(x.exists() for x in ensemble_exps)

        lib.experiment.run(bin.tune.main, None, tuning_exp, resume=resume)

        tuning_config = cast(bin.tune.Config, lib.experiment.load_config(tuning_exp))
        tuning_report = lib.experiment.load_report(tuning_exp)
        evaluation_config: bin.evaluate.Config = {
            'function': tuning_config['function'],
            'n_seeds': n_seeds,
            'base_config': tuning_report['best']['config'],
        }
        for key in ['n_workers', 'n_gpus_per_worker']:
            if key in tuning_config:
                evaluation_config[key] = tuning_config[key]
            del key
        evaluation_config['base_config'].pop('seed', None)
        if lib.experiment.get_config_path(evaluation_exp).exists():
            # NOTE
            # If tuning_exp was done, this branch is active. In theory, the existing
            # evaluation config must be identical to `evaluation_config`. In practice,
            # it is unclear if TOML and JSON serializations can lead to a difference.
            assert lib.experiment.load_config(evaluation_exp) == evaluation_config
        else:
            lib.experiment.create(evaluation_exp, config=evaluation_config, force=True)  # type: ignore

    lib.experiment.run(bin.evaluate.main, None, evaluation_exp, resume=resume)

    for ensemble_size in ensemble_sizes:
        ensemble_exp = evaluation_exp.with_name(f'ensemble-{ensemble_size}')
        ensemble_config: bin.ensemble.Config = {'ensemble_size': ensemble_size}
        if lib.experiment.get_config_path(ensemble_exp).exists():
            assert lib.experiment.load_config(ensemble_exp) == ensemble_config
        else:
            lib.experiment.create(ensemble_exp, config=ensemble_config, force=True)  # type: ignore

        lib.experiment.run(bin.ensemble.main, None, ensemble_exp, resume=resume)

    if clean or full_clean:
        experiments_to_clean = [evaluation_exp, *ensemble_exps]
        if is_tuning:
            experiments_to_clean.insert(0, tuning_exp)
        _clean_experiments(experiments_to_clean, full=full_clean)


def _clean_experiments(experiments: list[Path], *, full: bool = False) -> None:
    experiments = list(dict.fromkeys(exp.resolve() for exp in experiments))
    for exp in experiments:
        if not lib.experiment.is_done(exp):
            raise RuntimeError(f'Cannot clean an incomplete experiment: {exp}')

    for exp in experiments:
        logger.warning(f'Removing heavy artifacts (full={full}): {exp}')
        for path in exp.rglob('*'):
            if path.is_symlink():
                path.unlink()
                continue
            if not path.is_file():
                continue
            if full:
                is_heavy = path.name == 'checkpoint.pt' or path.suffix == '.npz'
            else:
                is_heavy = path.name in ('checkpoint.pt', 'predictions.npz')
            if not is_heavy:
                continue
            path.unlink()
        for path in sorted(exp.rglob('*'), reverse=True):
            if path.is_dir() and not path.is_symlink():
                try:
                    path.rmdir()
                except OSError:
                    pass
        lib.experiment.backup(exp)


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    parser.add_argument('--n-seeds', type=int, required=True)
    parser.add_argument('--ensemble-sizes', type=int, nargs='*')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='remove artifacts after a successful run, keeping only the results',
    )
    parser.add_argument(
        '--full_clean',
        action='store_true',
        help='remove all checkpoints and NumPy artifacts after a successful run',
    )

    main(**vars(parser.parse_args(sys.argv[1:])))
