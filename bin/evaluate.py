# NOTE[tabular]
import multiprocessing
import shutil
import tempfile
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import delu
from loguru import logger

import lib
import lib.experiment
import lib.parallel
import lib.util


class Config(TypedDict):
    function: str
    n_seeds: int
    base_config: dict[str, Any]
    # NOTE
    # Read the comments about `n_workers` and `n_gpus_per_worker` in `bin/tune.py`.
    n_workers: NotRequired[int]
    n_gpus_per_worker: NotRequired[int]


def _evaluate_seed(config: Config, exp: Path, seed: int) -> None:
    worker_id = lib.parallel.get_worker_id()
    function = lib.util.import_(config['function'])

    # All operations with the experiment directory must be performed under the lock.
    with lib.parallel.lock():
        checkpoint = lib.experiment.try_load_checkpoint(exp)
    if checkpoint is None:
        timer = delu.tools.Timer()
    else:
        del checkpoint['report']
        # Each worker resumes with the latest saved timer.
        timer = checkpoint.pop('timer')
        assert not checkpoint, (
            f'Some checkpoint fields were not used, namely: {sorted(checkpoint)}'
        )

    timer.run()

    seed_exp = exp / str(seed)
    if seed_exp.exists():
        logger.warning(f'Removing the incomplete experiment {seed_exp}')
        shutil.rmtree(seed_exp)

    seed_config: dict[str, Any] = {'seed': seed, **config['base_config']}
    if 'catboost' in config['function']:
        if seed_config['model']['task_type'] == 'GPU':
            seed_config['model']['task_type'] = (
                'CPU'  # This is crucial for good results.
            )
            seed_config['model']['thread_count'] = max(
                seed_config['model'].get('thread_count', 1),
                min(multiprocessing.cpu_count(), 4),
            )

    # Keep seed output on the evaluation filesystem for the final atomic rename.
    with tempfile.TemporaryDirectory(dir=exp, suffix=f'_evaluation_{seed}') as tmp_exp:
        tmp_exp = lib.experiment.create(tmp_exp, config=seed_config, force=True)
        seed_report = lib.experiment.run(function, None, tmp_exp)
        assert seed_report is not None
        if worker_id is not None:
            seed_report['evaluation'] = {'worker_id': worker_id}
        lib.experiment.remove_tracked_files(tmp_exp)
        lib.experiment.move(tmp_exp, seed_exp)

    # All operations with the experiment directory must be performed under the lock.
    with lib.parallel.lock():
        report = lib.experiment.load_report(exp)
        report.setdefault('experiments', []).append(
            {'config': seed_config, 'report': seed_report}
        )
        report['experiments'].sort(key=lambda x: x['config']['seed'])
        report['time'] = timer.elapsed()
        if worker_id is not None:
            report['worker_id'] = worker_id

        lib.experiment.dump_checkpoint(exp, {'report': report, 'timer': timer})
        lib.experiment.dump_report(exp, report)
        summary = lib.experiment.summarize(report)
        lib.experiment.dump_summary(exp, summary)
        lib.experiment.backup(exp)

    if len(report['experiments']) < config['n_seeds']:
        print(lib.util.add_frame(summary))


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)

    assert 'seed' not in config['base_config']
    assert exp.name == 'evaluation'

    checkpoint = lib.experiment.try_load_checkpoint(exp)
    if checkpoint is None:
        report = lib.experiment.create_report(main, add_gpu_info='n_workers' in config)
        completed_seeds = frozenset()
    else:
        report = checkpoint['report']
        completed_seeds = frozenset(x['config']['seed'] for x in report['experiments'])
        print(
            'Resuming from a checkpoint.'
            f' Completed {len(completed_seeds)}/{config["n_seeds"]} experiments'
        )
        report.setdefault('resumed_after_n_seeds', []).append(len(completed_seeds))

    lib.experiment.dump_report(exp, report)
    del report

    remaining_seeds = [x for x in range(config['n_seeds']) if x not in completed_seeds]
    n_remaining_seeds = len(remaining_seeds)

    if n_remaining_seeds == 0:
        return lib.experiment.load_report(exp)

    n_workers = config.get('n_workers')
    n_gpus_per_worker = config.get('n_gpus_per_worker')
    if n_gpus_per_worker is not None:
        assert n_workers is not None, (
            'The `n_gpus_per_worker` option cannot be used'
            ' without the `n_workers` option'
        )

    if n_workers is None:
        for seed in remaining_seeds:
            _evaluate_seed(config, exp, seed)
    else:
        if n_workers > n_remaining_seeds:
            logger.info(
                f'Reducing the number of workers'
                f' from {n_workers=} to {n_remaining_seeds=}'
            )
            n_workers = n_remaining_seeds
        logger.info(f'The number of workers: {n_workers}')

        lib.parallel.map(
            _evaluate_seed,
            [{'config': config, 'exp': exp, 'seed': seed} for seed in remaining_seeds],
            n_workers=n_workers,
            n_gpus_per_worker=1 if n_gpus_per_worker is None else n_gpus_per_worker,
        )

    report = lib.experiment.load_report(exp)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main, resumable=True)
