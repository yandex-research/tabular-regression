import shutil
import math
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import delu
import optuna
import optuna.samplers
import optuna.trial
from loguru import logger

import lib
import lib.experiment
import lib.parallel
import lib.types
import lib.util
from lib.types import KWArgs

type ConfigSpace = dict[str, Any]


def _get_storage_url(study_path: str | Path) -> str:
    study_path = Path(study_path)
    assert study_path.suffix == '.sqlite'
    return f'sqlite:///{study_path}'


def _get_seed(base_seed: int, worker_id: None | int) -> int:
    # The `1000000` multiplier is needed to avoid overlapping worker seeds for runs
    # with adjacent base seeds (e.g. base_seed=0 and base_seed=1).
    return base_seed if worker_id is None else base_seed + 1000000 * worker_id


def _filter_trials(
    study_path: Path, *, keep_trial_ids: list[int], **study_kwargs
) -> None:
    # This function can remove trials from the study file, and thus must be called
    # only from the main process when there are no trials running in parallel.
    assert lib.parallel.get_worker_id() is None, (
        'This function can be called only from the main process'
    )

    study_storage_url = _get_storage_url(study_path)
    study = optuna.load_study(study_name=None, storage=study_storage_url)

    # `all_trials` are sorted by ID,
    # which can be different from the order in `keep_trial_ids`
    # in the multi-process setup.
    all_trials = study.get_trials(deepcopy=False)
    id_to_trial = {t.number: t for t in all_trials}

    # Filter the trials by ID
    # and reorder the trials to follow the order in `keep_trial_ids`.
    trials = [id_to_trial[x] for x in keep_trial_ids]

    if len(trials) < len(all_trials):
        logger.info(
            f'Removing {len(all_trials) - len(trials)} trials'
            f' from the study "{study_path}"'
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_study_path = Path(tmp_dir) / study_path.name
            study = optuna.create_study(
                storage=_get_storage_url(tmp_study_path), **study_kwargs
            )
            study.add_trials(trials)
            tmp_study_path.rename(study_path)


def _sample_value(
    trial: optuna.trial.Trial,
    distribution: Literal['int', 'uniform', 'loguniform', 'categorical'],
    label: str,
    *args,
):
    trial_suggest, kwargs = {
        'int': (trial.suggest_int, {}),
        'uniform': (trial.suggest_float, {}),
        'loguniform': (trial.suggest_float, {'log': True}),
        'categorical': (trial.suggest_categorical, {}),
    }[distribution]
    if distribution in ('int', 'uniform', 'loguniform') and len(args) == 3:
        args, kwargs['step'] = args[:2], args[2]
    return trial_suggest(label, *args, **kwargs)


def _sample_config(
    trial: optuna.trial.Trial,
    space: bool | int | float | str | bytes | list | dict,
    label_parts: list,
) -> Any:
    if isinstance(space, bool | int | float | str | bytes):
        # This is a constant value, nothing to sample from.
        return space

    elif isinstance(space, list):
        if space and space[0] == '_tune_':
            # space: ["_tune_", distribution, arg_0, arg_1, ...]
            _, distribution, *args = space
            label = '.'.join(map(str, label_parts))

            # At this point, `distribution` can be one of the following:
            # 1. One of the built-in Optuna distributions expected in `_sample_value`.
            # 2. Same as 1., but prefixed with "?".
            # 3. Custom distributions. By convention, they start with "$".

            if distribution.startswith('?'):
                # space: ["_tune_", "?distribution", default_value, *actual_args]
                default, args_ = args[0], args[1:]
                if trial.suggest_categorical(f'?{label}', [False, True]):
                    return _sample_value(trial, distribution.lstrip('?'), label, *args_)
                else:
                    return default

            elif distribution == '$one_minus_loguniform':
                return 1 - _sample_value(trial, 'loguniform', label, *args)

            elif distribution == '$floor_loguniform':
                return math.floor(_sample_value(trial, 'loguniform', label, *args))

            elif distribution == '$power_of_two':
                # space: ["_tune_", "$power_of_two", low_exp, high_exp]
                # samples 2^k where k in [low_exp, high_exp]
                low_exp, high_exp = args
                return 2.0 ** trial.suggest_int(label, low_exp, high_exp)

            elif distribution == '$list':
                # space: ["_tune_", "$list", size, distribution, *actual_args]
                # A list of hyperparameters of a fixed size. For example, this
                # can be useful if a model allows configuring some hyperparameter
                # separately for each feature (in this case, `size` is the number
                # of features).
                size, item_distribution, *item_args = args
                return [
                    _sample_value(trial, item_distribution, f'{label}.{i}', *item_args)
                    for i in range(size)
                ]

            elif distribution == '$branched':
                # Branched parameter sampling for tabarena-new style search spaces.
                # Multiple params can share the same branch selector label.
                # Format: ["_tune_", "$branched", selector_label, true_choices, false_value]
                #
                # Example:
                #   plr_hidden_1 = ["_tune_", "$branched", "large_config", [8, 16, 32, 64], 16]
                #   plr_hidden_2 = ["_tune_", "$branched", "large_config", [8, 16, 32, 64], 4]
                #
                # Both params share the same "large_config" boolean decision.
                # Optuna caches suggest_categorical by label, so they get the same value.
                selector_label, true_choices, false_value = args
                is_true_branch = trial.suggest_categorical(selector_label, [False, True])

                if is_true_branch:
                    return trial.suggest_categorical(label, true_choices)
                else:
                    # Don't register a parameter - there's nothing to tune in false branch.
                    # Optuna handles conditional/missing parameters correctly.
                    return false_value

            else:
                return _sample_value(trial, distribution, label, *args)

        else:
            return [
                _sample_config(trial, subspace, [*label_parts, i])
                for i, subspace in enumerate(space)
            ]

    elif isinstance(space, dict):
        if '_tune_' in space:
            # A custom sampling rule of any complexity. For example, in config:
            #
            # [space.model]
            # _tune_ = "$hyperparameter-distribution-for-my-model"
            # a = 0    # <-- any key and value
            # b = 1.0  # <-- any key and value
            # c = '2'  # <-- any key and value
            distribution = space['_tune_']
            if distribution == '$hyperparameter-distribution-for-my-model':
                assert space['a'] == 0
                assert space['b'] == 1.0
                assert space['c'] == '2'
                model_config = {...}  # <-- Sample the model config based on a, b, c.
                raise NotImplementedError()
                return model_config
            else:
                raise ValueError(f'Unknown distibution: "{distribution}"')

        else:
            return {
                key: _sample_config(trial, subspace, [*label_parts, key])
                for key, subspace in space.items()
            }


def _objective(
    trial: optuna.trial.Trial,
    *,
    function: lib.experiment.MainFunction,
    space: ConfigSpace,
    timer: delu.tools.Timer,
) -> float:
    worker_id = lib.parallel.get_worker_id()
    trial_config = _sample_config(trial, space, [])

    with tempfile.TemporaryDirectory(suffix=f'_trial_{trial.number}') as tmp_exp:
        print()
        tmp_exp = lib.experiment.create(tmp_exp, config=trial_config, force=True)
        trial_report = lib.experiment.run(function, None, tmp_exp)

    assert trial_report is not None
    trial_report['tuning'] = {'trial_id': trial.number, 'time': timer.elapsed()}
    if worker_id is not None:
        trial_report['tuning']['worker_id'] = worker_id
    trial.set_user_attr('experiment', {'config': trial_config, 'report': trial_report})

    delu.cuda.free_memory()
    return trial_report['metrics']['val']['score']


def _summarize_study(
    study: optuna.study.Study,
    *,
    max_n_trials: None | int,
    previously_completed_trial_ids: None | list[int],
) -> tuple[list[optuna.trial.FrozenTrial], optuna.trial.FrozenTrial]:
    """
    The function extracts:

    * A list of completed trials from `study`.
    * The best trial within this list.

    The list has the following properties:

    * Trials with IDs from `previously_completed_trial_ids` will go before
      the rest of the trials. This property is important for the multi-process setup,
      where the order of completion can differ from the order of IDs
      (note that `study.get_trials` returns trials sorted by IDs).
    * The size of the list will be at most `max_n_trials` (if provided).
      This is also important for the multi-process setup,
      where the number of trials can exceed `n_trials`.
    """
    # `get_trials` must be the only query to the underlying study storage,
    # because the study storage is concurrently updated by other workers.
    # Thus, for example, the best trial must be found manually,
    # without using `study.best_trial`.
    original_completed_trials = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    direction = study.direction
    del study

    previously_completed_trial_ids_set = frozenset(
        [] if previously_completed_trial_ids is None else previously_completed_trial_ids
    )
    completed_trials = []
    new_completed_trials = []
    for trial in original_completed_trials:
        if trial.number in previously_completed_trial_ids_set:
            completed_trials.append(trial)
        else:
            new_completed_trials.append(trial)

    if max_n_trials is not None:
        if len(completed_trials) < max_n_trials:
            new_completed_trials = new_completed_trials[
                : max_n_trials - len(completed_trials)
            ]
        else:
            new_completed_trials = []

    completed_trials.extend(new_completed_trials)
    if max_n_trials is not None:
        # At this point, the following condition must always be true.
        assert len(completed_trials) <= max_n_trials

    if direction == optuna.study.StudyDirection.MAXIMIZE:
        find_best = max
    elif direction == optuna.study.StudyDirection.MINIMIZE:
        find_best = min
    else:
        raise ValueError(f'The study direction "{direction}" is not supported')
    best_trial = find_best(
        completed_trials,
        key=lambda x: x.value,  # type: ignore
    )

    return completed_trials, best_trial


def _callback(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    exp: Path,
    n_trials: None | int,
    timer: delu.tools.Timer,
):
    # NOTE
    # The only purpose of this function is to create a checkpoint based on the content
    # of the study file. Thus, in the multi-process setup, it is not important in what
    # order workers call this function.
    worker_id = lib.parallel.get_worker_id()

    # All operations with the experiment directory must be performed under the lock.
    with lib.parallel.lock():
        checkpoint = lib.experiment.try_load_checkpoint(exp)
        if checkpoint is None:
            checkpoint = {}

        # `_summarize_study` must be the only place where the study storage is queried.
        completed_trials, best_trial = _summarize_study(
            study,
            max_n_trials=n_trials,
            previously_completed_trial_ids=checkpoint.get('completed_trial_ids'),
        )
        checkpoint['completed_trial_ids'] = [x.number for x in completed_trials]

        # The following check ensures the overall correctness of the implementation,
        # especially when resuming from a checkpoint in the multi-process setup.
        # Note that there is a small chance of identical trials, when the processes
        # is interrupted just after one of the workers commits its completed trial
        # but before this worker saves a checkpoint with its new random state.
        # If this happens frequently, it should be possible to solve this issue
        # by baking the number of restarts to the random seed:
        # `seed = base_seed + 1000 * n_restarts + 1000000 * worker_id`
        for t in completed_trials:
            if t.number != trial.number:
                assert t.params != trial.params, (
                    f'The trials {t.number} and {trial.number}'
                    f' have identical sampled parameters: {trial.params}'
                )
        # The trial is not needed beyond the above check.
        del trial

        report = lib.experiment.load_report(exp)
        report['best'] = best_trial.user_attrs['experiment']
        report['time'] = timer.elapsed()
        report['n_completed_trials'] = len(completed_trials)
        if worker_id is not None:
            report['worker_id'] = worker_id

        checkpoint['report'] = report
        checkpoint['timer'] = timer
        if worker_id is None:
            checkpoint['random_state'] = delu.random.get_state()
            checkpoint['sampler'] = study.sampler
        else:
            workers_checkpoint = checkpoint.setdefault('workers', {})
            workers_checkpoint[worker_id] = {
                # Saving worker states does not give any reproducibility.
                # It only makes it possible to resume the run with diverse
                # states across workers, and can potentially help with debugging.
                'sampler': study.sampler,
                'timer': timer,
                'random_state': delu.random.get_state(),
            }

        lib.experiment.dump_checkpoint(exp, checkpoint)
        lib.experiment.dump_report(exp, report)
        summary = lib.experiment.summarize(report)
        lib.experiment.dump_summary(exp, summary)
        if report['n_completed_trials'] != n_trials:
            print(lib.util.add_frame(summary))

        # The "study.sqlite-journal" file is a temporary file that can be frequently
        # created and removed by SQLite to handle intensive writing workloads, which
        # can be the case when the number of workers is high. The journal file
        # should not be a part of the backup.
        lib.experiment.backup(exp, ignore=shutil.ignore_patterns('*.sqlite-journal'))


class Config(TypedDict):
    seed: int
    function: str
    space: ConfigSpace
    n_trials: NotRequired[int]
    timeout: NotRequired[int]
    sampler: NotRequired[KWArgs]
    enqueued_trials: NotRequired[list[dict[str, Any]]]
    # NOTE
    #
    # `n_workers` defines the number of processes ("workers") that will be evaluating
    # Optuna trials in parallel, which can significantly accelerate the tuning.
    # Carefully read this comment, in particular the "Limitations" paragraph,
    # before using this option.
    #
    # **General notes**
    #
    # - Lightweight workloads (e.g. tuning a plain MLP) that underutilize GPU can
    #   benefit from `n_workers > 1` even when all workers are running on the same GPU.
    #   Heavy workloads that fully utilize GPU can still benefit from `n_workers > 1`
    #   by running each worker on a separate GPU (see the next point).
    #
    # - To use multiple GPUs, set the `CUDA_VISIBLE_DEVICES` environment variable
    #   accordingly. For example, when `n_workers=4` and the command is as follows:
    #   ```
    #   CUDA_VISIBLE_DEVICES="0,1" uv run bin/tune.py path/to/exp
    #   ```
    #   then two workers will run with `CUDA_VISIBLE_DEVICES="0"`,
    #   and two workers will run with `CUDA_VISIBLE_DEVICES="1"`.
    #
    # **Practical tips**
    #
    # - Ensure that the machine has at least `n_workers + 1` CPU cores.
    # - For a plain MLP on A100, `n_workers=3` is a good starting point,
    #   i.e. it is faster then `n_workers=2` and comparable with `n_workers=4`.
    # - The execution time scales well with the number of GPUs, i.e. using
    #   N GPUs instead of 1 will reduce the tuning time almost by the factor of N.
    #
    # **Limitations**
    #
    # - Setting `n_workers > 4` can hurt the tuning results performance. Currently,
    #   there is no single explanation for why this happens. One hypothesis is that
    #   the multi-process tuning trajectory "lags" behind the single-process one,
    #   and the lag increases with the number of workers.
    # - When `n_workers` is used, bitwise reproducibility is not possible anymore.
    # - When `n_workers` is used, interrupting the tuning and resuming it from a
    #   checkpoint affects the optimization trajectory. For example, in the edge case
    #   of interrupting and resuming the run after each trial, the tuning behaves
    #   as there is only one worker.
    # - The workers are not allowed to spawn new processes. That in particular means
    #   that `torch.utils.data.DataLoader(..., num_workers > 0)` will lead to an error.
    #
    # P.S. One more technical thing to keep in mind. In `optuna.samplers.TPESampler`,
    # the argument `n_startup_trials` defines the number of _completed and pruned_
    # trials before activating the TPE algorithm, i.e. running trials are _not_ taken
    # into account. For example, `n_workers=32` and `n_startup_trials=20` will lead to
    # 32 random trials followed by 19 more (51 in total) random trials before starting
    # the TPE phase (modulo edge cases when workers finish trials simultaneously).
    n_workers: NotRequired[int]

    # The following option is implemented only for completeness. At the moment of
    # writing, the codebase does not offer scripts that run on multiple GPUs.
    n_gpus_per_worker: NotRequired[int]


def _make_sampler(config: Config, worker_id: None | int) -> optuna.samplers.BaseSampler:
    sampler_config = config.get('sampler', {})
    sampler_config = sampler_config.copy()
    sampler_cls = getattr(optuna.samplers, sampler_config.pop('type', 'TPESampler'))
    return sampler_cls(seed=_get_seed(config['seed'], worker_id), **sampler_config)


def _enqueue_initial_trials(
    study: optuna.Study, trials: list[dict[str, Any]]
) -> None:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for trial in trials:
        if not isinstance(trial, dict) or not trial:
            raise ValueError('Each enqueued trial must be a non-empty dictionary')
        if not all(isinstance(key, str) and key for key in trial):
            raise ValueError('Enqueued-trial parameter names must be non-empty strings')
        for value in trial.values():
            if not isinstance(value, bool | int | float | str):
                raise ValueError('Enqueued-trial values must be scalar')
            if isinstance(value, int | float) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise ValueError('Enqueued-trial numeric values must be finite')
        signature = tuple(sorted(trial.items()))
        if signature in seen:
            raise ValueError(f'Duplicate enqueued trial: {trial}')
        seen.add(signature)
        normalized.append(dict(trial))

    for trial in normalized:
        study.enqueue_trial(trial, skip_if_exists=True)


def _worker_main(exp: Path, config: Config, study_storage_url: str) -> None:
    worker_id = lib.parallel.get_worker_id()

    delu.random.seed(_get_seed(config['seed'], worker_id))
    function = lib.util.import_(config['function'])
    n_trials = config.get('n_trials')
    n_remaining_trials = n_trials
    timeout = config.get('timeout')

    # All operations with the experiment directory must be performed under the lock.
    with lib.parallel.lock():
        study = optuna.load_study(
            study_name=None,
            storage=study_storage_url,
            sampler=_make_sampler(config, worker_id),
        )
        checkpoint = lib.experiment.try_load_checkpoint(exp)

    if checkpoint is None:
        timer = delu.tools.Timer()

    else:
        del checkpoint['completed_trial_ids']
        if n_remaining_trials is not None:
            n_remaining_trials -= checkpoint.pop('report')['n_completed_trials']

        # Each worker resumes with the latest saved timer.
        timer = checkpoint.pop('timer')
        if timeout is not None:
            timeout -= timer.elapsed()

        if worker_id is None:
            delu.random.set_state(checkpoint.pop('random_state'))
            study.sampler = checkpoint.pop('sampler')
        else:
            workers_checkpoint = checkpoint.pop('workers')
            if worker_id in workers_checkpoint:
                # Restoring worker states does not give any reproducibility.
                # This is done only to obtain constructive diverse starting points.
                study.sampler = workers_checkpoint[worker_id]['sampler']
                delu.random.set_state(workers_checkpoint[worker_id]['random_state'])

        assert not checkpoint, (
            f'Some checkpoint fields were not used, namely: {sorted(checkpoint)}'
        )

    callbacks: list[Any] = [partial(_callback, exp=exp, n_trials=n_trials, timer=timer)]
    if n_trials is not None:
        # `study.optimize` runs `n_trials` in the current process, i.e. _per worker_.
        # The following callback stops the study after achieving `n_trials` _globally_.
        # However, workers will commit their latest active trials anyway, thus usually
        # exceeding `n_trials`. This is taken into account in two places:
        # - The extra trials are removed in `_summarize_study`.
        # - `_filter_trials` is used in `main`.
        callbacks.append(optuna.study.MaxTrialsCallback(n_trials))

    # NOTE
    # The study optimization loop roughly looks as follows:
    #
    # 1. Create a trial. Mark its state as "running" and add to the study storage.
    # 2. Call the objective function. The objective function is allowed to modify
    #    the trial object, e.g. by setting user attributes.
    # 3. Finalize all the trial data in the study storage.
    # 4. Call the callbacks with a read-only view of the trial.
    #
    # In the _single-process_ setup, a list of trial statuses in the study storage
    # usually looks like this:
    #
    # [COMPLETE, COMPLETE, ..., COMPLETE, <RUNNING OR COMPLETE>]
    #  ID=0      ID=1           ID=N      ID=N+1
    #
    # The above means that the order of trial IDs (`trial.number`) follows the order
    # in which the trials are completed. By contrast, in the _multi-process_ setup,
    # the following is possible (note that the trial N+1 is already completed,
    # while the trial N is still running):
    #
    # [COMPLETE, COMPLETE, ..., RUNNING, COMPLETE, ...]
    #  ID=0      ID=1           ID=N     ID=N+1
    #
    # From the "algorithmic" perspective, the order of completion is more important
    # (e.g. for analyzing how the best score evolves during the optimization process).
    # This is why there is the `completed_trial_ids` list in the checkpoint storing
    # the trials IDs in the order of completion, which is used, for example,
    # in `_filter_trials`.
    timer.run()
    study.optimize(  # type: ignore
        partial(_objective, function=function, space=config['space'], timer=timer),
        n_trials=n_remaining_trials,
        timeout=timeout,
        callbacks=callbacks,
    )


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    assert exp.name == 'tuning'

    study_path = exp / 'study.sqlite'
    study_storage_url = _get_storage_url(study_path)
    study_kwargs: lib.types.KWArgs = {
        'study_name': lib.util.get_function_full_name(main),
        'direction': 'maximize',
    }
    n_completed_trials = 0

    n_workers = config.get('n_workers')
    n_gpus_per_worker = config.get('n_gpus_per_worker')
    if n_gpus_per_worker is not None:
        assert n_workers is not None, (
            'The `n_gpus_per_worker` option cannot be used'
            ' without the `n_workers` option'
        )

    checkpoint = lib.experiment.try_load_checkpoint(exp)
    if checkpoint is None:
        report = lib.experiment.create_report(main, add_gpu_info='n_workers' in config)
        if study_path.exists():
            study_path.unlink()
        # NOTE: the sampler will be set in `_worker_main`.
        study = optuna.create_study(storage=study_storage_url, **study_kwargs)
        _enqueue_initial_trials(study, config.get('enqueued_trials', []))
    else:
        report = checkpoint['report']
        n_completed_trials = checkpoint['report']['n_completed_trials']
        print(
            'Resuming from a checkpoint.'
            f' Completed {n_completed_trials}/{config.get("n_trials", "inf")} trials'
        )
        # Remove incomplete trials from the study.
        _filter_trials(
            study_path,
            keep_trial_ids=checkpoint['completed_trial_ids'],
            **study_kwargs,
        )
        report.setdefault('resumed_after_n_trials', []).append(n_completed_trials)

    lib.experiment.dump_report(exp, report)
    del report

    if n_workers is None:
        _worker_main(exp, config, study_storage_url)
    else:
        n_trials = config.get('n_trials')
        if n_trials is not None:
            n_remaining_trials = n_trials - n_completed_trials
            if n_workers > n_remaining_trials:
                logger.info(
                    f'Reducing the number of workers'
                    f' from {n_workers=} to {n_remaining_trials=}'
                )
                n_workers = n_remaining_trials
        logger.info(f'The number of workers: {n_workers}')

        lib.parallel.map(
            _worker_main,
            [
                {
                    'exp': exp,
                    'config': config,
                    'study_storage_url': study_storage_url,
                }
                for _ in range(n_workers)
            ],
            n_workers=n_workers,
            n_gpus_per_worker=1 if n_gpus_per_worker is None else n_gpus_per_worker,
        )

    # In the multi-process setup, the total number of trials usually exceeds
    # config['n_trials']. The following function call prunes the extra trials.
    _filter_trials(
        study_path,
        keep_trial_ids=lib.experiment.load_checkpoint(exp)['completed_trial_ids'],
        **study_kwargs,
    )

    report = lib.experiment.load_report(exp)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main, resumable=True)
