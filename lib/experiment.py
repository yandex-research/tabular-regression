import argparse
import dataclasses
import datetime
import inspect
import json
import shutil
import statistics
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from pprint import pprint
from typing import Any, Protocol

import tomli_w
from loguru import logger

from . import env, util
from .types import JSONDict

type Report = JSONDict
type Summary = str

_CONFIG_FILE_NAME = 'config.toml'
_INPUT_FILE_NAMES = frozenset([_CONFIG_FILE_NAME])


def _check_exp(exp: str | Path) -> Path:
    exp = Path(exp)
    if not exp.is_dir():
        raise RuntimeError(
            f'The experiment does not exist: {exp}'
            if not exp.exists()
            else f'The experiment path is not a directory: {exp}'
            f'{f". Try passing {exp.parent} instead" if exp.name == "config.toml" else ""}'  # noqa: E501
        )
    return exp


# ======================================================================================
# IO
# ======================================================================================
def get_config_path(exp: str | Path) -> Path:
    return Path(exp).resolve() / _CONFIG_FILE_NAME


def get_report_path(exp: str | Path) -> Path:
    return Path(exp).resolve() / 'report.json'


def get_summary_path(exp: str | Path) -> Path:
    return Path(exp).resolve() / 'summary.txt'


def load_config(exp: str | Path) -> JSONDict:
    exp = _check_exp(exp)
    with open(get_config_path(exp), 'rb') as f:
        return tomllib.load(f)


def load_report(exp: str | Path) -> Report:
    exp = _check_exp(exp)
    with open(get_report_path(exp)) as f:
        return json.load(f)


def load_summary(exp: str | Path) -> Summary:
    exp = _check_exp(exp)
    return get_summary_path(exp).read_text()


def dump_config(exp: str | Path, config: JSONDict) -> None:
    exp = _check_exp(exp)
    with open(get_config_path(exp), 'wb') as f:
        tomli_w.dump(config, f)


def dump_report(exp: str | Path, report: Report) -> None:
    exp = _check_exp(exp)
    with open(get_report_path(exp), 'w') as f:
        json.dump(report, f, indent=4)


def dump_summary(exp: str | Path, summary: Summary) -> None:
    exp = _check_exp(exp)
    get_summary_path(exp).write_text(summary)


# ======================================================================================
# Status
# ======================================================================================
def is_experiment(exp: str | Path) -> bool:
    return get_config_path(exp).exists()


def _get_running_path(exp: str | Path) -> Path:
    return Path(exp).resolve().joinpath('_RUNNING')


def is_fresh(exp: str | Path) -> bool:
    exp = _check_exp(exp)
    return frozenset(x.name for x in Path(exp).iterdir()) == _INPUT_FILE_NAMES


def _is_running(exp: str | Path) -> bool:
    exp = _check_exp(exp)
    return _get_running_path(exp).exists()


def is_done(exp: str | Path) -> bool:
    exp = _check_exp(exp)
    return not _is_running(exp) and get_report_path(exp).exists()


# ======================================================================================
# Actions
# ======================================================================================
def create(
    exp: str | Path,
    *,
    config: None | JSONDict = None,
    parents: bool = False,
    force: bool = False,
) -> Path:
    exp = Path(exp).resolve()

    if exp.exists():
        if force:
            shutil.rmtree(exp)
        else:
            raise RuntimeError(f'The experiment already exists: {exp}')

    if parents:
        exp.parent.mkdir(parents=True, exist_ok=True)

    # The final rename must stay on the experiment's filesystem.
    with tempfile.TemporaryDirectory(dir=exp.parent) as tmp_exp:
        tmp_exp = Path(tmp_exp)
        if config is not None:
            dump_config(tmp_exp, config)
        move(tmp_exp, exp)

    return exp


def _run(exp: str | Path) -> None:
    exp = _check_exp(exp)
    _get_running_path(exp).touch()


def _stop_running(exp: str | Path) -> None:
    exp = _check_exp(exp)
    if is_fresh(exp):
        raise RuntimeError('Cannot stop a fresh experiment')
    if is_done(exp):
        raise RuntimeError('Cannot stop an experiment that is already done')
    _get_running_path(exp).unlink()


def finish(exp: str | Path, report: Report) -> None:
    exp = _check_exp(exp)

    dump_report(exp, report)
    dump_summary(exp, summarize(report))
    _stop_running(exp)

    print()
    print(util.add_frame(load_summary(exp)))
    backup(exp)


def reset(exp: str | Path) -> None:
    exp = _check_exp(exp)
    for path in Path(exp).iterdir():
        if path.name not in _INPUT_FILE_NAMES:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# ======================================================================================
# Experiment-level IO
# ======================================================================================
def copy(src: str | Path, dst: str | Path, **copytree_kwargs) -> None:
    # NOTE[exp]
    src = _check_exp(src)
    shutil.copytree(src, dst, **copytree_kwargs)


def move(src: str | Path, dst: str | Path) -> None:
    # NOTE[exp]
    src = _check_exp(src)
    Path(src).rename(dst)


def remove(exp: str | Path) -> None:
    # NOTE[exp]
    exp = _check_exp(exp)
    shutil.rmtree(exp)


def remove_tracked_files(exp: str | Path) -> None:
    """Remove files that are tracked by a VCS."""
    get_config_path(exp).unlink(missing_ok=True)
    get_report_path(exp).unlink(missing_ok=True)


def duplicate(
    src: str | Path,
    dst_project_dir: str | Path,
    *,
    force: bool = False,
    **copytree_kwargs,
) -> None:
    """Copy the experiment to another project."""
    # NOTE[exp]
    src = _check_exp(src)
    relative_path = src.relative_to(env.get_project_dir())
    dst = Path(dst_project_dir).joinpath(relative_path).resolve()
    if dst.exists() and not force:
        raise FileExistsError(f'The experiment already exists: "{dst}"')

    with tempfile.TemporaryDirectory() as tmp:
        # Copy the experiment before removing the existing one.
        tmp = Path(tmp).resolve() / dst.name
        copy(src, tmp, **copytree_kwargs)
        if dst.exists():
            assert force  # Already checked above.
            remove(dst)
        else:
            dst.parent.mkdir(exist_ok=True, parents=True)
        move(tmp, dst)


# ======================================================================================
# MainFunction
# ======================================================================================
class MainFunction[T](Protocol):
    def __call__(self, config: T, exp: str | Path) -> Report: ...


def _check_config_dataclass(cls, key: tuple[str, ...]):
    if not cls.__dataclass_params__.frozen:
        raise ValueError(
            'The config dataclass (and its nested dataclasses) must be defined with'
            ' frozen=True'
        )
    if not cls.__dataclass_params__.kw_only:
        raise ValueError(
            'The config dataclass (and its nested dataclasses) must be defined with'
            ' kw_only=True'
        )
    for field in dataclasses.fields(cls):
        if field.default not in (dataclasses.MISSING, None):
            raise ValueError(
                'Only None can be used as the default field value in config'
                ' dataclasses and their nested dataclasses.'
                f' However, the field {".".join((*key, field.name))}'
                f' has the default value {field.default}'
            )
        if dataclasses.is_dataclass(field.type):
            _check_config_dataclass(field.type, (*key, field.name))


def relabel_report(function: MainFunction, exp: str | Path, report: Report) -> Report:
    """Attribute a shared trainer's result to its public objective entrypoint."""
    report['function'] = util.get_function_full_name(function)
    dump_report(exp, report)
    return report


def create_report(function: MainFunction, *, add_gpu_info: bool) -> Report:
    report: Report = {'function': util.get_function_full_name(function)}

    if add_gpu_info:
        import torch

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            first_gpu_name = gpu_names[0]
            assert all(x == first_gpu_name for x in gpu_names)
            report['gpu'] = first_gpu_name
            if n_gpus > 1:
                report['n_gpus'] = n_gpus

    return report


def _get_config[T](config_type: type[T], config: None | T, exp: str | Path) -> T:
    config_path = get_config_path(exp)
    config_file_exists = config_path.exists()
    config_type_is_dataclass = dataclasses.is_dataclass(config_type)

    if config is not None and config_file_exists:
        raise ValueError(
            'The config is provided both as the `config` argument and as the text file'
            f' "{config_path}" simultaneously, which is not allowed.'
        )
    elif config is None and not config_file_exists:
        raise ValueError(
            'The config is not provided either as the `config` argument'
            f' or as the text file {config_path}'
        )
    if config_type_is_dataclass:
        _check_config_dataclass(config_type, ())

    if config is None:
        loaded_config = load_config(exp)
        if util.is_typed_dict(config_type):
            config = util.check_typed_dict(config_type, loaded_config)
        elif config_type_is_dataclass:
            config = util.dataclass_from_dict(config_type, loaded_config)
        else:
            raise ValueError(
                'When `config=None` config_type must be either TypedDict or a dataclass'
            )

    return config


def run[T](
    function: MainFunction[T],
    config: None | T,
    exp: str | Path,
    *,
    force: bool = False,
    resume: bool = False,
) -> None | Report:
    try:
        config_type = inspect.signature(function).parameters['config'].annotation
        assert config_type is not inspect._empty
    except Exception as err:
        raise RuntimeError(
            'Failed to infer the config type for the provided function.'
            ' The function must have the "config" argument with a type annotation'
        ) from err

    exp = _check_exp(exp).resolve()

    function_full_name = (
        None if function is None else util.get_function_full_name(function)
    )
    util.print_sep()
    print(
        f'{"" if function_full_name is None else f"{function_full_name} | "}'
        f'{util.try_get_relative_path(exp)}'
        f' | {datetime.datetime.now()}'
    )
    util.print_sep()

    # NOTE
    # The config is set up only when `function` will actually be run,
    # and before _run().
    def get_config():
        return _get_config(config_type, config, exp)

    if is_fresh(exp):
        config = get_config()
        logger.info('Running the experiment')
        _run(exp)

    else:
        if force:
            config = get_config()
            logger.warning('Resetting the experiment')
            reset(exp)
            logger.info('Running the experiment')
            _run(exp)

        elif resume:
            if is_done(exp):
                add_to_tmp_output(exp)
                logger.info('The experiment is already done')
                return None
            else:
                config = get_config()
                logger.info('Resuming the experiment')

        else:
            if is_done(exp):
                add_to_tmp_output(exp)
            logger.warning('The experiment already exists')
            return None

    assert config is not None
    if util.is_typed_dict(config_type):
        print('\nConfig')
        pprint(config, sort_dicts=False)
    else:
        print(f'\n{config}')

    return function(config, exp)


def run_cli[T](function: MainFunction[T], *, resumable: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', metavar='PATH')
    parser.add_argument('--force', action='store_true')
    if resumable:
        parser.add_argument('--resume', action='store_true')

    run(function, None, **vars(parser.parse_args(sys.argv[1:])))


# ======================================================================================
# Summary
# ======================================================================================
def _summarize_score(score: float) -> float:
    return round(float(score), 4)


def _summarize_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def _summarize_report(report: Report) -> JSONDict:
    summary = {}
    for key, value in report.items():
        if (key == 'time' or key.endswith('_time')) and isinstance(value, float):
            summary[key] = _summarize_time(value)

        elif key == 'gpu':
            n_gpus = report.get('n_gpus')
            summary['gpus'] = value.removeprefix('NVIDIA ') + (
                '' if n_gpus is None else f' x{n_gpus}'
            )

        elif key == 'best_step':
            summary[key] = value
            epoch_size = report.get('epoch_size')
            if epoch_size is not None:
                summary['best_epoch'] = int(value // epoch_size)

        elif key == 'metrics':
            summary['metrics'] = {
                part: {'score': _summarize_score(part_metrics['score'])}
                for part, part_metrics in value.items()
            }

        elif key == 'best':
            summary['best'] = _summarize_report(value['report'])

        elif key == 'experiments':
            summary['n_experiments'] = len(value)
            reports = [x['report'] for x in value]
            if reports and 'time' in reports[0]:
                summary['mean_time'] = _summarize_time(
                    statistics.mean(x['time'] for x in reports)
                )
            if reports and 'metrics' in reports[0]:
                summary['metrics'] = {}
                for part in reports[0]['metrics']:
                    part_scores = [x['metrics'][part]['score'] for x in reports]
                    part_score_mean = statistics.mean(part_scores)
                    part_score_std = (
                        statistics.stdev(part_scores) if len(part_scores) >= 2 else 0.0
                    )
                    summary['metrics'][part] = {
                        'score': (
                            f'{_summarize_score(part_score_mean)}'
                            f' +- {_summarize_score(part_score_std)}'
                        )
                    }
                    del part

        elif key in ('n_gpus',):
            continue

        else:
            summary[key] = value

    first = {x: summary.pop(x) for x in ('function', 'time', 'gpu') if x in summary}
    last = {x: summary.pop(x) for x in ('metrics', 'best') if x in summary}
    return {**first, **summary, **last}


def _dict_to_lines(dict_: JSONDict, current_indent: int, lines: list[str]) -> None:
    for key, value in dict_.items():
        prefix = f'{" " * current_indent}{key}:'
        if isinstance(value, dict):
            lines.append(prefix)
            _dict_to_lines(value, current_indent + 2, lines)
        else:
            lines.append(f'{prefix} {value}')


def summarize(report: Report) -> Summary:
    lines = []
    _dict_to_lines(_summarize_report(report), 0, lines)
    lines.append('')
    return '\n'.join(lines)


# ======================================================================================
def backup(exp: str | Path, **copytree_kwargs) -> None:
    """Compatibility no-op for the removed internal artifact infrastructure."""


# =================================================================================
# NOTE[tabular]
# =================================================================================
def get_checkpoint_path(exp: str | Path) -> Path:
    return Path(exp) / 'checkpoint.pt'


def get_predictions_path(exp: str | Path) -> Path:
    return Path(exp) / 'predictions.npz'


def load_checkpoint(exp: str | Path) -> dict[str, Any]:
    import torch

    exp = _check_exp(exp)
    return torch.load(get_checkpoint_path(exp), weights_only=False)


def try_load_checkpoint(exp: Path) -> None | dict[str, Any]:
    return load_checkpoint(exp) if get_checkpoint_path(exp).exists() else None


def load_predictions(exp: str | Path):  #  -> dict[PartKey, np.ndarray]
    import numpy as np

    exp = _check_exp(exp)
    x = np.load(get_predictions_path(exp))
    return {key: x[key] for key in x}


def dump_checkpoint(exp: str | Path, value: Any) -> None:
    import torch

    exp = _check_exp(exp)
    return torch.save(value, get_checkpoint_path(exp))


def dump_predictions(
    exp: str | Path,
    predictions,  # dict[PartKey, np.ndarray]
) -> None:
    import numpy as np

    exp = _check_exp(exp)
    np.savez(get_predictions_path(exp), **predictions)  # type: ignore[arg-type]
