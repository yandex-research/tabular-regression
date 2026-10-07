import dataclasses
import functools
import importlib
import inspect
import os
import subprocess
import sys
import types
import typing
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NotRequired, Required

import delu

# NOTE
# This file must NOT import anything from lib except for `env`,
# because all other submodules are allowed to import `util`.
from . import env
from .types import AMPDType

try:
    _TERMINAL_SIZE = os.get_terminal_size().columns
except OSError:
    # Jupyter
    _TERMINAL_SIZE = 80
_SEPARATOR = '─' * _TERMINAL_SIZE


def print_sep():
    print(_SEPARATOR)


def add_frame(text: str) -> str:
    lines = text.splitlines()
    width = max(map(len, lines))
    hline = '─' * (width + 2)
    return '\n'.join(
        [
            f'╭{hline}╮',
            *(f'│ {line}{" " * (width - len(line))} │' for line in lines),
            f'╰{hline}╯',
        ]
    )


def try_get_relative_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    project_dir = env.get_project_dir()
    return path.relative_to(project_dir) if project_dir in path.parents else path


def is_typed_dict(type_) -> bool:
    if not isinstance(type_, type):
        return False
    return (
        issubclass(type_, dict)
        and hasattr(type_, '__required_keys__')
        and hasattr(type_, '__optional_keys__')
        and hasattr(type_, '__annotations__')
    )


def check_typed_dict[T](type_: type[T], dictionary: dict) -> T:
    if not is_typed_dict(type_):
        raise ValueError('type_ must be inherited from `typing.TypedDict`')

    presented_keys = frozenset(dictionary)
    required_keys = type_.__required_keys__  # type: ignore
    optional_keys = type_.__optional_keys__  # type: ignore
    assert presented_keys >= required_keys, (
        'The following required keys are missing:'
        f' {", ".join(required_keys - presented_keys)}'
    )
    assert set(dictionary) <= (required_keys | optional_keys), (
        'The following keys are unknown:'
        f' {", ".join(presented_keys - required_keys - optional_keys)}'
    )

    for key, value in dictionary.items():
        annotation = type_.__annotations__[key]
        if typing.get_origin(annotation) in (NotRequired, Required):
            annotation = typing.get_args(annotation)[0]
        if isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        if typing.get_origin(type_) is not Literal and is_typed_dict(annotation):
            check_typed_dict(annotation, value)

    return typing.cast(T, dictionary)


def _make_parse_object_error(key, reason):
    return ValueError(
        f'Failed to parse the object.'
        f'{f' The problematic field: "{".".join(key)}".' if key else ""}'
        f'{"" if reason is None else f" Reason: {reason}"}'
    )


_TYPE_MISMATCH_REASON = (
    ' The expected type is {expected}, but the actual type is {actual}'
)


def _parse_object[T](
    type_: typing.TypeAliasType | type[T], obj: Any, key: tuple[str, ...]
) -> T:
    # NOTE
    # (1) The parsing is strict.
    #     In particular, types are checked exactly, i.e. without `isinstance`.
    #     For example, `_parse_object(int, True, '...')` will result in an error.
    # (2) The parsing is limited. Only basic types and containers are supported.
    while isinstance(type_, typing.TypeAliasType):
        type_ = type_.__value__

    if type_ is Any:
        return obj

    obj_type = type(obj)

    if type_ in (types.NoneType, types.EllipsisType, bool, int, float, str, bytes):
        if obj_type is not type_:
            raise _make_parse_object_error(
                key, _TYPE_MISMATCH_REASON.format(expected=type_, actual=obj_type)
            )
        return obj

    elif dataclasses.is_dataclass(type_):
        if obj_type is not dict:
            raise _make_parse_object_error(
                key, _TYPE_MISMATCH_REASON.format(expected=dict, actual=obj_type)
            )

        fields: dict[str, dataclasses.Field] = {}
        for field in dataclasses.fields(type_):
            if not field.init:
                raise _make_parse_object_error(
                    key, 'Dataclasses with non-init fields are not supported'
                )
            if field.default is dataclasses.MISSING and field.name not in obj:
                raise _make_parse_object_error(
                    key,
                    f'The value for the required dataclass field "{field.name}"'
                    ' is missing',
                )
            fields[field.name] = field

        kwargs = {}
        for field_key, field_value in obj.items():
            field = fields.get(field_key)
            if field is None:
                raise _make_parse_object_error(
                    key, f'The dataclass does not have the field "{field_key}"'
                )
            kwargs[field_key] = _parse_object(
                field.type,  # type: ignore
                field_value,
                (*key, field_key),
            )

        try:
            return type_(**kwargs)
        except Exception as err:
            raise _make_parse_object_error(key, None) from err

    else:
        type_origin = typing.get_origin(type_)
        type_args = typing.get_args(type_)

        if type_origin is tuple:
            if obj_type is not type_origin:
                raise _make_parse_object_error(
                    key, _TYPE_MISMATCH_REASON.format(expected=type_, actual=obj_type)
                )
            if len(obj) != len(type_args):
                raise _make_parse_object_error(
                    key,
                    f'The expected tuple size is {len(type_args)},'
                    f' but the actual is {len(obj)}',
                )
            return obj_type(  # type: ignore
                _parse_object(type_args[i], x, (*key, str(i)))
                for i, x in enumerate(obj)
            )

        if type_origin is list:
            if obj_type is not type_origin:
                raise _make_parse_object_error(
                    key, _TYPE_MISMATCH_REASON.format(expected=type_, actual=obj_type)
                )
            return obj_type(  # type: ignore
                _parse_object(type_args[0], x, (*key, str(i)))
                for i, x in enumerate(obj)
            )

        elif type_origin is dict:
            if obj_type is not type_origin:
                raise _make_parse_object_error(
                    key, _TYPE_MISMATCH_REASON.format(expected=type_, actual=obj_type)
                )
            return obj_type(
                (  # type: ignore
                    _parse_object(type_args[0], k, (*key, k, '<key>')),
                    _parse_object(type_args[1], v, (*key, k, '<value>')),
                )
                for k, v in obj.items()
            )

        elif type_origin in (
            types.UnionType,  # T1 | T2 | ...
            typing.Union,  # typing.Optional[T], typing.Union[T1, T2, ...]
        ):
            for type_candidate in type_args:
                try:
                    return _parse_object(type_candidate, obj, key)
                except Exception:
                    pass
            else:
                raise _make_parse_object_error(
                    key,
                    f'The value {obj} does not match any of the type variants:'
                    f' {type_args}',
                )

        else:
            raise _make_parse_object_error(
                key, f' unsupported type origin {type_origin} for the key "{key}"'
            )


def dataclass_from_dict[T](datacls: type[T], dict_: dict[str, Any]) -> T:
    assert dataclasses.is_dataclass(datacls), 'The first argument must be a dataclass.'

    return _parse_object(datacls, dict_, ())


def _flatten_dict(d: dict, key_prefix: str, result: dict) -> None:
    for k, v in d.items():
        new_k = f'{key_prefix}.{k}' if key_prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, new_k, result)
        else:
            if result.setdefault(new_k, v) is not v:
                RuntimeError(
                    'Different parts of the dictionary resulted'
                    f' in the same flat key "{new_k}"'
                )


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    flat_d: dict[str, Any] = {}
    _flatten_dict(d, '', flat_d)
    return flat_d


def import_(fullname: str) -> Any:
    """
    Examples:

    >>> import_('bin.demo.main')
    """
    if fullname.count('.') == 0:
        raise ValueError('qualname must contain at least one dot')
    module_name, attr = fullname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def get_function_full_name(function: Callable) -> str:
    """
    Examples:

    >>> # In the script bin/model.py
    >>> get_function_full_name(main) == 'bin.model.main'

    >>> # In the script a/b/c/foo.py
    >>> assert get_function_full_name(main) == 'a.b.c.foo.main'
    """
    module = inspect.getmodule(function)
    assert module is not None, 'Failed to locate the module of the function.'

    module_path = getattr(module, '__file__', None)
    assert module_path is not None, (
        'Failed to locate the module of the function.'
        ' This can happen if the code is running in a Jupyter notebook.'
    )

    module_path = Path(module_path).resolve()
    project_dir = env.get_project_dir()
    assert project_dir in module_path.parents, (
        'The module of the function must be located within the project directory: '
        f' {project_dir}'
    )

    module_full_name = str(
        module_path.relative_to(project_dir).with_suffix('')
    ).replace('/', '.')
    return f'{module_full_name}.{function.__name__}'


def get_device():  # -> torch.device
    import torch

    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        # else 'mps:0'
        # if torch.mps.is_available()
        else 'cpu'
    )


def get_amp_dtype(
    dtype: AMPDType,
    device,  # torch.device
):  # -> torch.dtype
    import torch

    if dtype == 'bfloat16':
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f' The current {device.type.upper()} device'
                f' does not support {dtype} as the AMP data type'
            )
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16
    else:
        raise ValueError(f'Unknown {dtype=}')


def is_oom_exception(err: RuntimeError) -> bool:
    import torch

    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def adjust_gpu_memory_usage[**P, T](
    memory_parameter: str,
) -> Callable[[Callable[P, T]], Callable[P, tuple[T, int]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, tuple[T, int]]:
        p = inspect.signature(f).parameters.get(memory_parameter)
        if p is None or p.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f'The function must have the keyword-only argument "{memory_parameter}"'
            )
        del p

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, int]:
            value: int = kwargs[memory_parameter]  # type: ignore
            if value <= 0:
                raise ValueError(f'{memory_parameter} must be a positive integer')
            while value:
                kwargs[memory_parameter] = value
                try:
                    return f(*args, **kwargs), value
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    delu.cuda.free_memory()
                    new_value = value // 2
                    message = (
                        f'Calling the function `{f.__name__}`'
                        f' with {memory_parameter}={value} triggers GPU OOM'
                    )
                    if new_value:
                        message += f'. Retrying with {memory_parameter}={new_value}'
                    import loguru

                    loguru.logger.warning(message)
                    value = new_value
            raise RuntimeError(f'Not enough memory even for {memory_parameter}=1')

        return wrapper

    return decorator


# NOTE: the following function should *not* cache its results.
def git_get_current_branch() -> str:
    return (
        subprocess.run(
            ['git', 'branch', '--show-current'], capture_output=True, check=True
        )
        .stdout.decode('utf-8')
        .strip()
    )


def configure_logging():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


def configure_torch():
    # NOTE[tabular]
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init(*, torch_: bool = True) -> None:
    if Path.cwd() != env.get_project_dir():
        raise RuntimeError('The code must be run from the project root')
    configure_logging()
    if torch_:
        import torch

        if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            warnings.warn(
                'When CUDA is available, CUDA_VISIBLE_DEVICES should be set explicitly'
            )
        configure_torch()


_IS_NOTEBOOK = 'ipykernel_launcher' in sys.argv[0]


def is_notebook() -> bool:
    return _IS_NOTEBOOK


def init_notebook(*, torch_: bool = True) -> None:
    assert is_notebook()
    os.chdir(env.get_project_dir())
    init(torch_=torch_)


# ==================================================================================
# NOTE[tabular]
# ==================================================================================


WORST_SCORE = -999999.0


def are_valid_predictions(predictions: dict) -> bool:
    # predictions: dict[PartKey, np.ndarray]
    import numpy as np

    assert all(isinstance(x, np.ndarray) for x in predictions.values())
    return all(np.isfinite(x).all() for x in predictions.values())
