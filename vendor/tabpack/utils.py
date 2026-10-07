# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import time
import typing
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np


def get_path_from_config(exp: str | Path, path: str | Path) -> Path:
    return (
        Path(exp) / path
        if isinstance(path, str) and path.startswith(('./', '../'))
        else Path(path)
    )


def time_now() -> float:
    """Get the current time."""
    return time.perf_counter()


@typing.overload
def time_elapsed_since(timepoint: float) -> float: ...


@typing.overload
def time_elapsed_since(timepoint: np.ndarray) -> np.ndarray: ...


def time_elapsed_since(timepoint):
    """Compute the time elapsed since a given time point.

    NOTE
    The provided timepoint must be produced by `time_now`.
    """
    return time_now() - timepoint


def unflatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    d_new = {}
    for key, value in d.items():
        d_dst = d_new
        key_parts = key.split('.')
        for key_part in key_parts[:-1]:
            d_dst = d_dst.setdefault(key_part, {})
        d_dst[key_parts[-1]] = value
    return d_new


def dict_merge_recursively[T: MutableMapping](this: T, other: T) -> None:
    """Merge two dictionaries recursively.

    Here, "merging" means "updating", but overlapping keys are forbidden.
    """
    for other_key, other_value in other.items():
        if other_key in this:
            assert isinstance(other_value, dict)
            value = this[other_key]
            assert isinstance(value, dict)
            dict_merge_recursively(value, other_value)
        else:
            this[other_key] = other_value


def transpose_list_of_dicts(value: list[dict[Any, Any]]) -> dict[Any, list]:
    assert value
    first = value[0]
    keys = frozenset(first)
    assert all(frozenset(x.keys()) == keys for x in value), (
        'All dictionaries must have the same set of keys'
    )
    return {key: [x[key] for x in value] for key in first.keys()}


def to_numpy(data):
    from torch import Tensor

    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, bool | int | float):
        return np.array(data)
    elif isinstance(data, Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return type(data)(to_numpy(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, to_numpy(v)) for k, v in data.items())
    else:
        raise ValueError(f'Cannot convert an instance of "{type(data)}" to NumPy')


def list_index(data, index):
    if isinstance(data, list):
        return data[index]  # type: ignore
    elif isinstance(data, dict):
        return type(data)((k, list_index(v, index)) for k, v in data.items())
    else:
        raise ValueError(f'Unsupported value type: {type(data)}')


def numpy_map(data, fn: Callable[[np.ndarray], Any]) -> Any:
    if isinstance(data, np.ndarray):
        return fn(data)  # type: ignore
    elif isinstance(data, list):
        return type(data)(numpy_map(x, fn) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, numpy_map(v, fn)) for k, v in data.items())
    else:
        raise ValueError(f'Unsupported value type: {type(data)}')


def numpy_index[T](data: T, index) -> T:
    return numpy_map(data, lambda x: x[index])


def numpy_stack[T](data: list[T]) -> T:
    if not data:
        return np.array([])  # type: ignore

    first = data[0]
    if isinstance(first, np.ndarray):
        return np.stack(data)  # type: ignore
    elif isinstance(first, list):
        return type(first)(numpy_stack([x[i] for x in data]) for i in range(len(first)))  # type: ignore
    elif isinstance(first, dict):
        return {k: numpy_stack([x[k] for x in data]) for k in first.keys()}  # type: ignore
    else:
        raise ValueError(f'Unsupported value type: {type(first)}')


def numpy_concatenate[T](data: list[T]) -> T:
    if not data:
        return np.array([])  # type: ignore

    first = data[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(data)  # type: ignore
    elif isinstance(first, list):
        return type(first)(
            numpy_concatenate([x[i] for x in data])  # type: ignore
            for i in range(len(first))
        )
    elif isinstance(first, dict):
        return {k: numpy_concatenate([x[k] for x in data]) for k in first.keys()}  # type: ignore
    else:
        raise ValueError(f'Unsupported value type: {type(first)}')
