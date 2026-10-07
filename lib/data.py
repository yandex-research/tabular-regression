# NOTE[tabular]
import enum
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from . import env
from .metrics import calculate_metrics as calculate_metrics_
from .types import DataKey, PartKey, PredictionType, TaskType


class Score(enum.Enum):
    ACCURACY = 'accuracy'
    CROSS_ENTROPY = 'cross-entropy'
    MAE = 'mae'
    R2 = 'r2'
    RMSE = 'rmse'
    ROC_AUC = 'roc-auc'


_SCORE_HIGHER_IS_BETTER = {
    Score.ACCURACY: True,
    Score.CROSS_ENTROPY: False,
    Score.MAE: False,
    Score.R2: True,
    Score.RMSE: False,
    Score.ROC_AUC: True,
}


@dataclass(frozen=True)
class Task:
    labels: dict[PartKey, np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls, path: str | Path) -> 'Task':
        path = Path(path).resolve()
        info = json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType(info['task_type'])
        score = info.get('score')
        if score is None:
            score = {
                TaskType.BINCLASS: Score.ACCURACY,
                TaskType.MULTICLASS: Score.ACCURACY,
                TaskType.REGRESSION: Score.RMSE,
            }[task_type]
        else:
            score = Score(score)
        return Task(
            {
                part: np.load(path / f'Y_{part}.npy')
                for part in ['train', 'val', 'test']
            },
            task_type,
            score,
        )

    def __post_init__(self):
        assert isinstance(self.type_, TaskType)
        assert isinstance(self.score, Score)
        if self.is_regression:
            assert all(
                value.dtype in (np.dtype('float32'), np.dtype('float64'))
                for value in self.labels.values()
            ), 'Regression labels must have dtype=float32'
            for key in self.labels:
                self.labels[key] = self.labels[key].astype('float32')

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    def compute_n_classes(self) -> int:
        assert self.is_binclass or self.is_classification
        return len(np.unique(self.labels['train']))

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        metrics = {
            part: calculate_metrics_(
                y_true=self.labels[part],
                y_pred=predictions[part],
                task_type=self.type_,
                prediction_type=prediction_type,
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_HIGHER_IS_BETTER[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics  # type: ignore


def load_data(path: str | Path) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    path = Path(path).resolve()
    return {  # type: ignore
        key.lower(): {
            part: np.load(path / f'{key}_{part}.npy', allow_pickle=True)
            for part in ['train', 'val', 'test']
        }
        for key in ['X_num', 'X_bin', 'X_cat', 'Y']
        if path.joinpath(f'{key}_train.npy').exists()
    }


@dataclass
class Dataset[T: np.ndarray | Tensor]:
    """Dataset = Data + Task + simple methods for convenience.

    The task is stored separately to ensure that the original labels never change.
    """

    data: dict[DataKey, dict[PartKey, T]]
    task: Task

    @classmethod
    def from_dir(cls, path: str | Path) -> 'Dataset[np.ndarray]':
        return Dataset(load_data(path), Task.from_dir(path))

    def __post_init__(self):
        """Validate the data."""
        data = self.data
        is_numpy = self._is_numpy()

        # >>> Check data types.
        for key, allowed_dtypes in {
            'x_num': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_bin': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_cat': [] if is_numpy else [torch.int64],
            'y': (
                [np.dtype('float32'), np.dtype('float64'), np.dtype('int64')]
                if is_numpy
                else [torch.float32, torch.int64]
            ),
        }.items():
            if key in data:
                for part, value in data[key].items():
                    if key == 'x_cat' and is_numpy:
                        assert value.dtype in (
                            np.dtype('int32'),
                            np.dtype('int64'),
                        ) or isinstance(
                            value.dtype,
                            np.dtypes.StrDType,  # type: ignore
                        )
                    else:
                        assert value.dtype in allowed_dtypes, (
                            f'The value data["{key}"]["{part}"] has dtype'
                            f' {value.dtype}, but it must be one of {allowed_dtypes}'
                        )

        # >>> Fix data types.
        if self.task.is_regression:
            for key in data['y']:
                data['y'][key] = (  # type: ignore
                    data['y'][key].astype('float32')  # type: ignore
                    if self._is_numpy()
                    else data['y'][key].to(torch.float32)  # type: ignore
                )
        if 'x_cat' in data and data['x_cat']['train'].dtype == np.dtype('int32'):
            for key in data['x_cat']:
                data['x_cat'][key] = data['x_cat'][key].astype('int64')  # type: ignore

        # >>> Check nans.
        isnan = np.isnan if is_numpy else torch.isnan
        for key in ['x_bin', 'y']:
            if key in data:
                for part, value in data[key].items():
                    assert not isnan(value).any(), (  # type: ignore
                        f'data["{key}"]["{part}"] contains nans'
                    )

    def _is_numpy(self) -> bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def to_torch(self, device: None | str | torch.device) -> 'Dataset[Tensor]':
        return Dataset(
            {
                key: {
                    part: torch.as_tensor(value).to(device)
                    for part, value in self.data[key].items()
                }
                for key in self.data
            },
            self.task,
        )

    @property
    def n_num_features(self) -> int:
        return self.data['x_num']['train'].shape[1] if 'x_num' in self.data else 0

    @property
    def n_bin_features(self) -> int:
        return self.data['x_bin']['train'].shape[1] if 'x_bin' in self.data else 0

    @property
    def n_cat_features(self) -> int:
        return self.data['x_cat']['train'].shape[1] if 'x_cat' in self.data else 0

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def size(self, part: None | PartKey) -> int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None
            else len(self.data['y'][part])
        )

    def parts(self) -> Iterable[PartKey]:
        return self.data['y'].keys()

    def compute_cat_cardinalities(self) -> list[int]:
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            return []
        unique = np.unique if self._is_numpy() else torch.unique
        return (
            []
            if x_cat is None
            else [len(unique(column)) for column in x_cat['train'].T]
        )

    def convert_bin_features_to_cat_(self: 'Dataset[np.ndarray]') -> None:
        assert self._is_numpy()

        x_bin = self.data.pop('x_bin', None)
        if x_bin is None:
            return

        x_bin_as_cat = {k: v.astype(np.int64) for k, v in x_bin.items()}
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            self.data['x_cat'] = x_bin_as_cat
        else:
            assert x_cat.keys() == x_bin_as_cat.keys()
            for part in x_bin_as_cat:
                x_cat[part] = np.column_stack([x_cat[part], x_bin_as_cat[part]])

    def standardize_labels_(
        self: 'Dataset[np.ndarray]',
    ) -> 'RegressionLabelStats':
        assert self._is_numpy()
        assert self.task.is_regression
        self.data['y'], regression_label_stats = standardize_labels(self.data['y'])
        return regression_label_stats

    def try_standardize_labels_(
        self: 'Dataset[np.ndarray]',
    ) -> 'None | RegressionLabelStats':
        assert self._is_numpy()
        return self.standardize_labels_() if self.task.is_regression else None

    def minmax_scale_labels_(
        self: 'Dataset[np.ndarray]',
    ) -> 'RegressionLabelStats':
        assert self._is_numpy()
        assert self.task.is_regression
        self.data['y'], regression_label_stats = minmax_scale_labels(self.data['y'])
        return regression_label_stats

    def try_minmax_scale_labels_(
        self: 'Dataset[np.ndarray]',
    ) -> 'None | RegressionLabelStats':
        assert self._is_numpy()
        return self.minmax_scale_labels_() if self.task.is_regression else None


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    NOISY_QUANTILE = 'noisy-quantile'


def transform_num(
    X_num: dict[PartKey, np.ndarray],
    policy: None | str | NumPolicy,
    seed: None | int,
    preserve_num_missing: bool = False,
) -> dict[PartKey, np.ndarray]:
    if preserve_num_missing and policy is not None:
        raise ValueError('preserve_num_missing requires num_policy=None')
    if policy is not None:
        policy = NumPolicy(policy)
        X_num_train = X_num['train']
        if policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()
        elif policy == NumPolicy.NOISY_QUANTILE:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
                output_distribution='normal',
                subsample=1_000_000_000,
                random_state=seed,
            )
            assert seed is not None
            X_num_train = X_num_train + np.random.RandomState(seed).normal(
                0.0, 1e-5, X_num_train.shape
            ).astype(X_num_train.dtype)
        else:
            raise ValueError(f'Unknown policy={policy}')

        normalizer.fit(X_num_train)
        X_num = {k: normalizer.transform(v) for k, v in X_num.items()}  # type: ignore

    if preserve_num_missing:
        X_num = {k: np.where(np.isinf(v), np.nan, v) for k, v in X_num.items()}
    else:
        # NOTE
        # (This is not a good way to process NaNs)
        # This is a quick hack to stop failing on some datasets because of NaNs.
        # NaNs are replaced with zeros (zero is the mean value for all features after
        # the conventional preprocessing techniques).
        X_num = {k: np.nan_to_num(v) for k, v in X_num.items()}

    # Remove columns with one constant value.
    mask = np.array([len(np.unique(x)) > 1 for x in X_num['train'].T])
    X_num = {k: v[:, mask] for k, v in X_num.items()}

    X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    return X_num


def _extract_bin_from_num(
    X_num: dict[PartKey, np.ndarray],
) -> tuple[None | dict[PartKey, np.ndarray], None | dict[PartKey, np.ndarray]]:
    """Split off the two-valued columns of X_num into binary features."""
    X_num_all = np.concatenate(list(X_num.values()))
    has_missing_values = np.any(np.isnan(X_num_all), 0)
    unique_values = [np.unique(x) for x in X_num_all.T]
    unique_counts = np.array([len(x) for x in unique_values])

    bin_mask = (unique_counts == 2) & ~has_missing_values
    bin_idx = np.nonzero(bin_mask)[0]

    if not len(bin_idx):
        return None, X_num

    transformer = sklearn.preprocessing.OrdinalEncoder(
        categories=[unique_values[i] for i in bin_idx]
    )
    transformer.fit(X_num['train'][:, bin_idx])
    if len(bin_idx) == X_num_all.shape[1]:
        # All the features are binary.
        return {
            k: transformer.transform(v).astype(bool) for k, v in X_num.items()
        }, None
    return (
        {
            k: transformer.transform(v[:, bin_idx]).astype(bool)
            for k, v in X_num.items()
        },
        {k: v[:, ~bin_mask] for k, v in X_num.items()},
    )


class BinPolicy(enum.Enum):
    CONVERT_TO_CAT = 'convert-to-cat'


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


def transform_cat(
    X_cat: dict[PartKey, np.ndarray], policy: None | str | CatPolicy
) -> dict[PartKey, np.ndarray]:
    if policy is None:
        return X_cat

    policy = CatPolicy(policy)

    # The first step is always the ordinal encoding,
    # even for the one-hot encoding.
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore
        unknown_value=unknown_value,  # type: ignore
        dtype='int64',  # type: ignore
    ).fit(X_cat['train'])
    X_cat = {k: encoder.transform(v) for k, v in X_cat.items()}
    max_values = X_cat['train'].max(axis=0)
    for part in ['val', 'test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat[part].shape[1]):
            X_cat[part][X_cat[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    if policy == CatPolicy.ORDINAL:
        return X_cat
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore',
            sparse=False,
            dtype=np.float32,  # type: ignore
        )
        encoder.fit(X_cat['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X_cat.items()}
    else:
        raise ValueError(f'Unknown policy={policy}')


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    y: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    assert y['train'].dtype == np.dtype('float32')
    mean = float(y['train'].mean())
    std = float(y['train'].std())
    return {k: (v - mean) / std for k, v in y.items()}, RegressionLabelStats(
        mean=mean, std=std
    )


def minmax_scale_labels(
    y: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    assert y['train'].dtype == np.dtype('float32')
    y_min = float(y['train'].min())
    y_range = float(y['train'].max()) - float(y['train'].min())
    return {k: (v - y_min) / y_range for k, v in y.items()}, RegressionLabelStats(
        mean=y_min, std=y_range
    )


def build_dataset(
    path: str | Path,
    *,
    extract_bin_from_num: bool = False,
    num_policy: None | str | NumPolicy = None,
    bin_policy: None | str | BinPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    preserve_num_missing: bool = False,
    seed: int = 0,
    cache: bool = False,
) -> Dataset[np.ndarray]:
    path = Path(path).resolve()
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = env.get_cache_dir() / (
            f'build_dataset__{path.name}__{hashlib.md5(str(args).encode("utf-8")).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = pickle.loads(cache_path.read_bytes())
            assert args == cached_args, f'Hash collision for {cache_path}'
            logger.info(f'Using cached dataset: {cache_path.name}')
            return cached_value
    else:
        args = None
        cache_path = None

    dataset = Dataset.from_dir(path)

    if 'x_num' in dataset.data and extract_bin_from_num:
        extracted_x_bin, remaining_x_num = _extract_bin_from_num(dataset.data['x_num'])
        if extracted_x_bin is not None:
            if remaining_x_num is None:
                del dataset.data['x_num']
            else:
                dataset.data['x_num'] = remaining_x_num
            x_bin = dataset.data.pop('x_bin', None)
            if x_bin is None:
                dataset.data['x_bin'] = extracted_x_bin
            else:
                assert extracted_x_bin.keys() == x_bin.keys()
                dataset.data['x_bin'] = {
                    k: np.concatenate([extracted_x_bin[k], x_bin[k]], axis=-1)
                    for k in extracted_x_bin
                }

    # The presence of "x_num" may change after the binary feature extraction,
    # so it must be checked again.
    if 'x_num' in dataset.data:
        dataset.data['x_num'] = transform_num(
            dataset.data['x_num'],
            num_policy,
            seed,
            preserve_num_missing=preserve_num_missing,
        )
    if 'x_bin' in dataset.data:
        if bin_policy is not None:
            bin_policy = BinPolicy(bin_policy)
            if bin_policy == BinPolicy.CONVERT_TO_CAT:
                dataset.convert_bin_features_to_cat_()
            else:
                raise ValueError(f'Unknown {bin_policy=}')
    if 'x_cat' in dataset.data:
        dataset.data['x_cat'] = transform_cat(dataset.data['x_cat'], cat_policy)

    if cache_path is not None:
        with tempfile.NamedTemporaryFile('wb') as tmp_cache_file:
            tmp_cache_file.write(pickle.dumps((args, dataset)))
            os.rename(tmp_cache_file.name, cache_path)
    return dataset
