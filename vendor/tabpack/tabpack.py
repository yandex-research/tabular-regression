# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import dataclasses
import datetime
import gc
import inspect
import itertools
import json
import math
import statistics
import time
import typing
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, Protocol, TypedDict

import delu
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.utils.checkpoint
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import bin.tune
import lib.data
import lib.deep
import lib.env
import lib.experiment
import lib.uncertainty
import lib.util
import vendor.tabpack.ensemble_utils
import vendor.tabpack.ensemble_utils_torch
import vendor.tabpack.metrics
import vendor.tabpack.nn
import vendor.tabpack.optim
import vendor.tabpack.utils
from lib.types import AMPDType, JSONDict, KWArgs, PartKey, PredictionType, TaskType
from vendor.tabpack.metrics import Metrics
from vendor.tabpack.nn import BATCH_DIM, PACK_DIM, get_pack_size
from vendor.tabpack.types import ConfigDict


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Model
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _make_num_module(type: str, **kwargs) -> nn.Module:
    classes = (
        vendor.tabpack.nn.LinearEmbeddingsPack,
        vendor.tabpack.nn.LinearReLUEmbeddingsPack,
        vendor.tabpack.nn.CosineEmbeddingsPack,
    )
    cls = {x.__name__: x for x in classes}[type]
    return cls(**kwargs)


class ModelPack(vendor.tabpack.nn.ModulePack):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        pack_size: int,
        num_embeddings: None | KWArgs = None,
        gradient_checkpointing: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        # Checkpoint the numerical embeddings during training: their fp32
        # intermediates are [pack, batch, n_features, d_embedding] each and
        # dominate GPU memory on wide datasets.
        self._gradient_checkpointing = gradient_checkpointing

        if num_embeddings is None or n_num_features == 0:
            # No numerical embeddings when there are no numerical features (e.g. datasets
            # whose features are all binary and get moved to x_cat by the preprocessing).
            self.num_module = None
            d_num_feature = 1
        else:
            self.num_module = _make_num_module(
                n_features=n_num_features, pack_size=pack_size, **num_embeddings
            )
            d_num_feature = num_embeddings.get('max_d_embedding')
            if d_num_feature is None:
                d_num_feature = num_embeddings['d_embedding']
            assert isinstance(d_num_feature, int)
        self.cat_module = (
            vendor.tabpack.nn.OneHotEncoding(cat_cardinalities)
            if cat_cardinalities
            else None
        )
        self.pack_view = vendor.tabpack.nn.PackView(pack_size=pack_size)
        self.backbone = vendor.tabpack.nn.MLPBackbonePack(
            d_in=d_num_feature * n_num_features + sum(cat_cardinalities),
            pack_size=pack_size,
            **backbone_kwargs,
        )
        self.output = vendor.tabpack.nn.LinearPack(
            backbone_kwargs['d_block'],
            1 if n_classes is None or n_classes == 2 else n_classes,
            max_in_features=backbone_kwargs.get('max_d_block'),
            pack_size=pack_size,
        )

        self._n_num_features = n_num_features
        self._cat_cardinalities = cat_cardinalities

    @property
    def pack_size(self) -> int:
        return self.backbone.pack_size

    def forward(
        self, x_num: None | Tensor, x_cat: None | Tensor, pack_idx: None | Tensor = None
    ) -> Tensor:
        assert x_num is not None or x_cat is not None

        x_list: list[Tensor] = []
        pack_view_used = False

        if x_num is None:
            assert self._n_num_features == 0
            # assert self.num_module is None
        else:
            assert self._n_num_features > 0
            if self.num_module is None:
                x_list.append(x_num)
            else:
                x_num_ = self.pack_view(x_num, pack_idx)
                pack_view_used = True
                if (
                    self._gradient_checkpointing
                    and pack_idx is None
                    and torch.is_grad_enabled()
                ):
                    x_num_ = torch.utils.checkpoint.checkpoint(
                        self.num_module, x_num_, use_reentrant=False
                    )
                else:
                    x_num_ = self.num_module(x_num_, pack_idx)
                x_list.append(x_num_.flatten(-2))
                del x_num_

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_cat_ = self.cat_module(x_cat).to(torch.get_default_dtype())
            if pack_view_used:
                x_cat_ = self.pack_view(x_cat_, pack_idx)
            x_list.append(x_cat_)
            del x_cat_

        x: Tensor = torch.cat(x_list, dim=-1)
        if not pack_view_used:
            x = self.pack_view(x, pack_idx)
        x = self.backbone(x, pack_idx)
        x = self.output(x, pack_idx)
        return x


class ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        batch_idx: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: nn.Module, dataset: lib.data.Dataset, *, part: PartKey, batch_idx: Tensor
) -> Tensor:
    return (
        model(
            dataset.data['x_num'][part][batch_idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][batch_idx] if 'x_cat' in dataset.data else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()
    )


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Hyperparameter sampler
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class HyperparameterSampler:
    """A simpler wrapper around `optuna.study.Study`."""

    _WARMUP_SEED_SHIFT = 1000
    _BASIC_SAMPLERS = (
        'BruteForceSampler',
        'GridSampler',
        'RandomSampler',
        'QMCSampler',
    )

    @staticmethod
    def _make_sampler(type: str, **kwargs) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, type)(**kwargs)

    def __init__(
        self,
        *,
        space: dict[str, Any],
        type: str = 'TPESampler',
        strict_n_startup_trials: bool = False,
        study_kwargs: None | KWArgs = None,
        **kwargs,
    ) -> None:
        self._study = optuna.create_study(
            sampler=HyperparameterSampler._make_sampler(type, **kwargs),
            **({} if study_kwargs is None else study_kwargs),
        )
        self._strict_n_startup_trials = strict_n_startup_trials
        self._space = space
        self._trials: dict[int, optuna.trial.Trial] = {}

    @property
    def n_startup_trials(self) -> None | int:
        return getattr(self._study.sampler, '_n_startup_trials', None)

    def _load_n_finished_trials(self) -> int:
        return len(
            self._study.get_trials(
                False,
                (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
            )
        )

    def ask(self, index: int) -> dict[str, Any]:
        assert index not in self._trials
        trial = self._study.ask()
        if self.n_startup_trials is not None:
            trial.set_user_attr(
                'startup', self._load_n_finished_trials() < self.n_startup_trials
            )
        self._trials[index] = trial
        return bin.tune._sample_config(trial, self._space, [])

    def tell(self, index: int, value: float) -> bool:
        trial = self._trials[index]
        should_skip = (
            self._strict_n_startup_trials
            and self.n_startup_trials is not None
            and trial.user_attrs.get('startup', False)
            and self._load_n_finished_trials() >= self.n_startup_trials
        )
        should_tell = not should_skip
        if should_tell:
            self._study.tell(self._trials[index], value)
        return should_tell


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Pack
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _find_arrays_and_tensors(x: Any) -> Generator[np.ndarray | Tensor]:
    if isinstance(x, np.ndarray | Tensor):
        yield x
    elif x is None or x is ... or isinstance(x, bool | int | float | str | bytes):
        return
    elif isinstance(x, Sequence):
        for item in x:
            yield from _find_arrays_and_tensors(item)
    elif isinstance(x, Mapping):
        for item in itertools.chain(x.keys(), x.values()):
            yield from _find_arrays_and_tensors(item)
    elif hasattr(x, '__dict__'):
        yield from _find_arrays_and_tensors(vars(x))


class StatePack:
    _BEST_STEP_INIT = np.iinfo(np.int64).min

    def __init__(self, *, pack_size: int, configs: None | list[ConfigDict]) -> None:
        assert pack_size >= 0

        # Static properties.
        self._ids = np.arange(pack_size)
        self._configs = deepcopy(configs)

        # Dynamic properties.
        self._steps = np.zeros(pack_size, dtype=np.int64)
        self._n_consequtive_bad_updates = np.zeros(pack_size, dtype=np.int64)
        self._best_metrics = {}
        self._best_steps = np.full(pack_size, StatePack._BEST_STEP_INIT)
        self._best_predictions = {}
        self._best_predictions_torch = {}
        self._best_model_state_dicts = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def configs(self) -> None | list[dict]:
        return self._configs

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def n_consequtive_bad_updates(self) -> np.ndarray:
        return self._n_consequtive_bad_updates

    @property
    def best_metrics(self) -> dict[PartKey, Metrics]:
        return self._best_metrics

    @property
    def best_steps(self) -> np.ndarray:
        return self._best_steps

    @property
    def best_predictions(self) -> dict[PartKey, np.ndarray]:
        return self._best_predictions

    @property
    def best_predictions_torch(self) -> dict[PartKey, Tensor]:
        return self._best_predictions_torch

    @property
    def best_model_state_dicts(self) -> dict[str, Tensor]:
        return self._best_model_state_dicts

    @property
    def pack_size(self) -> int:
        return len(self.ids)

    def validate(self) -> None:
        # Validate the pack size.
        pack_size = self.pack_size
        if self.configs is not None:
            assert len(self.configs) == pack_size
        assert all(
            x.shape[PACK_DIM] == pack_size for x in _find_arrays_and_tensors(self)
        )

        # Validate the static properties.
        assert np.all(self.ids >= 0)
        assert len(np.unique(self.ids)) == pack_size

        # Validate the dynamic properties.
        assert np.all(self.steps >= 0)
        assert np.all(self.n_consequtive_bad_updates >= 0)
        assert np.all(self.best_steps <= self.steps)

    # NOTE
    # All the following methods MUTATE `self`.

    def step(self) -> None:
        self._steps += 1

    def remove(self, pack_idx: np.ndarray) -> None:
        assert len(pack_idx) > 0
        assert (self.steps[pack_idx] > 0).all()

        device = next(iter(self.best_model_state_dicts.values())).device
        keep_pack_idx_torch = vendor.tabpack.nn.make_keep_pack_idx(
            self.pack_size, torch.as_tensor(pack_idx)
        )
        keep_pack_idx = keep_pack_idx_torch.numpy()
        keep_pack_idx_torch = keep_pack_idx_torch.to(device)
        del pack_idx

        # NOTE
        # All static and dynamic properties corresponding to `pack_idx`
        # must be removed.

        # Remove the static properties.
        self._ids = self.ids[keep_pack_idx].copy()
        if self.configs is not None:
            self.configs[:] = [self.configs[i] for i in map(int, keep_pack_idx)]

        # Remove the dynamic properties.
        self._steps = self.steps[keep_pack_idx].copy()
        self._n_consequtive_bad_updates = self.n_consequtive_bad_updates[
            keep_pack_idx
        ].copy()
        for part_metrics in self.best_metrics.values():
            for key in list(part_metrics):
                part_metrics[key] = part_metrics[key][keep_pack_idx].copy()
        self._best_steps = self.best_steps[keep_pack_idx].copy()
        for part in list(self.best_predictions):
            self.best_predictions[part] = self.best_predictions[part][
                keep_pack_idx
            ].copy()
        for part in list(self.best_predictions):
            self.best_predictions_torch[part] = self.best_predictions_torch[part][
                keep_pack_idx_torch
            ].clone()
        for key, value in list(self.best_model_state_dicts.items()):
            self.best_model_state_dicts[key] = value[keep_pack_idx_torch]

    def update(
        self,
        metrics: dict[PartKey, Metrics],
        *,
        predictions: dict[PartKey, np.ndarray],
        predictions_torch: dict[PartKey, Tensor],
        model_state_dict: dict[str, Tensor],
    ) -> None:
        assert (self.steps > 0).all()

        if self.best_metrics:
            device = next(iter(model_state_dict.values())).device
            improved_mask = metrics['val']['score'] > self.best_metrics['val']['score']
            improved_pack_idx = np.nonzero(improved_mask)[0]

            if len(improved_pack_idx) > 0:
                improved_pack_idx_torch = torch.tensor(improved_pack_idx, device=device)
                self.n_consequtive_bad_updates[improved_pack_idx] = 0

                for part in self.best_metrics:
                    for key in self.best_metrics[part]:
                        self.best_metrics[part][key][improved_pack_idx] = metrics[part][
                            key
                        ][improved_pack_idx]
                self.best_steps[improved_pack_idx] = self.steps[improved_pack_idx]
                for part in self.best_predictions:
                    self.best_predictions[part][improved_pack_idx] = predictions[part][
                        improved_pack_idx
                    ]
                for part in self.best_predictions_torch:
                    self.best_predictions_torch[part][improved_pack_idx_torch] = (
                        predictions_torch[part][improved_pack_idx_torch]
                    )
                for key, value in model_state_dict.items():
                    self.best_model_state_dicts[key][improved_pack_idx_torch] = value[
                        improved_pack_idx_torch
                    ]
            self.n_consequtive_bad_updates[~improved_mask] += 1

        else:
            assert np.all(self.n_consequtive_bad_updates == 0)
            self._best_metrics = metrics
            self._best_steps[:] = self.steps
            self._best_predictions = deepcopy(predictions)
            self._best_predictions_torch = deepcopy(predictions_torch)
            self._best_model_state_dicts = deepcopy(model_state_dict)


class FinalStatePack:
    def __init__(self) -> None:
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions = {}
        self._predictions_torch = {}

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def steps(self) -> np.ndarray:
        return self._steps

    @property
    def predictions(self) -> dict[str, np.ndarray]:
        return self._predictions

    @property
    def predictions_torch(self) -> dict[str, Tensor]:
        return self._predictions_torch

    def __len__(self) -> int:
        return len(self.ids)

    def extend(
        self,
        *,
        ids: np.ndarray,
        steps: np.ndarray,
        predictions: dict[str, np.ndarray],
        predictions_torch: dict[str, Tensor],
    ) -> None:
        self._ids = np.concat([self._ids, ids])
        self._steps = np.concat([self._steps, steps])
        self._predictions = (
            {
                k: np.concat([self.predictions[k], predictions[k]])
                for k in predictions.keys()
            }
            if self.predictions
            else predictions
        )
        self._predictions_torch = (
            {
                k: torch.cat([self.predictions_torch[k], predictions_torch[k]])
                for k in predictions_torch.keys()
            }
            if self.predictions_torch
            else predictions_torch
        )


def pack_validate(
    model: ModelPack, optimizer: torch.optim.Optimizer, state: StatePack
) -> None:
    pack_size = model.pack_size
    assert state.pack_size == pack_size

    for x in itertools.chain(model.parameters(), model.buffers()):
        if isinstance(
            x, vendor.tabpack.nn.ParameterPack | vendor.tabpack.nn.BufferPack
        ):
            assert get_pack_size(x) == pack_size

    assert pack_size == state.pack_size
    for x in itertools.chain.from_iterable(
        group['params'] for group in optimizer.param_groups
    ):
        assert get_pack_size(x) == pack_size

    state.validate()


def pack_remove(
    model: ModelPack,
    optimizer: torch.optim.Optimizer,
    state: StatePack,
    *,
    pack_idx: np.ndarray,
    loss_fn: None | Callable[[Tensor, Tensor], Tensor] = None,
) -> None:
    assert len(pack_idx) > 0
    pack_idx_torch = torch.as_tensor(pack_idx, device=next(model.parameters()).device)
    old_to_new = vendor.tabpack.nn.module_pack_remove(model, pack_idx_torch)
    vendor.tabpack.optim.optimizer_pack_remove(optimizer, pack_idx_torch, old_to_new)
    state.remove(pack_idx)
    if loss_fn is not None and hasattr(loss_fn, 'pack_remove'):
        loss_fn.pack_remove(pack_idx_torch)


def compute_stop_pack_idx(
    state: StatePack, *, epoch_size: int, n_epochs: int, patience: int
) -> None | np.ndarray:
    early_stop_mask = (
        state.n_consequtive_bad_updates > patience if patience >= 0 else None
    )
    epoch_stop_mask = state.steps // epoch_size >= n_epochs if n_epochs >= 0 else None
    stop_mask = (
        early_stop_mask
        if epoch_stop_mask is None
        else epoch_stop_mask
        if early_stop_mask is None
        else (early_stop_mask | epoch_stop_mask)
    )
    if stop_mask is None:
        return None
    stop_pack_idx = np.nonzero(stop_mask)[0]
    return None if len(stop_pack_idx) == 0 else stop_pack_idx


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Experiments
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class ExperimentDict(TypedDict):
    config: NotRequired[JSONDict]
    report: lib.experiment.Report


def assemble_experiments(
    *,
    state: StatePack,
    final_metrics: dict[PartKey, Metrics],
    pack_idx: np.ndarray,
    time_elapsed: float,
) -> list[ExperimentDict]:
    experiments = []
    for i, index in enumerate(pack_idx):
        experiment = {}
        if state.configs is not None:
            experiment['config'] = state.configs[index]
        experiment['report'] = {
            'id': int(state.ids[index]),
            'best_step': int(state.best_steps[index]),
            'metrics': {
                part: {k: float(v[i]) for k, v in part_metrics.items()}
                for part, part_metrics in final_metrics.items()
            },
            # NOTE
            # The time computed below is not representative of the time
            # that running this one experiment would take in isolation.
            'time': time_elapsed,
        }
        experiments.append(experiment)  # ty:ignore[invalid-argument-type]
    return experiments


def get_experiment_val_score(experiment: ExperimentDict) -> float:
    return experiment['report']['metrics']['val']['score']


def get_best_experiment(
    current_best_experiment: None | ExperimentDict,
    new_experiments: list[ExperimentDict],
) -> tuple[ExperimentDict, bool]:
    assert new_experiments

    best_experiment = current_best_experiment
    del current_best_experiment

    best_experiment_improved = False
    for experiment in new_experiments:
        if best_experiment is None or get_experiment_val_score(
            experiment
        ) > get_experiment_val_score(best_experiment):
            best_experiment = experiment
            best_experiment_improved = True

    assert best_experiment is not None
    return best_experiment, best_experiment_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Ensembles
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class OnlineEnsemble:
    def __init__(
        self,
        *,
        type: str,
        options: None | KWArgs = None,
        algorithm_score_fn: None
        | vendor.tabpack.ensemble_utils_torch.EnsembleScoreFn = None,
        update_type: Literal['final', 'best', 'latest'],
        update_part: PartKey,
        include_current_ensemble_in_pool: bool = False,
        prediction_type: PredictionType,
        score_fn: vendor.tabpack.ensemble_utils_torch.EnsembleScoreFn,
        patience: int,
    ) -> None:
        # Algorithm details
        self._ensemble_fn = vendor.tabpack.ensemble_utils_torch.get_ensemble_fn(type)
        self._ensemble_fn_kwargs = {} if options is None else options
        self._ensemble_fn_kwargs['score_fn'] = (
            score_fn if algorithm_score_fn is None else algorithm_score_fn
        )
        if type == 'autotopk' or (
            type == 'greedy'
            and options is not None
            and options.get('init_autotopk', False)
        ):
            self._ensemble_fn_kwargs['prediction_type'] = prediction_type
        self._update_type = update_type
        self._include_current_ensemble_in_pool = include_current_ensemble_in_pool

        # Current ensemble
        self._ids = np.array([], dtype=np.int64)
        self._steps = np.array([], dtype=np.int64)
        self._predictions: dict[PartKey, np.ndarray] = {}
        self._predictions_torch: dict[PartKey, Tensor] = {}
        self._weights: None | np.ndarray = None
        self._score: None | float = None

        # Task properties
        self._prediction_type = prediction_type

        # Update settings
        self._update_part = update_part
        self._score_fn = score_fn
        self._patience = patience
        self._remaining_patience = patience

        #
        self._time_elapsed = 0.0

    @property
    def ids(self):
        return self._ids

    @property
    def steps(self):
        return self._steps

    @property
    def predictions(self):
        return self._predictions

    @property
    def weights(self):
        return self._weights

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def is_running(self) -> bool:
        return self._remaining_patience >= 0

    def _prepare_pool(
        self,
        running_ids: np.ndarray,
        running_steps: np.ndarray,
        running_latest_predictions: dict[str, np.ndarray],
        running_latest_predictions_torch: dict[str, Tensor],
        running_best_predictions: dict[str, np.ndarray],
        running_best_predictions_torch: dict[str, Tensor],
        finished_ids: np.ndarray,
        finished_steps: np.ndarray,
        finished_predictions: dict[str, np.ndarray],
        finished_predictions_torch: dict[str, Tensor],
    ):
        if self._update_type == 'final':
            pool_ids = finished_ids
            pool_steps = finished_steps
            pool_predictions = finished_predictions
            pool_predictions_torch = finished_predictions_torch

        else:
            if self._update_type == 'best':
                running_predictions = running_best_predictions
                running_predictions_torch = running_best_predictions_torch
            else:
                assert self._update_type == 'latest'
                running_predictions = running_latest_predictions
                running_predictions_torch = running_latest_predictions_torch

            if len(finished_ids) == 0:
                pool_ids = running_ids
                pool_steps = running_steps
                pool_predictions = running_predictions
                pool_predictions_torch = running_predictions_torch

            else:
                pool_ids = np.concat([finished_ids, running_ids])
                pool_steps = np.concat([finished_steps, running_steps])
                pool_predictions = {
                    k: np.concat([finished_predictions[k], running_predictions[k]])
                    for k in running_predictions.keys()
                }
                pool_predictions_torch = {
                    k: torch.cat(
                        [finished_predictions_torch[k], running_predictions_torch[k]]
                    )
                    for k in running_predictions_torch.keys()
                }

            del running_predictions, running_predictions_torch
        del finished_predictions, finished_predictions_torch
        del running_best_predictions, running_best_predictions_torch
        del running_latest_predictions, running_latest_predictions_torch

        if self._include_current_ensemble_in_pool and len(self.ids) > 0:
            pool_ids = np.concat([self._ids, pool_ids])
            pool_steps = np.concat([self._steps, pool_steps])
            if self._predictions:
                pool_predictions = {
                    k: np.concat([self._predictions[k], pool_predictions[k]])
                    for k in pool_predictions.keys()
                }
            if self._predictions_torch:
                pool_predictions_torch = {
                    k: torch.cat(
                        [self._predictions_torch[k], pool_predictions_torch[k]]
                    )
                    for k in pool_predictions_torch.keys()
                }

        return pool_ids, pool_steps, pool_predictions, pool_predictions_torch

    def update(self, **kwargs) -> bool:
        assert self.is_running
        if self._update_type == 'final' and len(kwargs['finished_ids']) == 0:
            return False

        start_time = time.perf_counter()

        (pool_ids, pool_steps, pool_predictions, pool_predictions_torch) = (
            self._prepare_pool(**kwargs)
        )
        ensemble = self._ensemble_fn(
            pool_predictions_torch[self._update_part], **self._ensemble_fn_kwargs
        )
        ensemble_idx, ensemble_weights = (
            (ensemble, None) if isinstance(ensemble, Tensor) else ensemble
        )

        ensemble_part_prediction = (
            vendor.tabpack.ensemble_utils_torch.compute_ensemble_prediction(
                pool_predictions_torch[self._update_part][ensemble_idx],
                ensemble_weights,
                self._prediction_type,
            )
        )
        score = self._score_fn(ensemble_part_prediction[None]).item()
        improved = self._score is None or score > self._score

        if improved:
            self._remaining_patience = self._patience

            ensemble_idx_numpy = ensemble_idx.cpu().numpy()
            self._ids = pool_ids[ensemble_idx_numpy]
            self._steps = pool_steps[ensemble_idx_numpy]
            self._predictions = {
                k: v[ensemble_idx_numpy] for k, v in pool_predictions.items()
            }
            self._predictions_torch = {
                k: v[ensemble_idx] for k, v in pool_predictions_torch.items()
            }
            self._weights = (
                ensemble_weights
                if ensemble_weights is None
                else ensemble_weights.cpu().numpy()
            )
            self._score = score

        else:
            self._remaining_patience -= 1

        self._time_elapsed += time.perf_counter() - start_time
        return improved


def update_online_ensembles(
    online_ensembles: dict[str, OnlineEnsemble],
    *,
    task: lib.data.Task,
    step: int,
    timer: delu.tools.Timer,
    **kwargs,
) -> tuple[dict[str, lib.experiment.Report], bool]:
    reports = {}
    first_online_ensemble_improved = False
    for i_ensemble, (ensemble_name, ensemble) in enumerate(online_ensembles.items()):
        if not ensemble.is_running:
            continue

        ensemble_improved = ensemble.update(**kwargs)
        if i_ensemble == 0:
            first_online_ensemble_improved = ensemble_improved

        if ensemble_improved:
            ensemble_predictions = {
                k: vendor.tabpack.ensemble_utils.compute_ensemble_prediction(
                    v, ensemble.weights, ensemble._prediction_type
                )
                for k, v in ensemble._predictions.items()
            }
            metrics = task.calculate_metrics(
                ensemble_predictions, ensemble._prediction_type
            )

            reports[ensemble_name] = {
                'ids': ensemble.ids.tolist(),
                **(
                    {}
                    if ensemble.weights is None
                    else {'weights': ensemble.weights.tolist()}
                ),
                'step': step,
                'steps': ensemble.steps.tolist(),
                'time': timer.elapsed(),
                'ensemble_time': ensemble.time_elapsed,
                'metrics': metrics,
            }

    return reports, first_online_ensemble_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Training
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def generate_training_batches(
    *,
    train_size: int,
    batch_size: int,
    batch_generator: torch.Generator,
    pack_size: int,
) -> list[Tensor]:
    """Generate training batches for one epoch."""
    random_values = torch.rand(
        (pack_size, train_size),
        generator=batch_generator,
        device=batch_generator.device,
    )
    batches = random_values.argsort(dim=BATCH_DIM).split(batch_size, dim=BATCH_DIM)
    batches = list(batches)
    return batches


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Evaluation
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class _EvaluateOutput(NamedTuple):
    metrics: dict[PartKey, Metrics]
    predictions: dict[PartKey, np.ndarray]
    predictions_torch: dict[PartKey, Tensor]


@lib.util.adjust_gpu_memory_usage('batch_size')
def _evaluate(
    apply_model: ApplyModel,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: lib.data.Dataset,
    *,
    parts: list[PartKey],
    regression_label_stats: None | lib.data.RegressionLabelStats,
    prediction_type: str | PredictionType,
    batch_size: int,
    device: torch.device,
    get_scalar_prediction: None | Callable[[Tensor], Tensor] = None,
) -> _EvaluateOutput:
    model.eval()
    del optimizer

    metrics = {}
    predictions = {}
    predictions_torch = {}

    # Reduce a wide regression head (quantiles / bins [+ logsigma]) to a scalar per sample
    # *per batch*, before concatenation — otherwise the full (pack, N, n_channels) tensor is
    # materialized and OOMs on large datasets.
    reduce_per_batch = dataset.task.is_regression and get_scalar_prediction is not None

    for part in parts:
        batch_predictions = []
        for batch_idx in torch.arange(dataset.size(part), device=device).split(
            batch_size
        ):
            out = apply_model(model, dataset, part=part, batch_idx=batch_idx)
            if reduce_per_batch:
                out = get_scalar_prediction(out)
            batch_predictions.append(out)
        y_pred_torch = torch.cat(batch_predictions, dim=BATCH_DIM)

        if dataset.task.is_regression:
            assert regression_label_stats is not None
            y_pred_torch *= regression_label_stats.std
            y_pred_torch += regression_label_stats.mean

        elif dataset.task.is_binclass:
            y_pred_torch = torch.special.expit(y_pred_torch)

        else:
            assert dataset.task.is_multiclass
            y_pred_torch = torch.special.softmax(y_pred_torch, dim=-1)

        y_pred = y_pred_torch.cpu().numpy()

        assert np.isfinite(y_pred).all()
        metrics[part] = vendor.tabpack.metrics.calculate_metrics_pack(
            y_true=dataset.task.labels[part],
            y_pred=y_pred,
            task_type=dataset.task.type_,
            prediction_type=prediction_type,
            score=dataset.task.score,
        )
        predictions[part] = y_pred
        predictions_torch[part] = y_pred_torch

    return _EvaluateOutput(metrics, predictions, predictions_torch)


@torch.inference_mode()
def _evaluate_adaqr_uncertainty(
    apply_model: ApplyModel,
    model: nn.Module,
    dataset: lib.data.Dataset,
    *,
    parts: list[PartKey],
    batch_size: int,
    loss_fn: '_AdaQRLossPack',
    muls_init: Tensor,
    label_std: float,
    device: torch.device,
) -> dict[PartKey, dict[str, np.ndarray]]:
    """Evaluate compact uncertainty diagnostics without saving the quantile grid."""
    model.eval()
    result = {}
    for part in parts:
        batches: dict[str, list[Tensor]] = {}
        for batch_idx in torch.arange(dataset.size(part), device=device).split(
            batch_size
        ):
            diagnostics = loss_fn.get_uncertainty_predictions(
                apply_model(model, dataset, part=part, batch_idx=batch_idx),
                muls_init=muls_init,
            )
            for name, values in diagnostics.items():
                batches.setdefault(name, []).append(values)
        result[part] = {
            name: (torch.cat(values, dim=BATCH_DIM) * label_std).cpu().numpy()
            for name, values in batches.items()
        }
    return result


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Utilities
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _free_mps_memory() -> None:
    if torch.mps.is_available():
        gc.collect()
        torch.mps.empty_cache()


def _make_Y_train(dataset: lib.data.Dataset[Tensor]) -> Tensor:
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )
    return Y_train


def _make_autocast(amp_dtype: AMPDType, device: torch.device) -> torch.autocast:
    dtype = lib.util.get_amp_dtype(amp_dtype, device)
    # It is unclear how to implement gradient scaling,
    # so FP16 is not supported for now.
    assert dtype is torch.bfloat16, 'For now, only "bfloat16" is supported as amp_dtype'
    return torch.autocast(device.type, dtype)


def _compute_muls_init_offset_fix(
    raw_sigmas: list[Tensor], *, muls_init_add: float, min_sigma: float
) -> Tensor:
    """Pack-aware "offset fix" warm-start for the adaqr raw_sigma channel.

    Ported from the `_compute_memberwise_muls_init` variant of bin/adaqr.py
    that solves, per pack member, the exact equation

        mean_i | raw_sigma_i + offset | == muls_init_add - min_sigma,

    i.e. after the shift the *mean predicted sigma* equals `muls_init_add`
    (recall AdaQR's sigma = |raw_sigma| + min_sigma). This is stricter than the plain
    warm-start (`offset = muls_init_add - mean(raw_sigma)`), which only matches the
    mean of the pre-abs values.

    `mean_i |raw_i + c|` is piecewise linear and convex in `c`, so the solution is found
    by walking `k` = the number of samples that fall below the kink, from the branch
    where all shifted values are non-negative towards the median. If the target is below
    the attainable minimum (a very spread-out raw_sigma channel), the minimizer
    (`-median`) is returned.

    `raw_sigmas`: per-batch tensors of shape (pack, batch); returns (pack,).
    """
    target = muls_init_add - min_sigma
    assert target > 0.0, 'muls_init_add must be greater than min_sigma'

    raw = torch.cat(raw_sigmas, dim=BATCH_DIM).float()  # (pack, n_samples)
    raw = raw.sort(dim=BATCH_DIM).values
    n_samples = raw.shape[BATCH_DIM]
    assert n_samples > 0
    total_sum = raw.sum(dim=BATCH_DIM)  # (pack,)

    lower_median = -raw[:, (n_samples - 1) // 2]  # (pack,)
    minimum_mean_abs = (raw + lower_median.unsqueeze(-1)).abs().mean(dim=BATCH_DIM)

    offsets = lower_median.clone()
    unresolved = minimum_mean_abs <= target  # (pack,)
    prefix_sum = torch.zeros_like(total_sum)
    eps = 16 * torch.finfo(raw.dtype).eps

    for k in range((n_samples - 1) // 2 + 1):
        # The candidate assumes exactly `k` smallest samples end up non-positive.
        candidate = (n_samples * target - total_sum + 2 * prefix_sum) / (
            n_samples - 2 * k
        )
        left_is_nonpositive = (
            torch.ones_like(unresolved) if k == 0 else raw[:, k - 1] + candidate <= eps
        )
        right_is_nonnegative = raw[:, k] + candidate >= -eps
        valid = unresolved & left_is_nonpositive & right_is_nonnegative
        offsets = torch.where(valid, candidate, offsets)
        unresolved = unresolved & ~valid
        if not bool(unresolved.any()):
            break
        prefix_sum = prefix_sum + raw[:, k]

    assert not bool(unresolved.any())
    return offsets


def _make_online_ensembles(
    online_ensemble_configs: dict[str, KWArgs],
    *,
    task: lib.data.Task,
    prediction_type: PredictionType,
    update_part: PartKey,
    device: torch.device,
) -> dict[str, OnlineEnsemble]:
    score_fn = vendor.tabpack.ensemble_utils_torch.make_emsemble_score_fn(
        task,
        prediction_type,
        part=update_part,
        device=device,
    )
    loss_score_fn = (
        vendor.tabpack.ensemble_utils_torch.make_emsemble_score_fn(
            dataclasses.replace(task, score=lib.data.Score.CROSS_ENTROPY),
            prediction_type,
            part=update_part,
            device=device,
        )
        if task.is_classification
        else None
    )

    online_ensembles = {}
    for name, ensemble_config in online_ensemble_configs.items():
        algorithm_score_fn = ensemble_config.get('algorithm_score_fn')
        if algorithm_score_fn == 'loss':
            ensemble_config['algorithm_score_fn'] = loss_score_fn
        online_ensembles[name] = OnlineEnsemble(
            update_part=update_part,
            prediction_type=prediction_type,
            score_fn=score_fn,
            **ensemble_config,
        )

    return online_ensembles


def _make_muon_scale(linear: vendor.tabpack.nn.LinearPack) -> Tensor:
    pack_size = linear.pack_size
    out_f = (
        linear.out_features.float()
        if linear.out_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_out_features),
            device=linear.weight.device,
        )
    )
    in_f = (
        linear.in_features.float()
        if linear.in_features is not None
        else torch.full(
            (pack_size,),
            float(linear.max_in_features),
            device=linear.weight.device,
        )
    )
    return (out_f / in_f).clamp_(min=1.0).sqrt_()


def _default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
):
    return lib.deep.default_zero_weight_decay_condition(
        module_name, module, parameter_name, parameter
    ) or isinstance(module, vendor.tabpack.nn.LinearEmbeddingsPack)


def _make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    optimizer_cls = getattr(torch.optim, type, None)
    if optimizer_cls is None:
        optimizer_cls = {
            x.__name__: x
            for x in [
                vendor.tabpack.optim.AdamWPack,
                vendor.tabpack.optim.MuonAdamWPack,
            ]
        }[type]
    if 'pack_size' not in inspect.signature(optimizer_cls.__init__).parameters:
        kwargs.pop('pack_size', None)
    return optimizer_cls(**kwargs)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Gradient clipping
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Same semantics as `gradient_clipping_norm` / `always_clip` in bin/*.py, but applied
# per pack member instead of globally: the pack members are independent models trained
# on a summed loss, so a single global norm would couple them and make the effective
# clipping depend on `n_models`. Parameter packs are indexed along `PACK_DIM`; any
# non-pack parameter (e.g. a shared `CosineEmbeddings`) is clipped globally.
#
# `max_norm` is a (pack,) tensor, so the threshold can be tuned per pack member
# (`sampler.space.gradient_clipping_norm`). Shared parameters have no per-member
# gradient, so they use the smallest threshold in the pack (the most conservative
# choice: no member's shared-parameter update is clipped less than it asked for).
def _clip_grad_norm_pack(
    model: nn.Module, max_norm: Tensor, *, always_clip: bool, pack_size: int
) -> None:
    pack_grads: list[Tensor] = []
    shared_grads: list[Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        if (
            isinstance(p, vendor.tabpack.nn.ParameterPack)
            and get_pack_size(p) == pack_size
        ):
            pack_grads.append(p.grad)
        else:
            shared_grads.append(p.grad)

    if pack_grads:
        # (pack,): the norm of each member's own gradient.
        total_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(g.flatten(start_dim=1), dim=-1)
                    for g in pack_grads
                ]
            ),
            dim=0,
        )
        scale = max_norm / (total_norm + 1e-6)
        if not always_clip:
            scale = scale.clamp(max=1.0)
        for g in pack_grads:
            g.mul_(scale.view(-1, *((1,) * (g.ndim - 1))))

    if shared_grads:
        total_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in shared_grads])
        )
        scale = max_norm.min() / (total_norm + 1e-6)
        if not always_clip:
            scale = scale.clamp(max=1.0)
        for g in shared_grads:
            g.mul_(scale)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# AdaQR loss: sigma_coef
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Multiplier for the sigma-only part of the adaqr loss (`q` = the number of quantiles).
# Ported from bin/adaqr.py::_resolve_sigma_coef. Both the symbolic options
# and a plain number are accepted, so `sigma_coef` can either be fixed in the config
# or sampled per pack member, e.g.
#   sampler.space.loss.sigma_coef: ["_tune_", "categorical", ["1", "1/q", "1/sqrt(q)"]]
#   sampler.space.loss.sigma_coef: ["_tune_", "loguniform", 0.001, 1.0]
_SIGMA_COEF_FORMULAS: dict[str, Callable[[int], float]] = {
    '1': lambda q: 1.0,
    '1/q': lambda q: 1.0 / q,
    '1/sqrt(q)': lambda q: 1.0 / math.sqrt(q),
}


def _resolve_sigma_coef(sigma_coef: str | float | int, n_quantiles: int) -> float:
    if isinstance(sigma_coef, bool):
        raise ValueError(f'Unknown sigma_coef: {sigma_coef}')
    if isinstance(sigma_coef, int | float):
        return float(sigma_coef)
    formula = _SIGMA_COEF_FORMULAS.get(sigma_coef)
    if formula is None:
        raise ValueError(
            f'Unknown sigma_coef: {sigma_coef!r}.'
            f' Valid options: {sorted(_SIGMA_COEF_FORMULAS)} or a number'
        )
    return formula(n_quantiles)


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def _monotone_expand(raw_q: Tensor, inc_bias: float, scale_reg: str = 'left') -> Tensor:
    """Raw head channels (..., n_q) -> non-decreasing quantiles (..., n_q).

    scale_reg='left':  q_0 = raw[..., 0]; q_k = q_0 + sum_{j<=k} softplus(raw+j+bias).
    scale_reg='median': the middle channel is the base; increments fan out both ways,
    q_mid±k = base ± cumsum(softplus(raw+j+bias)), so the best-estimated quantile
    (not the 0.1% tail) scale_regs the whole function and the init is symmetric.

    Unlike independent quantile heads (which are free to cross), every channel here
    contributes to all the quantiles above it, so each parameter is trained by all the
    pinball terms at once instead of only its own. `inc_bias` sets the init spacing:
    softplus(inc_bias) per step at raw == 0, i.e. an init fan-out of
    (n_q - 1) * softplus(inc_bias).
    """
    if scale_reg == 'left':
        base = raw_q[..., 0:1]
        increments = nn.functional.softplus(raw_q[..., 1:] + inc_bias)
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
    mid = (raw_q.shape[-1] - 1) // 2
    base = raw_q[..., mid : mid + 1]
    up = nn.functional.softplus(raw_q[..., mid + 1 :] + inc_bias)
    down = nn.functional.softplus(raw_q[..., :mid] + inc_bias)
    above = base + torch.cumsum(up, dim=-1)
    below = base - torch.cumsum(down.flip(-1), dim=-1).flip(-1)
    return torch.cat([below, base, above], dim=-1)


def _make_monotone_inc_bias(
    monotone: bool,
    monotone_init_spread: float,
    n_quantiles: int,
    monotone_reference: str = 'left',
) -> float:
    """Validate the shared monotone switches and resolve the increment bias."""
    assert isinstance(monotone, bool), (
        'loss.monotone is a shared switch, it cannot be tuned per pack member'
    )
    assert isinstance(monotone_init_spread, int | float) and monotone_init_spread > 0, (
        'loss.monotone_init_spread must be a positive number (shared across the pack)'
    )
    assert monotone_reference in ('left', 'median'), (
        f"loss.monotone_reference must be 'left' or 'median', got {monotone_reference!r}"
    )
    assert n_quantiles > 1
    return _inverse_softplus(monotone_init_spread / (n_quantiles - 1))


class _QuantileLossPack:
    """Pinball (quantile) loss, pack-aware. Ported from bin/quantile_regression.py.

    Expects `y_pred` of shape (pack, batch, n_quantiles) and `y_true` (pack, batch);
    returns a per-pack-member loss vector (pack,) — same contract as the native loss.

    `monotone` (shared switch, off by default) reparameterizes the head as
    `_monotone_expand`, making the predicted quantiles non-decreasing by construction;
    `monotone_init_spread` is the init fan-out of the quantile function (targets are
    standardized, so ~4 matches the scale of tau = 0.001..0.999 under a normal).
    `monotone_reference` picks the base channel: 'left' (q at tau=0.001) or 'median'
    (the middle channel, with increments fanning out both ways — the best-estimated
    quantile scale_regs the rest and the init mean-of-quantiles is unshifted).
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        agg_mode: str = 'sum',
        monotone: bool = False,
        monotone_init_spread: float = 4.0,
        monotone_reference: str = 'left',
        device: torch.device,
    ) -> None:
        assert agg_mode in ('sum', 'mean'), f'Unsupported agg_mode: {agg_mode}'
        self.quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        self.agg_mode = agg_mode
        self.monotone = monotone
        self.monotone_reference = monotone_reference
        self._inc_bias = _make_monotone_inc_bias(
            monotone, monotone_init_spread, n_quantiles, monotone_reference
        )

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if self.monotone:
            y_pred = _monotone_expand(y_pred, self._inc_bias, self.monotone_reference)
        errors = y_true.unsqueeze(-1) - y_pred  # (pack, batch, n_quantiles)
        pinball = torch.maximum(
            self.quantiles * errors, (self.quantiles - 1.0) * errors
        )
        per_example = pinball.sum(-1) if self.agg_mode == 'sum' else pinball.mean(-1)
        return per_example.mean(BATCH_DIM)  # (pack,)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # (pack, batch, n_quantiles) -> (pack, batch): mean over the quantile axis.
        if self.monotone:
            y_pred = _monotone_expand(y_pred, self._inc_bias, self.monotone_reference)
        return y_pred.mean(-1)


class _GaussianTailQuantileLossPack(_QuantileLossPack):
    """Pack-aware pinball loss with free centers and independent Gaussian tails."""

    def __init__(
        self,
        *,
        n_quantiles: int,
        min_sigma: float = 1e-5,
        agg_mode: str = 'sum',
        device: torch.device,
    ) -> None:
        if type(n_quantiles) is not int or n_quantiles < 18:
            raise ValueError('n_quantiles must be an integer >= 18 (C//18 tail values)')
        if not isinstance(min_sigma, int | float) or min_sigma <= 0.0:
            raise ValueError('min_sigma must be a positive number')
        self.n_center_quantiles = n_quantiles
        self.n_tail_quantiles = n_quantiles // 18
        self.n_outputs = n_quantiles + 4
        self.min_sigma = float(min_sigma)
        super().__init__(
            n_quantiles=n_quantiles + 2 * self.n_tail_quantiles,
            agg_mode=agg_mode,
            device=device,
        )
        normal = torch.distributions.Normal(0.0, 1.0)
        self.left_icdf_values = normal.icdf(self.quantiles[: self.n_tail_quantiles])
        self.right_icdf_values = normal.icdf(self.quantiles[-self.n_tail_quantiles :])

    def reconstruct_quantiles(self, raw: Tensor) -> Tensor:
        if raw.shape[-1] != self.n_outputs:
            raise ValueError(
                'Expected C center values followed by four tail parameters'
            )
        raw = raw.float()
        centers = raw[..., : self.n_center_quantiles]
        left_mean = raw[..., -4:-3]
        left_scale = raw[..., -3:-2].abs() + self.min_sigma
        right_mean = raw[..., -2:-1]
        right_scale = raw[..., -1:].abs() + self.min_sigma
        return torch.cat(
            (
                left_mean + left_scale * self.left_icdf_values,
                centers,
                right_mean + right_scale * self.right_icdf_values,
            ),
            dim=-1,
        )

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return super().__call__(self.reconstruct_quantiles(y_pred), y_true)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        return self.reconstruct_quantiles(y_pred).mean(-1)


class _OrthogonalQuantileLossPack:
    def __init__(
        self,
        *,
        n_quantiles: int,
        parameterization: str,
        mean_loss_weight: float | None = None,
        detach_location: bool = True,
        n_spline_coefficients: int | None = None,
        spline_slope_scale: float = 4.0,
        spline_slope_bias: float = -2.0,
        agg_mode: str = 'sum',
        device: torch.device,
    ) -> None:
        assert n_quantiles >= 2
        assert parameterization in ('grid', 'linear_spline')
        assert agg_mode in ('sum', 'mean')
        self.n_quantiles = n_quantiles
        self.parameterization = parameterization
        self.mean_loss_weight = (
            math.sqrt(n_quantiles) if mean_loss_weight is None else mean_loss_weight
        )
        assert self.mean_loss_weight >= 0.0
        assert isinstance(detach_location, bool)
        self.detach_location = detach_location
        self.agg_mode = agg_mode
        self.quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)

        if parameterization == 'grid':
            assert n_spline_coefficients is None
            self.n_shape_outputs = n_quantiles
            self.spline_basis = None
        else:
            assert n_spline_coefficients is not None
            assert n_spline_coefficients >= 2
            assert spline_slope_scale > 0.0
            self.n_shape_outputs = n_spline_coefficients
            self.spline_slope_scale = spline_slope_scale
            self.spline_slope_bias = spline_slope_bias

            positions = torch.linspace(0.0, 1.0, n_quantiles, device=device)
            edges = torch.linspace(0.0, 1.0, n_spline_coefficients + 1, device=device)
            width = 1.0 / n_spline_coefficients
            self.spline_basis = (positions[:, None] - edges[:-1]).clamp(
                min=0.0, max=width
            )

    @property
    def n_outputs(self) -> int:
        return 1 + self.n_shape_outputs

    def decode(
        self, y_pred: Tensor, *, detach_location: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert y_pred.shape[-1] == self.n_outputs
        mean = y_pred[..., 0].float()
        raw_shape = y_pred[..., 1:].float()

        if self.parameterization == 'grid':
            curve = raw_shape
        else:
            assert self.spline_basis is not None
            slopes = self.spline_slope_scale * nn.functional.softplus(
                raw_shape + self.spline_slope_bias
            )
            curve = torch.einsum('...j,qj->...q', slopes, self.spline_basis)

        residual_quantiles = curve - curve.mean(dim=-1, keepdim=True)
        location = mean.detach() if detach_location else mean
        quantiles = location.unsqueeze(-1) + residual_quantiles
        return mean, residual_quantiles, quantiles

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mean, _, quantiles = self.decode(y_pred, detach_location=self.detach_location)
        errors = y_true.float().unsqueeze(-1) - quantiles
        pinball = torch.maximum(
            self.quantiles * errors, (self.quantiles - 1.0) * errors
        )
        shape_loss = (
            pinball.sum(-1) if self.agg_mode == 'sum' else pinball.mean(-1)
        ).mean(BATCH_DIM)
        mean_loss = (mean - y_true.float()).square().mean(BATCH_DIM)
        return shape_loss + self.mean_loss_weight * mean_loss

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        return y_pred[..., 0]


class _SigmaHuberLossPack:
    """Sigma-Huberized pinball loss, pack-aware. Ported from bin/quantile_sigma_huber.py.

    A simplification of the adaqr loss: it keeps only the per-sample gradient
    attenuation (the part that drives the adaqr loss's wins) and drops the
    AdaQR wrapper, the Gaussian scale_reg and the `regularizer`. The
    quantile-branch gradient w.r.t. pred_tau is

        |tau - 1[err < 0]| * clamp(err / (sigma_scale * sigma), -1, 1)   (kernel='huber')
        |tau - 1[err < 0]| * (2 * Phi(err / (sigma_scale * sigma)) - 1)  (kernel='gauss')

    i.e. a pinball gradient squashed at the per-sample scale sigma(x). Sigma is always
    detached in this branch, so there is no 1/sigma**2 blow-up.

    Contract: `y_pred` (pack, batch, n_quantiles + 1) with the last channel the
    per-sample `logsigma` (as in `_AdaQRLossPack`), `y_true` (pack, batch); returns a
    per-pack-member loss vector (pack,). With `sigma_objective='spread'` the logsigma
    channel is ignored (still emitted, for a shared output width across the pack).

    `sigma_objective` (how sigma is obtained; per-pack-member):
    - 'nll':    Gaussian NLL on the mean-of-quantiles residual (proper score, but the
                classic heteroscedastic-NLL pathology applies: sigma can grow instead
                of the signal being learned).
    - 'mse':    MSE between log sigma and log|residual| (no 1/sigma**2 weighting and no
                inflation channel).
    - 'spread': no sigma head training at all; sigma is the predicted quantile spread
                0.5 * (q_0.841 - q_0.159), detached.

    Spread estimators (shared; how sigma is read off the quantile curve; all constants
    are Gaussian-consistent and derived from the actual tau grid):
    - spread_estimator='pair': sigma = agg_k[(q_{1-a_k} - q_{a_k}) / (z_hi - z_lo)] over
      the tail fractions in `spread_pairs` (z = Phi^-1(tau)). a=0.159 reproduces the
      original 0.5 * (q_0.841 - q_0.159); a=0.25 is the IQR estimator (more stable,
      blind to the tails). `spread_agg`: 'mean' (efficient) or 'median' (robust to one
      stuck pair).
    - spread_estimator='slope': sigma = OLS slope of preds regressed on z (uses all
      quantiles; the statistically optimal version of the pair estimator).
    - spread_asymmetric=True: separate sigma_L / sigma_R from the left / right half of
      the curve (per-side pair widths or per-side OLS slopes), applied to quantiles by
      side (tau <= 0.5 gets sigma_L). For skewed conditionals, where a single sigma
      over-attenuates one side.

    center_expectile = [lo, hi]: quantiles with tau in [lo, hi] are trained with the
    expectile loss tilt * err**2 instead of the Huberized pinball (MSE-like efficiency
    at the center -> better mean-of-quantiles prediction). Sigma pairs must lie outside
    [lo, hi] (asserted); the slope estimator drops the center points. Expectiles are
    not quantiles, so the center must not feed sigma.

    Per-pack-member knobs (a single value broadcasts, or a list with one value per pack
    member, e.g. sampled via `sampler.space.loss.sigma_scale`): `sigma_weight`,
    `sigma_scale`, `sigma_floor`. Shared (they cannot vary across the pack):
    `n_quantiles` (output width), `kernel`, `sigma_objective`, `sigma_param`,
    `min_sigma`.

    `sigma_scale` moves only the Huber transition point to |err| = sigma_scale * sigma;
    the tail gradient stays at full pinball force for any value. `sigma_floor > 0`
    (spread mode only) sets sigma = max(spread, sigma_floor * |resid|), which breaks the
    spread circularity on datasets where the quantiles never fan out.

    `muls_init` is a per-pack-member warm-start offset for logsigma, estimated once
    before training exactly as for `_AdaQRLossPack` (unused when
    `sigma_objective='spread'`).
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        kernel: str = 'huber',
        sigma_objective: str = 'nll',
        sigma_param: str = 'abs',
        sigma_weight: float | list[float] = 1.0,
        sigma_scale: float | list[float] = 1.0,
        sigma_floor: float | list[float] = 0.0,
        min_sigma: float = 1e-4,
        spread_estimator: str = 'pair',
        spread_pairs: float | list[float] = 0.159,
        spread_agg: str = 'mean',
        spread_asymmetric: bool = False,
        center_expectile: list[float] | None = None,
        monotone: bool = False,
        monotone_init_spread: float = 4.0,
        monotone_reference: str = 'left',
        pack_size: int,
        device: torch.device,
    ) -> None:
        assert kernel in ('huber', 'gauss'), f'Unsupported kernel: {kernel}'
        assert sigma_objective in ('nll', 'mse', 'spread'), (
            f'Unsupported sigma_objective: {sigma_objective}'
        )
        assert sigma_param in ('abs', 'softplus'), (
            f'Unsupported sigma_param: {sigma_param}'
        )
        assert spread_estimator in ('pair', 'slope'), (
            f'Unsupported spread_estimator: {spread_estimator}'
        )
        assert spread_agg in ('mean', 'median'), f'Unsupported spread_agg: {spread_agg}'
        assert not spread_asymmetric or sigma_objective == 'spread', (
            'spread_asymmetric makes sense only with sigma_objective="spread"'
        )
        assert pack_size > 0

        def _per_member(value: float | list[float], name: str) -> Tensor:
            if isinstance(value, list):
                assert len(value) == pack_size, f'loss.{name} has a wrong pack size'
            else:
                value = [value] * pack_size
            # (pack, 1) to broadcast against (pack, batch) / (pack, batch, 1) tensors.
            return torch.tensor(value, device=device).float().unsqueeze(-1)

        quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        self.quantiles = quantiles
        self.kernel = kernel
        self.sigma_objective = sigma_objective
        self.sigma_param = sigma_param
        self.sigma_weight = _per_member(sigma_weight, 'sigma_weight')  # (pack, 1)
        self.sigma_scale = _per_member(sigma_scale, 'sigma_scale')  # (pack, 1)
        self.sigma_floor = _per_member(sigma_floor, 'sigma_floor')  # (pack, 1)
        self.min_sigma = min_sigma
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf
        self.spread_estimator = spread_estimator
        self.spread_agg = spread_agg
        self.spread_asymmetric = spread_asymmetric
        self.monotone = monotone
        self.monotone_reference = monotone_reference
        self._inc_bias = _make_monotone_inc_bias(
            monotone, monotone_init_spread, n_quantiles, monotone_reference
        )

        z = (2**0.5) * torch.erfinv(2 * quantiles - 1)  # Phi^-1(tau) on the grid
        self._q_med = int(torch.argmin((quantiles - 0.5).abs()))
        pairs = (
            [float(spread_pairs)]
            if isinstance(spread_pairs, int | float)
            else list(spread_pairs)
        )
        assert pairs and all(0.0 < a < 0.5 for a in pairs), (
            'spread_pairs must be tail fractions in (0, 0.5)'
        )
        # (i_lo, i_hi, c_sym, c_left, c_right); constants from the actual grid.
        self._pair_specs = [
            (
                (i_lo := int(torch.argmin((quantiles - a).abs()))),
                (i_hi := int(torch.argmin((quantiles - (1 - a)).abs()))),
                1.0 / (z[i_hi] - z[i_lo]).item(),
                1.0 / (z[self._q_med] - z[i_lo]).item(),
                1.0 / (z[i_hi] - z[self._q_med]).item(),
            )
            for a in pairs
        ]

        if center_expectile is not None:
            lo, hi = center_expectile
            assert 0.0 < lo < hi < 1.0, 'center_expectile must lie in (0, 1)'
            assert all(
                quantiles[i_lo] <= lo and quantiles[i_hi] >= hi
                for i_lo, i_hi, *_ in self._pair_specs
            ), 'spread_pairs must lie outside the expectile center'
            self.center_mask = (quantiles >= lo) & (quantiles <= hi)
        else:
            self.center_mask = None

        # Slope estimator precomputations (per-side OLS with intercept).
        sel = (
            ~self.center_mask
            if self.center_mask is not None
            else torch.ones_like(quantiles, dtype=torch.bool)
        )
        self._slope_sel = sel
        z_sel = z[sel]
        self._z_slope = z_sel - z_sel.mean()
        self._z_slope_denom = (self._z_slope**2).sum().item()
        self._slope_left = sel & (z < 0)
        self._slope_right = sel & (z > 0)
        z_l, z_r = z[self._slope_left], z[self._slope_right]
        self._z_left = z_l - z_l.mean()
        self._z_right = z_r - z_r.mean()
        self._z_left_denom = (self._z_left**2).sum().item()
        self._z_right_denom = (self._z_right**2).sum().item()

        # Overwritten in main() right after construction (as for `_AdaQRLossPack`).
        self.muls_init = torch.zeros(pack_size, 1, device=device)

    def _agg(self, widths: Tensor) -> Tensor:
        # widths: (..., n_pairs) -> (...,)
        if self.spread_agg == 'median':
            return widths.median(dim=-1).values
        return widths.mean(dim=-1)

    def _spread_sigma(self, preds: Tensor) -> tuple[Tensor, Tensor]:
        # preds: (pack, batch, n_quantiles) -> (sigma_l, sigma_r), each
        # (pack, batch, 1); the two are equal unless spread_asymmetric=True.
        if self.spread_estimator == 'pair':
            if not self.spread_asymmetric:
                widths = torch.stack(
                    [
                        (preds[..., i_hi] - preds[..., i_lo]).abs() * c_sym
                        for i_lo, i_hi, c_sym, _, _ in self._pair_specs
                    ],
                    dim=-1,
                )
                sigma_l = sigma_r = self._agg(widths)
            else:
                lefts = torch.stack(
                    [
                        (preds[..., self._q_med] - preds[..., i_lo]).abs() * c_l
                        for i_lo, _, _, c_l, _ in self._pair_specs
                    ],
                    dim=-1,
                )
                rights = torch.stack(
                    [
                        (preds[..., i_hi] - preds[..., self._q_med]).abs() * c_r
                        for _, i_hi, _, _, c_r in self._pair_specs
                    ],
                    dim=-1,
                )
                sigma_l, sigma_r = self._agg(lefts), self._agg(rights)
        else:
            if not self.spread_asymmetric:
                q = preds[..., self._slope_sel]
                q_c = q - q.mean(dim=-1, keepdim=True)
                sigma = (q_c * self._z_slope).sum(-1) / self._z_slope_denom
                sigma_l = sigma_r = sigma.abs()
            else:
                q_l = preds[..., self._slope_left]
                q_r = preds[..., self._slope_right]
                q_lc = q_l - q_l.mean(dim=-1, keepdim=True)
                q_rc = q_r - q_r.mean(dim=-1, keepdim=True)
                sigma_l = ((q_lc * self._z_left).sum(-1) / self._z_left_denom).abs()
                sigma_r = ((q_rc * self._z_right).sum(-1) / self._z_right_denom).abs()
        return (
            sigma_l.unsqueeze(-1) + self.min_sigma,
            sigma_r.unsqueeze(-1) + self.min_sigma,
        )

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            get_pack_size(self.sigma_scale), remove_pack_idx
        )
        self.sigma_weight = self.sigma_weight[keep_pack_idx].clone()
        self.sigma_scale = self.sigma_scale[keep_pack_idx].clone()
        self.sigma_floor = self.sigma_floor[keep_pack_idx].clone()
        self.muls_init = self.muls_init[keep_pack_idx].clone()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        preds = y_pred[..., :-1].float()  # (pack, batch, n_quantiles)
        if self.monotone:
            # The spread estimator below becomes crossing-free by construction; the
            # .abs() is then a no-op but is kept so that monotone=False stays exact.
            preds = _monotone_expand(preds, self._inc_bias, self.monotone_reference)
        y_true_ = y_true.float().unsqueeze(-1)  # (pack, batch, 1)

        if self.sigma_objective == 'spread':
            sigma_l, sigma_r = self._spread_sigma(preds)  # (pack, batch, 1) each
            floor = self.sigma_floor.unsqueeze(-1)  # (pack, 1, 1)
            if bool((floor > 0).any()):
                resid = y_true_ - preds.detach().mean(dim=-1, keepdim=True)
                sigma_l = torch.where(
                    floor > 0, torch.maximum(sigma_l, floor * resid.abs()), sigma_l
                )
                sigma_r = torch.where(
                    floor > 0, torch.maximum(sigma_r, floor * resid.abs()), sigma_r
                )
        else:
            logsigma = (y_pred[..., -1].float() + self.muls_init).unsqueeze(-1)
            if self.sigma_param == 'softplus':
                # softplus(0) ~= 0.69: no singularity at init, unlike |head| + min_sigma
                sigma = nn.functional.softplus(logsigma) + self.min_sigma
            else:
                sigma = logsigma.abs() + self.min_sigma  # (pack, batch, 1)
            sigma_l = sigma_r = sigma

        err = preds - y_true_  # (pack, batch, n_quantiles)
        tilt = torch.where(err >= 0, self.quantiles, 1.0 - self.quantiles)

        # Quantile branch: Huberized pinball, sigma detached. The `scale` factor in front
        # keeps the tail gradient at pinball force for any sigma_scale; only the
        # transition point moves to |err| = sigma_scale * sigma.
        if self.spread_asymmetric:
            # Per-side sigma by the side the quantile belongs to: (pack, batch, n_q).
            sigma_eff = torch.where(self.quantiles <= 0.5, sigma_l, sigma_r)
        else:
            sigma_eff = sigma_l  # (pack, batch, 1)
        scale = self.sigma_scale.unsqueeze(-1) * sigma_eff.detach()
        z = err / scale
        if self.kernel == 'huber':
            core = torch.where(z.abs() <= 1, 0.5 * z**2, z.abs() - 0.5)
        else:
            core = (2.0 / torch.pi) ** 0.5 * torch.exp(-(z**2) / 2.0) + z * (
                2.0 * self.cdf(z) - 1.0
            )
        loss_matrix = tilt * scale * core
        if self.center_mask is not None:
            loss_matrix = torch.where(self.center_mask, tilt * err**2, loss_matrix)
        # Reduce over batch and quantiles per pack member (keep the pack axis).
        quantile_loss = loss_matrix.mean(dim=(BATCH_DIM, -1))  # (pack,)

        if self.sigma_objective == 'spread':
            return quantile_loss

        # Sigma branch on the mean-of-quantiles residual, preds detached.
        resid = y_true_ - preds.detach().mean(dim=-1, keepdim=True)
        if self.sigma_objective == 'nll':
            sigma_loss = torch.log(sigma) + 0.5 * (resid / sigma) ** 2
        else:
            sigma_loss = (
                torch.log(sigma) - torch.log(resid.abs().clamp_min(self.min_sigma))
            ) ** 2
        sigma_loss = sigma_loss.mean(dim=(BATCH_DIM, -1))  # (pack,)
        return quantile_loss + self.sigma_weight.squeeze(-1) * sigma_loss

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # Drop the logsigma channel, average the quantile predictions.
        preds = y_pred[..., :-1]
        if self.monotone:
            preds = _monotone_expand(preds, self._inc_bias, self.monotone_reference)
        return preds.mean(-1)


class _AdaQRSpreadLossPack:
    """AdaQR distance to a Gaussian target whose scale comes from two quantiles.

    This combines the analytic AdaQR objective from ``_AdaQRLossPack``
    with sigma-Huber's ``sigma_objective='spread'`` idea: the model emits only
    ``n_quantiles`` channels and its conditional scale is estimated as

        sigma = |q_(1-a) - q_a| / (Phi^-1(tau_hi) - Phi^-1(tau_lo)).

    ``a`` defaults to 0.159, so this is the familiar half-width between the roughly
    16th and 84th percentiles, with an exact correction for the quantiles available on
    the actual grid. The Gaussian target is ``N(y_true, sigma_n**2)`` with
    ``sigma_n = noise_scale * sigma`` (``noise_scale=0.5`` by default).

    The returned objective is one half of the conventional adaqr distance, matching
    the scale used by the existing adaqr losses:

        E|X-Y| - 0.5 E|X-X'| - 0.5 E|Y-Y'|.

    For a normal ``Y``, the last term is ``-sigma_n / sqrt(pi)``. ``sigma_n`` is
    detached, as the spread sigma is in sigma-Huber: it defines the target smoothing
    for the current step, but does not add a second gradient path through the two
    quantiles used to estimate it. ``noise_scale`` is deliberately a shared scalar;
    passing a per-pack list (and therefore tuning it per member) is rejected.
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        noise_scale: float = 0.5,
        spread_pair: float = 0.159,
        min_sigma: float = 1e-4,
        monotone: bool = False,
        monotone_init_spread: float = 4.0,
        monotone_reference: str = 'left',
        device: torch.device,
    ) -> None:
        assert n_quantiles >= 2
        assert (
            not isinstance(noise_scale, bool)
            and isinstance(noise_scale, int | float)
            and noise_scale > 0.0
        ), 'loss.noise_scale must be a positive shared scalar (not tuned per member)'
        assert (
            not isinstance(spread_pair, bool)
            and isinstance(spread_pair, int | float)
            and 0.0 < spread_pair < 0.5
        ), 'loss.spread_pair must be a shared scalar in (0, 0.5)'
        assert (
            not isinstance(min_sigma, bool)
            and isinstance(min_sigma, int | float)
            and min_sigma > 0.0
        ), 'loss.min_sigma must be a positive shared scalar'

        quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        z = (2.0**0.5) * torch.erfinv(2.0 * quantiles - 1.0)
        self.quantiles = quantiles
        self.noise_scale = float(noise_scale)
        self.min_sigma = float(min_sigma)
        self.monotone = monotone
        self.monotone_reference = monotone_reference
        self._inc_bias = _make_monotone_inc_bias(
            monotone, monotone_init_spread, n_quantiles, monotone_reference
        )

        self._spread_lo = int(torch.argmin((quantiles - spread_pair).abs()))
        self._spread_hi = int(torch.argmin((quantiles - (1.0 - spread_pair)).abs()))
        assert self._spread_lo < self._spread_hi
        self._spread_normalizer = 1.0 / (z[self._spread_hi] - z[self._spread_lo]).item()
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf

    def _spread_sigma(self, preds: Tensor) -> Tensor:
        sigma = (
            preds[..., self._spread_hi] - preds[..., self._spread_lo]
        ).abs() * self._spread_normalizer + self.min_sigma
        # The target distribution is rebuilt from the current quantile curve, but is
        # held fixed while differentiating this training step (as in sigma-Huber).
        return sigma.detach()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        preds = y_pred.float()  # (pack, batch, n_quantiles)
        if self.monotone:
            preds = _monotone_expand(preds, self._inc_bias, self.monotone_reference)
        y_true = y_true.float()

        sigma_n = self.noise_scale * self._spread_sigma(preds)  # (pack, batch)
        sigma_n_ = sigma_n.unsqueeze(-1)  # (pack, batch, 1)
        errors = preds - y_true.unsqueeze(-1)
        external = (2.0 / torch.pi) ** 0.5 * sigma_n_ * torch.exp(
            -(errors**2) / (2.0 * sigma_n_**2)
        ) + errors * (2.0 * self.cdf(errors / sigma_n_) - 1.0)

        # Each reduction keeps the pack axis, matching the loss-pack contract.
        external_term = external.mean(dim=(BATCH_DIM, -1))
        predicted_self_term = preds.mean(dim=(BATCH_DIM, -1)) - 2.0 * (
            self.quantiles * preds
        ).mean(dim=(BATCH_DIM, -1))
        target_self_term = -sigma_n.mean(dim=BATCH_DIM) / math.sqrt(math.pi)
        return external_term + predicted_self_term + target_self_term

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        preds = y_pred
        if self.monotone:
            preds = _monotone_expand(preds, self._inc_bias, self.monotone_reference)
        return preds.mean(-1)


class _AdaQRGTLossPack:
    """AdaQR distance with fixed per-row sigma supplied by an external producer."""

    def __init__(
        self,
        *,
        n_quantiles: int,
        sigma_train: Tensor,
        sigma_scale: float | list[float],
        min_sigma: float = 1e-5,
        pack_size: int,
        device: torch.device,
    ) -> None:
        assert n_quantiles > 1
        assert pack_size > 0
        if sigma_train.ndim != 1:
            raise ValueError('External train sigma must be one-dimensional')
        if not torch.isfinite(sigma_train).all() or not (sigma_train > 0.0).all():
            raise ValueError('External train sigma must be finite and positive')
        if not isinstance(sigma_scale, list):
            sigma_scale = [sigma_scale] * pack_size
        if len(sigma_scale) != pack_size:
            raise ValueError('sigma_scale must have one value per pack member')

        self.quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        self.sigma_train = sigma_train.to(device=device, dtype=torch.float32)
        self.sigma_scale = torch.as_tensor(
            sigma_scale, device=device, dtype=torch.float32
        )
        if (
            not torch.isfinite(self.sigma_scale).all()
            or not (self.sigma_scale > 0.0).all()
        ):
            raise ValueError('sigma_scale must be finite and positive')
        if not math.isfinite(min_sigma) or min_sigma < 0.0:
            raise ValueError('min_sigma must be finite and non-negative')
        self.min_sigma = min_sigma
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            get_pack_size(self.sigma_scale), remove_pack_idx
        )
        self.sigma_scale = self.sigma_scale[keep_pack_idx].clone()

    def get_batch_sigma(self, batch_idx: Tensor) -> Tensor:
        if batch_idx.ndim != 2 or batch_idx.shape[0] != len(self.sigma_scale):
            raise ValueError('batch_idx must have shape (pack, batch)')
        return (
            self.sigma_train[batch_idx] * self.sigma_scale.unsqueeze(-1)
        ).abs() + self.min_sigma

    def __call__(self, y_pred: Tensor, y_true: Tensor, *, batch_idx: Tensor) -> Tensor:
        preds = y_pred.float()
        y_true = y_true.float()
        if preds.ndim != 3 or preds.shape[-1] != len(self.quantiles):
            raise ValueError(
                'AdaQR GT predictions must have shape (pack, batch, n_quantiles)'
            )
        if y_true.shape != preds.shape[:2] or batch_idx.shape != preds.shape[:2]:
            raise ValueError('Targets and batch indices must match prediction rows')

        sigma = self.get_batch_sigma(batch_idx)
        sigma_ = sigma.unsqueeze(-1)
        errors = preds - y_true.unsqueeze(-1)
        external = (2.0 / torch.pi) ** 0.5 * sigma_ * torch.exp(
            -(errors**2) / (2.0 * sigma_**2)
        ) + errors * (2.0 * self.cdf(errors / sigma_) - 1.0)
        external_term = external.mean(dim=(BATCH_DIM, -1))
        internal_term = preds.mean(dim=(BATCH_DIM, -1)) - 2.0 * (
            self.quantiles * preds
        ).mean(dim=(BATCH_DIM, -1))
        return external_term + internal_term

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        return y_pred.mean(-1)


def _gaussian_crps_zero_mean(residual: Tensor, scale: Tensor) -> Tensor:
    z = residual / scale
    return (
        residual * torch.erf(z / math.sqrt(2.0))
        + math.sqrt(2.0 / math.pi) * scale * torch.exp(-0.5 * z.square())
        - scale / math.sqrt(math.pi)
    )


class _AdaQRLossPack:
    """AdaQR loss, pack-aware. Ported from bin/adaqr.py.

    Expects `y_pred` (pack, batch, n_quantiles + 1) — the last channel is per-sample
    `raw_sigma` — and `y_true` (pack, batch); returns a per-pack-member loss vector (pack,).
    The adaqr loss has batch-global terms, so it is reduced per member (over batch and
    quantiles) while keeping the pack axis. `regularizer` is per-pack-member (like `sigma`
    in `_HLGaussLossPack`/`lr` in `AdamWPack`): a single float broadcasts to the whole
    pack, or a list with one value per pack member (e.g. sampled independently via
    `sampler.space.loss.regularizer`). `muls_init` is likewise per-pack-member: a
    warm-start offset for raw_sigma, estimated once before training from each member's
    own predictions (see main()) — it naturally differs across the pack, since each
    member has its own weights (and, when tuned, its own `regularizer`).

    `sigma_coef` (optional) splits the objective into a "quantile" part (with raw_sigma
    detached) and a "sigma" part (with the quantile predictions detached) and reweights
    the latter:

        loss = L(preds, sg(raw_sigma)) + sigma_coef * L(sg(preds), raw_sigma)

    With `sigma_coef == 1` this is gradient-equivalent to the plain loss (the value is
    simply doubled), while smaller values (e.g. "1/q", "1/sqrt(q)") slow down the sigma
    head relative to the quantile head. `sigma_coef` is per-pack-member exactly like
    `regularizer`: a single value (str/float) broadcasts, or a list with one value per
    pack member (e.g. sampled via `sampler.space.loss.sigma_coef`). When `sigma_coef`
    is omitted (the default), the original single-pass loss is used unchanged.
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        y_true_mode: str = 'identity',
        regularizer: float | list[float] | None = None,
        kappa: float | list[float] | None = None,
        sigma_coef: None | str | float | list[str | float] = None,
        raw_sigma_scale: float = 1.0,
        min_sigma: float = 1e-4,
        scale_reg_weight: float | list[float] = 0.0,
        scale_reg_balance: str = 'equal_loss',
        pack_size: int,
        device: torch.device,
    ) -> None:
        assert y_true_mode in ('identity', 'sampling'), (
            f'Unsupported y_true_mode: {y_true_mode}'
        )
        assert pack_size > 0

        if kappa is not None:
            assert regularizer is None
            if isinstance(kappa, list):
                regularizer = [k / math.sqrt(1 + k**2) for k in kappa]
            else:
                regularizer = kappa / math.sqrt(1 + kappa**2)

        if isinstance(regularizer, list):
            assert len(regularizer) == pack_size
        else:
            regularizer = [regularizer] * pack_size

        self.quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        self._spread_lo = int(torch.argmin((self.quantiles - 0.159).abs()))
        self._spread_hi = int(torch.argmin((self.quantiles - 0.841).abs()))
        self.y_true_mode = y_true_mode
        self.regularizer = torch.tensor(regularizer, device=device).float()  # (pack,)
        self.raw_sigma_scale = raw_sigma_scale
        self.min_sigma = min_sigma
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf

        if not isinstance(scale_reg_weight, list):
            scale_reg_weight = [scale_reg_weight] * pack_size
        assert len(scale_reg_weight) == pack_size
        self.scale_reg_weight = torch.tensor(
            scale_reg_weight, device=device
        ).float()
        if (self.scale_reg_weight < 0.0).any():
            raise ValueError('scale_reg_weight must be nonnegative')
        if scale_reg_balance not in ('equal_loss', 'output_jacobian'):
            raise ValueError(f'Unknown scale-reg balance: {scale_reg_balance!r}')
        scale_reg_enabled = self.scale_reg_weight > 0.0
        self.has_scale_reg = bool(scale_reg_enabled.any())
        invalid_regularizer = (self.regularizer <= 0.0) | (self.regularizer >= 1.0)
        if (scale_reg_enabled & invalid_regularizer).any():
            raise ValueError('The scale scale_reg requires 0 < regularizer < 1')
        if self.has_scale_reg and y_true_mode != 'identity':
            raise ValueError('The scale scale_reg requires y_true_mode="identity"')
        self.scale_reg_balance = scale_reg_balance
        normal = torch.distributions.Normal(0.0, 1.0)
        normal_scores = normal.icdf(self.quantiles)
        self._normal_score_rms = normal_scores.square().mean().sqrt()
        self._n_quantiles_sqrt = math.sqrt(n_quantiles)

        # Per-pack-member weight of the sigma-only part of the loss; None == disabled.
        if sigma_coef is None:
            self.sigma_coef = None
        else:
            if not isinstance(sigma_coef, list):
                sigma_coef = [sigma_coef] * pack_size
            assert len(sigma_coef) == pack_size
            self.sigma_coef = torch.tensor(
                [_resolve_sigma_coef(x, n_quantiles) for x in sigma_coef],
                device=device,
            ).float()  # (pack,)

        # Overwritten in main() right after construction; (pack, 1) to broadcast
        # against `y_pred[..., -1]` of shape (pack, batch).
        self.muls_init = torch.zeros(pack_size, 1, device=device)

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            get_pack_size(self.regularizer), remove_pack_idx
        )
        self.regularizer = self.regularizer[keep_pack_idx].clone()
        self.scale_reg_weight = self.scale_reg_weight[keep_pack_idx].clone()
        self.has_scale_reg = bool((self.scale_reg_weight > 0.0).any())
        self.muls_init = self.muls_init[keep_pack_idx].clone()
        if self.sigma_coef is not None:
            self.sigma_coef = self.sigma_coef[keep_pack_idx].clone()

    def _compute_scale_reg(
        self, *, preds: Tensor, y_true: Tensor, sigma: Tensor
    ) -> Tensor:
        scale_reg_regularizer = torch.where(
            self.scale_reg_weight > 0.0,
            self.regularizer,
            torch.full_like(self.regularizer, 0.5),
        )
        q_factor = (1.0 - scale_reg_regularizer.square()).sqrt()
        center = preds.mean(dim=-1, keepdim=True)
        quantile_rms = ((preds - center).square().mean(dim=-1) + 1e-12).sqrt()
        q_scale = (
            q_factor.unsqueeze(-1) * quantile_rms / self._normal_score_rms
        ).clamp_min(1e-6)
        sigma_scale = (q_factor / scale_reg_regularizer).unsqueeze(-1) * sigma
        sigma_scale = sigma_scale.clamp_min(1e-6)
        residual = y_true - center.squeeze(-1).detach()

        q_weight: float | Tensor = 1.0
        sigma_weight: float | Tensor = 1.0
        if self.scale_reg_balance == 'output_jacobian':
            q_weight = self._normal_score_rms * self._n_quantiles_sqrt
            sigma_weight = scale_reg_regularizer
        return 0.5 * (
            q_weight * _gaussian_crps_zero_mean(residual, q_scale).mean(BATCH_DIM)
            + sigma_weight
            * _gaussian_crps_zero_mean(residual, sigma_scale).mean(BATCH_DIM)
        )

    def _compute_loss(
        self, *, preds: Tensor, y_true: Tensor, raw_sigma: Tensor
    ) -> Tensor:
        sigma = raw_sigma.abs() + self.min_sigma  # (pack, batch)
        sigma_ = sigma.unsqueeze(-1)  # (pack, batch, 1)
        if self.y_true_mode == 'sampling':
            errors = preds - (torch.randn_like(preds) * sigma_ + y_true.unsqueeze(-1))
        else:  # 'identity'
            errors = preds - y_true.unsqueeze(-1)  # (pack, batch, n_quantiles)

        external = (2.0 / torch.pi) ** 0.5 * sigma_ * torch.exp(
            -(errors**2) / 2.0 / sigma_**2
        ) + errors * (
            2.0 * self.cdf(errors / sigma_) - 1.0
        )  # (pack, batch, n_quantiles)

        # Reduce over batch and quantiles per pack member (keep the pack axis).
        external_term = external.mean(dim=(BATCH_DIM, -1))  # (pack,)
        internal_terms = (
            preds.mean(dim=(BATCH_DIM, -1))
            - 2.0 * (self.quantiles * preds).mean(dim=(BATCH_DIM, -1))
            - sigma.mean(dim=BATCH_DIM) / (torch.pi**0.5) * self.regularizer
        )  # (pack,)
        result = external_term + internal_terms
        if self.has_scale_reg:
            result = result + self.scale_reg_weight * self._compute_scale_reg(
                preds=preds, y_true=y_true, sigma=sigma
            )
        return result  # (pack,)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        preds = y_pred[..., :-1].float()  # (pack, batch, n_quantiles)
        raw_sigma = (
            y_pred[..., -1].float() + self.muls_init
        ) * self.raw_sigma_scale  # (pack, batch)
        y_true = y_true.float()

        if self.sigma_coef is None:
            return self._compute_loss(preds=preds, y_true=y_true, raw_sigma=raw_sigma)

        quantile_loss = self._compute_loss(
            preds=preds, y_true=y_true, raw_sigma=raw_sigma.detach()
        )
        sigma_loss = self._compute_loss(
            preds=preds.detach(), y_true=y_true, raw_sigma=raw_sigma
        )
        return quantile_loss + self.sigma_coef * sigma_loss  # (pack,)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # Drop the raw_sigma channel, average the quantile predictions.
        return y_pred[..., :-1].mean(-1)

    def get_uncertainty_predictions(
        self, y_pred: Tensor, *, muls_init: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Return learned and quantile-derived scales in standardized target units."""
        preds = y_pred[..., :-1].float()
        offset = self.muls_init if muls_init is None else muls_init
        raw_sigma = (y_pred[..., -1].float() + offset) * self.raw_sigma_scale
        learned_sigma = raw_sigma.abs() + self.min_sigma
        quantile_spread = (
            0.5 * (preds[..., self._spread_hi] - preds[..., self._spread_lo]).abs()
        )
        quantile_std = (
            (preds - preds.mean(dim=-1, keepdim=True)).square().mean(dim=-1).sqrt()
        )
        return {
            'learned_sigma': learned_sigma,
            'quantile_spread': quantile_spread,
            'quantile_std': quantile_std,
        }


class _GaussianTailAdaQRLossPack(_AdaQRLossPack):
    """Pack-aware AdaQR with free centers and independent Gaussian tails."""

    def __init__(self, *, n_quantiles: int, **kwargs: Any) -> None:
        if type(n_quantiles) is not int or n_quantiles < 18:
            raise ValueError('n_quantiles must be an integer >= 18 (C//18 tail values)')
        self.n_center_quantiles = n_quantiles
        self.n_tail_quantiles = n_quantiles // 18
        super().__init__(
            n_quantiles=n_quantiles + 2 * self.n_tail_quantiles,
            **kwargs,
        )
        normal = torch.distributions.Normal(0.0, 1.0)
        self.left_icdf_values = normal.icdf(self.quantiles[: self.n_tail_quantiles])
        self.right_icdf_values = normal.icdf(self.quantiles[-self.n_tail_quantiles :])

    def reconstruct_quantiles(self, raw: Tensor) -> Tensor:
        if raw.shape[-1] != self.n_center_quantiles + 4:
            raise ValueError(
                'Expected C center values followed by four tail parameters'
            )
        raw = raw.float()
        centers = raw[..., : self.n_center_quantiles]
        left_mean = raw[..., -4:-3]
        left_scale = raw[..., -3:-2].abs() + self.min_sigma
        right_mean = raw[..., -2:-1]
        right_scale = raw[..., -1:].abs() + self.min_sigma
        return torch.cat(
            (
                left_mean + left_scale * self.left_icdf_values,
                centers,
                right_mean + right_scale * self.right_icdf_values,
            ),
            dim=-1,
        )

    def _as_adaqr_output(self, y_pred: Tensor) -> Tensor:
        quantiles = self.reconstruct_quantiles(y_pred[..., :-1])
        return torch.cat((quantiles, y_pred[..., -1:].float()), dim=-1)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return super().__call__(self._as_adaqr_output(y_pred), y_true)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        return self.reconstruct_quantiles(y_pred[..., :-1]).mean(-1)

    def get_uncertainty_predictions(
        self, y_pred: Tensor, *, muls_init: Tensor | None = None
    ) -> dict[str, Tensor]:
        return super().get_uncertainty_predictions(
            self._as_adaqr_output(y_pred), muls_init=muls_init
        )


class _AdaQRLossV9Pack:
    """AdaQR loss (v9), pack-aware. Ported from bin/adaqr_dist_v9.py.

    The v9 counterpart of `_AdaQRLossPack`: same AdaQR objective and
    the same contract (`y_pred` (pack, batch, n_quantiles + 1) with the last channel a
    per-sample `logsigma`; `y_true` (pack, batch); returns a per-pack-member loss (pack,)).
    The only substantive difference is how the per-sample data uncertainty is
    parameterized: v9 treats the last channel as a genuine *log*-std and uses
    `sigma = exp(logsigma).clamp(min=min_sigma)`, whereas AdaQR uses
    `sigma = |logsigma| + min_sigma`. Both keep the small `min_sigma` floor for
    numerical stability (for v9 it stops exp(logsigma) from underflowing to 0, which
    would blow up `errors / sigma`). `regularizer` is per-pack-member exactly as in
    `_AdaQRLossPack` (a float broadcasts, or a per-member list from
    `sampler.space.loss.regularizer`). `muls_init` is likewise per-pack-member, but for v9
    the warm-start is disabled by default (use_muls_init=False in bin/adaqr_dist_v9.py,
    and its configs don't set it), so it stays zero unless main() fills it in; when
    enabled, its offset is `log(muls_init_add)` (not `muls_init_add` directly, as in
    AdaQR).

    `sigma_coef` (optional) works exactly as in `_AdaQRLossPack`: it splits the
    objective into a "quantile" part (with logsigma detached) and a "sigma" part (with
    the quantile predictions detached) and reweights the latter:

        loss = L(preds, sg(logsigma)) + sigma_coef * L(sg(preds), logsigma)

    With `sigma_coef == 1` this is gradient-equivalent to the plain loss (the value is
    simply doubled), while smaller values (e.g. "1/q", "1/sqrt(q)") slow down the sigma
    head relative to the quantile head. It is per-pack-member like `regularizer`: a single
    value (str/float) broadcasts, or a list with one value per pack member (e.g. sampled
    via `sampler.space.loss.sigma_coef`). When omitted (the default), the original
    single-pass loss is used unchanged.
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        y_true_mode: str = 'identity',
        regularizer: float | list[float] = 1.0,
        sigma_coef: None | str | float | list[str | float] = None,
        raw_sigma_scale: float = 1.0,
        min_sigma: float = 1e-4,
        max_sigma: float | None = None,
        pack_size: int,
        device: torch.device,
    ) -> None:
        assert y_true_mode in ('identity', 'sampling'), (
            f'Unsupported y_true_mode: {y_true_mode}'
        )
        assert pack_size > 0
        if isinstance(regularizer, list):
            assert len(regularizer) == pack_size
        else:
            regularizer = [regularizer] * pack_size

        self.quantiles = torch.linspace(0.001, 0.999, n_quantiles, device=device)
        self.y_true_mode = y_true_mode
        self.regularizer = torch.tensor(regularizer, device=device).float()  # (pack,)
        self.raw_sigma_scale = raw_sigma_scale
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        # validate_args=False as a safety net: the `min_sigma` floor (see __call__)
        # already stops `errors / sigma` from blowing up, but disabling validation keeps a
        # pathological pack member's loss finite/NaN (and thus stoppable) rather than
        # crashing the whole pack. bin/adaqr_dist_v9.py uses the same Normal(...).cdf.
        self.cdf = torch.distributions.Normal(0.0, 1.0, validate_args=False).cdf

        # Per-pack-member weight of the sigma-only part of the loss; None == disabled.
        if sigma_coef is None:
            self.sigma_coef = None
        else:
            if not isinstance(sigma_coef, list):
                sigma_coef = [sigma_coef] * pack_size
            assert len(sigma_coef) == pack_size
            self.sigma_coef = torch.tensor(
                [_resolve_sigma_coef(x, n_quantiles) for x in sigma_coef],
                device=device,
            ).float()  # (pack,)

        # Zero by default (v9's use_muls_init defaults to False); optionally overwritten
        # in main() when use_muls_init is enabled. (pack, 1) to broadcast against
        # `y_pred[..., -1]` of shape (pack, batch).
        self.muls_init = torch.zeros(pack_size, 1, device=device)

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            get_pack_size(self.regularizer), remove_pack_idx
        )
        self.regularizer = self.regularizer[keep_pack_idx].clone()
        self.muls_init = self.muls_init[keep_pack_idx].clone()
        if self.sigma_coef is not None:
            self.sigma_coef = self.sigma_coef[keep_pack_idx].clone()

    def _compute_loss(
        self, *, preds: Tensor, y_true: Tensor, logsigma: Tensor
    ) -> Tensor:
        # v9: log-parameterized std, floored at `min_sigma` (the analog of AdaQR's
        # `min_sigma`): exp(logsigma) would otherwise underflow to 0 and make
        # `errors / sigma` blow up to inf/NaN. Equivalent to
        # `exp(logsigma).clamp(min=min_sigma)`. No upper clamp on logsigma (the `max`
        # guard was removed). Healthy logsigma is O(1), so this is inactive in practice.
        sigma = torch.exp(
            logsigma.clamp(
                min=math.log(self.min_sigma),
                max=math.log(self.max_sigma) if self.max_sigma is not None else None,
            )
        )  # (pack, batch)
        sigma_ = sigma.unsqueeze(-1)  # (pack, batch, 1)
        if self.y_true_mode == 'sampling':
            errors = preds - (torch.randn_like(preds) * sigma_ + y_true.unsqueeze(-1))
        else:  # 'identity'
            errors = preds - y_true.unsqueeze(-1)  # (pack, batch, n_quantiles)

        external = (2.0 / torch.pi) ** 0.5 * sigma_ * torch.exp(
            -(errors**2) / 2.0 / sigma_**2
        ) + errors * (
            2.0 * self.cdf(errors / sigma_) - 1.0
        )  # (pack, batch, n_quantiles)

        # Reduce over batch and quantiles per pack member (keep the pack axis).
        external_term = external.mean(dim=(BATCH_DIM, -1))  # (pack,)
        internal_terms = (
            preds.mean(dim=(BATCH_DIM, -1))
            - 2.0 * (self.quantiles * preds).mean(dim=(BATCH_DIM, -1))
            - sigma.mean(dim=BATCH_DIM) / (torch.pi**0.5) * self.regularizer
        )  # (pack,)
        return external_term + internal_terms  # (pack,)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        preds = y_pred[..., :-1].float()  # (pack, batch, n_quantiles)
        logsigma = (
            y_pred[..., -1].float() + self.muls_init
        ) * self.raw_sigma_scale  # (pack, batch)
        y_true = y_true.float()

        if self.sigma_coef is None:
            return self._compute_loss(preds=preds, y_true=y_true, logsigma=logsigma)

        quantile_loss = self._compute_loss(
            preds=preds, y_true=y_true, logsigma=logsigma.detach()
        )
        sigma_loss = self._compute_loss(
            preds=preds.detach(), y_true=y_true, logsigma=logsigma
        )
        return quantile_loss + self.sigma_coef * sigma_loss  # (pack,)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # Drop the logsigma channel, average the quantile predictions.
        return y_pred[..., :-1].mean(-1)


class _AdaQRLossNightlyPack(_AdaQRLossV9Pack):
    """Pack-aware port of bin.adaqr_dist_vNightly.AdaQRLoss."""

    def _compute_loss(
        self, *, preds: Tensor, y_true: Tensor, logsigma: Tensor
    ) -> Tensor:
        variance = torch.exp(2.0 * logsigma) + self.min_sigma
        sigma = variance.sqrt()
        sigma_ = sigma.unsqueeze(-1)
        if self.y_true_mode == 'sampling':
            errors = preds - (torch.randn_like(preds) * sigma_ + y_true.unsqueeze(-1))
        else:
            errors = preds - y_true.unsqueeze(-1)

        external = (2.0 / torch.pi) ** 0.5 * sigma_ * torch.exp(
            -errors.square() / (2.0 * variance.unsqueeze(-1))
        ) + errors * (2.0 * self.cdf(errors / sigma_) - 1.0)
        external_term = external.mean(dim=(BATCH_DIM, -1))
        regularizer_sigma = (
            torch.exp(2.0 * logsigma)
            + self.regularizer.square().unsqueeze(-1) * self.min_sigma
        ).sqrt()
        internal_terms = (
            preds.mean(dim=(BATCH_DIM, -1))
            - 2.0 * (self.quantiles * preds).mean(dim=(BATCH_DIM, -1))
            - self.regularizer
            * regularizer_sigma.mean(dim=BATCH_DIM)
            / math.sqrt(math.pi)
        )
        return external_term + internal_terms


class _HLGaussLossPack:
    """HL-Gauss (histogram) loss, pack-aware. Vendored from vendor/hl_loss.py.

    The model emits `num_bins` logits per sample; the target is a discretized Gaussian
    over a fixed support [min_value, max_value] (shared across the pack, not tuned,
    since `num_bins` sets the model's output width). `sigma`, in contrast, is a
    per-pack-member hyperparameter, exactly like `lr`/`weight_decay` in `AdamWPack`:
    a single float broadcasts to the whole pack, or a list with one value per pack
    member (e.g. sampled independently via `sampler.space.loss.sigma`, matching how
    `bin/hl_gaussian.py` tunes `sigma` per trial).
    Loss = cross-entropy(log_softmax(logits), soft target), reduced per pack member.
    Prediction = softmax(logits) · bin_centers (the predicted distribution's expected
    value). The support is computed once in main() from the training targets.
    """

    def __init__(
        self,
        *,
        num_bins: int,
        min_value: float,
        max_value: float,
        sigma: float | list[float],
        pack_size: int,
        clamp_to_range: bool = True,
        eps: float = 1e-10,
        device: torch.device,
    ) -> None:
        assert num_bins > 1
        assert min_value < max_value
        assert pack_size > 0
        if isinstance(sigma, list):
            assert len(sigma) == pack_size
            assert all(x > 0.0 for x in sigma)
        else:
            assert sigma > 0.0
            sigma = [sigma] * pack_size

        support = torch.linspace(
            min_value, max_value, num_bins + 1, device=device
        ).float()
        self.support = support
        self.centers = (support[:-1] + support[1:]) / 2.0
        sigma_tensor = torch.tensor(sigma, device=device).float()
        self.sigma_times_sqrt_two = (math.sqrt(2.0) * sigma_tensor).view(-1, 1, 1)
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.clamp_to_range = clamp_to_range
        self.eps = eps

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            get_pack_size(self.sigma_times_sqrt_two), remove_pack_idx
        )
        self.sigma_times_sqrt_two = self.sigma_times_sqrt_two[keep_pack_idx].clone()

    def _target_probs(self, target: Tensor) -> Tensor:
        # target (...,) -> (..., num_bins): discretized Gaussian (Imani et al. 2018).
        cdf = torch.erf(
            (self.support - target.unsqueeze(-1)) / self.sigma_times_sqrt_two
        )
        z = (cdf[..., -1] - cdf[..., 0]).clamp(min=self.eps)
        bin_probs = (cdf[..., 1:] - cdf[..., :-1]).clamp(min=0.0)
        return bin_probs / z.unsqueeze(-1)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_pred: (pack, batch, num_bins) logits; y_true: (pack, batch).
        logits = y_pred.float()
        target = y_true.float()
        if self.clamp_to_range:
            target = target.clamp(self.min_value, self.max_value)
        target_probs = self._target_probs(target)  # (pack, batch, num_bins)
        # Soft-target cross-entropy per sample, then per-pack-member mean over the batch.
        cross_entropy = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(-1)
        return cross_entropy.mean(BATCH_DIM)  # (pack,)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # (pack, batch, num_bins) -> (pack, batch): expected value of the predicted pmf.
        probs = torch.softmax(y_pred.float(), dim=-1)
        return (probs * self.centers).sum(-1)


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Mixed loss: one loss type per pack member
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# The head is split into disjoint sections, so that loss types with different output
# requirements can coexist in one pack (`ModelPack` has a single shared output layer):
#
#     [0 : n_q]                     quantiles       (quantile, sigma_huber, adaqr*, mse)
#     [n_q]                         logsigma        (sigma_huber, adaqr*)
#     [n_q + 1 : n_q + 1 + n_bins]  HL-Gauss logits (hl_gauss; present only if requested)
#
# Loss types that additionally consume the logsigma channel (see `_member_input`);
# 'hl_gauss' is handled separately, since its slice does not start at zero, and every
# other type reads the quantile section alone.
_MIXED_LOGSIGMA_TYPES = ('sigma_huber', 'adaqr', 'adaqr_v9')


def _mixed_head_width(
    n_quantiles: int, types: Iterable[str], hl_num_bins: None | int
) -> int:
    """Total head width required by a set of member loss types."""
    width = n_quantiles + 1  # quantiles + logsigma (always emitted, cheap)
    if 'hl_gauss' in set(types):
        assert hl_num_bins is not None, (
            'loss.hl_num_bins is required when "hl_gauss" is among the mixed loss types'
        )
        width += hl_num_bins
    return width


class _MixedLossPack:
    """Portfolio loss: each pack member is trained with its own loss type.

    `types` is the per-pack-member list of loss types (length `pack_size`), produced by
    the `loss_config` merge in main() from `sampler.space.loss.type`
    (["_tune_", "categorical", [...]]) or by explicit per-member `configs`. The base
    `loss.type` must be "mixed", because the head width is derived from it before the
    per-member merge happens (see the n_classes computation in main()).

    Rationale: the best loss is dataset-dependent (mse on some, adaqr/quantile on
    others), and the greedy online ensemble already extracts more from member diversity
    than any single loss gains on its own. A portfolio pack turns "which loss wins here"
    into diversity the ensemble can arbitrage, instead of a choice made up front.

    Sections of the head that a member does not consume receive no gradient from the
    loss (only weight decay acts on them) — see `_MIXED_SECTION_END` above.

    Sub-losses are built once per unique type at the FULL pack size with the full
    per-member parameter lists (entries belonging to members of other types are junk and
    get masked out). `__call__` evaluates each sub-loss on the whole pack and combines
    them with float (pack,) masks: `out = sum_t mask_t * sub_t(...)`. Every sub-loss
    reduces per member (none has cross-member terms), so member i's gradient is exactly
    the gradient of its own loss.

    Per-member losses live on different scales (pinball summed over n_q vs MSE vs
    cross-entropy). That is harmless here: main() sums the per-member losses, so the
    gradients stay independent, and the optimizer and gradient clipping are per-member.
    """

    _MEMBER_CLASSES = {
        'quantile': _QuantileLossPack,
        'sigma_huber': _SigmaHuberLossPack,
        'adaqr_spread': _AdaQRSpreadLossPack,
        'adaqr': _AdaQRLossPack,
        'adaqr_v9': _AdaQRLossV9Pack,
        'hl_gauss': _HLGaussLossPack,
    }
    # 'mse' is implemented inline: the native branch returns a bare closure without
    # `pack_remove`/`get_scalar_prediction`, and mse members must train on the same
    # mean-over-quantiles readout that every other member is evaluated with.
    _MSE_TYPE = 'mse'
    # `_HLGaussLossPack` names its bin count `num_bins`; the config key is `hl_num_bins`.
    _RENAMED_PARAMS = {'hl_gauss': {'num_bins': 'hl_num_bins'}}

    def __init__(
        self,
        *,
        types: list[str],
        loss_config: KWArgs,
        pack_size: int,
        device: torch.device,
    ) -> None:
        assert pack_size > 0
        assert len(types) == pack_size, (
            f'loss.type has {len(types)} entries, expected one per pack member'
            f' ({pack_size})'
        )
        assert all(isinstance(x, str) for x in types), (
            'mixed loss types must be strings'
        )
        known = {*self._MEMBER_CLASSES, self._MSE_TYPE}
        unknown = sorted(set(types) - known)
        assert not unknown, (
            f'Unknown mixed loss types: {unknown} (known: {sorted(known)})'
        )

        n_quantiles = loss_config.get('n_quantiles')
        assert isinstance(n_quantiles, int), (
            'loss.n_quantiles must be a shared int for the mixed loss: it defines the'
            ' quantile section of the head'
        )
        assert not loss_config.get('monotone', False), (
            'loss.monotone is not supported by the mixed loss yet: the adaqr members'
            ' are not monotone-parameterized, so the shared readout would be wrong'
        )
        if 'sigma_huber' in types:
            assert loss_config.get('sigma_param', 'abs') != 'softplus', (
                'sigma_param="softplus" is unsupported in a mixed pack: the muls_init'
                ' warm-start estimator assumes sigma = |logsigma| + min_sigma'
            )

        self.n_quantiles = n_quantiles
        self.types = list(types)
        self._hl_slice: None | slice = None
        if 'hl_gauss' in types:
            hl_num_bins = loss_config['hl_num_bins']
            self._hl_slice = slice(n_quantiles + 1, n_quantiles + 1 + hl_num_bins)

        self.sub_losses: dict[str, Any] = {}
        for loss_type in dict.fromkeys(types):  # unique, order-preserving
            if loss_type == self._MSE_TYPE:
                self.sub_losses[loss_type] = None
                continue
            member_class = self._MEMBER_CLASSES[loss_type]
            # Filter by the constructor signature: irrelevant keys for this type (e.g.
            # `sigma_objective` for adaqr, `regularizer` for quantile) must not reach it,
            # and the existing loss classes are intentionally left untouched.
            allowed = set(inspect.signature(member_class.__init__).parameters) - {
                'self',
                'device',
            }
            available = {**loss_config, 'pack_size': pack_size}
            for target, source in self._RENAMED_PARAMS.get(loss_type, {}).items():
                if source in available:
                    available[target] = available[source]
            params = {k: v for k, v in available.items() if k in allowed}
            self.sub_losses[loss_type] = member_class(device=device, **params)

        self.masks = {
            loss_type: torch.tensor(
                [float(x == loss_type) for x in types], device=device
            )
            for loss_type in self.sub_losses
        }
        # Overwritten in main() right after construction; the setter propagates it.
        self._muls_init = torch.zeros(pack_size, 1, device=device)

    @property
    def muls_init(self) -> Tensor:
        return self._muls_init

    @muls_init.setter
    def muls_init(self, value: Tensor) -> None:
        self._muls_init = value
        for sub_loss in self.sub_losses.values():
            if sub_loss is not None and hasattr(sub_loss, 'muls_init'):
                sub_loss.muls_init = value

    def pack_remove(self, remove_pack_idx: Tensor) -> None:
        keep_pack_idx = vendor.tabpack.nn.make_keep_pack_idx(
            len(self.types), remove_pack_idx
        )
        self.types = [self.types[i] for i in keep_pack_idx.tolist()]
        self.masks = {k: v[keep_pack_idx].clone() for k, v in self.masks.items()}
        self._muls_init = self._muls_init[keep_pack_idx].clone()
        for sub_loss in self.sub_losses.values():
            if sub_loss is not None and hasattr(sub_loss, 'pack_remove'):
                sub_loss.pack_remove(remove_pack_idx)

    def _member_input(self, y_pred: Tensor, loss_type: str) -> Tensor:
        if loss_type == 'hl_gauss':
            assert self._hl_slice is not None
            return y_pred[..., self._hl_slice]
        if loss_type in _MIXED_LOGSIGMA_TYPES:
            return y_pred[..., : self.n_quantiles + 1]
        return y_pred[..., : self.n_quantiles]

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        out = torch.zeros(
            get_pack_size(y_pred), dtype=torch.float32, device=y_pred.device
        )
        for loss_type, sub_loss in self.sub_losses.items():
            member_input = self._member_input(y_pred, loss_type)
            if sub_loss is None:  # mse, on the shared readout
                prediction = member_input.float().mean(-1)  # (pack, batch)
                sub_out = (prediction - y_true.float()).square().mean(BATCH_DIM)
            else:
                sub_out = sub_loss(member_input, y_true)  # (pack,)
            out = out + self.masks[loss_type] * sub_out
        return out  # (pack,)

    def get_scalar_prediction(self, y_pred: Tensor) -> Tensor:
        # Per-member readout: HL-Gauss members predict the mean of their categorical
        # distribution, every other type the mean of its quantiles. A single shared
        # readout would make the HL members' metrics meaningless and the greedy
        # ensemble's comparisons invalid.
        prediction = y_pred[..., : self.n_quantiles].mean(-1)  # (pack, batch)
        hl_loss = self.sub_losses.get('hl_gauss')
        if hl_loss is not None:
            hl_prediction = hl_loss.get_scalar_prediction(
                self._member_input(y_pred, 'hl_gauss')
            )
            hl_mask = self.masks['hl_gauss'].bool().unsqueeze(-1)  # (pack, 1)
            prediction = torch.where(hl_mask, hl_prediction, prediction)
        return prediction


def _make_loss_fn_pack(
    task_type: TaskType,
    loss_config: None | KWArgs = None,
    device: None | torch.device = None,
    pack_size: None | int = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    # Custom regression losses (mse is the native default, handled below).
    loss_type = (loss_config or {}).get('type', 'mse')

    # Loss-config keys that are consumed by main() (warm-start of muls_init), not by the
    # adaqr loss constructors.
    _ADAQR_MAIN_ONLY_KEYS = (
        'type',
        'quantile_init',
        'muls_init_add',
        'use_muls_init',
        'muls_init_offset_fix',
    )

    # Mixed loss: `type` is a per-pack-member list after the `loss_config` merge in
    # main(). This must precede the branch below, where a list would fall through every
    # string comparison and raise "Unknown regression loss type".
    if task_type == TaskType.REGRESSION and (
        isinstance(loss_type, list) or loss_type == 'mixed'
    ):
        assert device is not None and pack_size is not None
        loss_config = loss_config or {}
        # Canonical: main()'s merge turned `type` into per-member strings. Fallback:
        # an explicit shared config may use type="mixed" with a `types` list (no
        # sampler and no per-member configs).
        types = loss_type if isinstance(loss_type, list) else loss_config.get('types')
        if not (isinstance(types, list) and all(isinstance(x, str) for x in types)):
            raise ValueError(
                'The mixed loss needs per-member types: either sample'
                ' `sampler.space.loss.type` (categorical) or set `loss.types` to a'
                ' list of strings with one entry per pack member'
            )
        return _MixedLossPack(
            types=types,
            loss_config=loss_config,
            pack_size=pack_size,
            device=device,
        )

    if task_type == TaskType.REGRESSION and loss_type != 'mse':
        assert device is not None
        loss_config = loss_config or {}
        if loss_type == 'quantile':
            params = {
                k: v
                for k, v in loss_config.items()
                if k
                in (
                    'n_quantiles',
                    'agg_mode',
                    'monotone',
                    'monotone_init_spread',
                    'monotone_reference',
                )
            }
            return _QuantileLossPack(device=device, **params)
        elif loss_type == 'quantile_gaussian_tails':
            params = {
                k: v
                for k, v in loss_config.items()
                if k in ('n_quantiles', 'min_sigma', 'agg_mode')
            }
            return _GaussianTailQuantileLossPack(device=device, **params)
        elif loss_type == 'quantile_orthogonal':
            params = {
                k: v
                for k, v in loss_config.items()
                if k
                in (
                    'n_quantiles',
                    'parameterization',
                    'mean_loss_weight',
                    'detach_location',
                    'n_spline_coefficients',
                    'spline_slope_scale',
                    'spline_slope_bias',
                    'agg_mode',
                )
            }
            return _OrthogonalQuantileLossPack(device=device, **params)
        elif loss_type == 'sigma_huber':
            # Simplified adaqr loss: sigma-Huberized pinball. `sigma_weight`,
            # `sigma_scale` and `sigma_floor` are per-pack-member (see the
            # `loss_config` merge in main()); the rest is shared.
            assert pack_size is not None
            params = {
                k: v for k, v in loss_config.items() if k not in _ADAQR_MAIN_ONLY_KEYS
            }
            return _SigmaHuberLossPack(device=device, pack_size=pack_size, **params)
        elif loss_type == 'adaqr_spread':
            # AdaQR distance to N(y_true, (noise_scale * spread_sigma)**2), where
            # spread_sigma comes from two already emitted quantiles. No sigma head.
            params = {
                k: v
                for k, v in loss_config.items()
                if k
                in (
                    'n_quantiles',
                    'noise_scale',
                    'spread_pair',
                    'min_sigma',
                    'monotone',
                    'monotone_init_spread',
                    'monotone_reference',
                )
            }
            return _AdaQRSpreadLossPack(device=device, **params)
        elif loss_type == 'adaqr_gt':
            # The source itself is resolved and validated once in main(). Only the
            # standardized train sigma and per-member scale reach the loss object.
            assert pack_size is not None
            unexpected = set(loss_config) - {
                'type',
                'n_quantiles',
                'sigma_train',
                'sigma_scale',
                'min_sigma',
            }
            if unexpected:
                raise ValueError(
                    f'Unsupported adaqr_gt loss parameters: {sorted(unexpected)}'
                )
            return _AdaQRGTLossPack(
                n_quantiles=loss_config['n_quantiles'],
                sigma_train=loss_config['sigma_train'],
                sigma_scale=loss_config['sigma_scale'],
                min_sigma=loss_config.get('min_sigma', 1e-5),
                pack_size=pack_size,
                device=device,
            )
        elif loss_type == 'adaqr':
            # `regularizer` and `sigma_coef` are per-pack-member
            # (see `loss_config` merge in main()).
            assert pack_size is not None
            params = {
                k: v for k, v in loss_config.items() if k not in _ADAQR_MAIN_ONLY_KEYS
            }
            return _AdaQRLossPack(device=device, pack_size=pack_size, **params)
        elif loss_type == 'adaqr_gaussian_tails':
            assert pack_size is not None
            params = {
                k: v for k, v in loss_config.items() if k not in _ADAQR_MAIN_ONLY_KEYS
            }
            return _GaussianTailAdaQRLossPack(
                device=device, pack_size=pack_size, **params
            )
        elif loss_type == 'adaqr_v9':
            # As `adaqr`, but sigma = exp(logsigma).clamp(min=min_sigma).
            # `use_muls_init` (default False) gates the warm-start, applied in main().
            # `regularizer` and `sigma_coef` are per-pack-member.
            assert pack_size is not None
            params = {
                k: v for k, v in loss_config.items() if k not in _ADAQR_MAIN_ONLY_KEYS
            }
            return _AdaQRLossV9Pack(device=device, pack_size=pack_size, **params)
        elif loss_type == 'adaqr_vNightly':
            assert pack_size is not None
            params = {
                k: v for k, v in loss_config.items() if k not in _ADAQR_MAIN_ONLY_KEYS
            }
            return _AdaQRLossNightlyPack(device=device, pack_size=pack_size, **params)
        elif loss_type == 'hl_gauss':
            # `min_value` / `max_value` are the shared bin support computed in main().
            # `sigma` is per-pack-member (see `loss_config` merge in main()).
            assert pack_size is not None
            return _HLGaussLossPack(
                device=device,
                num_bins=loss_config['hl_num_bins'],
                min_value=loss_config['min_value'],
                max_value=loss_config['max_value'],
                sigma=loss_config['sigma'],
                pack_size=pack_size,
                clamp_to_range=loss_config.get('clamp_to_range', True),
            )
        else:
            raise ValueError(f'Unknown regression loss type: {loss_type}')

    base_loss_fn = {
        TaskType.REGRESSION: nn.functional.mse_loss,
        TaskType.BINCLASS: nn.functional.binary_cross_entropy_with_logits,
        TaskType.MULTICLASS: nn.functional.cross_entropy,
    }[task_type]

    def loss_fn_pack(y_pred: Tensor, y_true: Tensor) -> Tensor:
        pack_size = get_pack_size(y_pred)
        losses = base_loss_fn(
            y_pred.flatten(0, 1), y_true.flatten(0, 1), reduction='none'
        )
        losses = losses.unflatten(0, (pack_size, y_true.shape[BATCH_DIM]))
        losses = losses.flatten(BATCH_DIM)
        return losses.mean(BATCH_DIM)

    return loss_fn_pack


def _get_mean_scores(
    current_mean_scores: None | dict[PartKey, float],
    experiments: list[ExperimentDict],
    state: StatePack,
    eval_parts: list[PartKey],
) -> tuple[dict[PartKey, float], bool]:
    scores = {}
    for experiment in experiments:
        for part in eval_parts:
            scores.setdefault(part, []).append(
                experiment['report']['metrics'][part]['score']
            )
    for part in eval_parts:
        scores.setdefault(part, []).extend(
            state.best_metrics[part]['score'][state.steps > 0].tolist()
        )
    assert scores

    mean_scores = {k: statistics.mean(v) for k, v in scores.items()}
    return mean_scores, (
        current_mean_scores is None or mean_scores['val'] > current_mean_scores['val']
    )


def _get_best_scores(
    current_best_experiment: None | ExperimentDict,
    current_best_scores: None | dict[PartKey, float],
    state: StatePack,
) -> tuple[dict[PartKey, float], bool]:
    if current_best_experiment is not None and (
        current_best_scores is None
        or get_experiment_val_score(current_best_experiment)
        > current_best_scores['val']
    ):
        current_best_scores = {
            part: current_best_experiment['report']['metrics'][part]['score']
            for part in state.best_metrics.keys()
        }
        best_scores_improved = True
    else:
        best_scores_improved = False

    if state.pack_size == 0 or (state.steps == 0).all():
        assert current_best_scores is not None
        return current_best_scores, best_scores_improved

    else:
        val_scores = state.best_metrics['val']['score']
        best_index = np.argmax(val_scores)
        if (
            current_best_scores is None
            or val_scores[best_index] > current_best_scores['val']
        ):
            return (
                {
                    part: float(part_metrics['score'][best_index])
                    for part, part_metrics in state.best_metrics.items()
                },
                True,
            )
        else:
            assert current_best_scores is not None
            return current_best_scores, best_scores_improved


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# Main
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
class Config(TypedDict):
    seed: int
    data: KWArgs

    # Model
    n_models: int
    model: KWArgs

    # Loss (regression only). Nested dict: {"type": "mse" | "quantile"
    # | "quantile_gaussian_tails" | "quantile_orthogonal" | "adaqr_spread"
    # | "adaqr_gt" | "adaqr"
    # | "adaqr_gaussian_tails" | "adaqr_v9" | "hl_gauss", plus
    # loss-specific params, e.g. "n_quantiles", "agg_mode", "hl_num_bins", "sigma"}.
    # For "hl_gauss": "hl_num_bins" and the bin-support params (bin_support_mode,
    # hl_min_value_relative, hl_max_value_relative) must stay shared across the pack,
    # since they determine the model's output width. "sigma" may instead be a single
    # float (shared) or sampled per pack member via `sampler.space.loss.sigma`, exactly
    # like `model`/`optimizer` hyperparameters.
    # For "adaqr"/"adaqr_v9": "n_quantiles" must stay shared (output width),
    # same as "hl_num_bins" above. For "adaqr_gaussian_tails", it is the
    # number C of free center outputs; the head adds four tail parameters and smoothing
    # sigma. "regularizer" may be a single float (shared) or sampled per pack member via
    # `sampler.space.loss.regularizer`. All three take "min_sigma"
    # (shared/fixed): AdaQR as sigma = |logsigma| + min_sigma, v9 as
    # sigma = exp(logsigma).clamp(min=min_sigma). "adaqr_v9" also disables the muls_init
    # warm-start by default (use_muls_init=False).
    # Both adaqr losses support the optional "sigma_coef": weight of the sigma-only
    # part of the split objective ("1" | "1/q" | "1/sqrt(q)" or a number, shared or per
    # pack member); omitting it keeps the original single-pass loss.
    # "adaqr" additionally supports the following optional tricks (all disabled
    # when omitted, i.e. a config without them behaves exactly as before):
    #   - "kappa": reparametrized regularizer, regularizer = kappa / sqrt(1 + kappa**2);
    #     mutually exclusive with "regularizer", also tunable per pack member;
    #   - "muls_init_offset_fix" (shared bool): exact warm-start that makes the mean
    #     predicted sigma start at "muls_init_add" instead of matching the mean of the
    #     pre-abs logsigma values.
    #   - "scale_reg_weight" (float, shared or per member): weight of the paired
    #     Gaussian-CRPS scale_regs for quantile-grid scale and learned sigma scale;
    #     "scale_reg_balance" is shared ("equal_loss" or "output_jacobian").
    # "adaqr_spread" has no sigma head. It estimates sigma from the two quantiles at
    # `spread_pair` and `1 - spread_pair` (default 0.159), and uses the target
    # N(y_true, (noise_scale * sigma)^2). `noise_scale` defaults to the fixed shared
    # value 0.5; it, `spread_pair`, and `min_sigma` cannot be tuned per pack member.
    # "adaqr_gt" also has no sigma head. It loads raw-target per-row uncertainty
    # from `loss.sigma_source`, converts it to standardized target units, and indexes
    # train sigma with each member's independently shuffled batch indices.
    # `sigma_scale` can be fixed or sampled per member.
    # "quantile", "sigma_huber", and "adaqr_spread" additionally accept "monotone"
    # (shared bool, default false) and "monotone_init_spread" (shared float, default
    # 4.0): the head is
    # reparameterized as q_0 + cumsum(softplus(raw + bias)), so the quantiles cannot
    # cross and every channel is trained by all the pinball terms above it.
    # "monotone_reference" ("left" | "median", shared, default "left") picks the base
    # channel of the cumsum: "median" scale_regs on the middle quantile with increments
    # fanning out both ways, so the tail quantile at tau=0.001 is not the scale_reg.
    # "quantile_init" = "empirical" (shared) initializes the quantile output weights
    # to zero and biases from the centered empirical target quantiles. For adaqr
    # losses, the extra logsigma channel keeps its default initialization.
    # For "mixed" (portfolio pack: one loss type per pack member) the per-member types
    # come from `sampler.space.loss.type` (categorical) or from a shared `loss.types`
    # list; the candidate types must be known upfront because they set the head width.
    # The head is sectioned as [0:n_q] quantiles, [n_q] logsigma, and — only when
    # "hl_gauss" is a candidate — [n_q+1:n_q+1+hl_num_bins] HL logits. Every loss-specific
    # key of the participating types is accepted (shared or per-pack-member as usual) and
    # routed to the type that understands it. "monotone" is not supported yet.
    # For "sigma_huber" (the simplified adaqr loss: sigma-Huberized pinball, no
    # regularizer / adaqr wrapper): "n_quantiles" must stay shared (output width, +1
    # logsigma channel as for the adaqr losses). Shared switches: "kernel"
    # ("huber" | "gauss"), "sigma_objective" ("nll" | "mse" | "spread"), "sigma_param"
    # ("abs" | "softplus"), "min_sigma". Per-pack-member (a float, or sampled via
    # `sampler.space.loss.<name>`): "sigma_weight", "sigma_scale", "sigma_floor"
    # ("spread" only; sigma = max(spread, sigma_floor * |resid|), 0 == disabled).
    # The muls_init warm-start applies as for "adaqr", except with
    # "sigma_objective" = "spread" (no sigma head is trained) where it is skipped.
    # Defaults to {"type": "mse"} (the native TabPack regression loss).
    loss: NotRequired[KWArgs]

    # Training
    optimizer: KWArgs
    batch_size: int
    n_epochs: int
    patience: int

    # Gradient clipping (omitted -> no clipping, as before). The norm is computed per
    # pack member, so the result does not depend on `n_models`. If `always_clip` is
    # true, each member's gradient is renormalized to `gradient_clipping_norm` every
    # step (scaled up or down) instead of only when it exceeds the threshold.
    # `gradient_clipping_norm` may be a single float (shared) or sampled per pack member
    # via `sampler.space.gradient_clipping_norm`, like `model`/`optimizer`/`loss` ones.
    # `always_clip` is a shared switch and cannot be tuned per member.
    gradient_clipping_norm: NotRequired[float]
    always_clip: NotRequired[bool]

    # Split each training step into forward/backward chunks of this many samples
    # (gradients accumulate across chunks; per-member losses are combined into the
    # exact full-batch mean). Needed for datasets whose activations do not fit into
    # GPU memory at the full batch (e.g. high-cardinality one-hot cat features).
    # Not supported for losses with batch-global terms or per-member state
    # (adaqr*/sigma_huber/hl_gauss/mixed).
    train_batch_chunk_size: NotRequired[int]

    # Evaluation
    eval_parts: NotRequired[list[PartKey]]
    eval_batch_size: NotRequired[int]

    # Ensembles
    online_ensembles: NotRequired[KWArgs]

    # Configs
    configs: NotRequired[list[ConfigDict]]
    sampler: NotRequired[KWArgs]

    # Efficiency
    amp_dtype: NotRequired[AMPDType]
    timeout: NotRequired[int]

    # Report
    track_experiments: NotRequired[bool]
    track_best_experiment: NotRequired[bool]
    track_online_ensemble_history: NotRequired[bool]
    track_memory_usage: NotRequired[bool]

    # Output
    save_final_predictions: NotRequired[bool]
    save_all_predictions: NotRequired[bool]
    save_uncertainty_predictions: NotRequired[bool]


def _validate_config(config: Config) -> None:
    configs = config.get('configs')
    if configs is not None:
        assert len(configs) == config['n_models']

    sampler_config = config.get('sampler')
    if (
        sampler_config is not None
        and sampler_config.get('type') not in HyperparameterSampler._BASIC_SAMPLERS
    ):
        raise ValueError(
            'Given the provided sampler config, pack_size must be set explicitly,'
            ' because it can have non-trivial impact on the results'
        )


def _prepare_configs(
    config: Config, hyperparameter_sampler: None | HyperparameterSampler
) -> None | list[ConfigDict]:
    configs = config.get('configs')
    if configs is None:
        if hyperparameter_sampler is None:
            return None
        else:
            return [hyperparameter_sampler.ask(x) for x in range(config['n_models'])]
    else:
        assert hyperparameter_sampler is None
        return configs


def _prepare_model_config(
    config: Config, model_configs_T: None | dict[str, list[Any]]
) -> ConfigDict:
    def infer_max_dimension_(
        subconfig: ConfigDict,
        key: str,
        value: None | int | list[int],
        value_space: None | list[Any],
        value_list: None | list[int],
    ) -> None:
        max_key = f'max_{key}'
        if value is None:
            if value_space is None:
                assert value_list is not None
                assert value_list
                subconfig[max_key] = max(value_list)

                assert max_key in subconfig, (
                    f'The model argument `{max_key}` cannot be inferred from the'
                    ' config, so it must be provided explicitly'
                )
            else:
                # value_space: ["_tune_", "int", min_value, max_value[, step]]
                subconfig[max_key] = value_space[3]
        else:
            assert max_key not in subconfig, (
                f'When the model argument `{key}` is provided explicitly,'
                f' the model argument `{max_key}` must be omitted, which is not true'
            )
            assert value_space is None, (
                f'The model argument `{key}` is presented as both a specific value and'
                ' as a part of the sampler space, which is not allowed'
            )
            subconfig[max_key] = max(value) if isinstance(value, list) else None

    model_config = deepcopy(config['model'])

    # Infer the maximum feature embedding dimension.
    model_num_embeddings_config = model_config.get('num_embeddings')
    if model_num_embeddings_config is not None:
        d_embedding_value = model_num_embeddings_config.get('d_embedding')
        d_embedding_value_space = (
            config.get('sampler', {})
            .get('space', {})
            .get('model', {})
            .get('num_embeddings', {})
            .get('d_embedding')
        )
        d_embedding_value_list = (
            None
            if model_configs_T is None
            or 'num_embeddings' not in model_configs_T
            or 'd_embedding' not in model_configs_T['num_embeddings'][0]
            else [x['d_embedding'] for x in model_configs_T['num_embeddings']]
        )
        if (
            d_embedding_value is not None
            or d_embedding_value_space is not None
            or d_embedding_value_list is not None
        ):
            infer_max_dimension_(
                model_num_embeddings_config,
                'd_embedding',
                d_embedding_value,
                d_embedding_value_space,
                d_embedding_value_list,
            )

    # Infer the maximum backbone dimensions.
    for key in ['n_blocks', 'd_block']:
        infer_max_dimension_(
            model_config,
            key,
            model_config.get(key),
            config.get('sampler', {}).get('space', {}).get('model', {}).get(key),
            None if model_configs_T is None else model_configs_T.get(key),
        )

    if model_configs_T is not None:
        # Merge the configs to the main config.
        vendor.tabpack.utils.dict_merge_recursively(
            model_config,
            (
                model_configs_T
                | {
                    key: vendor.tabpack.utils.transpose_list_of_dicts(
                        model_configs_T[key]
                    )
                    for key in ('num_embeddings', 'activation')
                    if key in model_configs_T
                }
            ),
        )

    return model_config


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    _validate_config(config)

    # >>> Start
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])

    # >>> Efficiency
    device = lib.util.get_device()
    logger.info(f'Device: {device}')

    amp_dtype = config.get('amp_dtype')
    if amp_dtype is None:
        autocast = None
    else:
        autocast = _make_autocast(amp_dtype, device)
        logger.info('Enabled AMP')

    timer = delu.tools.Timer()
    timeout = config.get('timeout')

    if config.get('track_memory_usage', False):
        torch.cuda.reset_peak_memory_stats(device)

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    loss_config = dict(config.get('loss') or {})
    loss_type = loss_config.get('type', 'mse')
    raw_targets = (
        {part: values.copy() for part, values in dataset.data['y'].items()}
        if dataset.task.is_regression and loss_type == 'adaqr_gt'
        else None
    )
    regression_label_stats = dataset.try_standardize_labels_()
    if raw_targets is not None:
        assert regression_label_stats is not None
        sigma_source = loss_config.pop('sigma_source', None)
        if not isinstance(sigma_source, dict):
            raise ValueError('loss.sigma_source is required for adaqr_gt')
        if 'sigma_scale' in sigma_source:
            raise ValueError(
                'Set loss.sigma_scale, not loss.sigma_source.sigma_scale, for adaqr_gt'
            )
        loaded_sigma = lib.uncertainty.load_sigma_source(
            sigma_source,
            dataset_name=Path(config['data']['path']).name,
            expected_targets=raw_targets,
            label_std=regression_label_stats.std,
        )
        loss_config['sigma_train'] = torch.as_tensor(
            loaded_sigma.values['train'], device=device
        )
        report['sigma_source_artifact'] = str(loaded_sigma.artifact)
        logger.info(f'Loaded external sigma from {loaded_sigma.artifact}')
    dataset = dataset.to_torch(device)
    n_classes = dataset.task.try_compute_n_classes()
    # Custom regression losses need a wider output head.
    # The mixed loss assigns a loss type per pack member, but the head width must be
    # known here — before the sampler runs and long before the per-member `loss_config`
    # merge. So the candidate types are read from the base config (`loss.types`) or from
    # the categorical choices of `sampler.space.loss.type`, mirroring how
    # `_prepare_model_config` peeks at the space to infer `max_*` dimensions.
    mixed_candidate_types: None | list[str] = None
    if dataset.task.is_regression and loss_type == 'mixed':
        space_type = (
            (config.get('sampler') or {}).get('space', {}).get('loss', {}).get('type')
        )
        if isinstance(space_type, list) and space_type[:2] == ['_tune_', 'categorical']:
            mixed_candidate_types = list(space_type[2])
        elif isinstance(loss_config.get('types'), list):
            mixed_candidate_types = list(loss_config['types'])
        else:
            raise ValueError(
                'loss.type = "mixed" requires the candidate types to be known upfront'
                ' (they determine the head width): set'
                ' `sampler.space.loss.type = ["_tune_", "categorical", [...]]`'
                ' or `loss.types = [...]`'
            )
    if dataset.task.is_regression and loss_type == 'mixed':
        assert mixed_candidate_types is not None
        n_classes = _mixed_head_width(
            loss_config['n_quantiles'],
            mixed_candidate_types,
            loss_config.get('hl_num_bins'),
        )
    elif dataset.task.is_regression and loss_type in (
        'quantile',
        'quantile_gaussian_tails',
        'adaqr_spread',
        'adaqr_gt',
        'adaqr',
        'adaqr_gaussian_tails',
        'adaqr_v9',
        'adaqr_vNightly',
        'sigma_huber',
    ):
        n_quantiles = loss_config['n_quantiles']
        if loss_type in (
            'quantile_gaussian_tails',
            'adaqr_gaussian_tails',
        ):
            if type(n_quantiles) is not int or n_quantiles < 18:
                raise ValueError(
                    'n_quantiles must be an integer >= 18 (C//18 tail values)'
                )
            n_classes = n_quantiles + (
                4 if loss_type == 'quantile_gaussian_tails' else 5
            )
        else:
            n_classes = n_quantiles + (
                1
                if loss_type in (
                    'adaqr',
                    'adaqr_v9',
                    'adaqr_vNightly',
                    'sigma_huber',
                )
                else 0
            )
    elif dataset.task.is_regression and loss_type == 'quantile_orthogonal':
        parameterization = loss_config['parameterization']
        if parameterization == 'grid':
            n_classes = 1 + loss_config['n_quantiles']
        elif parameterization == 'linear_spline':
            n_classes = 1 + loss_config['n_spline_coefficients']
        else:
            raise ValueError(
                'loss.parameterization must be "grid" or "linear_spline", got'
                f' {parameterization!r}'
            )
    elif dataset.task.is_regression and loss_type == 'hl_gauss':
        # HL-Gauss classifies over `hl_num_bins` bins.
        n_classes = loss_config['hl_num_bins']
    Y_train = _make_Y_train(dataset)
    train_size = dataset.size('train')

    needs_hl_support = loss_type == 'hl_gauss' or (
        mixed_candidate_types is not None and 'hl_gauss' in mixed_candidate_types
    )
    if dataset.task.is_regression and needs_hl_support:
        # Shared bin support (not tuned): computed once from the training targets, in
        # the standardized space of Y_train (predictions are de-standardized at eval).
        # `sigma`, unlike the bin support, is per-pack-member and merged separately
        # below (once `configs_T` is available), after the pack size is known.
        support_mode = loss_config.get('bin_support_mode', 'y_quantile')
        if support_mode == 'full':
            target_min = Y_train.min().item()
            target_max = Y_train.max().item()
        elif support_mode == 'y_quantile':
            target_min = torch.quantile(Y_train, 0.005).item()
            target_max = torch.quantile(Y_train, 0.995).item()
        else:
            raise ValueError(f'Unknown bin_support_mode: {support_mode}')
        target_range = target_max - target_min
        lo = target_min + loss_config['hl_min_value_relative'] * target_range
        hi = target_min + loss_config['hl_max_value_relative'] * target_range
        loss_config['min_value'], loss_config['max_value'] = min(lo, hi), max(lo, hi)

    # >>> Hyperparameter sampler
    hyperparameter_sampler_config = config.get('sampler')
    if hyperparameter_sampler_config is None:
        hyperparameter_sampler = None
    else:
        hyperparameter_sampler = HyperparameterSampler(
            seed=config['seed'],
            study_kwargs={'study_name': report['function'], 'direction': 'maximize'},
            **hyperparameter_sampler_config,
        )
        logger.info('Using hyperparameter sampler')

    # >>> State
    state = StatePack(
        pack_size=config['n_models'],
        configs=_prepare_configs(config, hyperparameter_sampler),
    )
    logger.debug('Created state')
    final_state = FinalStatePack()
    logger.debug('Created final_state')
    # Save all configs before training
    all_configs = deepcopy(state.configs) if state.configs is not None else None
    configs_T = (
        None
        if state.configs is None
        else vendor.tabpack.utils.transpose_list_of_dicts(deepcopy(state.configs))
    )
    logger.debug('Transposed the configs')

    # >>> Model
    model = ModelPack(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=n_classes,
        pack_size=state.pack_size,
        **_prepare_model_config(
            config,
            (
                None
                if configs_T is None or 'model' not in configs_T
                else vendor.tabpack.utils.transpose_list_of_dicts(
                    configs_T.pop('model')
                )
            ),
        ),
    )
    logger.debug('Created the model')
    model.to(device)
    logger.debug('Moved the model to the device')
    quantile_init = loss_config.get('quantile_init')
    if quantile_init is not None:
        if quantile_init != 'empirical':
            raise ValueError(f'Unknown loss.quantile_init: {quantile_init!r}')
        if loss_type not in ('quantile', 'adaqr'):
            raise ValueError(
                'loss.quantile_init="empirical" is supported only for quantile and'
                ' adaqr'
            )
        if loss_config.get('monotone', False):
            raise ValueError(
                'loss.quantile_init="empirical" initializes direct quantile channels'
                ' and is incompatible with loss.monotone'
            )
        if (
            configs_T is not None
            and 'loss' in configs_T
            and any('quantile_init' in x for x in configs_T['loss'])
        ):
            raise ValueError('loss.quantile_init must be shared across the pack')
        quantile_bias = lib.deep.init_empirical_quantile_output_(
            model.output.weight,
            model.output.bias,
            Y_train,
            loss_config['n_quantiles'],
            weight_output_dim=1,
            bias_output_dim=1,
        )
        logger.info(
            'Initialized quantile output from the empirical target distribution:'
            f' [{quantile_bias[0].item():.4g}, {quantile_bias[-1].item():.4g}]'
        )

    # NOTE
    # Predictions must be stored in aggregation-friendly units
    # for ensembling purposes (raw logits do _not_ meet this requirement).
    prediction_type = (
        PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS
    )
    apply_model = apply_model_impl if autocast is None else autocast(apply_model_impl)

    # >>> Ensembles
    online_ensemble_configs = config.get('online_ensembles')
    if online_ensemble_configs is None:
        online_ensembles = None
        online_ensemble_history = None
    else:
        online_ensembles = _make_online_ensembles(
            online_ensemble_configs,
            task=dataset.task,
            prediction_type=prediction_type,
            update_part='val',
            device=device,
        )
        online_ensemble_history = {}
    logger.debug('Created the ensembles')

    # >>> Training
    muon_optimizer_parameter_groups = (
        [
            {
                'params': [block.linear.weight],
                'muon': True,
                'muon_scale': _make_muon_scale(block.linear),
            }
            for block in model.backbone._iter_blocks()
        ]
        if config['optimizer']['type'] == 'MuonAdamWPack'
        else []
    )
    optimizer = _make_optimizer(
        params=lib.deep.make_parameter_groups(
            model,
            _default_zero_weight_decay_condition,
            custom_groups=muon_optimizer_parameter_groups,
        ),
        pack_size=model.pack_size,
        **config['optimizer'],  # type: ignore
        **(
            {}
            if configs_T is None or 'optimizer' not in configs_T
            else vendor.tabpack.utils.transpose_list_of_dicts(
                configs_T.pop('optimizer')
            )
        ),
    )
    logger.debug('Created the optimizer')

    # Gradient clipping. The threshold is either shared (a float in the config) or
    # sampled per pack member; either way it is stored as a (pack,) tensor, so it must
    # be shrunk alongside the pack when models are early-stopped (see `pack_remove`).
    always_clip = config.get('always_clip', False)
    assert isinstance(always_clip, bool), (
        '`always_clip` is a shared switch, it cannot be tuned per pack member'
    )
    if configs_T is not None and 'gradient_clipping_norm' in configs_T:
        assert 'gradient_clipping_norm' not in config, (
            '`gradient_clipping_norm` is sampled per pack member, so it must not also'
            ' be set as a shared value'
        )
        gradient_clipping_norm = torch.tensor(
            configs_T.pop('gradient_clipping_norm'), dtype=torch.float32, device=device
        )
    elif config.get('gradient_clipping_norm') is not None:
        gradient_clipping_norm = torch.full(
            (state.pack_size,),
            config['gradient_clipping_norm'],
            dtype=torch.float32,
            device=device,
        )
    else:
        gradient_clipping_norm = None

    if gradient_clipping_norm is None:
        assert not always_clip, (
            '`always_clip` requires `gradient_clipping_norm` to be set'
        )
    else:
        assert (gradient_clipping_norm > 0.0).all(), (
            '`gradient_clipping_norm` must be positive'
        )
        logger.info(
            'Gradient clipping (per pack member):'
            f' {gradient_clipping_norm.tolist()}'
            f'{" (always)" if always_clip else ""}'
        )

    # Per-pack-member loss hyperparameters (e.g. hl_gauss's `sigma`), sampled the same
    # way as `model`/`optimizer` ones.
    if configs_T is not None and 'loss' in configs_T:
        loss_config.update(
            vendor.tabpack.utils.transpose_list_of_dicts(configs_T.pop('loss'))
        )

    assert not configs_T, (
        'The following fields of the generated configs were not used:'
        f' {", ".join(configs_T)}'
    )
    del configs_T

    loss_fn = _make_loss_fn_pack(
        dataset.task.type_, loss_config, device, pack_size=state.pack_size
    )
    save_uncertainty_predictions = config.get('save_uncertainty_predictions', False)
    if save_uncertainty_predictions and not isinstance(loss_fn, _AdaQRLossPack):
        raise ValueError(
            'save_uncertainty_predictions is supported only for adaqr'
        )
    # Custom regression losses emit multi-channel outputs (quantiles [+ logsigma]); the
    # evaluation must reduce them to a single scalar prediction per sample before metrics.
    get_scalar_prediction = getattr(loss_fn, 'get_scalar_prediction', None)
    train_batch_chunk_size = config.get('train_batch_chunk_size')
    if train_batch_chunk_size is not None:
        if train_batch_chunk_size <= 0:
            raise ValueError('train_batch_chunk_size must be a positive integer')
        if hasattr(loss_fn, 'pack_remove') and not isinstance(
            loss_fn, _AdaQRGTLossPack
        ):
            raise ValueError(
                'train_batch_chunk_size is not supported for losses with per-member'
                ' state or batch-global terms (adaqr*/sigma_huber/hl_gauss/mixed):'
                ' chunking the batch would change the loss values'
            )
    epoch_size = math.ceil(train_size / config['batch_size'])

    # >>> Evaluation
    # The following order of `torch.inference_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = torch.inference_mode()(
        partial(
            _evaluate,
            apply_model,
            model,
            optimizer,
            dataset,
            regression_label_stats=regression_label_stats,
            prediction_type=prediction_type,
            device=device,
            get_scalar_prediction=get_scalar_prediction,
        )
    )
    logger.debug('Created the evaluation function')

    eval_parts = config.get('eval_parts', ['val', 'test'])
    assert 'val' in eval_parts
    eval_batch_size = config.get('eval_batch_size', 32768)

    # >>> Global state
    step = 0
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    experiments: list[ExperimentDict] = []
    online_ensemble_predictions: None | dict[str, dict[PartKey, np.ndarray]] = (
        None
        if online_ensembles is None
        else {name: {} for name in online_ensembles.keys()}
    )
    logger.debug('Prepared the global state')

    # >>> adaqr warm-start: muls_init is a per-pack-member offset for logsigma,
    # estimated once before training from each member's own predictions over one epoch
    # (as in bin/adaqr.py and the logarithmic-scale variant, generalized from one trial to the whole
    # pack). A separate generator keeps `batch_generator` untouched.
    #
    # Two estimators are available (`loss.muls_init_offset_fix`, default False):
    #   * False ("mean"):      offset = muls_init_add - mean(raw_sigma).
    #   * True  ("offset fix"): offset solves mean|raw_sigma + offset| ==
    #     muls_init_add - min_sigma, i.e. the *mean predicted sigma* starts exactly
    #     at `muls_init_add` (AdaQR only: sigma = |logsigma| + min_sigma).
    #
    # AdaQR always applies the warm-start; v9 gates it behind `use_muls_init`
    # (default False, matching bin/adaqr_dist_v9.py) and adds `log(muls_init_add)`.
    # sigma_huber shares AdaQR's estimator (same sigma = |logsigma| + min_sigma), but
    # only when its sigma head is actually trained: 'spread' derives sigma from the
    # quantile predictions and ignores the logsigma channel, so warming it up is a no-op.
    # The mixed loss reuses AdaQR's estimator: its head always carries a logsigma
    # channel, and the offsets computed for members whose loss ignores it are unused.
    if isinstance(
        loss_fn,
        _AdaQRLossPack | _AdaQRLossV9Pack | _SigmaHuberLossPack | _MixedLossPack,
    ):
        is_v9 = isinstance(loss_fn, _AdaQRLossV9Pack)
        is_sigma_huber = isinstance(loss_fn, _SigmaHuberLossPack)
        is_mixed = isinstance(loss_fn, _MixedLossPack)
        muls_init_offset_fix = loss_config.get('muls_init_offset_fix', False)
        assert isinstance(muls_init_offset_fix, bool), (
            'loss.muls_init_offset_fix is a shared switch, it cannot be tuned per'
            ' pack member'
        )
        assert not (muls_init_offset_fix and is_v9), (
            'muls_init_offset_fix is defined only for "adaqr"'
            ' (sigma = |logsigma| + min_sigma)'
        )
        assert not (muls_init_offset_fix and is_sigma_huber), (
            'muls_init_offset_fix is defined only for "adaqr"'
        )
        assert not (muls_init_offset_fix and is_mixed), (
            'muls_init_offset_fix is defined only for "adaqr"'
        )
        # NOTE: the two sigma_huber-specific rules below must not fire for a mixed pack:
        # disabling the warm-start pack-wide because of its spread-sigma members would
        # starve the adaqr members that do need it.
        if is_sigma_huber and loss_fn.sigma_objective == 'spread':
            # No sigma head to warm up (and softplus would need its own estimator).
            loss_config = {**loss_config, 'use_muls_init': False}
        elif is_sigma_huber and loss_fn.sigma_param == 'softplus':
            assert not loss_config.get('use_muls_init', True), (
                'the muls_init warm-start assumes sigma = |logsigma| + min_sigma;'
                ' set loss.use_muls_init = false for sigma_param = "softplus"'
            )
        if loss_config.get('use_muls_init', not is_v9):
            muls_init_add = loss_config.get('muls_init_add', 0.05 if is_v9 else 0.5)
            if is_v9:
                muls_init_add = math.log(muls_init_add)
            init_generator = torch.Generator(device).manual_seed(config['seed'] + 1)
            # Run in train mode with pack_idx=None (full pack), as PackView requires;
            # matches the step==0 estimation in bin/adaqr.py and the logarithmic-scale variant.
            model.train()
            with torch.no_grad():
                # For the "offset fix" estimator the whole logsigma channel is needed
                # (it solves a piecewise-linear equation over the sorted values), so the
                # raw per-batch tensors are kept; otherwise only per-batch means are.
                logsigmas: list[Tensor] = []
                for batch_idx in generate_training_batches(
                    train_size=train_size,
                    batch_size=config['batch_size'],
                    batch_generator=init_generator,
                    pack_size=state.pack_size,
                ):
                    logsigma = apply_model(
                        model, dataset, part='train', batch_idx=batch_idx
                    )[..., -1]  # (pack, batch)
                    logsigmas.append(
                        logsigma
                        if muls_init_offset_fix
                        else logsigma.mean(dim=BATCH_DIM)  # (pack,)
                    )
                if muls_init_offset_fix:
                    muls_init = _compute_muls_init_offset_fix(
                        logsigmas,
                        muls_init_add=muls_init_add,
                        min_sigma=loss_config.get('min_sigma', 1e-4),
                    )
                else:
                    muls_init = (
                        -torch.stack(logsigmas).mean(dim=0) + muls_init_add
                    )  # (pack,)
                del logsigmas
                # (pack,) -> (pack, 1) to broadcast against `y_pred[..., -1]`.
                loss_fn.muls_init = muls_init.unsqueeze(-1)
            logger.info(
                f'{"adaqr_v9" if is_v9 else "mixed" if is_mixed else "adaqr"}'
                ' muls_init'
                f' ({"offset fix" if muls_init_offset_fix else "mean"}) ='
                f' {loss_fn.muls_init.squeeze(-1).tolist()}'
            )

    # >>> Numerical logs
    #
    # Numerical log is a list of Python objects of the same type
    # (usually, dictionaries holding numerical data, e.g. arrays, tensors, etc.).
    #
    # During training, simply append new records and avoid expensive operations,
    # such as CUDA-to-CPU conversions. After the training, transform the
    # logs as needed (e.g. to a more memory-efficient format) and save to disk.
    steps_numlog = []
    epochs_numlog = []
    pack_epochs_numlog = []
    uncertainty_prediction_chunks: dict[str, list[np.ndarray]] = {}

    # `mean_scores` and `best_scores` are maintained only for printing
    # and rely on incomplete experiments (i.e. on the data from `training_state`).
    mean_scores = None
    mean_scores_improved = False
    best_scores = None
    best_scores_improved = False
    first_online_ensemble_scores = None
    first_online_ensemble_improved = False

    # >>> Report
    report['n_models'] = 0
    if hyperparameter_sampler is not None:
        report['n_trials'] = 0
    report['prediction_type'] = prediction_type.value
    report['epoch_size'] = epoch_size
    if online_ensembles is not None:
        report['online_ensembles'] = typing.cast(
            dict[str, ExperimentDict],
            {k: {'report': {}} for k in online_ensembles.keys()},
        )
    track_experiments = config.get('track_experiments', hyperparameter_sampler is None)
    track_best_experiment = config.get(
        'track_best_experiment',
        # If individual models have different hyperparameters,
        # save the best experiment by default.
        state.configs is not None,
    )
    logger.debug('Filled the report')

    # >>> Training loop
    print()
    timer.run()
    pack_validate(model, optimizer, state)

    logger.debug('Starting the training loop')
    while (
        report['n_models'] < config['n_models']
        and (timeout is None or timer.elapsed() < timeout)
        and (
            online_ensembles is None
            or any(x.is_running for x in online_ensembles.values())
        )
    ):
        # >>> Validation
        n_remaining_models = config['n_models'] - report['n_models']
        assert model.pack_size <= n_remaining_models

        # >>> Training phase
        epoch_training_start_time = time.perf_counter()
        model.train()

        batches = generate_training_batches(
            train_size=train_size,
            batch_size=config['batch_size'],
            batch_generator=batch_generator,
            pack_size=state.pack_size,
        )
        batch_losses = []
        batch_sizes = []
        for batch_idx in tqdm(
            batches,
            desc=str(lib.util.try_get_relative_path(exp)),
            leave=False,
            disable=not lib.env.is_local(),
        ):
            if train_batch_chunk_size is None:
                predictions = apply_model(
                    model, dataset, part='train', batch_idx=batch_idx
                )
                losses = (
                    loss_fn(predictions, Y_train[batch_idx], batch_idx=batch_idx)
                    if isinstance(loss_fn, _AdaQRGTLossPack)
                    else loss_fn(predictions, Y_train[batch_idx])
                )
                # The scale of the gradients should not depend on the number of models,
                # so the individual losses are summed, not averaged.
                loss = losses.sum()
                optimizer.zero_grad()
                loss.backward()
            else:
                # Same as above, but forward/backward run in batch chunks to reduce
                # peak activation memory; gradients accumulate across chunks. The
                # per-member losses are means over the chunk, so they are weighted by
                # the relative chunk size to recover the exact full-batch mean.
                optimizer.zero_grad()
                loss = None
                for chunk_batch_idx in batch_idx.split(
                    train_batch_chunk_size, dim=BATCH_DIM
                ):
                    chunk_weight = (
                        chunk_batch_idx.shape[BATCH_DIM] / batch_idx.shape[BATCH_DIM]
                    )
                    chunk_predictions = apply_model(
                        model, dataset, part='train', batch_idx=chunk_batch_idx
                    )
                    chunk_losses = (
                        loss_fn(
                            chunk_predictions,
                            Y_train[chunk_batch_idx],
                            batch_idx=chunk_batch_idx,
                        )
                        if isinstance(loss_fn, _AdaQRGTLossPack)
                        else loss_fn(chunk_predictions, Y_train[chunk_batch_idx])
                    )
                    chunk_loss = chunk_losses.sum() * chunk_weight
                    chunk_loss.backward()
                    loss = (
                        chunk_loss.detach()
                        if loss is None
                        else loss + chunk_loss.detach()
                    )
            if gradient_clipping_norm is not None:
                _clip_grad_norm_pack(
                    model,
                    gradient_clipping_norm,
                    always_clip=always_clip,
                    pack_size=model.pack_size,
                )
            optimizer.step()

            step += 1
            state.step()

            loss_detached = loss.detach()
            batch_losses.append(loss_detached)
            batch_sizes.append(batch_idx.shape[BATCH_DIM])
            steps_numlog.append(
                {
                    'step': step,
                    'time': timer.elapsed(),
                    'batch_size': batch_sizes[-1],
                    'loss': loss_detached,
                }
            )

        epoch_training_duration = time.perf_counter() - epoch_training_start_time

        del batches, batch_idx, losses, loss, loss_detached  # pyright: ignore[reportPossiblyUnboundVariable]
        _free_mps_memory()

        if config.get('track_memory_usage', False):
            report['memory-usage'] = torch.cuda.memory_stats(device)
            break

        # >>> Evaluation phase
        epoch_evaluation_start_time = time.perf_counter()

        # Evaluate the models.
        (
            (eval_metrics, eval_predictions, eval_predictions_torch),
            eval_batch_size,
        ) = evaluate(parts=eval_parts, batch_size=eval_batch_size)
        report['eval_batch_size'] = eval_batch_size
        state.update(
            eval_metrics,
            predictions=eval_predictions,
            predictions_torch=eval_predictions_torch,
            model_state_dict=model.state_dict(),
        )

        pack_epochs_numlog.append(
            deepcopy(
                {
                    'step': state.steps,
                    'time': np.full((state.pack_size,), timer.elapsed()),
                    'size': np.array([state.pack_size]),
                    'id': state.ids,
                    'metrics': eval_metrics,
                    **(
                        {'predictions': eval_predictions}
                        if config.get('save_all_predictions', False)
                        else {}
                    ),
                }
            )
        )

        # Determine which models to stop.
        pack_size_before_stopping = model.pack_size
        stop_pack_idx = compute_stop_pack_idx(
            state,
            epoch_size=epoch_size,
            n_epochs=config['n_epochs'],
            patience=config['patience'],
        )

        if stop_pack_idx is None:
            latest_predictions = eval_predictions
            latest_predictions_torch = eval_predictions_torch

        else:
            stop_pack_idx_torch = torch.as_tensor(stop_pack_idx)
            n_models_to_stop = len(stop_pack_idx)

            keep_pack_idx_torch = vendor.tabpack.nn.make_keep_pack_idx(
                state.pack_size, stop_pack_idx_torch
            )
            keep_pack_idx = keep_pack_idx_torch.numpy()

            stop_pack_idx_torch = stop_pack_idx_torch.to(device)
            keep_pack_idx_torch = keep_pack_idx_torch.to(device)

            # Filter predictions for ensembles.
            latest_predictions = {
                k: v[keep_pack_idx] for k, v in eval_predictions.items()
            }
            latest_predictions_torch = {
                k: v[keep_pack_idx_torch] for k, v in eval_predictions_torch.items()
            }

            # Evaluate the best checkpoints of the stopped models.
            vendor.tabpack.nn.module_pack_load_state_dict(
                model, state.best_model_state_dicts, pack_idx=stop_pack_idx_torch
            )
            with vendor.tabpack.nn.module_pack_select(model, stop_pack_idx_torch):
                (
                    (final_metrics, final_predictions, final_predictions_torch),
                    eval_batch_size,
                ) = evaluate(parts=['train', 'val', 'test'], batch_size=eval_batch_size)
                final_uncertainty_predictions = (
                    _evaluate_adaqr_uncertainty(
                        apply_model,
                        model,
                        dataset,
                        parts=['train', 'val', 'test'],
                        batch_size=eval_batch_size,
                        loss_fn=loss_fn,
                        muls_init=loss_fn.muls_init[stop_pack_idx_torch],
                        label_std=float(regression_label_stats.std),
                        device=device,
                    )
                    if save_uncertainty_predictions
                    else None
                )
            report['eval_batch_size'] = eval_batch_size

            if final_uncertainty_predictions is not None:
                for part, diagnostics in final_uncertainty_predictions.items():
                    for name, values in diagnostics.items():
                        uncertainty_prediction_chunks.setdefault(
                            f'{part}__{name}', []
                        ).append(values)

            # Assemble the new results.
            final_state.extend(
                ids=state.ids[stop_pack_idx],
                steps=state.best_steps[stop_pack_idx],
                predictions=final_predictions,
                predictions_torch=final_predictions_torch,
            )
            new_experiments = assemble_experiments(
                state=state,
                final_metrics=final_metrics,
                pack_idx=stop_pack_idx,
                time_elapsed=timer.elapsed(),
            )
            experiments.extend(new_experiments)
            if hyperparameter_sampler is not None:
                for experiment in new_experiments:
                    report['n_trials'] += hyperparameter_sampler.tell(
                        experiment['report']['id'],
                        get_experiment_val_score(experiment),
                    )

            # Remove the stopped models.
            pack_remove(
                model, optimizer, state, pack_idx=stop_pack_idx, loss_fn=loss_fn
            )
            if gradient_clipping_norm is not None:
                # Per-member thresholds must follow the pack as it shrinks.
                gradient_clipping_norm = gradient_clipping_norm[
                    keep_pack_idx_torch
                ].clone()

            # After the removal, all existing pack indices become invalid.
            del stop_pack_idx

            # Update the report.
            report['n_models'] += n_models_to_stop
            report['step'] = step
            if track_experiments:
                report['experiments'] = experiments
            if track_best_experiment:
                report['best'], _ = get_best_experiment(
                    report.get('best'), new_experiments
                )
            report['time'] = timer.elapsed()

            # Make the update visible.
            lib.experiment.dump_report(exp, report)

            del final_predictions, final_predictions_torch
            _free_mps_memory()

        del eval_predictions, eval_predictions_torch
        _free_mps_memory()

        # Update online ensembles.
        if online_ensembles is None:
            first_online_ensemble_improved = False
        else:
            ensemble_reports, first_online_ensemble_improved = update_online_ensembles(
                online_ensembles,
                task=dataset.task,
                step=step,
                timer=timer,
                running_ids=state.ids,
                running_steps=state.steps,
                running_best_predictions=state.best_predictions,
                running_best_predictions_torch=state.best_predictions_torch,
                finished_ids=final_state.ids,
                finished_steps=final_state.steps,
                finished_predictions=final_state.predictions,
                finished_predictions_torch=final_state.predictions_torch,
                running_latest_predictions=latest_predictions,
                running_latest_predictions_torch=latest_predictions_torch,
            )

            assert online_ensemble_history is not None
            for ensemble_name, ensemble_report in ensemble_reports.items():
                report['online_ensembles'][ensemble_name]['report'] = ensemble_report
                online_ensemble_history.setdefault(ensemble_name, []).append(
                    deepcopy(ensemble_report)
                )
                if config.get('track_online_ensemble_history', False):
                    report['online_ensembles'][ensemble_name]['report']['history'] = (
                        online_ensemble_history[ensemble_name]
                    )
                del ensemble_name, ensemble_report

        del latest_predictions, latest_predictions_torch
        _free_mps_memory()

        # Compute statistics.
        training_throughput = math.trunc(epoch_size / epoch_training_duration)
        total_training_throughput = training_throughput * pack_size_before_stopping
        epoch_mean_loss = statistics.fmean(
            torch.stack(batch_losses).tolist(), batch_sizes
        )
        if track_experiments:
            mean_scores, mean_scores_improved = _get_mean_scores(
                mean_scores, experiments, state, eval_parts
            )
            del mean_scores_improved  # Currently unused.
        if track_best_experiment:
            best_scores, best_scores_improved = _get_best_scores(
                report.get('best'), best_scores, state
            )
        if online_ensembles is not None:
            # Here, we rely on the ordered nature of Python dicts.
            first_online_ensemble_experiment = next(
                iter(report['online_ensembles'].values())
            )
            first_online_ensemble_scores = {
                part: part_metrics['score']
                for part, part_metrics in (
                    first_online_ensemble_experiment['report']
                    .get('metrics', {})
                    .items()
                )
            }

        epochs_numlog.append(
            {'step': step, 'time': timer.elapsed(), 'loss': epoch_mean_loss}
        )

        # Print metrics.
        epoch_evaluation_duration = time.perf_counter() - epoch_evaluation_start_time
        mean_scores_message = (
            None
            if mean_scores is None
            else ' '.join(
                f'[{part[0]}] {score:.3f}' for part, score in mean_scores.items()
            )
        )
        best_scores_message = (
            None
            if best_scores is None
            else ' '.join(
                f'[{part[0]}*] {score:.3f}' for part, score in best_scores.items()
            )
        )
        first_online_ensemble_scores_message = (
            None
            if first_online_ensemble_scores is None or not first_online_ensemble_scores
            else ' '.join(
                f'[{part[0]}$] {score:.3f}'
                for part, score in first_online_ensemble_scores.items()
            )
        )
        print(
            f'{"$" if first_online_ensemble_improved else " "}'
            f'{"*" if best_scores_improved else " "}'
            f' [E] {step // epoch_size:<3}'
            f' [T] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [L] {epoch_mean_loss / pack_size_before_stopping:.3f}'
            f' [M] {report["n_models"]:<3}'
            f'{"" if mean_scores_message is None else f" {mean_scores_message}"}'
            f'{"" if best_scores_message is None else f" {best_scores_message}"}'
            f'{"" if first_online_ensemble_scores_message is None else f" {first_online_ensemble_scores_message}"}'  # noqa: E501
            f' [it/s] {training_throughput:<3} | {total_training_throughput:<5}'
            # f' [e/t] {epoch_evaluation_duration / epoch_training_duration:.3f}'
        )
        del epoch_evaluation_duration

        # >>> Validation
        pack_validate(model, optimizer, state)

    pack_validate(model, optimizer, state)
    report['time'] = timer.elapsed()

    # >>> Main artifacts
    lib.experiment.dump_checkpoint(
        exp,
        {
            'report': report,
            'step': step,
            'random_state': delu.random.get_state(),
            'batch_generator': batch_generator.get_state(),
            'hyperparameter_sampler': hyperparameter_sampler,
            'timer': timer,
        },
    )

    # NOTE
    # The order of values in pack-related artifacts follows the order of finishing,
    # NOT the order of IDs.

    # Patch experiments to include all configs (even for unfinished models)
    experiments_with_all_configs = []
    for model_id in range(config['n_models']):
        # Try to find finished experiment with this ID
        finished_experiment = next(
            (exp for exp in experiments if exp['report']['id'] == model_id), None
        )

        if finished_experiment is not None:
            experiments_with_all_configs.append(finished_experiment)
        else:
            experiment = {}
            if all_configs is not None:
                experiment['config'] = all_configs[model_id]
            experiments_with_all_configs.append(experiment)

    exp.joinpath('experiments.json').write_text(
        json.dumps(experiments_with_all_configs, indent=4)
    )
    if config.get('save_final_predictions', True):
        np.savez(exp / 'predictions.npz', **final_state.predictions)  # type: ignore

    if save_uncertainty_predictions:
        np.savez_compressed(
            exp / 'uncertainty_predictions.npz',
            ids=final_state.ids,
            **{
                key: np.concatenate(values, axis=PACK_DIM)
                for key, values in uncertainty_prediction_chunks.items()
            },
            **{
                f'{part}__target': np.asarray(dataset.task.labels[part])
                for part in ('train', 'val', 'test')
            },
        )

    if online_ensemble_predictions is not None:
        assert online_ensemble_history is not None
        np.savez(
            exp / 'online_ensemble_predictions.npz',
            **lib.util.flatten_dict(
                {str(i): x for i, x in enumerate(online_ensemble_predictions)}
            ),
        )
        exp.joinpath('online_ensemble_history.json').write_text(
            json.dumps(online_ensemble_history, indent=4)
        )

    # >>> Save numerical logs
    np.savez(
        exp / 'numlog.npz',
        **lib.util.flatten_dict(
            {
                'steps': vendor.tabpack.utils.numpy_stack(
                    vendor.tabpack.utils.to_numpy(steps_numlog)  # type: ignore
                ),
                'epochs': vendor.tabpack.utils.numpy_stack(
                    vendor.tabpack.utils.to_numpy(epochs_numlog)  # type: ignore
                ),
                'pack': {
                    'epochs': vendor.tabpack.utils.numpy_concatenate(
                        pack_epochs_numlog
                    ),
                },
            }
        ),
    )

    # >>> Finish
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
