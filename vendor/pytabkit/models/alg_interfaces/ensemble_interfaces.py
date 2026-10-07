# Vendored pytabkit snapshot, adapted for the regression benchmark.
# Source: https://github.com/dholzmueller/pytabkit
# License: Apache-2.0; see LICENSE.txt. Local changes: see VENDOR.md.

import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch


def _get_mp_context():
    """Get multiprocessing context that works with CUDA."""
    return mp.get_context("spawn")


from pytabkit.models.alg_interfaces.alg_interfaces import (
    SingleSplitAlgInterface,
    AlgInterface,
)
from pytabkit.models.alg_interfaces.base import (
    SplitIdxs,
    InterfaceResources,
    RequiredResources,
)
from pytabkit.models.data.data import DictDataset, TaskType
from pytabkit.models.torch_utils import cat_if_necessary
from pytabkit.models.training.logging import Logger, StdoutLogger
from pytabkit.models.training.metrics import Metrics
from pytabkit.models.utils import ObjectLoadingContext, serialize, deserialize


def _fit_and_eval_alg_worker(
    alg_idx: int,
    alg_interface: AlgInterface,
    ds: DictDataset,
    idxs_list: List[SplitIdxs],
    interface_resources: InterfaceResources,
    tmp_folder: Optional[Path],
    val_idxs: torch.Tensor,
    y_oob: torch.Tensor,
    val_metric_name: str,
    name: str,
) -> Dict[str, Any]:
    """
    Worker function for parallel HPO. Trains one AlgInterface and computes validation loss.
    Runs in a separate process when parallel training is enabled.
    """
    start = time.time()

    sub_tmp_folders = [tmp_folder / str(alg_idx) if tmp_folder is not None else None]
    worker_logger = StdoutLogger(verbosity_level=0)

    try:
        alg_interface.fit(
            ds,
            idxs_list,
            interface_resources,
            worker_logger,
            sub_tmp_folders,
            name + f"sub-alg-{alg_idx}",
        )
        y_preds = alg_interface.predict(ds)
        y_pred_oob = cat_if_necessary(
            [y_preds[j, val_idxs[j]] for j in range(val_idxs.shape[0])], dim=0
        )
        loss = Metrics.apply(y_pred_oob.cpu(), y_oob.cpu(), val_metric_name).item()
        sub_fit_params = alg_interface.get_fit_params()[0]

        del y_preds, y_pred_oob
        torch.cuda.empty_cache()

        serialize(f"{tmp_folder}/model_{alg_idx}", alg_interface)

        return dict(
            alg_idx=alg_idx,
            loss=loss,
            sub_fit_params=sub_fit_params,
            duration_s=time.time() - start,
            error=None,
        )
    except Exception as e:
        print(e)
        return dict(
            alg_idx=alg_idx,
            loss=np.inf,
            sub_fit_params=None,
            duration_s=time.time() - start,
            error=str(e),
            fitted_interface=None,
        )


class WeightedPrediction:
    def __init__(self, y_pred_list: List[torch.Tensor], task_type: TaskType):
        self.task_type = task_type
        self.y_pred_converted_list = (
            y_pred_list
            if task_type == TaskType.REGRESSION
            else [torch.softmax(y_pred, dim=-1) for y_pred in y_pred_list]
        )

    def predict_for_weights(self, weights: np.ndarray):
        weights = weights.astype(np.float32)
        norm_weights = weights / np.sum(weights)
        weighted_sum = sum(
            [w * y_pred for w, y_pred in zip(norm_weights, self.y_pred_converted_list)]
        )
        if self.task_type == TaskType.CLASSIFICATION:
            weighted_sum = torch.log(weighted_sum + 1e-30)
        return weighted_sum


class CaruanaEnsembleAlgInterface(SingleSplitAlgInterface):
    """
    Following a simple variant of Caruana et al. (2004), "Ensemble selection from libraries of models"
    without pre-selection of candidates
    """

    def __init__(
        self,
        alg_interfaces: List[AlgInterface],
        fit_params: Optional[List[Dict]] = None,
        **config,
    ):
        super().__init__(fit_params=fit_params, **config)
        self.alg_interfaces = alg_interfaces
        self.task_type = None

    def get_refit_interface(
        self, n_refit: int, fit_params: Optional[List[Dict]] = None
    ) -> "AlgInterface":
        return CaruanaEnsembleAlgInterface(
            [
                alg_interface.get_refit_interface(n_refit=n_refit)
                for alg_interface in self.alg_interfaces
            ],
            fit_params=fit_params or self.fit_params,
        )

    def fit(
        self,
        ds: DictDataset,
        idxs_list: List[SplitIdxs],
        interface_resources: InterfaceResources,
        logger: Logger,
        tmp_folders: List[Optional[Path]],
        name: str,
    ) -> None:
        assert len(idxs_list) == 1

        # if tmp_folders is specified, then models will be saved there instead of holding all of them in memory
        tmp_folder = tmp_folders[0]
        self.alg_contexts_ = [
            ObjectLoadingContext(
                ai, None if tmp_folder is None else tmp_folder / f"model_{i}"
            )
            for i, ai in enumerate(self.alg_interfaces)
        ]
        # store copies here, but the ones that will actually be trained are in alg_contexts_
        # this means that models should not be held in RAM all the time
        self.alg_interfaces = copy.deepcopy(self.alg_interfaces)

        sub_fit_params = []

        # train sub-models
        for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
            with alg_ctx as alg_interface:
                sub_tmp_folders = [
                    tmp_folder / str(alg_idx) if tmp_folder is not None else None
                    for tmp_folder in tmp_folders
                ]
                if self.config.get("diversify_seeds", False):
                    sub_idxs_list = [
                        SplitIdxs(
                            train_idxs=idxs.train_idxs,
                            val_idxs=idxs.val_idxs,
                            test_idxs=idxs.test_idxs,
                            split_seed=idxs.split_seed + alg_idx,
                            sub_split_seeds=[
                                sss + alg_idx for sss in idxs.sub_split_seeds
                            ],
                            split_id=idxs.split_id,
                        )
                        for idxs in idxs_list
                    ]
                else:
                    sub_idxs_list = idxs_list
                alg_interface.fit(
                    ds,
                    sub_idxs_list,
                    interface_resources,
                    logger,
                    sub_tmp_folders,
                    name + f"sub-alg-{alg_idx}",
                )
                sub_fit_params.append(alg_interface.get_fit_params()[0])

        if self.fit_params is not None:
            # this is the refit stage, there is no validation data set to determine the weights on,
            # instead the weights are already in fit_params
            return
        if idxs_list[0].val_idxs is None:
            raise ValueError(
                "CaruanaEnsembleAlgInterface.fit(): Neither a validation set "
                "nor ensemble weights were provided"
            )

        self.task_type = (
            TaskType.CLASSIFICATION
            if ds.tensor_infos["y"].get_cat_size_product() > 0
            else TaskType.REGRESSION
        )
        val_metric_name = self.config.get(
            "ens_weight_metric_name", self.config.get("val_metric_name", None)
        )
        if val_metric_name is None:
            val_metric_name = Metrics.default_val_metric_name(task_type=self.task_type)

        n_caruana_steps = self.config.get(
            "n_caruana_steps", 40
        )  # default value is taken from TabRepo paper (IIRC)

        y_preds_oob_list = []

        time_limit_s: Optional[float] = self.config.get("time_limit_s", None)
        start_time = time.time()

        for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
            if (
                alg_idx > 0
                and time_limit_s is not None
                and (alg_idx + 1) / alg_idx * (time.time() - start_time) > time_limit_s
            ):
                break
            with alg_ctx as alg_interface:
                y_preds = alg_interface.predict(ds)
                # get out-of-bag predictions
                y_preds_oob_list.append(
                    cat_if_necessary(
                        [
                            y_preds[j, idxs_list[0].val_idxs[j]]
                            for j in range(idxs_list[0].val_idxs.shape[0])
                        ],
                        dim=0,
                    )
                )

        # get out-of-bag labels
        y = ds.tensors["y"]
        y_oob = cat_if_necessary(
            [
                y[idxs_list[0].val_idxs[j]]
                for j in range(idxs_list[0].val_idxs.shape[0])
            ],
            dim=0,
        )

        weights = np.zeros(len(self.alg_contexts_), dtype=np.int32)
        best_weights = np.copy(weights)
        best_loss = np.inf

        wp = WeightedPrediction(y_preds_oob_list, self.task_type)

        allow_negative_weights = self.config.get("allow_negative_weights", False)

        for step_idx in range(n_caruana_steps):
            best_step_weights = None
            best_step_loss = np.inf
            for weight_idx in range(weights.shape[0]):
                weights[weight_idx] += 1

                y_pred_oob = wp.predict_for_weights(weights)
                loss = Metrics.apply(
                    y_pred_oob.cpu(), y_oob.cpu(), val_metric_name
                ).item()
                # print(f'{weights=}, {loss=}')
                if loss < best_step_loss:
                    best_step_loss = loss
                    best_step_weights = np.copy(weights)

                weights[weight_idx] -= 1

                # negative weights option
                # check weights >= 2 allowing for floating-point errors
                if allow_negative_weights and np.sum(weights) >= 1.5:
                    weights[weight_idx] -= 1

                    y_pred_oob = wp.predict_for_weights(weights)
                    loss = Metrics.apply(
                        y_pred_oob.cpu(), y_oob.cpu(), val_metric_name
                    ).item()
                    # print(f'{weights=}, {loss=}')
                    if loss < best_step_loss:
                        best_step_loss = loss
                        best_step_weights = np.copy(weights)

                    weights[weight_idx] += 1

            if best_step_loss < best_loss:
                best_loss = best_step_loss
                best_weights = np.copy(best_step_weights)

            weights = best_step_weights

        logger.log(2, f"Obtained ensemble weights: {best_weights}")

        self.fit_params = [
            dict(alg_weights=best_weights.tolist(), sub_fit_params=sub_fit_params)
        ]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        weights = self.fit_params[0]["alg_weights"]
        sparse_weights = []
        sparse_preds = []
        for i, w in enumerate(weights):
            if w != 0:
                with self.alg_contexts_[i] as alg_interface:
                    sparse_preds.append(alg_interface.predict(ds))
                    sparse_weights.append(w)
        wp = WeightedPrediction(sparse_preds, task_type=self.task_type)
        return wp.predict_for_weights(weights=np.asarray(sparse_weights))

    def get_required_resources(
        self,
        ds: DictDataset,
        n_cv: int,
        n_refit: int,
        n_splits: int,
        split_seeds: List[int],
        n_train: int,
    ) -> RequiredResources:
        single_resources = [
            ssi.get_required_resources(
                ds,
                n_cv,
                n_refit,
                n_splits=n_splits,
                split_seeds=split_seeds,
                n_train=n_train,
            )
            for ssi in self.alg_interfaces
        ]
        return RequiredResources.combine_sequential(single_resources)

    def to(self, device: str) -> None:
        for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
            with alg_ctx as alg_interface:
                alg_interface.to(device)


class AlgorithmSelectionAlgInterface(SingleSplitAlgInterface):
    """
    Picks the best model out of a list of candidates.
    """

    def __init__(
        self,
        alg_interfaces: List[AlgInterface],
        fit_params: Optional[List[Dict]] = None,
        **config,
    ):
        super().__init__(fit_params=fit_params, **config)
        self.alg_interfaces = alg_interfaces
        self.task_type = None
        self.progress_log_level = config.get("alg_sel_progress_log_level", 1)
        self.max_parallel_algs = config.get("max_parallel_algs", 1)

    def get_refit_interface(
        self, n_refit: int, fit_params: Optional[List[Dict]] = None
    ) -> "AlgInterface":
        # todo: could use sub_fit_params
        refit_interfaces = []
        for alg_context in self.alg_contexts_:
            with alg_context as alg_interface:
                refit_interfaces.append(
                    alg_interface.get_refit_interface(n_refit=n_refit)
                )
        return AlgorithmSelectionAlgInterface(
            refit_interfaces, fit_params=fit_params or self.fit_params
        )

    def fit(
        self,
        ds: DictDataset,
        idxs_list: List[SplitIdxs],
        interface_resources: InterfaceResources,
        logger: Logger,
        tmp_folders: List[Optional[Path]],
        name: str,
    ) -> None:
        assert len(idxs_list) == 1

        # if tmp_folders is specified, then models will be saved there instead of holding all of them in memory
        tmp_folder = tmp_folders[0]
        self.alg_contexts_ = [
            ObjectLoadingContext(
                ai, None if tmp_folder is None else tmp_folder / f"model_{i}"
            )
            for i, ai in enumerate(self.alg_interfaces)
        ]
        # store copies here, but the ones that will actually be trained are in alg_contexts_
        # this means that models should not be held in RAM all the time
        self.alg_interfaces = copy.deepcopy(self.alg_interfaces)

        if self.fit_params is not None:
            # this is the refit stage, there is no validation data set to determine the best model on,
            # instead the best model index is already in fit_params
            best_alg_idx = self.fit_params[0]["best_alg_idx"]
            sub_tmp_folders = [
                tmp_folder / str(best_alg_idx) if tmp_folder is not None else None
                for tmp_folder in tmp_folders
            ]
            with self.alg_contexts_[best_alg_idx] as alg_interface:
                alg_interface.fit(
                    ds,
                    idxs_list,
                    interface_resources,
                    logger,
                    sub_tmp_folders,
                    name + f"sub-alg-{best_alg_idx}",
                )

            return

        if idxs_list[0].val_idxs is None:
            raise ValueError(
                "CaruanaEnsembleAlgInterface.fit(): Neither a validation set "
                "nor fit_params were provided"
            )

        self.task_type = (
            TaskType.CLASSIFICATION
            if ds.tensor_infos["y"].get_cat_size_product() > 0
            else TaskType.REGRESSION
        )
        val_metric_name = self.config.get(
            "alg_sel_metric_name", self.config.get("val_metric_name", None)
        )
        if val_metric_name is None:
            val_metric_name = Metrics.default_val_metric_name(task_type=self.task_type)

        # get out-of-bag labels
        y = ds.tensors["y"]
        y_oob = cat_if_necessary(
            [
                y[idxs_list[0].val_idxs[i]]
                for i in range(idxs_list[0].val_idxs.shape[0])
            ],
            dim=0,
        )

        best_alg_idx = 0
        best_alg_loss = np.inf
        best_sub_fit_params = None

        time_limit_s: Optional[float] = self.config.get("time_limit_s", None)
        start_time = time.time()
        n_algs = len(self.alg_contexts_)
        trial_durations = []
        val_idxs = idxs_list[0].val_idxs

        if self.max_parallel_algs > 1:
            # Parallel path: train multiple models concurrently
            best_alg_idx, best_alg_loss, best_sub_fit_params = self._fit_parallel(
                ds,
                idxs_list,
                interface_resources,
                logger,
                tmp_folder,
                tmp_folders,
                val_idxs,
                y_oob,
                val_metric_name,
                name,
                n_algs,
                time_limit_s,
                start_time,
            )
        else:
            # Sequential path (original behavior with logging)
            old_best_alg_idx = None  # Track previous best for cleanup
            for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
                if (
                    alg_idx > 0
                    and time_limit_s is not None
                    and (alg_idx + 1) / alg_idx * (time.time() - start_time)
                    > time_limit_s
                ):
                    logger.log(
                        self.progress_log_level,
                        f"[alg_sel] Stopping early due to time limit. "
                        f"Completed {alg_idx}/{n_algs} trials.",
                    )
                    break

                trial_start = time.time()
                with alg_ctx as alg_interface:
                    sub_tmp_folders = [
                        tmp_folder / str(alg_idx) if tmp_folder is not None else None
                        for tmp_folder in tmp_folders
                    ]
                    alg_interface.fit(
                        ds,
                        idxs_list,
                        interface_resources,
                        logger,
                        sub_tmp_folders,
                        name + f"sub-alg-{alg_idx}",
                    )
                    y_preds = alg_interface.predict(ds)
                    # get out-of-bag predictions
                    y_pred_oob = cat_if_necessary(
                        [y_preds[j, val_idxs[j]] for j in range(val_idxs.shape[0])],
                        dim=0,
                    )
                    loss = Metrics.apply(
                        y_pred_oob.cpu(), y_oob.cpu(), val_metric_name
                    ).item()

                    improved = loss < best_alg_loss
                    if improved:
                        best_alg_loss = loss
                        best_alg_idx = alg_idx
                        best_sub_fit_params = alg_interface.get_fit_params()[0]
                    else:
                        # Cleanup: prevent serialization for non-best trials
                        alg_ctx.filename = None

                # After context exit: cleanup old best file if we have a new best
                if improved:
                    if old_best_alg_idx is not None and tmp_folder is not None:
                        old_path = tmp_folder / f"model_{old_best_alg_idx}"
                        if old_path.exists():
                            old_path.unlink()
                    old_best_alg_idx = alg_idx

                trial_duration = time.time() - trial_start
                trial_durations.append(trial_duration)
                elapsed = time.time() - start_time
                avg_trial_time = np.mean(trial_durations)
                remaining_trials = n_algs - (alg_idx + 1)
                eta_seconds = remaining_trials * avg_trial_time

                improved_marker = "*" if improved else " "
                logger.log(
                    self.progress_log_level,
                    f"{improved_marker} [alg_sel] {alg_idx + 1}/{n_algs} "
                    f"loss={loss:.4f} best={best_alg_loss:.4f} (idx={best_alg_idx}) "
                    f"time={trial_duration:.1f}s elapsed={elapsed:.0f}s ETA={eta_seconds:.0f}s",
                )

        self.fit_params = [
            dict(best_alg_idx=best_alg_idx, sub_fit_params=best_sub_fit_params)
        ]
        logger.log(2, f"Best algorithm has index {best_alg_idx}")
        logger.log(2, f"Algorithm selection fit parameters: {self.fit_params[0]}")

    def _fit_parallel(
        self,
        ds: DictDataset,
        idxs_list: List[SplitIdxs],
        interface_resources: InterfaceResources,
        logger: Logger,
        tmp_folder: Optional[Path],
        tmp_folders: List[Optional[Path]],
        val_idxs: torch.Tensor,
        y_oob: torch.Tensor,
        val_metric_name: str,
        name: str,
        n_algs: int,
        time_limit_s: Optional[float],
        start_time: float,
    ) -> Tuple[int, float, Optional[Dict]]:
        """Parallel HPO path using ProcessPoolExecutor."""
        best_alg_idx = 0
        best_alg_loss = np.inf
        best_sub_fit_params = None
        old_best_alg_idx = None  # Track previous best for cleanup
        trial_durations = []
        completed = 0

        max_workers = min(self.max_parallel_algs, n_algs)
        logger.log(
            self.progress_log_level,
            f"[alg_sel] Starting parallel HPO with {max_workers} workers for {n_algs} trials",
        )

        mp_context = _get_mp_context()
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ) as executor:
            futures = {}
            for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
                with alg_ctx as alg_interface:
                    future = executor.submit(
                        _fit_and_eval_alg_worker,
                        alg_idx,
                        alg_interface,
                        ds,
                        idxs_list,
                        interface_resources,
                        tmp_folder,
                        val_idxs,
                        y_oob,
                        val_metric_name,
                        name,
                    )
                    futures[future] = alg_idx

            for future in as_completed(futures):
                res = future.result()
                completed += 1
                trial_durations.append(res["duration_s"])

                alg_idx = res["alg_idx"]

                if res["error"] is not None:
                    logger.log(
                        self.progress_log_level,
                        f"  [alg_sel] Trial {alg_idx} failed: {res['error']}",
                    )
                    print("some error")
                    continue

                loss = res["loss"]
                improved = loss < best_alg_loss

                if improved:
                    # Cleanup: delete old best's file
                    if old_best_alg_idx is not None and tmp_folder is not None:
                        old_path = tmp_folder / f"model_{old_best_alg_idx}"
                        if old_path.exists():
                            old_path.unlink()
                    old_best_alg_idx = alg_idx
                    best_alg_loss = loss
                    best_alg_idx = alg_idx
                    best_sub_fit_params = res["sub_fit_params"]
                else:
                    # Cleanup: this trial is not best, delete immediately
                    if tmp_folder is not None:
                        trial_path = tmp_folder / f"model_{alg_idx}"
                        if trial_path.exists():
                            trial_path.unlink()

                elapsed = time.time() - start_time
                avg_trial_time = np.mean(trial_durations)
                remaining_trials = n_algs - completed
                eta_seconds = remaining_trials * avg_trial_time / max_workers

                improved_marker = "*" if improved else " "
                logger.log(
                    self.progress_log_level,
                    f"{improved_marker} [alg_sel] {completed}/{n_algs} "
                    f"(idx={alg_idx}) loss={loss:.4f} best={best_alg_loss:.4f} "
                    f"time={res['duration_s']:.1f}s elapsed={elapsed:.0f}s ETA={eta_seconds:.0f}s",
                )

                if time_limit_s is not None and elapsed + eta_seconds > time_limit_s:
                    logger.log(
                        self.progress_log_level,
                        f"[alg_sel] Stopping early due to time limit. "
                        f"Completed {completed}/{n_algs} trials.",
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        return best_alg_idx, best_alg_loss, best_sub_fit_params

    def predict(self, ds: DictDataset) -> torch.Tensor:
        alg_idx = self.fit_params[0]["best_alg_idx"]
        with self.alg_contexts_[alg_idx] as alg_interface:
            print(alg_interface.alg_interface)
            return alg_interface.predict(ds)

    def get_required_resources(
        self,
        ds: DictDataset,
        n_cv: int,
        n_refit: int,
        n_splits: int,
        split_seeds: List[int],
        n_train: int,
    ) -> RequiredResources:
        # too pessimistic for refit...
        single_resources = [
            ssi.get_required_resources(
                ds,
                n_cv,
                n_refit,
                n_splits=n_splits,
                split_seeds=split_seeds,
                n_train=n_train,
            )
            for ssi in self.alg_interfaces
        ]
        return RequiredResources.combine_sequential(single_resources)

    def to(self, device: str) -> None:
        for alg_idx, alg_ctx in enumerate(self.alg_contexts_):
            with alg_ctx as alg_interface:
                alg_interface.to(device)


class PrecomputedPredictionsAlgInterface(SingleSplitAlgInterface):
    def __init__(
        self,
        y_preds_cv: torch.Tensor,
        y_preds_refit: Optional[torch.Tensor],
        fit_params_cv: Dict,
        fit_params_refit: Optional[Dict],
    ):
        super().__init__()
        self.y_preds_cv = y_preds_cv
        self.y_preds_refit = y_preds_refit
        self.is_refit = None
        self.fit_params_cv = fit_params_cv
        self.fit_params_refit = fit_params_refit

    def get_refit_interface(
        self, n_refit: int, fit_params: Optional[List[Dict]] = None
    ) -> "AlgInterface":
        return self  # todo: does this work?

    def fit(
        self,
        ds: DictDataset,
        idxs_list: List[SplitIdxs],
        interface_resources: InterfaceResources,
        logger: Logger,
        tmp_folders: List[Optional[Path]],
        name: str,
    ) -> None:
        self.is_refit = idxs_list[0].val_idxs is None
        self.fit_params = (
            [self.fit_params_refit] if self.is_refit else [self.fit_params_cv]
        )

    def predict(self, ds: DictDataset) -> torch.Tensor:
        if ds.n_samples != self.y_preds_cv.shape[1]:
            raise ValueError(
                "Prediction can only be performed on the exact same dataset "
                "because this uses precomputed predictions"
            )
        return self.y_preds_refit if self.is_refit else self.y_preds_cv

    def get_required_resources(
        self,
        ds: DictDataset,
        n_cv: int,
        n_refit: int,
        n_splits: int,
        split_seeds: List[int],
        n_train: int,
    ) -> RequiredResources:
        return RequiredResources(
            time_s=1e-5 * ds.n_samples, cpu_ram_gb=2.0, n_threads=1
        )
