# Vendored pytabkit snapshot, adapted for the regression benchmark.
# Source: https://github.com/dholzmueller/pytabkit
# License: Apache-2.0; see LICENSE.txt. Local changes: see VENDOR.md.

from pytabkit.models.training.lightning_callbacks import ModelCheckpointCallback

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from typing import List, Optional, Dict, Any
import datetime
import time
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from pytabkit.models.data.data import ParallelDictDataLoader, DictDataset
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
from pytabkit.models.nn_models.base import Layer
from pytabkit.models.optim.optimizers import get_opt_class
from pytabkit.models.training.nn_creator import NNCreator
from pytabkit.models.training.logging import StdoutLogger, Logger
from pytabkit.models.training.metrics import Metrics
from pytabkit.models.training.scheduling import LearnerProgress


def postprocess_multiquantile(
    y_pred: torch.Tensor,
    val_metric_name: Optional[str] = None,
    sort_quantile_predictions: bool = True,
    **config,
):
    if (
        val_metric_name is None
        or not val_metric_name.startswith('multi_pinball(')
        or not sort_quantile_predictions
    ):
        return y_pred

    quantiles = [
        float(q_str) for q_str in val_metric_name[len('multi_pinball(') : -1].split(',')
    ]
    if not all([a <= b for a, b in zip(quantiles[:-1], quantiles[1:])]):
        raise ValueError(f'Quantiles {quantiles} must be sorted')

    return y_pred.sort(dim=-1)[0]


class TabNNModule(pl.LightningModule):
    def __init__(
        self,
        n_epochs: int = 256,
        logger: Optional[Logger] = None,
        fit_params: Optional[List[Dict[str, Any]]] = None,
        **config,
    ):
        """
        Pytorch Lightning Module for building and training a pytorch NN for tabular data.
        The core of the module is the NNCreatorInterface, which is used to create the model, the callbacks,
        the hyperparameter manager and the dataloaders. The TabNNModule is responsible for the training loop,
        (optional) validation and inference.
        """
        super().__init__()
        self.my_logger = logger or StdoutLogger(
            verbosity_level=config.get('verbosity', 0)
        )
        # todo: improve this
        self.creator = NNCreator(n_epochs=n_epochs, fit_params=fit_params, **config)

        self.hp_manager = self.creator.hp_manager
        self.model: Optional[Layer] = None
        self.criterion = None
        self.train_dl = None

        self.progress = LearnerProgress()
        self.progress.max_epochs = n_epochs
        self.fit_params = fit_params

        # Validation
        self.val_preds = []
        self.old_training = None
        self.val_dl = None
        self.save_best_params = True
        self.val_metric_names = None
        self.epoch_mean_val_errors = None
        self.best_mean_val_errors = None
        self.best_mean_val_epochs = None
        self.best_val_errors = None
        self.best_val_epochs = None
        self.has_stopped_list = None
        self.callbacks = None
        # will contain {val_metric_name: ModelCheckpointCallback(..., val_metric_name)}
        self.ckpt_callbacks = dict()

        # LightningModule
        self.automatic_optimization = False

        self.config = config

        # EMA model (initialized in compile_model if ema_momentum is set)
        self.ema_model: Optional[AveragedModel] = None
        self.ema_momentum: Optional[float] = config.get('ema_momentum', None)

        # Logging state
        self._epoch_batch_losses: List[torch.Tensor] = []
        self._epoch_start_time: float = 0.0
        self._fit_start_time: float = 0.0

    def compile_model(
        self,
        ds: DictDataset,
        idxs_list: List[SplitIdxs],
        interface_resources: InterfaceResources,
    ):
        """
        Method to create the model and all other training dependencies given the dataset and the assigned resources.
        Once this is called, the module is ready for training.
        """
        self.creator.setup_from_dataset(
            ds, idxs_list=idxs_list, interface_resources=interface_resources
        )
        self.is_classification_ = ds.tensor_infos['y'].is_cat()
        self.model = self.creator.create_model(ds, idxs_list=idxs_list)
        self.train_dl, self.val_dl = self.creator.create_dataloaders(ds)
        self.criterion, self.val_metric_names = self.creator.get_criterions()

        # Initialize EMA model if ema_momentum is set
        if self.ema_momentum is not None:
            self.ema_model = AveragedModel(
                self.model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_momentum),
                use_buffers=True,
            )

    def create_callbacks(self):
        """Helper method to return callbacks for the trainer.fit callback argument."""
        assert self.val_metric_names is not None
        self.callbacks = self.creator.create_callbacks(
            self.model, self.my_logger, self.val_metric_names
        )
        self.ckpt_callbacks = {}
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpointCallback):
                self.ckpt_callbacks[callback.val_metric_name] = callback
        return self.callbacks

    def get_predict_dataloader(self, ds: DictDataset):
        """Helper method to create a dataloader for inference."""
        ds_x, _ = ds.split_xy()
        ds_x = self.creator.static_model.forward_ds(ds_x)
        # ds_x = self.static_model.forward_ds(ds_x)
        idxs_single = torch.arange(ds.n_samples, dtype=torch.long)
        n_ens = self.config.get('n_ens', 1)
        idxs = idxs_single[None, :].expand(
            self.creator.n_tt_splits * self.creator.n_tv_splits * n_ens, -1
        )

        return ParallelDictDataLoader(
            ds=ds_x,
            idxs=idxs,
            batch_size=self.creator.config.get('predict_batch_size', 1024),
        )

    # ----- Start LightningModule Methods -----
    def on_fit_start(self):
        self.model.train()
        self.optimizers().train()
        self._fit_start_time = time.perf_counter()
        # mean val errors will not be accurate if all epochs after this yield NaN
        self.best_mean_val_errors = {
            val_metric_name: [np.inf] * self.creator.n_tt_splits
            for val_metric_name in self.val_metric_names
        }
        # epoch 0 counts as before training, epoch 1 is first epoch
        self.best_mean_val_epochs = {
            val_metric_name: [0] * self.creator.n_tt_splits
            for val_metric_name in self.val_metric_names
        }
        # don't use simpler notation of the form [[]] * 2 because this will have two references to the same inner array!
        self.best_val_errors = {
            val_metric_name: [
                [np.inf] * self.creator.n_tv_splits
                for i in range(self.creator.n_tt_splits)
            ]
            for val_metric_name in self.val_metric_names
        }
        self.best_val_epochs = {
            val_metric_name: [
                [0] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)
            ]
            for val_metric_name in self.val_metric_names
        }
        self.has_stopped_list = {
            val_metric_name: [
                [False] * self.creator.n_tv_splits
                for i in range(self.creator.n_tt_splits)
            ]
            for val_metric_name in self.val_metric_names
        }

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()
        self._epoch_batch_losses = []

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        opt = self.optimizers()

        if self.config.get('train_metric_name', None) in ('softbin', 'sbvar', 'hlgauss', 'quantile', 'adaqr_v9', 'adaqr'):
            assert self.criterion is None
            loss = output['loss']
        else:
            loss = self.criterion(output['x_cont'], output['y']).sum()  # type: ignore

        self.manual_backward(loss)
        opt.step(loss=loss)
        opt.zero_grad()

        # Update EMA model if enabled
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)

        self.progress.total_samples += batch['y'].shape[-2]
        self.progress.epoch_float = (
            self.progress.total_samples / self.train_dl.get_num_iterated_samples()
        )

        # Track batch loss for logging
        self._epoch_batch_losses.append(loss.detach())

        return loss

    def on_validation_start(self):
        self.val_preds = []
        self.model.eval()
        if self.ema_model is not None:
            self.ema_model.eval()

    def validation_step(self, batch, batch_idx):
        # Use EMA model for validation if available
        model_for_val = self.ema_model if self.ema_model is not None else self.model
        model_for_val.eval()
        self.val_preds.append(model_for_val(batch)['x_cont'])

    def on_validation_epoch_end(self):
        self.model.train()

        y_pred = self._postprocess_ens_pred(torch.cat(self.val_preds, dim=-2))
        y_pred = postprocess_multiquantile(y_pred, **self.config)

        n_ens = self.config.get('n_ens', 1)
        # y is duplicated by the dataloader as well in the ensemble case, deduplicate it
        y = self.val_dl.val_y[::n_ens]

        use_early_stopping = self.config.get('use_early_stopping', False)
        early_stopping_additive_patience = self.config.get(
            'early_stopping_additive_patience', 20
        )
        early_stopping_multiplicative_patience = self.config.get(
            'early_stopping_multiplicative_patience', 2
        )

        val_score_improved = False
        current_val_error_for_log: Optional[float] = None

        for val_metric_name in self.val_metric_names:
            val_errors = torch.as_tensor(
                [
                    Metrics.apply(y_pred[i, :, :], y[i, :, :], val_metric_name)
                    for i in range(y_pred.shape[0])
                ]
            )
            val_errors = val_errors.view(
                self.creator.n_tt_splits, self.creator.n_tv_splits
            )
            mean_val_errors = val_errors.mean(dim=-1)  # mean over cv/refit dimension
            mean_val_error = mean_val_errors.mean().item()

            # Store val error for logging (use first metric)
            if current_val_error_for_log is None:
                current_val_error_for_log = mean_val_error

            current_epoch = self.progress.epoch + 1

            for tt_split_idx in range(self.creator.n_tt_splits):
                use_last_best_epoch = self.config.get('use_last_best_epoch', True)

                has_stopped = self.has_stopped_list[val_metric_name][tt_split_idx]

                # compute best single-split validation errors
                for tv_split_idx in range(self.creator.n_tv_splits):
                    if use_early_stopping and not has_stopped[tv_split_idx]:
                        if (
                            current_epoch
                            > early_stopping_multiplicative_patience
                            * self.best_val_epochs[val_metric_name][tt_split_idx][
                                tv_split_idx
                            ]
                            + early_stopping_additive_patience
                        ):
                            has_stopped[tv_split_idx] = True

                    if not has_stopped[tv_split_idx]:
                        # compute best validation errors
                        current_err = val_errors[tt_split_idx, tv_split_idx].item()
                        best_err = self.best_val_errors[val_metric_name][tt_split_idx][
                            tv_split_idx
                        ]
                        # use <= on purpose such that latest epoch among tied best epochs is kept
                        # this has been slightly beneficial for accuracy in previous experiments
                        improved = (
                            current_err <= best_err
                            if use_last_best_epoch
                            else current_err < best_err
                        )
                        if improved:
                            self.best_val_errors[val_metric_name][tt_split_idx][
                                tv_split_idx
                            ] = current_err
                            self.best_val_epochs[val_metric_name][tt_split_idx][
                                tv_split_idx
                            ] = self.progress.epoch + 1

                if not any(has_stopped):
                    # compute best mean validation errors (averaged over sub-splits (cv/refit))
                    # use <= on purpose such that latest epoch among tied best epochs is kept
                    # this has been slightly beneficial for accuracy in previous experiments
                    improved = (
                        mean_val_errors[tt_split_idx]
                        <= self.best_mean_val_errors[val_metric_name][tt_split_idx]
                        if use_last_best_epoch
                        else mean_val_errors[tt_split_idx]
                        < self.best_mean_val_errors[val_metric_name][tt_split_idx]
                    )
                    if improved:
                        val_score_improved = True
                        self.best_mean_val_errors[val_metric_name][tt_split_idx] = (
                            mean_val_errors[tt_split_idx]
                        )
                        self.best_mean_val_epochs[val_metric_name][tt_split_idx] = (
                            self.progress.epoch + 1
                        )

        # Compute epoch loss and timing for logging
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - self._epoch_start_time
        total_elapsed = epoch_end_time - self._fit_start_time

        epoch_loss = 0.0
        if self._epoch_batch_losses:
            epoch_loss = torch.stack(self._epoch_batch_losses).mean().item()

        n_batches = len(self._epoch_batch_losses) if self._epoch_batch_losses else 1
        it_per_sec = int(n_batches / epoch_duration) if epoch_duration > 0 else 0

        current_epoch = self.progress.epoch + 1

        # Format: * [epoch] 1   [val] -0.603 [loss] 0.4687 [time] 0:00:00 [it/s] 131
        improved_marker = '*' if val_score_improved else ' '
        time_str = str(datetime.timedelta(seconds=int(total_elapsed)))
        val_score = (
            -current_val_error_for_log if current_val_error_for_log is not None else 0.0
        )

        print(
            f'{improved_marker} [epoch] {current_epoch:<3} '
            f'[val] {val_score:.5f} '
            f'[loss] {epoch_loss:.4f} '
            f'[time] {time_str} '
            f'[it/s] {it_per_sec:>3}'
        )

        self.progress.epoch += 1

        if use_early_stopping and all(
            all([all(sub_lst) for sub_lst in lst])
            for lst in self.has_stopped_list.values()
        ):
            self.trainer.should_stop = True

    def on_fit_end(self):
        if self.creator.config.get('use_best_epoch', True):
            self.fit_params = [
                {
                    'stop_epoch': {
                        val_metric_name: self.best_mean_val_epochs[val_metric_name][i]
                        for val_metric_name in self.val_metric_names
                    },
                    'best_indiv_stop_epochs': {
                        val_metric_name: self.best_val_epochs[val_metric_name][i]
                        for val_metric_name in self.val_metric_names
                    },
                }
                for i in range(self.creator.n_tt_splits)
            ]
        else:
            self.fit_params = [
                {
                    'stop_epoch': {
                        val_metric_name: self.progress.max_epochs
                        for val_metric_name in self.val_metric_names
                    }
                }
                for i in range(self.creator.n_tt_splits)
            ]

        # put in eval() mode for predict(), so we don't need to save the trainer and the optimizer state
        self.optimizers(use_pl_optimizer=False).eval()

        # delete stuff so we don't save the dataset when pickling RealMLP
        del self.creator.train_idxs
        del self.creator.val_idxs
        del self.train_dl
        del self.val_dl
        del self.val_preds
        del self.callbacks
        del self.ckpt_callbacks

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Use EMA model for prediction if available
        model_for_pred = self.ema_model if self.ema_model is not None else self.model
        model_for_pred.eval()
        with torch.no_grad():
            return self._postprocess_ens_pred(model_for_pred(batch)['x_cont'].to('cpu'))

    def _postprocess_ens_pred(self, y_pred: torch.Tensor) -> torch.Tensor:
        # if n_ens > 1, we need to average the predictions of the ensemble members
        n_ens = self.config.get('n_ens', 1)
        if n_ens == 1:
            return y_pred

        y_pred = y_pred.reshape(y_pred.shape[0] // n_ens, n_ens, *y_pred.shape[1:])
        if self.is_classification_ and not self.config.get(
            'ens_av_before_softmax', False
        ):
            y_pred = torch.softmax(y_pred, dim=-1)
            y_pred = y_pred.mean(dim=1)
            y_pred = torch.log(y_pred + 1e-30)
        else:
            y_pred = y_pred.mean(dim=1)
        return y_pred

    def configure_optimizers(self):
        param_groups = [{'params': [p], 'lr': 0.01} for p in self.model.parameters()]
        return get_opt_class(self.config.get('opt', 'adam'))(
            param_groups, self.hp_manager, **self.config
        )

    def restore_ckpt_for_val_metric_name(self, val_metric_name: str):
        self.ckpt_callbacks[val_metric_name].restore(self)

    # from https://github.com/Lightning-AI/pytorch-lightning/discussions/19759
    # def on_fit_start(self) -> None:
    #     self.optimizers().train()  # already above

    # def on_predict_start(self) -> None:
    #     print(f'predict start')
    #     self.optimizers(use_pl_optimizer=False).eval()

    def on_validation_model_eval(self) -> None:
        self.model.eval()
        self.optimizers(use_pl_optimizer=False).eval()

    def on_validation_model_train(self) -> None:
        model_for_pred: nn.Module = (  # type: ignore
            self.ema_model if self.ema_model is not None else self.model
        )
        model_for_pred.train()
        self.optimizers(use_pl_optimizer=False).train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        self.optimizers(use_pl_optimizer=False).eval()

    def on_test_model_train(self) -> None:
        model_for_pred: nn.Module = (  # type: ignore
            self.ema_model if self.ema_model is not None else self.model
        )
        self.optimizers(use_pl_optimizer=False).train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        self.model.eval()
        # don't do it here in case we don't have the optimizers at predict time
        # self.optimizers(use_pl_optimizer=False).eval()

    def to(self, *args: Any, **kwargs: Any) -> 'TabNNModule':
        super().to(*args, **kwargs)
        # print(f'moving static model to {args} {kwargs}')
        self.creator.static_model.to(*args, **kwargs)
