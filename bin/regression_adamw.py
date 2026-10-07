import datetime
import math
import statistics
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, Protocol, TypedDict

import delu
import numpy as np
import rtdl_num_embeddings
import tabm
import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from torch import Tensor
from tqdm import tqdm

import lib
import lib.data
import lib.deep
import lib.env
import lib.experiment
import lib.util
import vendor.tabm_reference as tabm_reference
from lib.types import AMPDType, KWArgs, PartKey, PredictionType


def hard_labels(y, centers):
    dists = (y.unsqueeze(1) - centers.T) ** 2
    return dists.argmin(dim=1)


def earth_mover_loss_l2(y_true, y_pred):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
    )


def earth_mover_loss_l1(y_true, y_pred):
    batch_losses = torch.mean(
        torch.abs(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1
    )
    return torch.mean(batch_losses)


def soft_labels(y, centers, sigma):
    dists = ((y.unsqueeze(1) - centers.T) ** 2) / (2 * sigma**2)
    w = torch.exp(-dists)
    return w / w.sum(dim=1, keepdim=True)


def normal_ppf(p):
    # p in [0, 1]
    return torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * p - 1)


def generate_centers(
    *,
    mode: str,
    n_bins: int,
    bin_bound: float,
    dataset: lib.data.Dataset | None = None,
) -> Tensor | None:
    fixed_centers: Tensor
    if mode == 'baseline':
        return
    if mode in ('hard_binning', 'soft_binning'):
        quantiles = torch.linspace(0.01, 0.99, n_bins)
        fixed_centers = normal_ppf(quantiles).unsqueeze(1).float()
    elif mode in ('hard_binning_quantile', 'soft_binning_quantile'):
        assert dataset is not None, 'Dataset is needed for quantile-based binning'
        Y_train = dataset.data['y']['train']
        quantiles = (
            torch.quantile(
                Y_train, torch.linspace(0.0, 1.0, n_bins + 1, device=Y_train.device)
            )
            .unsqueeze(1)
            .float()
        )
        fixed_centers = (quantiles[:-1] + quantiles[1:]) / 2
    elif mode in ('soft_binning_uniform', 'hard_binning_uniform'):
        fixed_centers = (
            torch.linspace(-bin_bound, bin_bound, n_bins).unsqueeze(1).float()
        )
    else:
        return
    return fixed_centers


class Model(nn.Module):
    """
    mode:
        'baseline'         - usual MSE regression
        'softmax_mu'       - softmax + mu-regression
        'hard_binning'     - one-hot binning
        'soft_binning'     - soft-binning with Normal-distribution-quantile bins
        'pretrained_enc'   - pre-trained encoder classification
        'joint_end_to_end'    - end-to-end encoder + classification
        'soft_binning_uniform' - soft-binning with uniform binning
        'hard_binning_uniform' - hard-binning with uniform binning
        'soft_binning_quantile' - soft-binning with Y_train distribution quantile bins
        'hard_binning_quantile' - hard-binning with Y_train distribution quantile bins
    """

    def __init__(
        self,
        *,
        mode: str,
        temperature: float = 1.0,
        n_num_features: int,
        cat_cardinalities: list[int],
        m: int = 1,  # number of dimensions in the regression task (only m = 1 is
        bins: None | list[Tensor],
        # implemented as of now)
        soft_sigma: float = 0.1,  # smoothness of the softmax in the soft_bining mode
        bin_bound: float = 3,  # right bound for the bin space
        n_bins: int = 8,  # number of bins to reformulate the regression tasks
        backbone: KWArgs,
        arch: KWArgs,
        fixed_centers: Tensor | None = None,
        centers: Tensor | None = None,
    ) -> None:
        super().__init__()
        assert n_num_features > 0 or cat_cardinalities
        super().__init__()
        self.cat_module = (
            lib.deep.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        self._n_num_features = n_num_features
        if m > 1:
            raise NotImplementedError('m > 1 is not implemented yet')
        self.m = m
        self.mode = mode
        self.temperature = temperature
        self.n_bins = n_bins

        self.enc_sigma = soft_sigma
        self.model = tabm_reference.Model(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            bins=bins,
            n_classes=(m if mode in ('baseline') else n_bins),
            **arch,
            backbone=backbone,
        )

        if mode == 'baseline':
            assert fixed_centers is None and centers is None, (
                'No centers needed for baseline'
            )
            return
        if mode in (
            'hard_binning',
            'soft_binning',
            'soft_binning_uniform',
            'hard_binning_uniform',
            'soft_binning_quantile',
            'hard_binning_quantile',
        ):
            assert fixed_centers is not None, (
                f'fixed_centers are required for mode: {mode}'
            )
            assert centers is None, f'centers are fixed in {mode=}'
            self.register_buffer('fixed_centers', fixed_centers)
        else:
            assert fixed_centers is None, (
                f'Centers are learnable (not fixed) for mode: {mode}'
            )
            assert centers is not None, f'centers are required for {mode=}'
            self.mu = nn.Parameter(centers)
            self.y_encoder = nn.Linear(m, n_bins)
            with torch.no_grad():
                # centers: [n_bins, 1]
                w1 = centers / self.enc_sigma**2  # [n_bins, 1]
                w2 = -(centers.pow(2).sum(dim=1)) / (2 * self.enc_sigma**2)

                # centers: [n_bins, 1] → y_encoder.weight: [n_bins, 1]
                self.y_encoder.weight.copy_(w1)  # [n_bins, 1]
                self.y_encoder.bias.copy_(w2)  # [n_bins]

    def forward(self, x_num: None | Tensor, x_cat: None | Tensor) -> Tensor:
        x = self.model(x_num, x_cat)

        if self.mode == 'baseline':
            return x

        if self.mode == 'softmax_mu':
            probs = nn.functional.softmax(x / self.temperature, dim=-1)
            out = probs @ self.mu
            return out
        if self.mode in (
            'hard_binning',
            'hard_binning_uniform',
            'hard_binning_quantile',
            'soft_binning',
            'soft_binning_uniform',
            'soft_binning_quantile',
            'pretrained_enc',
            'joint_end_to_end',
        ):
            probs = nn.functional.softmax(x / self.temperature, dim=-1)
            # for training:
            if self.training:
                return x / self.temperature
            # for eval:
            if self.mode in (
                'hard_binning',
                'soft_binning',
                'soft_binning_uniform',
                'hard_binning_uniform',
                'soft_binning_quantile',
                'hard_binning_quantile',
            ):
                return probs @ self.fixed_centers

            return probs @ self.mu
        raise ValueError(f'Unknown mode {self.mode}')

    def get_loss_fn(
        self,
        lambda_auto=1.0,
        lambda_kl=1.0,
        lambda_pred=1.0,
        alpha=0.0,
        loss_mode='logsoftmax_kl',
        huber_delta=1,
    ):
        """Reterns loss function suited for every mode"""
        if self.mode in ['baseline', 'softmax_mu']:
            if loss_mode == 'huber':
                base_loss_fn = partial(nn.functional.huber_loss, delta=huber_delta)
            elif loss_mode == 'l_1':
                base_loss_fn = nn.functional.l1_loss
            else:
                base_loss_fn = nn.functional.mse_loss

        elif self.mode in (
            'hard_binning',
            'hard_binning_uniform',
            'hard_binning_quantile',
        ):

            def hard_bin_loss_fn(logits, y):
                dists = (y.unsqueeze(1) - self.fixed_centers.T) ** 2
                idx = dists.argmin(dim=1)
                return nn.functional.cross_entropy(logits, idx)

            base_loss_fn = hard_bin_loss_fn

        elif self.mode in (
            'soft_binning',
            'soft_binning_uniform',
            'soft_binning_quantile',
            'joint_end_to_end',
        ):
            if loss_mode == 'logsoftmax_kl':
                if self.mode == 'soft_binning_uniform':

                    def soft_bin_loss_fn_logsoftmax_kl(logits, y):
                        dists = ((y.unsqueeze(1) - self.fixed_centers.T) ** 2) / (
                            2 * self.enc_sigma**2
                        )
                        # dists = dists.unsqueeze(1).expand(
                        #     -1, (1 if self.model.arch_type == 'plain' else self.model.k), -1
                        # )
                        slabels = nn.functional.softmax(-dists, dim=1)
                        log_probs = nn.functional.log_softmax(logits, dim=-1)
                        return nn.functional.kl_div(
                            log_probs, slabels, reduction='batchmean'
                        )

                    base_loss_fn = soft_bin_loss_fn_logsoftmax_kl

                if self.mode == 'joint_end_to_end':

                    def joint_end_to_end_logsoftmax_kl(logits, y):
                        y = y.unsqueeze(1)
                        log_probs = nn.functional.log_softmax(logits, dim=-1)
                        log_psi_y = nn.functional.log_softmax(self.y_encoder(y), dim=-1)
                        recon_y = torch.exp(log_psi_y) @ self.mu
                        auto_loss = nn.functional.mse_loss(recon_y, y) - alpha * (
                            -(torch.exp(log_probs) * log_psi_y).sum(dim=-1).mean()
                        )
                        kl_loss = nn.functional.kl_div(
                            log_probs, log_psi_y, reduction='batchmean', log_target=True
                        )
                        # regression loss on prediction
                        pred_y = torch.exp(log_probs) @ self.mu
                        pred_loss = nn.functional.mse_loss(pred_y, y)
                        return (
                            lambda_auto * auto_loss
                            + lambda_kl * kl_loss
                            + lambda_pred * pred_loss
                        )

                    base_loss_fn = joint_end_to_end_logsoftmax_kl

            elif loss_mode == 'earth_mover_l1':

                def soft_bin_earth_mover_l1_loss(logits, y):
                    dists = ((y.unsqueeze(1) - self.fixed_centers.T) ** 2) / (
                        2 * self.enc_sigma**2
                    )
                    slabels = nn.functional.softmax(-dists, dim=1)
                    probs = nn.functional.softmax(logits, dim=-1)
                    return earth_mover_loss_l1(slabels, probs)

                base_loss_fn = soft_bin_earth_mover_l1_loss

            elif loss_mode == 'earth_mover_l2':

                def soft_bin_earth_mover_l2_loss(logits, y):
                    dists = ((y.unsqueeze(1) - self.fixed_centers.T) ** 2) / (
                        2 * self.enc_sigma**2
                    )
                    slabels = nn.functional.softmax(-dists, dim=1)
                    probs = nn.functional.softmax(logits, dim=-1)
                    return earth_mover_loss_l2(slabels, probs)

                base_loss_fn = soft_bin_earth_mover_l2_loss

            elif loss_mode == 'log+softmax_kl':

                def soft_bin_loss_fn(logits, y):
                    dists = ((y.unsqueeze(1) - self.fixed_centers.T) ** 2) / (
                        2 * self.enc_sigma**2
                    )
                    # dists = dists.unsqueeze(1).expand(
                    #     -1, (1 if self.model.arch_type == 'plain' else self.model.k), -1
                    # )
                    probs = nn.functional.softmax(logits, dim=-1)
                    slabels = nn.functional.softmax(-dists, dim=1)
                    return nn.functional.kl_div(
                        (probs + 1e-12).log(), slabels + 1e-12, reduction='batchmean'
                    )

                base_loss_fn = soft_bin_loss_fn
            else:
                raise ValueError('this loss is not implemented')

        elif self.mode == 'pretrained_enc':

            def pretrained_loss_fn(probs, y):
                with torch.no_grad():
                    psi_y = nn.functional.softmax(self.y_encoder(y), dim=-1)
                return nn.functional.kl_div(probs.log(), psi_y, reduction='batchmean')

            base_loss_fn = pretrained_loss_fn

        elif self.mode == 'joint_end_to_end':

            def joint_loss_fn(probs, y):
                y = y.unsqueeze(1)
                psi_y = nn.functional.softmax(self.y_encoder(y), dim=-1)
                recon_y = psi_y @ self.mu
                auto_loss = nn.functional.mse_loss(recon_y, y) - alpha * (
                    -(psi_y * (psi_y + 1e-12).log()).sum(dim=-1).mean()
                )
                kl_loss = nn.functional.kl_div(
                    (probs + 1e-12).log(), psi_y + 1e-12, reduction='batchmean'
                )
                # regression loss on prediction
                pred_y = probs @ self.mu
                pred_loss = nn.functional.mse_loss(pred_y, y)
                return (
                    lambda_auto * auto_loss
                    + lambda_kl * kl_loss
                    + lambda_pred * pred_loss
                )

            base_loss_fn = joint_loss_fn
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        def loss_fn(y_pred: Tensor, y_true: Tensor):
            y_pred = y_pred.flatten(0, 1)
            if self.model.share_training_batches:
                # (batch_size,) -> (batch_size * k,)
                k = self.model.k if self.model.arch_type != 'plain' else 1
                y_true = y_true.repeat_interleave(k)
            else:
                # (batch_size, k) -> (batch_size * k,)
                y_true = y_true.flatten(0, 1)
            return base_loss_fn(y_pred, y_true)

        return loss_fn


class ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        idx: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: nn.Module, dataset: lib.data.Dataset, *, part: PartKey, idx: Tensor
) -> Tensor:
    return (
        model(
            dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()
    )


class EvaluateImplOutput(NamedTuple):
    metrics: dict[PartKey, Any]
    predictions: dict[PartKey, np.ndarray]


@lib.util.adjust_gpu_memory_usage('batch_size')
def evaluate_impl(
    apply_model: ApplyModel,
    model: nn.Module,
    dataset: lib.data.Dataset,
    *,
    parts: list[PartKey],
    regression_label_stats: None | lib.data.RegressionLabelStats,
    prediction_type: str | PredictionType,
    batch_size: int,
    device: torch.device,
    target_transform,
) -> EvaluateImplOutput:
    model.eval()

    predictions = {
        part: (
            target_transform.inverse_transform(
                torch.cat(
                    [
                        apply_model(model, dataset, part=part, idx=idx)
                        for idx in torch.arange(
                            dataset.size(part), device=device
                        ).split(batch_size)
                    ]
                )
                .mean(1)
                .cpu()
                .numpy()
                .reshape(-1, 1)
            ).squeeze()
        )
        for part in parts
    }

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        for part in predictions:
            predictions[part] *= regression_label_stats.std
            predictions[part] += regression_label_stats.mean
    else:
        raise NotImplementedError('only regression tasks are supported')
    metrics = (
        dataset.task.calculate_metrics(predictions, prediction_type)
        if lib.util.are_valid_predictions(predictions)
        else {x: {'score': lib.util.WORST_SCORE} for x in predictions}
    )

    return EvaluateImplOutput(metrics, predictions)


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    bins: NotRequired[KWArgs]
    loss_fn: KWArgs
    optimizer: KWArgs
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    compute_parameter_statistics: NotRequired[bool]
    # NOTE
    # For models like MLP or TabM only amp_dtype="bfloat16" was tested,
    # so "float16" is supported only for completeness
    # (in theory, it can be a better choice for some models).
    target_transform: NotRequired[Literal['power', 'quantile', 'identity']]
    amp_dtype: NotRequired[AMPDType]
    compile: NotRequired[bool]


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])
    device = lib.util.get_device()
    logger.info(f'Device: {device}')
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    assert dataset.task.is_regression
    regression_label_stats = dataset.try_standardize_labels_()
    target_transform = config['model'].pop('target_transform', None)
    if target_transform == 'power':
        target_transform = Pipeline(
            [('power', PowerTransformer()), ('standard', StandardScaler())]
        ).fit(dataset.data['y']['train'].reshape(-1, 1))
    elif target_transform == 'quantile':
        target_transform = QuantileTransformer(
            output_distribution='normal', random_state=config['seed']
        ).fit(dataset.data['y']['train'].reshape(-1, 1))
    elif target_transform is None or target_transform == 'identity':
        target_transform = FunctionTransformer(func=None)
    else:
        raise ValueError(f'Unknown target_transform {target_transform}')

    dataset.data['y'] = {
        part: target_transform.transform(dataset.data['y'][part].reshape(-1, 1))
        .astype(np.float32)  # pyright: ignore
        .squeeze()
        for part in dataset.data['y']
    }
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(torch.float)

    target_min = torch.max(Y_train).item()
    target_max = torch.min(Y_train).item()
    target_range = target_max - target_min

    sb_min_value = (
        target_min + config['loss_fn'].pop('sb_min_value_relative', -0.1) * target_range
    )
    sb_max_value = (
        target_max + config['loss_fn'].pop('sb_max_value_relative', 0.1) * target_range
    )

    if sb_min_value > sb_max_value:
        sb_min_value, sb_max_value = sb_max_value, sb_min_value


    if 'bins' in config:
        # Compute the bins for PiecewiseLinearEncoding and PiecewiseLinearEmbeddings.
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['bins']
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset.data['x_num']['train'], **config['bins'], **compute_bins_kwargs
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in bin_edges]}')
    else:
        bin_edges = None

    if 'bin_bound' in config['model'].keys():
        bin_quantile = torch.quantile(Y_train, 0.995)
        config['model']['bin_bound'] *= bin_quantile

    # Model
    if config['model'].get('mode', None) in (
        'soft_binning',
        'soft_binning_uniform',
        'soft_binning_quantile',
        'hard_binning_quantile',
        'hard_binning_uniform',
        'joint_end_to_end',
    ):
        support = (
            torch.linspace(sb_min_value, sb_max_value, config['model']['n_bins'] + 1)
            .float()
            .to(device)
        )
        centers = None
        fixed_centers = ((support[:-1] + support[1:]) / 2).unsqueeze(1)
        if config['model'].get('mode', None) == 'joint_end_to_end':
            centers = fixed_centers
            fixed_centers = None
    else:
        centers = None
        fixed_centers = None
    model: Model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        bins=bin_edges,
        fixed_centers=fixed_centers,
        centers=centers,
        **config['model'],
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'logits'
    model.to(device)
    share_training_batches = config['model']['arch'].get('share_training_batches', True)
    # mode = config['model']['mode']

    # >>> Training
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    gradient_clipping_norm = config.get('gradient_clipping_norm')

    loss_fn = model.get_loss_fn(**config.get('loss_fn'))

    step = 0
    if config['batch_size'] != -1:
        batch_size = config['batch_size']
    else:
        batch_size = dataset.size('train')
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = config.get(
        'eval_batch_size',
        # With torch.compile,
        # the largest possible evaluation batch size is noticeably smaller.
        2048 if config.get('compile', False) else 32768,
    )

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    compute_parameter_statistics = config.get('compute_parameter_statistics', False)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(exp)
    best_checkpoint = None
    amp_dtype = config.get('amp_dtype')
    if amp_dtype == 'bfloat16':
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'amp_dtype is set to "bfloat16" in the config.'
                f' However, the current {device.type.upper()} device'
                ' does not support bfloat16'
            )
        amp_dtype = torch.bfloat16
    elif amp_dtype == 'float16':
        amp_dtype = torch.float16
    # For float16, the gradient scaler must be used.
    grad_scaler = (
        torch.amp.GradScaler(device.type) if amp_dtype is torch.float16 else None  # type: ignore
    )
    logger.info(f'AMP dtype: {amp_dtype}')
    autocast = None if amp_dtype is None else torch.autocast(device.type, amp_dtype)
    if config.get('compile', False):
        # NOTE
        # - `torch.compile` is intentionally called without the `mode` argument,
        #   because it causes issues with training.
        # - `type: ignore` is a rough solution to the problem that torch.compile changes
        #   the type of `model`. The alternative is to mark all the following usage of
        #   `model` with `type: ignore`.
        model = torch.compile(model)  # type: ignore
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    # >>> Functions
    apply_model = apply_model_impl
    if autocast is not None:
        apply_model = autocast(apply_model)
    # The following order of `evaluation_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = evaluation_mode()(
        partial(
            evaluate_impl,
            apply_model,
            model,
            dataset,
            regression_label_stats=regression_label_stats,
            prediction_type='labels',
            device=device,
            target_transform=target_transform,
        )
    )

    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'batch_generator': batch_generator.get_state(),
                'random_state': delu.random.get_state(),
                'early_stopping': early_stopping,
                'report': report,
                'timer': timer,
                'training_log': training_log,
                **(
                    {}
                    if grad_scaler is None
                    else {'grad_scaler': grad_scaler.state_dict()}
                ),
            }
        )

    print()
    timer.run()
    compute_parameter_statistics = config.get('compute_parameter_statistics', None)

    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        model.train()

        batch_losses = []
        batches = (
            torch.randperm(
                dataset.size('train'),
                generator=batch_generator,
                device=device,
            ).split(batch_size)
            if share_training_batches
            else [
                x.transpose(0, 1).flatten()
                for x in torch.rand(
                    (config['model']['k'], dataset.size('train')),
                    generator=batch_generator,
                    device=device,
                )
                .argsort(dim=1)
                .split(batch_size, dim=1)
            ]
        )
        epoch_start_time = timer.elapsed()
        for batch_idx in tqdm(
            batches,
            desc=str(lib.util.try_get_relative_path(exp)),
            leave=False,
            disable=not lib.env.is_local(),
        ):
            optimizer.zero_grad()
            loss = loss_fn(
                apply_model(model, dataset, part='train', idx=batch_idx),
                Y_train[batch_idx],
            )
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

            if compute_parameter_statistics and (
                step // epoch_size == 0  # The first epoch.
                or step % epoch_size == 0  # The first batch of the epoch.
            ):
                for k, v in lib.deep.compute_parameter_statistics(model).items():
                    writer.add_scalars(k, v, step, timer.elapsed())
                    del k, v

            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )

            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

            step += 1
            batch_losses.append(loss.detach())
        epoch_end_time = timer.elapsed()

        batch_losses = torch.stack(batch_losses).tolist()
        epoch_loss = statistics.mean(batch_losses)

        (metrics, predictions), eval_batch_size = evaluate(
            parts=['val', 'test'], batch_size=eval_batch_size
        )

        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        )

        training_log.append(
            {'batch-losses': batch_losses, 'metrics': metrics, 'time': timer.elapsed()}
        )
        print(
            f'{"*" if val_score_improved else " "}'
            f' [epoch] {step // epoch_size:<3}'
            f' [val] {metrics["val"]["score"]:.3f}'
            f' [test] {metrics["test"]["score"]:.3f}'
            f' [loss] {epoch_loss:.4f}'
            f' [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [it/s] {math.trunc(epoch_size / (epoch_end_time - epoch_start_time)):>3}'
        )
        writer.add_scalars('loss', {'train': epoch_loss}, step, timer.elapsed())
        for part in metrics:
            writer.add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
            )

        if val_score_improved:
            report['best_step'] = step
            report['metrics'] = metrics
            best_checkpoint = make_checkpoint()

        early_stopping.update(metrics['val']['score'])
        if early_stopping.should_stop() or not lib.util.are_valid_predictions(
            predictions
        ):
            break

    report['time'] = timer.elapsed()

    # >>>
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model'])
    (metrics, predictions), eval_batch_size = evaluate(
        parts=['train', 'val', 'test'], batch_size=eval_batch_size
    )
    report['eval_batch_size'] = eval_batch_size
    report['metrics'] = metrics
    lib.experiment.dump_checkpoint(exp, make_checkpoint())
    lib.experiment.dump_predictions(exp, predictions)

    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
