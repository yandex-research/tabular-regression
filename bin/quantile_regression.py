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
import torch.nn.functional as F
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
import lib.project.optim
import lib.util
import lib.verbose_stats
import vendor.tabm_reference as tabm_reference

# from vendor.sb_var_scheme import HLGaussLossVar
from lib.types import AMPDType, KWArgs, PartKey, PredictionType

_CUSTOM_OPTIMIZERS = {
    x.__name__: x
    for x in [
        lib.project.optim.AdamWSharpnessAware,
        lib.project.optim.Adan,
        lib.project.optim.AdEMAMix,
        lib.project.optim.ADOPT,
        lib.project.optim.conda_torch.Conda,
        lib.project.optim.Kron,
        lib.project.optim.Lion,
        lib.project.optim.Signum,
        lib.project.optim.SOAP,
        lib.project.optim.SPlus,
    ]
}
_CUSTOM_OPTIMIZERS['C-AdamW'] = lib.project.optim.c_adamw.AdamW
_CUSTOM_OPTIMIZERS['MARS-M'] = lib.project.optim.MARS_M


def get_optimizer_class(type: str) -> type[torch.optim.Optimizer]:
    Optimizer = getattr(torch.optim, type, None)
    if Optimizer is None:
        Optimizer = _CUSTOM_OPTIMIZERS[type]
    return Optimizer


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    return get_optimizer_class(type)(**kwargs)


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


class PredictionAdapter(Protocol):
    n_outputs: int

    def __call__(self, preds: Tensor, y_true: Tensor) -> Tensor: ...

    def get_scalar_prediction(
        self, logits_prediction: Tensor | np.ndarray
    ) -> Tensor: ...


class QuantileLoss(nn.Module):
    def __init__(
        self,
        num_quantiles: int = 999,
        eta_min: float = 0.001,
        eta_max: float = 0.999,
        device: torch.device = torch.device('cuda:0'),
        agg_mode: str = 'sum',
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.register_buffer(
            'quantiles', torch.linspace(eta_min, eta_max, num_quantiles, device=device)
        )
        self.agg_mode = agg_mode
        assert agg_mode in ['sum', 'mean', 'scaled_mean'], (
            'Unsupported loss aggregation'
        )

    def forward(self, preds: Tensor, y_true: Tensor) -> Tensor:
        # if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

        errors = y_true - preds
        taus = self.quantiles.unsqueeze(0)
        loss = torch.max(taus * errors, (taus - 1) * errors)

        if self.agg_mode == 'mean':
            return loss.mean()
        elif self.agg_mode == 'sum':
            return loss.sum(dim=-1).mean()
        elif self.agg_mode == 'scaled_mean':
            return (loss * 1024).mean()

    # def get_scalar_prediction()


class AdaQRSpreadLoss(nn.Module):
    """AdaQR distance to N(y_true, (noise_scale * spread_sigma)**2).

    ``spread_sigma`` is estimated from two already predicted quantiles, so this loss
    uses the same ``n_quantiles``-wide model head as plain quantile regression. The
    target scale is detached exactly as in sigma-Huber's spread mode.
    """

    def __init__(
        self,
        num_quantiles: int = 999,
        device: torch.device = torch.device('cuda:0'),
        noise_scale: float = 0.5,
        spread_pair: float = 0.159,
        min_sigma: float = 1e-4,
    ) -> None:
        super().__init__()
        assert num_quantiles >= 2
        assert (
            not isinstance(noise_scale, bool)
            and isinstance(noise_scale, int | float)
            and noise_scale > 0.0
        ), 'noise_scale must be a positive fixed scalar'
        assert (
            not isinstance(spread_pair, bool)
            and isinstance(spread_pair, int | float)
            and 0.0 < spread_pair < 0.5
        ), 'spread_pair must be a fixed scalar in (0, 0.5)'
        assert (
            not isinstance(min_sigma, bool)
            and isinstance(min_sigma, int | float)
            and min_sigma > 0.0
        ), 'min_sigma must be a positive fixed scalar'

        quantiles = torch.linspace(0.001, 0.999, num_quantiles, device=device)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        self.register_buffer('quantiles', quantiles)
        self.noise_scale = float(noise_scale)
        self.min_sigma = float(min_sigma)
        self.spread_lo = int(torch.argmin((quantiles - spread_pair).abs()))
        self.spread_hi = int(torch.argmin((quantiles - (1.0 - spread_pair)).abs()))
        assert self.spread_lo < self.spread_hi
        self.spread_normalizer = 1.0 / (z[self.spread_hi] - z[self.spread_lo]).item()
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf

    def spread_sigma(self, preds: Tensor) -> Tensor:
        sigma = (
            preds[..., self.spread_hi] - preds[..., self.spread_lo]
        ).abs() * self.spread_normalizer + self.min_sigma
        return sigma.detach()

    def forward(self, preds: Tensor, y_true: Tensor) -> Tensor:
        preds = preds.float()
        y_true = y_true.float()
        sigma_n = self.noise_scale * self.spread_sigma(preds)
        sigma_n_ = sigma_n.unsqueeze(-1)
        errors = preds - y_true.unsqueeze(-1)

        external_term = (
            (2.0 / torch.pi) ** 0.5
            * sigma_n_
            * torch.exp(-(errors**2) / (2.0 * sigma_n_**2))
            + errors * (2.0 * self.cdf(errors / sigma_n_) - 1.0)
        ).mean()
        predicted_self_term = preds.mean() - 2.0 * (self.quantiles * preds).mean()
        target_self_term = -sigma_n.mean() / math.sqrt(math.pi)
        return external_term + predicted_self_term + target_self_term


class OrthogonalQuantileDecoder(nn.Module):
    def __init__(
        self,
        num_quantiles: int,
        parameterization: Literal['grid', 'linear_spline'],
        *,
        n_spline_coefficients: int | None = None,
        spline_slope_scale: float = 4.0,
        spline_slope_bias: float = -2.0,
        device: torch.device = torch.device('cuda:0'),
    ) -> None:
        super().__init__()
        assert num_quantiles >= 2
        assert parameterization in ['grid', 'linear_spline']
        self.num_quantiles = num_quantiles
        self.parameterization = parameterization
        self.spline_slope_scale = spline_slope_scale
        self.spline_slope_bias = spline_slope_bias

        if parameterization == 'grid':
            assert n_spline_coefficients is None
            self.n_shape_outputs = num_quantiles
            self.spline_basis = None
        else:
            assert n_spline_coefficients is not None
            assert n_spline_coefficients >= 2
            assert spline_slope_scale > 0.0
            self.n_shape_outputs = n_spline_coefficients

            positions = torch.linspace(0.0, 1.0, num_quantiles, device=device)
            edges = torch.linspace(0.0, 1.0, n_spline_coefficients + 1, device=device)
            width = 1.0 / n_spline_coefficients
            basis = (positions[:, None] - edges[:-1]).clamp(min=0.0, max=width)
            self.register_buffer('spline_basis', basis)

    @property
    def n_outputs(self) -> int:
        return 1 + self.n_shape_outputs

    def forward(
        self, logits: Tensor, *, detach_location: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert logits.shape[-1] == self.n_outputs
        mean = logits[..., 0]
        raw_shape = logits[..., 1:]

        if self.parameterization == 'grid':
            curve = raw_shape
        else:
            assert self.spline_basis is not None
            slopes = self.spline_slope_scale * F.softplus(
                raw_shape + self.spline_slope_bias
            )
            curve = torch.einsum('...j,qj->...q', slopes, self.spline_basis)

        residual_quantiles = curve - curve.mean(dim=-1, keepdim=True)
        location = mean.detach() if detach_location else mean
        quantiles = location.unsqueeze(-1) + residual_quantiles
        return mean, residual_quantiles, quantiles


class OrthogonalQuantileLoss(nn.Module):
    def __init__(
        self,
        decoder: OrthogonalQuantileDecoder,
        *,
        mean_loss_weight: float,
        agg_mode: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert mean_loss_weight > 0.0
        self.decoder = decoder
        self.mean_loss_weight = mean_loss_weight
        self.quantile_loss = QuantileLoss(
            decoder.num_quantiles, device=device, agg_mode=agg_mode
        )

    def forward(self, logits: Tensor, y_true: Tensor) -> Tensor:
        mean, _, quantiles = self.decoder(logits, detach_location=True)
        shape_loss = self.quantile_loss(quantiles, y_true)
        mean_loss = F.mse_loss(mean, y_true)
        return shape_loss + self.mean_loss_weight * mean_loss


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
    get_scalar_prediction,
    target_transform,
) -> EvaluateImplOutput:
    model.eval()

    predictions = {
        part: (
            # target_transform.inverse_transform(
            torch.cat(
                [
                    get_scalar_prediction(
                        apply_model(model, dataset, part=part, idx=idx)
                    )
                    for idx in torch.arange(dataset.size(part), device=device).split(
                        batch_size
                    )
                ]
            )
            .mean(1)
            .cpu()
            .numpy()
            #     .reshape(-1, 1)
            # ).squeeze()
        )
        for part in parts
    }
    # assert False

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
    bins: NotRequired[KWArgs]
    model: KWArgs
    optimizer: KWArgs
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    # If true, always renormalize the global gradient to gradient_clipping_norm
    # (scaling up or down every step), instead of only clipping when it exceeds.
    always_clip: NotRequired[bool]
    parameter_statistics: NotRequired[bool]
    # NOTE
    # For models like MLP or TabM only amp_dtype="bfloat16" was tested,
    # so "float16" is supported only for completeness
    # (in theory, it can be a better choice for some models).
    amp_dtype: NotRequired[AMPDType]
    compile: NotRequired[bool]

    n_quantiles: int
    eta_min: NotRequired[float]
    eta_max: NotRequired[float]
    agg_mode: str
    quantile_init: NotRequired[Literal['empirical']]
    loss: NotRequired[Literal['quantile', 'adaqr_spread']]
    noise_scale: NotRequired[float]
    spread_pair: NotRequired[float]
    min_sigma: NotRequired[float]
    orthogonal_parameterization: NotRequired[Literal['grid', 'linear_spline']]
    mean_loss_weight: NotRequired[float]
    n_spline_coefficients: NotRequired[int]
    spline_slope_scale: NotRequired[float]
    spline_slope_bias: NotRequired[float]

    target_transform: NotRequired[Literal['power', 'quantile', 'identity']]
    y_scale_method: NotRequired[str]
    quantile_output_distribution: NotRequired[str]
    verbose: NotRequired[bool]
    # Store test predictions after every epoch for post-hoc training-dynamics
    # analyses. Disabled for standard tuning/evaluation runs.
    save_epoch_predictions: NotRequired[bool]
    global_target_noise_mode: NotRequired[Literal['dynamic', 'fixed']]
    global_target_noise_sigma: NotRequired[float]


class GlobalTargetNoise:
    """Adds a fixed or freshly sampled global Gaussian perturbation to targets."""

    def __init__(
        self,
        *,
        mode: Literal['dynamic', 'fixed'],
        sigma: float,
        n_train: int,
        device: torch.device,
        seed: int,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError('global_target_noise_sigma must be positive')
        self.mode = mode
        self.sigma = float(sigma)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.fixed_noise = (
            torch.randn(n_train, device=device, generator=self.generator) * self.sigma
            if mode == 'fixed'
            else None
        )

    def apply(self, targets: Tensor, indices: Tensor) -> Tensor:
        if self.mode == 'fixed':
            assert self.fixed_noise is not None
            return targets + self.fixed_noise[indices].to(dtype=targets.dtype)
        return targets + torch.randn(
            targets.shape,
            device=targets.device,
            dtype=targets.dtype,
            generator=self.generator,
        ) * self.sigma


def _main(
    config: Config,
    exp: str | Path,
    *,
    prediction_adapter_factory=None,
    report_function=None,
) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(
        main if report_function is None else report_function, add_gpu_info=True
    )

    delu.random.seed(config['seed'])
    device = lib.util.get_device()
    logger.info(f'Device: {device}')
    prediction_adapter: PredictionAdapter | None = (
        None
        if prediction_adapter_factory is None
        else prediction_adapter_factory(config, device)
    )
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
    noise_mode = config.get('global_target_noise_mode')
    noise_sigma = config.get('global_target_noise_sigma')
    if (noise_mode is None) != (noise_sigma is None):
        raise ValueError(
            'global_target_noise_mode and global_target_noise_sigma must be set together'
        )
    target_noise = (
        None
        if noise_mode is None
        else GlobalTargetNoise(
            mode=noise_mode,
            sigma=noise_sigma,
            n_train=dataset.size('train'),
            device=device,
            seed=config['seed'] + 17,
        )
    )

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

    model_config = deepcopy(config['model'])
    use_bias = model_config.pop('use_bias', True)

    loss_type = config.get('loss', 'quantile')
    assert loss_type in ('quantile', 'adaqr_spread')
    orthogonal_parameterization = config.get('orthogonal_parameterization')
    if prediction_adapter is not None and orthogonal_parameterization is not None:
        raise ValueError(
            'A custom prediction adapter is incompatible with'
            ' orthogonal_parameterization'
        )
    assert loss_type == 'quantile' or orthogonal_parameterization is None, (
        'adaqr_spread is defined directly on the quantile head and is incompatible'
        ' with orthogonal_parameterization'
    )
    orthogonal_decoder = (
        None
        if orthogonal_parameterization is None
        else OrthogonalQuantileDecoder(
            config['n_quantiles'],
            orthogonal_parameterization,
            n_spline_coefficients=config.get('n_spline_coefficients'),
            spline_slope_scale=config.get('spline_slope_scale', 4.0),
            spline_slope_bias=config.get('spline_slope_bias', -2.0),
            device=device,
        )
    )
    n_outputs = (
        prediction_adapter.n_outputs
        if prediction_adapter is not None
        else config['n_quantiles']
        if orthogonal_decoder is None
        else orthogonal_decoder.n_outputs
    )

    model = tabm_reference.Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=n_outputs,
        **model_config,
        bins=bin_edges,
    )

    if not use_bias:
        logger.info('Removing bias parameters from all Linear layers')
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.bias = None

    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'logits'
    model.to(device)
    quantile_init = config.get('quantile_init')
    if quantile_init is not None:
        if prediction_adapter is not None:
            raise ValueError(
                'quantile_init is incompatible with a custom prediction adapter'
            )
        if quantile_init != 'empirical':
            raise ValueError(f'Unknown quantile_init: {quantile_init!r}')
        if orthogonal_decoder is not None:
            raise ValueError(
                'quantile_init="empirical" is incompatible with'
                ' orthogonal_parameterization'
            )
        quantile_bias = lib.deep.init_empirical_quantile_output_(
            model.output.weight,
            model.output.bias,
            Y_train,
            config['n_quantiles'],
            weight_output_dim=0 if model.arch_type == 'plain' else 2,
            bias_output_dim=0 if model.arch_type == 'plain' else 1,
        )
        logger.info(
            'Initialized quantile output from the empirical target distribution:'
            f' [{quantile_bias[0].item():.4g}, {quantile_bias[-1].item():.4g}]'
        )
    share_training_batches = config['model'].get('share_training_batches', True)
    # mode = config['model']['mode']

    # >>> Training
    zero_weight_decay_condition = (
        lib.deep.default_zero_weight_decay_condition
    )  # policy A
    if 'muon' in config['optimizer']:
        assert 'type' not in config['optimizer']
        optimizer_config = deepcopy(config['optimizer'])
        optimizer_is_normuon = optimizer_config['muon'].pop('normuon', False)
        optimizer_is_cautious = optimizer_config.pop('cautious', False)
        optimizer_muon_config = optimizer_config.pop('muon')
        optimizer_muon_tuned_abc = optimizer_muon_config.pop('tuned_abc', False)
        optimizer_muon_original_precision = optimizer_muon_config.pop(
            'original_precision', False
        )
        assert (
            optimizer_is_normuon
            + optimizer_is_cautious
            + optimizer_muon_tuned_abc
            + optimizer_muon_original_precision
        ) <= 1

        # Muon API requires explicitly setting all hyperparameters
        # (lr, weight_decay, ...) for all parameter groups.

        # Inherit Muon hyperparameters from AdamW hyperparameters.
        for key in ['lr', 'weight_decay']:
            optimizer_muon_config.setdefault(key, optimizer_config[key])
            del key

        # Configure Muon parameter groups.
        # for module in model.model.backbone.modules():
        #     print(module)
        # print(f'{model.model.output=}')
        muon_linear_types = (
            nn.Linear,
            *(
                getattr(tabm, name)
                for name in ('LinearBatchEnsemble', 'LinearEnsemble')
                if hasattr(tabm, name)
            ),
            tabm_reference.LinearEfficientEnsemble,
        )
        muon_parameter_group = {
            'params': [
                x.weight
                for x in model.backbone.modules()
                if isinstance(x, muon_linear_types)
            ],
            **optimizer_muon_config,
            'use_muon': True,
        }
        # print(f'{muon_parameter_group=}')
        if config.get('optimizer_muon_num_embeddings', False):
            assert model.num_module is not None
            muon_parameter_group['params'].extend(
                x.weight for x in model.num_module.modules() if isinstance(x, nn.Linear)
            )
        parameter_groups = lib.deep.make_parameter_groups(
            model, zero_weight_decay_condition, custom_groups=[muon_parameter_group]
        )

        # Configure AdamW parameter groups.
        for parameter_group in parameter_groups:
            parameter_group.setdefault('use_muon', False)
            if parameter_group['use_muon']:
                # Muon parameter groups are already configured.
                continue
            for k, v in optimizer_config.items():
                parameter_group.setdefault(k, v)
                del k, v
            del parameter_group

        optimizer = (
            lib.project.optim.normuon.SingleDeviceNorMuonWithAuxAdam
            if optimizer_is_normuon
            else lib.project.optim.c_muon.SingleDeviceMuonWithAuxAdam
            if optimizer_is_cautious
            else lib.project.optim.muon_tuned_abc.SingleDeviceMuonWithAuxAdam
            if optimizer_muon_tuned_abc
            else lib.project.optim.muon_original_precision.SingleDeviceMuonWithAuxAdam
            if optimizer_muon_original_precision
            else lib.project.optim.muon.SingleDeviceMuonWithAuxAdam
        )(parameter_groups)
    else:
        optimizer = lib.deep.make_optimizer(
            **config['optimizer'], params=lib.deep.make_parameter_groups(model)
        )
    gradient_clipping_norm = config.get('gradient_clipping_norm')
    always_clip = config.get('always_clip', False)

    # def base_loss_fn(logits: Tensor, y_true):
    #     mu, logvar = logits[..., 0], logits[..., 1]
    #     loss_vec = logvar / 2 + ((mu - y_true) ** 2) / (2 * (logvar.exp()))
    #     return loss_vec.mean()

    if prediction_adapter is not None:
        if loss_type != 'quantile':
            raise ValueError('A custom prediction adapter requires quantile loss')
        base_loss = prediction_adapter
    elif loss_type == 'adaqr_spread':
        base_loss = AdaQRSpreadLoss(
            config['n_quantiles'],
            device=device,
            noise_scale=config.get('noise_scale', 0.5),
            spread_pair=config.get('spread_pair', 0.159),
            min_sigma=config.get('min_sigma', 1e-4),
        )
    elif orthogonal_decoder is None:
        base_loss = QuantileLoss(
            config['n_quantiles'],
            eta_min=config.get('eta_min', 0.001),
            eta_max=config.get('eta_max', 0.999),
            device=device,
            agg_mode=config.get('agg_mode', 'sum'),
        )
    else:
        base_loss = OrthogonalQuantileLoss(
            orthogonal_decoder,
            mean_loss_weight=config.get(
                'mean_loss_weight', math.sqrt(config['n_quantiles'])
            ),
            agg_mode=config.get('agg_mode', 'sum'),
            device=device,
        )

    def loss_fn(y_pred: Tensor, y_true: Tensor):
        y_pred = y_pred.flatten(0, 1)
        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            k = model.k if model.arch_type != 'plain' else 1
            y_true = y_true.repeat_interleave(k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)
        return base_loss(y_pred, y_true)

    def get_scalar_prediction(logits_prediction: Tensor | np.ndarray):
        if not isinstance(logits_prediction, Tensor):
            logits_prediction = torch.from_numpy(logits_prediction).to(device)

        if prediction_adapter is not None:
            return prediction_adapter.get_scalar_prediction(logits_prediction)
        return (
            logits_prediction.mean(-1)
            if orthogonal_decoder is None
            else logits_prediction[..., 0]
        )

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
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    compute_parameter_statistics = config.get('compute_parameter_statistics', False)
    training_log = []
    epoch_predictions_path = exp / 'epoch_predictions'
    if config.get('save_epoch_predictions', False):
        epoch_predictions_path.mkdir(exist_ok=True)
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

    apply_model = apply_model_impl
    if autocast is not None:
        apply_model = autocast(apply_model)

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
            get_scalar_prediction=get_scalar_prediction,
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
        _verbose_grad_norms: list[float] = []
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
            targets = Y_train[batch_idx]
            if target_noise is not None:
                targets = target_noise.apply(targets, batch_idx)
            loss = loss_fn(
                apply_model(model, dataset, part='train', idx=batch_idx), targets
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

            if config.get('verbose'):
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                _verbose_grad_norms.append(
                    lib.verbose_stats.collect_batch_grad_norm(model)
                )

            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                if always_clip:
                    # Renormalize the global gradient to gradient_clipping_norm
                    # every step (scale up or down), so clipping always applies.
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
                    scale = gradient_clipping_norm / (total_norm + 1e-6)
                    for g in grads:
                        g.mul_(scale)
                else:
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

        eval_parts = (
            ['train', 'val', 'test'] if config.get('verbose') else ['val', 'test']
        )
        (metrics, predictions), eval_batch_size = evaluate(
            parts=eval_parts, batch_size=eval_batch_size
        )

        if config.get('save_epoch_predictions', False):
            np.save(
                epoch_predictions_path / f'{step // epoch_size:04d}.npy',
                predictions['test'],
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

        if config.get('verbose'):
            lib.verbose_stats.log_grad_norm_stats(
                _verbose_grad_norms,
                writer,
                step,
                clip_threshold=gradient_clipping_norm,
            )
            lib.verbose_stats.collect_epoch_eff_rank(
                model, dataset, writer, step, is_clustering=False
            )
            model.train()

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

    if config.get('verbose'):
        lib.verbose_stats.run_verbose_stats(
            model,
            dataset,
            exp,
            writer,
            is_clustering=False,
            data_path=Path(config['data']['path']),
        )

    lib.experiment.finish(exp, report)
    return report


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    return _main(config, exp)


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
