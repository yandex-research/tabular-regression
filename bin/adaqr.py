"""Train and evaluate Adaptive Quantile Regression (AdaQR).

The model predicts quantiles and an input-dependent raw smoothing scale.
The standard scale parameterization is abs(raw_sigma) + min_sigma; the
quantile and scale objectives and optional regularizer are configured below.
"""
import datetime
import json
import math
import statistics
from collections.abc import Iterable
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
import lib.uncertainty
import lib.util
import lib.verbose_stats
import vendor.tabm_reference as tabm_reference
from bin.quantile_regression_cohort_early_stopping import (
    _build_cohorts_for_run,
    _record_stitched_metrics,
)
from lib.cohorts import CohortEarlyStoppingController, stitch_predictions

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
        ).float()  # (B, k, n_quantiles + 1)
    )


def _compute_memberwise_muls_init(
    raw_sigmas: Iterable[Tensor], muls_init_add: float
) -> Tensor:
    raw_sigma_sum = None
    n_samples = 0
    for batch_raw_sigma in raw_sigmas:
        batch_sum = batch_raw_sigma.sum(dim=0)
        raw_sigma_sum = batch_sum if raw_sigma_sum is None else raw_sigma_sum + batch_sum
        n_samples += batch_raw_sigma.shape[0]

    assert raw_sigma_sum is not None and n_samples > 0
    return muls_init_add - raw_sigma_sum / n_samples


def _compute_softplus_muls_init(
    raw_sigmas: Iterable[Tensor],
    *,
    target_sigma: float,
    min_sigma: float,
    softplus_scale: float,
    raw_scale: float,
) -> Tensor:
    """Choose a memberwise offset with the requested mean initial sigma."""
    if target_sigma <= min_sigma:
        raise ValueError('muls_init_add must be greater than min_sigma')
    if softplus_scale <= 0.0 or raw_scale <= 0.0:
        raise ValueError('softplus_scale and raw_sigma_scale must be positive')

    values = torch.cat(list(raw_sigmas), dim=0)
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    target_softplus = (target_sigma - min_sigma) / softplus_scale
    target_raw = math.log(math.expm1(target_softplus))
    lo = target_raw / raw_scale - values.amax(dim=0)
    hi = target_raw / raw_scale - values.amin(dim=0)

    # The mean is monotone in the offset. These bounds put every item below or
    # above the target respectively, so bisection is guaranteed to bracket it.
    for _ in range(40):
        mid = (lo + hi) / 2
        mean_softplus = F.softplus((values + mid) * raw_scale).mean(dim=0)
        move_lo = mean_softplus < target_softplus
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_lo, hi, mid)
    return (lo + hi) / 2


def _n_model_outputs(config: 'Config') -> int:
    return config['n_quantiles'] + int(config.get('sigma_source') is None)


def _split_training_outputs(
    outputs: Tensor,
    *,
    batch_idx: Tensor,
    external_sigma_train: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if external_sigma_train is None:
        return outputs[..., :-1], outputs[..., -1]
    if external_sigma_train.ndim != 1:
        raise ValueError('External training sigma must be one-dimensional')
    sigma = external_sigma_train[batch_idx].unsqueeze(-1)
    return outputs, sigma.expand(outputs.shape[:-1])


class QuantileSigmaOutput(nn.Module):
    """Apply independent quantile and sigma heads to the same hidden tensor."""

    def __init__(self, quantile_output: nn.Module, sigma_output: nn.Module) -> None:
        super().__init__()
        self.quantile_output = quantile_output
        self.sigma_output = sigma_output

    def forward(self, hidden: Tensor) -> Tensor:
        return torch.cat(
            [self.quantile_output(hidden), self.sigma_output(hidden)], dim=-1
        )


class JointScaleOutput(nn.Module):
    """One global radial coordinate around the original linear Q+sigma head.

    The raw sigma output is transformed around the fixed initialization offset,
    so the unchanged downstream abs decoder scales only its above-floor part.
    """

    def __init__(self, base_output: nn.Module) -> None:
        super().__init__()
        self.base_output = base_output
        self.log_gain = nn.Parameter(torch.zeros(()))
        self.register_buffer('sigma_offset', torch.zeros(1))

    @torch.no_grad()
    def set_sigma_offset(self, offset: Tensor) -> None:
        if offset.numel() != 1:
            raise ValueError('The joint scale currently supports plain MLP only')
        self.sigma_offset.copy_(offset.detach().reshape_as(self.sigma_offset))

    def forward(self, hidden: Tensor) -> Tensor:
        output = self.base_output(hidden)
        q, raw_sigma = output[..., :-1], output[..., -1:]
        delta = self.log_gain.expm1()
        q = q + delta * (q - q.mean(dim=-1, keepdim=True))
        raw_sigma = raw_sigma + delta * (raw_sigma + self.sigma_offset)
        return torch.cat((q, raw_sigma), dim=-1)


class RankScaleOutput(JointScaleOutput):
    """One common gain table routed by detached base-sigma train CDF."""

    def __init__(
        self, base_output, *, n_bins=4, interpolation='linear', min_sigma=1e-5
    ):
        if (
            type(n_bins) is not int
            or n_bins < 2
            or interpolation not in ('hard', 'linear')
        ):
            raise ValueError('Expected integer n_bins>=2 and hard/linear interpolation')
        if not math.isfinite(min_sigma) or min_sigma <= 0:
            raise ValueError('min_sigma must be finite and positive')
        super().__init__(base_output)
        self.n_bins = n_bins
        self.interpolation = interpolation
        self.min_sigma = min_sigma
        self.log_gain = nn.Parameter(torch.zeros(n_bins))
        self.register_buffer('log_reference', torch.zeros(257))

    @torch.no_grad()
    def set_reference(self, sigma):
        sigma = sigma.detach().reshape(-1).to(self.log_reference)
        if (
            not sigma.numel()
            or not torch.isfinite(sigma).all()
            or not (sigma > 0).all()
        ):
            raise ValueError('Reference sigma must be nonempty, finite and positive')
        self.log_reference.copy_(
            torch.quantile(
                sigma.log(),
                torch.linspace(0, 1, 257, device=sigma.device, dtype=sigma.dtype),
            )
        )

    @torch.no_grad()
    def ranks(self, sigma):
        values = sigma.detach().clamp_min(self.min_sigma).log().contiguous()
        reference = self.log_reference.contiguous()
        left = torch.searchsorted(reference, values, right=False)
        right = torch.searchsorted(reference, values, right=True)
        lo = (left - 1).clamp(0, 255)
        width = (reference[lo + 1] - reference[lo]).clamp_min(
            torch.finfo(values.dtype).eps
        )
        rank = (lo + ((values - reference[lo]) / width).clamp(0, 1)) / 256
        return torch.where(right > left, (left + right - 1) / 512, rank).clamp(0, 1)

    def effective_log_gain(self):
        limit = math.log(4.0)
        return limit * torch.tanh(self.log_gain / limit)

    def gains(self, sigma):
        rank = self.ranks(sigma)
        table = self.effective_log_gain()
        if self.interpolation == 'hard':
            return table[(rank * self.n_bins).long().clamp(0, self.n_bins - 1)].exp()
        position = (rank * self.n_bins - 0.5).clamp(0, self.n_bins - 1)
        lo = position.floor().long().clamp(0, self.n_bins - 2)
        fraction = position - lo
        return (table[lo] + fraction * (table[lo + 1] - table[lo])).exp()

    def forward(self, hidden):
        output = self.base_output(hidden)
        q, raw = output[..., :-1], output[..., -1:]
        gain = self.gains((raw.detach() + self.sigma_offset).abs() + self.min_sigma)
        q = q + (gain - 1) * (q - q.mean(dim=-1, keepdim=True))
        raw = raw + (gain - 1) * (raw + self.sigma_offset)
        return torch.cat((q, raw), dim=-1)


@torch.no_grad()
def refresh_rank_scale_reference(model, dataset, apply_model, batch_size):
    """Refresh from train inputs only, without consuming dropout RNG."""
    head = model.output
    values = []
    was_training = model.training
    model.eval()
    hook = head.base_output.register_forward_hook(
        lambda _module, _inputs, output: values.append(
            output[..., -1].detach().reshape(-1)
        )
    )
    try:
        device = next(model.parameters()).device
        for idx in torch.arange(dataset.size('train'), device=device).split(batch_size):
            apply_model(model, dataset, part='train', idx=idx)
        head.set_reference(
            (torch.cat(values) + head.sigma_offset).abs() + head.min_sigma
        )
    finally:
        hook.remove()
        model.train(was_training)


def make_adaqr_model(
    *,
    n_num_features: int,
    cat_cardinalities: list[int],
    n_quantiles: int,
    backbone: dict[str, Any],
    bins: list[Tensor] | None,
    sigma_head: dict[str, Any] | None = None,
    include_sigma: bool = True,
    joint_scale: bool = False,
    rank_scale: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> tabm_reference.Model:
    """Construct the original joint head or a parallel sigma-only MLP head."""
    if sigma_head is None:
        model = tabm_reference.Model(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_quantiles + int(include_sigma),
            backbone=backbone,
            bins=bins,
            **model_kwargs,
        )
        if joint_scale:
            if not include_sigma or model.arch_type != 'plain':
                raise ValueError('The joint scale requires a plain Q+sigma model')
            model.output = (
                JointScaleOutput(model.output)
                if rank_scale is None
                else RankScaleOutput(model.output, **rank_scale)
            )
        return model
    if joint_scale:
        raise ValueError('The joint scale requires the original linear sigma head')
    if not include_sigma:
        raise ValueError('A separate sigma head cannot be used with external sigma')
    if sigma_head.get('type') != 'mlp':
        raise ValueError(f'Unknown sigma head type: {sigma_head.get("type")!r}')

    model = tabm_reference.Model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_quantiles,
        backbone=backbone,
        bins=bins,
        **model_kwargs,
    )
    sigma_output = tabm_reference.MLP(
        d_in=backbone['d_block'],
        d_out=1,
        n_blocks=sigma_head.get('n_blocks', 1),
        d_block=sigma_head['d_hidden'],
        dropout=sigma_head.get('dropout', 0.0),
        activation=sigma_head.get('activation', 'ReLU'),
    )
    model.output = QuantileSigmaOutput(model.output, sigma_output)
    return model


class EvaluateImplOutput(NamedTuple):
    metrics: dict[PartKey, Any]
    predictions: dict[PartKey, np.ndarray]


def gaussian_crps(residual: Tensor, scale: Tensor) -> Tensor:
    """CRPS of a Gaussian centered at zero, evaluated at ``residual``."""
    z = residual / scale
    return (
        residual * torch.erf(z / math.sqrt(2.0))
        + math.sqrt(2.0 / math.pi) * scale * torch.exp(-0.5 * z.square())
        - scale / math.sqrt(math.pi)
    )


class DualScaleCRPSRegularizer(nn.Module):
    """Regularizer both Proposition-1 estimates of the conditional data scale."""

    def __init__(
        self,
        quantiles: Tensor,
        *,
        regularizer: float,
        balance: Literal['equal_loss', 'output_jacobian'] = 'equal_loss',
        min_scale: float = 1e-6,
        factor_cap: float | None = None,
        weight_beta: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < regularizer < 1.0:
            raise ValueError('The scale scale_reg requires 0 < regularizer < 1')
        if min_scale <= 0.0:
            raise ValueError('min_scale must be positive')
        if balance not in ('equal_loss', 'output_jacobian'):
            raise ValueError(f'Unknown scale-reg balance: {balance!r}')
        if factor_cap is not None and factor_cap < 1.0:
            raise ValueError('factor_cap must be at least 1')
        if not math.isfinite(weight_beta) or weight_beta < 0.0:
            raise ValueError('The scale_reg weight beta must be finite and nonnegative')
        normal = torch.distributions.Normal(0.0, 1.0)
        normal_scores = normal.icdf(quantiles)
        self.register_buffer('normal_score_rms', normal_scores.square().mean().sqrt())
        self.n_quantiles_sqrt = math.sqrt(quantiles.numel())
        self.regularizer = regularizer
        raw_q_factor = math.sqrt(1.0 - regularizer**2)
        raw_sigma_factor = raw_q_factor / regularizer
        if factor_cap is None:
            self.q_factor = raw_q_factor
            self.sigma_factor = raw_sigma_factor
        else:
            floor = 1.0 / factor_cap
            self.q_factor = max(raw_q_factor, floor)
            self.sigma_factor = min(max(raw_sigma_factor, floor), factor_cap)
        self.balance = balance
        self.min_scale = min_scale
        self.weight_beta = weight_beta

    def implied_data_scales(
        self, preds: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor]:
        center = preds.mean(dim=-1, keepdim=True)
        quantile_rms = (
            (preds - center).square().mean(dim=-1) + self.min_scale**2
        ).sqrt()
        q_scale = self.q_factor * quantile_rms / self.normal_score_rms
        sigma_scale = self.sigma_factor * sigma.squeeze(-1)
        return q_scale.clamp_min(self.min_scale), sigma_scale.clamp_min(self.min_scale)

    def forward(self, *, preds: Tensor, y_true: Tensor, sigma: Tensor) -> Tensor:
        q_scale, sigma_scale = self.implied_data_scales(preds, sigma)
        residual = y_true - preds.mean(dim=-1).detach()
        q_weight: float | Tensor = 1.0
        sigma_weight = 1.0
        if self.balance == 'output_jacobian':
            # At equal implied scales, these coefficients exactly equalize the
            # direct q-vector and raw-sigma gradient norms. Without them, their
            # ratio is r / (z_rms * sqrt(Q)).
            q_weight = self.normal_score_rms * self.n_quantiles_sqrt
            sigma_weight = self.regularizer
        if self.weight_beta:
            log_scale = 0.5 * (q_scale.log() + sigma_scale.log())
            weights = (self.weight_beta * log_scale.detach()).softmax(dim=0)
            return 0.5 * (
                q_weight * (weights * gaussian_crps(residual, q_scale)).sum()
                + sigma_weight * (weights * gaussian_crps(residual, sigma_scale)).sum()
            )
        return 0.5 * (
            q_weight * gaussian_crps(residual, q_scale).mean()
            + sigma_weight * gaussian_crps(residual, sigma_scale).mean()
        )


class AdaQRLoss(nn.Module):
    def __init__(
        self,
        num_quantiles: int = 999,
        eta_min: float = 0.001,
        eta_max: float = 0.999,
        device: torch.device = torch.device('cuda:0'),
        y_true_mode: str = 'identity',
        agg_mode: str = 'sum',
        raw_sigma_scale: float = 1.0,
        regularizer: float = 1.0,
        scale_loss_by_n_quantiles: bool = False,
        min_sigma: float = 1e-4,
        sigma_param: Literal['abs', 'softplus', 'profiled'] = 'abs',
        softplus_scale: float = 1.0,
        profile_scale: float = 1.0,
        profile_bisection_steps: int = 16,
        scale_reg_weight: float = 0.0,
        scale_reg_balance: Literal['equal_loss', 'output_jacobian'] = 'equal_loss',
        scale_reg_factor_cap: float | None = None,
        scale_reg_weight_shape: Literal['regularizer'] | None = None,
        scale_reg_beta: float = 0.0,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.register_buffer(
            'quantiles', torch.linspace(eta_min, eta_max, num_quantiles, device=device)
        )
        self.min_sigma = min_sigma
        self.agg_mode = agg_mode
        self.regularizer = regularizer
        if scale_reg_weight_shape == 'regularizer':
            scale_reg_weight = scale_reg_weight * regularizer
        self.raw_sigma_scale = raw_sigma_scale
        self.sigma_param = sigma_param
        self.softplus_scale = softplus_scale
        self.profile_scale = profile_scale
        self.profile_bisection_steps = profile_bisection_steps
        self.scale_reg_weight = scale_reg_weight
        self.y_true_mode = y_true_mode
        self.cdf = torch.distributions.Normal(0.0, 1.0).cdf
        self.loss_scale = float(num_quantiles) if scale_loss_by_n_quantiles else 1.0
        assert y_true_mode in ['identity', 'sampling'], 'Unsupported y_true mode'
        assert agg_mode in ['mean'], 'Unsupported loss aggregation'
        assert sigma_param in ['abs', 'softplus', 'profiled']
        assert softplus_scale > 0.0
        assert profile_scale > 0.0
        assert profile_bisection_steps > 0
        if scale_reg_weight < 0.0:
            raise ValueError('scale_reg_weight must be nonnegative')
        if scale_reg_weight > 0.0 and y_true_mode != 'identity':
            raise ValueError('The scale scale_reg requires y_true_mode="identity"')
        self.scale_reg = (
            None
            if scale_reg_weight == 0.0
            else DualScaleCRPSRegularizer(
                self.quantiles,
                regularizer=regularizer,
                balance=scale_reg_balance,
                factor_cap=scale_reg_factor_cap,
                weight_beta=scale_reg_beta,
            )
        )
        if sigma_param == 'profiled':
            assert y_true_mode == 'identity', (
                'Profiled sigma is incompatible with sampled y_true'
            )
            assert 0.0 <= regularizer < math.sqrt(2.0), (
                'Profiled sigma requires 0 <= regularizer < sqrt(2)'
            )

    def _profile_sigma(self, errors: Tensor) -> Tensor:
        """Minimize the per-sample loss over sigma with detached residuals."""
        errors = errors.detach()
        if self.regularizer == 0.0:
            return torch.full_like(errors[..., 0], self.min_sigma)

        target = self.regularizer / math.sqrt(2.0)
        lo = torch.full_like(errors[..., 0], self.min_sigma)

        def kernel_mean(sigma: Tensor) -> Tensor:
            return torch.exp(-(errors**2) / (2 * sigma.unsqueeze(-1) ** 2)).mean(dim=-1)

        at_floor = kernel_mean(lo) >= target
        max_error = errors.abs().amax(dim=-1)
        denominator = math.sqrt(2.0 * math.log(1.0 / target))
        hi = torch.maximum(lo, max_error / denominator)
        hi = torch.where(at_floor, lo, hi)

        for _ in range(self.profile_bisection_steps):
            mid = (lo + hi) / 2
            move_lo = kernel_mean(mid) < target
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(move_lo, hi, mid)

        sigma = ((lo + hi) / 2) * self.profile_scale
        return sigma.clamp_min(self.min_sigma)

    def _sigma_from_raw(self, raw_sigma: Tensor) -> Tensor:
        if self.sigma_param == 'abs':
            return raw_sigma.abs() + self.min_sigma
        if self.sigma_param == 'softplus':
            return self.softplus_scale * F.softplus(raw_sigma) + self.min_sigma
        raise RuntimeError('Profiled sigma does not use the raw sigma head')

    def forward(
        self, *, preds: Tensor, y_true: Tensor, raw_sigma: Tensor, muls_init: float = 0.0
    ) -> Tensor:
        """
        : raw_sigma -- data uncertainty of the real conditional distribution y | x
        : preds -- quantile predictions of the shape (B, num_quantiles)
        """
        preds = preds.float()
        y_true = y_true.float()
        raw_sigma = ((raw_sigma.float() + muls_init) * self.raw_sigma_scale).unsqueeze(-1)

        if self.sigma_param == 'profiled':
            errors = preds - y_true.unsqueeze(-1)
            sigma = self._profile_sigma(errors).unsqueeze(-1)
        else:
            sigma = self._sigma_from_raw(raw_sigma)
        if self.y_true_mode == 'sampling':
            errors = preds - (
                torch.randn_like(y_true.unsqueeze(-1)) * sigma + y_true.unsqueeze(-1)
            )  # (B, n_q)
        elif self.sigma_param != 'profiled':
            errors = preds - y_true.unsqueeze(-1)  # (B, n_q)

        # estimating sigma_pred
        # var_pred = (preds * preds).mean(dim=1) - (preds.mean(dim=1)) ** 2  # (B, )
        # log_total_std = torch.log(var_pred + torch.exp(2 * raw_sigma)) / 2

        assert raw_sigma.ndim == 2 and errors.ndim == 2
        external_term = (2 / torch.pi) ** 0.5 * sigma * torch.exp(
            -(errors**2) / 2 / sigma**2
        ) + errors * (2 * self.cdf(errors / sigma) - 1)  # (B, n_q)
        external_term = external_term.mean()

        quantiles = self.quantiles.unsqueeze(0)  # (1, n_quantiles)
        internal_terms = (
            preds.mean()
            - 2 * (quantiles * preds).mean()
            - sigma.mean() / (torch.pi) ** 0.5 * self.regularizer
        )

        result = external_term + internal_terms
        if self.scale_reg is not None:
            result = result + self.scale_reg_weight * self.scale_reg(
                preds=preds, y_true=y_true, sigma=sigma
            )
        return result * self.loss_scale

    # def get_scalar_prediction()


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


class SigmaHeadConfig(TypedDict):
    type: Literal['mlp']
    d_hidden: int
    n_blocks: NotRequired[int]
    dropout: NotRequired[float]
    activation: NotRequired[str]


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
    parameter_statistics: NotRequired[bool]
    # NOTE
    # For models like MLP or TabM only amp_dtype="bfloat16" was tested,
    # so "float16" is supported only for completeness
    # (in theory, it can be a better choice for some models).
    amp_dtype: NotRequired[AMPDType]
    compile: NotRequired[bool]

    n_quantiles: int
    quantile_init: NotRequired[Literal['empirical']]
    eta_min: NotRequired[float]
    eta_max: NotRequired[float]
    regularizer: NotRequired[float]
    raw_sigma_scale: NotRequired[float]
    y_true_mode: str
    agg_mode: str
    muls_init_add: NotRequired[float]
    use_muls_init: NotRequired[bool]
    scale_loss_by_n_quantiles: NotRequired[bool]
    min_sigma: NotRequired[float]
    sigma_param: NotRequired[Literal['abs', 'softplus', 'profiled']]
    softplus_scale: NotRequired[float]
    profile_scale: NotRequired[float]
    profile_bisection_steps: NotRequired[int]
    scale_reg_weight: NotRequired[float]
    scale_reg_balance: NotRequired[Literal['equal_loss', 'output_jacobian']]
    scale_reg_factor_cap: NotRequired[float]
    scale_reg_weight_shape: NotRequired[Literal['regularizer']]
    scale_reg_beta: NotRequired[float]
    joint_scale_lr: NotRequired[float]
    rank_scale: NotRequired[dict[str, Any]]
    sigma_head: NotRequired[SigmaHeadConfig]
    sigma_source: NotRequired[lib.uncertainty.SigmaSourceConfig]

    target_transform: NotRequired[Literal['power', 'quantile', 'identity']]
    y_scale_method: NotRequired[str]
    quantile_output_distribution: NotRequired[str]
    verbose: NotRequired[bool]
    source_evaluation_config: NotRequired[str]
    cohort_strategy: NotRequired[str]
    n_cohorts: NotRequired[int]
    cohort_rng_seed: NotRequired[int]
    density_k: NotRequired[int]
    cohort_cache_dir: NotRequired[str]
    cohort_uncertainty_path: NotRequired[str]
    cohort_uncertainty_kind: NotRequired[str]
    compute_density_signals: NotRequired[bool]
    balance_ties_on_val: NotRequired[bool]


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    joint_scale_lr = config.get('joint_scale_lr')
    if config.get('rank_scale') is not None:
        if (
            joint_scale_lr is None
            or config.get('sigma_head') is not None
            or config.get('sigma_source') is not None
            or config.get('raw_sigma_scale', 1.0) != 1.0
            or config.get('compile', False)
            or config.get('amp_dtype') is not None
            or config['model']['arch_type'] != 'plain'
        ):
            raise ValueError(
                'Rank scale requires plain linear sigma, joint_scale_lr, '
                'raw_sigma_scale=1, no compile/AMP/external sigma'
            )
    if joint_scale_lr is not None:
        if not math.isfinite(joint_scale_lr) or joint_scale_lr <= 0:
            raise ValueError('joint_scale_lr must be finite and positive')
        if config.get('sigma_param', 'abs') != 'abs':
            raise ValueError('The joint scale requires the original abs decoder')
    if config.get('scale_reg_beta', 0.0) and not config.get(
        'scale_reg_weight', 0.0
    ):
        raise ValueError('Detached scale_reg weighting requires a positive scale_reg weight')
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])
    device = lib.util.get_device()
    logger.info(f'Device: {device}')
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    assert dataset.task.is_regression
    raw_targets = {part: values for part, values in dataset.data['y'].items()}
    sigma_source = config.get('sigma_source')
    configured_target_transform = config['model'].get('target_transform')
    if sigma_source is not None:
        if configured_target_transform not in (None, 'identity'):
            raise ValueError(
                'External sigma currently requires target_transform="identity"'
            )
        if 'regularizer' in config:
            raise ValueError(
                'regularizer has no effect and is forbidden with sigma_source'
            )
        if config.get('scale_reg_weight', 0.0) > 0.0:
            raise ValueError('The scale scale_reg is incompatible with external sigma')
    regression_label_stats = dataset.try_standardize_labels_()
    assert regression_label_stats is not None
    loaded_sigma = (
        None
        if sigma_source is None
        else lib.uncertainty.load_sigma_source(
            sigma_source,
            dataset_name=Path(config['data']['path']).name,
            expected_targets=raw_targets,
            label_std=regression_label_stats.std,
        )
    )
    if loaded_sigma is not None:
        report['sigma_source_artifact'] = str(loaded_sigma.artifact)
        logger.info(f'External sigma: {loaded_sigma.artifact}')
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
    external_sigma = (
        None
        if loaded_sigma is None
        else {
            part: torch.as_tensor(values, device=device)
            for part, values in loaded_sigma.values.items()
        }
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

    model = make_adaqr_model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_quantiles=config['n_quantiles'],
        sigma_head=config.get('sigma_head'),
        include_sigma=sigma_source is None,
        joint_scale=joint_scale_lr is not None,
        rank_scale=(
            None if config.get('rank_scale') is None
            else {**config['rank_scale'], 'min_sigma': config.get('min_sigma', 1e-4)}
        ),
        bins=bin_edges,
        **config['model'],
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'logits'
    model.to(device)
    quantile_init = config.get('quantile_init')
    if quantile_init is not None:
        if quantile_init != 'empirical':
            raise ValueError(f'Unknown quantile_init: {quantile_init!r}')
        quantile_output = (
            model.output.quantile_output
            if isinstance(model.output, QuantileSigmaOutput)
            else model.output
        )
        if isinstance(quantile_output, JointScaleOutput):
            quantile_output = quantile_output.base_output
        quantile_bias = lib.deep.init_empirical_quantile_output_(
            quantile_output.weight,
            quantile_output.bias,
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
    joint_scale_groups = (
        [
            {
                'params': [model.output.log_gain],
                'lr': joint_scale_lr,
                'weight_decay': 0.0,
            }
        ]
        if joint_scale_lr is not None
        else []
    )
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
        muon_parameter_group = {
            'params': [
                x.weight
                for x in model.backbone.modules()
                if isinstance(
                    x,
                    nn.Linear
                    | tabm.LinearBatchEnsemble
                    | tabm.LinearEnsemble
                    | tabm_reference.LinearEfficientEnsemble,
                )
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
            model,
            zero_weight_decay_condition,
            custom_groups=[muon_parameter_group, *joint_scale_groups],
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
            **config['optimizer'],
            params=lib.deep.make_parameter_groups(
                model, custom_groups=joint_scale_groups
            ),
        )
    gradient_clipping_norm = config.get('gradient_clipping_norm')

    # def base_loss_fn(logits: Tensor, y_true):
    #     mu, logvar = logits[..., 0], logits[..., 1]
    #     loss_vec = logvar / 2 + ((mu - y_true) ** 2) / (2 * (logvar.exp()))
    #     return loss_vec.mean()

    base_loss = AdaQRLoss(
        config['n_quantiles'],
        eta_min=config.get('eta_min', 0.001),
        eta_max=config.get('eta_max', 0.999),
        device=device,
        agg_mode=config.get('agg_mode', 'sum'),
        regularizer=0.0 if external_sigma is not None else config.get('regularizer', 1.0),
        raw_sigma_scale=(
            1.0 if external_sigma is not None else config.get('y_scale', 1.0)
        ),
        y_true_mode=config['y_true_mode'],
        scale_loss_by_n_quantiles=config.get('scale_loss_by_n_quantiles', False),
        min_sigma=config.get('min_sigma', 1e-4),
        sigma_param=config.get('sigma_param', 'abs'),
        softplus_scale=config.get('softplus_scale', 1.0),
        profile_scale=config.get('profile_scale', 1.0),
        profile_bisection_steps=config.get('profile_bisection_steps', 16),
        scale_reg_weight=config.get('scale_reg_weight', 0.0),
        scale_reg_balance=config.get('scale_reg_balance', 'equal_loss'),
        scale_reg_factor_cap=config.get('scale_reg_factor_cap'),
        scale_reg_weight_shape=config.get('scale_reg_weight_shape'),
        scale_reg_beta=config.get('scale_reg_beta', 0.0),
    )

    def loss_fn(
        *,
        preds: Tensor,
        y_true: Tensor,
        raw_sigma: Tensor,
        muls_init: float | Tensor = 0.0,
    ):
        preds = preds.flatten(0, 1)
        raw_sigma = (raw_sigma + muls_init).flatten(0, 1)
        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            k = model.k if model.arch_type != 'plain' else 1
            y_true = y_true.repeat_interleave(k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)
        return base_loss(preds=preds, y_true=y_true, raw_sigma=raw_sigma)

    def get_scalar_prediction(logits_prediction: Tensor | np.ndarray):
        if external_sigma is None:
            logits_prediction = logits_prediction[
                ..., :-1
            ]  # deleting raw_sigma on inference

        if not isinstance(logits_prediction, Tensor):
            logits_prediction = torch.from_numpy(logits_prediction).to(device)

        return logits_prediction.mean(-1)

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
    writer = torch.utils.tensorboard.SummaryWriter(exp)
    best_checkpoint = None
    muls_init = torch.zeros(1 if model.arch_type == 'plain' else model.k, device=device)
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
                'muls_init': muls_init.detach().cpu(),
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
    cohort_controller = None
    val_cohort = test_cohort = None
    original_y_val = original_y_test = None
    if 'cohort_strategy' in config:
        cohort_dir = _build_cohorts_for_run(config, exp)  # type: ignore[arg-type]
        val_cohort = np.load(cohort_dir / 'val_cohorts.npy')
        test_path = cohort_dir / 'test_cohorts.npy'
        test_cohort = np.load(test_path) if test_path.exists() else None
        n_cohorts = int(config['n_cohorts'])
        if len(val_cohort) != dataset.size('val'):
            raise ValueError('val cohort IDs do not match the validation split')
        if np.any(np.bincount(val_cohort, minlength=n_cohorts) == 0):
            raise ValueError('at least one validation cohort is empty')
        if test_cohort is not None and len(test_cohort) != dataset.size('test'):
            raise ValueError('test cohort IDs do not match the test split')
        cohort_controller = CohortEarlyStoppingController(
            patience=config['patience'], n_cohorts=n_cohorts
        )
        original_y_val = (
            dataset.data['y']['val'].detach().cpu().numpy()
            * regression_label_stats.std
            + regression_label_stats.mean
        )
        original_y_test = (
            dataset.data['y']['test'].detach().cpu().numpy()
            * regression_label_stats.std
            + regression_label_stats.mean
        )

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
        if external_sigma is None and config.get('sigma_param', 'abs') != 'profiled':
            with torch.no_grad():
                if step == 0 and config.get('use_muls_init', True):
                    raw_sigmas = (
                        apply_model(model, dataset, part='train', idx=batch_idx)[
                            ..., -1
                        ]
                        for batch_idx in tqdm(
                            batches,
                            desc=str(lib.util.try_get_relative_path(exp)),
                            leave=False,
                            disable=not lib.env.is_local(),
                        )
                    )
                    if config.get('sigma_param', 'abs') == 'softplus':
                        muls_init = _compute_softplus_muls_init(
                            raw_sigmas,
                            target_sigma=config.get('muls_init_add', 0.5),
                            min_sigma=config.get('min_sigma', 1e-4),
                            softplus_scale=config.get('softplus_scale', 1.0),
                            raw_scale=config.get('y_scale', 1.0),
                        )
                    else:
                        muls_init = _compute_memberwise_muls_init(
                            raw_sigmas,
                            config.get('muls_init_add', 0.5),
                        )
                    if joint_scale_lr is not None:
                        model.output.set_sigma_offset(muls_init)

        if config.get('rank_scale') is not None:
            refresh_rank_scale_reference(model, dataset, apply_model, eval_batch_size)

        for batch_idx in tqdm(
            batches,
            desc=str(lib.util.try_get_relative_path(exp)),
            leave=False,
            disable=not lib.env.is_local(),
        ):
            optimizer.zero_grad()

            outputs = apply_model(model, dataset, part='train', idx=batch_idx)
            quantile_outputs, batch_sigma = _split_training_outputs(
                outputs,
                batch_idx=batch_idx,
                external_sigma_train=(
                    None if external_sigma is None else external_sigma['train']
                ),
            )
            batch_muls_init = muls_init if external_sigma is None else 0.0
            loss = loss_fn(
                preds=quantile_outputs,
                y_true=Y_train[batch_idx],
                raw_sigma=batch_sigma,
                muls_init=batch_muls_init,
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
        if cohort_controller is not None:
            assert val_cohort is not None and original_y_val is not None
            cohort_mse = np.asarray(
                [
                    (
                        (
                            predictions['val'][val_cohort == cohort]
                            - original_y_val[val_cohort == cohort]
                        )
                        ** 2
                    ).mean()
                    for cohort in range(cohort_controller.n_cohorts)
                ]
            )
            cohort_controller.update(
                step // epoch_size,
                ((predictions['val'] - original_y_val) ** 2).mean(),
                cohort_mse,
                predictions['val'],
                predictions['test'],
            )

        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        )

        training_log.append(
            {'batch-losses': batch_losses, 'metrics': metrics, 'time': timer.elapsed()}
        )
        if joint_scale_lr is not None:
            training_log[-1]['joint_scale_gain'] = (
                model.output.effective_log_gain().detach().exp().tolist()
                if config.get('rank_scale') is not None
                else model.output.log_gain.detach().exp().item()
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
            lib.verbose_stats.log_grad_norm_stats(_verbose_grad_norms, writer, step)
            lib.verbose_stats.collect_epoch_eff_rank(
                model, dataset, writer, step, is_clustering=False
            )
            model.train()

        early_stopping.update(metrics['val']['score'])
        if (
            (
                cohort_controller.all_cohorts_stopped
                if cohort_controller is not None
                else early_stopping.should_stop()
            )
            or not lib.util.are_valid_predictions(predictions)
        ):
            break

    report['time'] = timer.elapsed()

    if cohort_controller is not None:
        assert val_cohort is not None and original_y_val is not None
        assert original_y_test is not None
        global_selection = cohort_controller.finalize_global()
        if any(x is None for x in cohort_controller.cohort_selections):
            raise RuntimeError('cohort early stopping did not freeze every selection')
        selections = [x for x in cohort_controller.cohort_selections if x is not None]
        predictions = {
            'val': global_selection.val_predictions,
            'test': global_selection.test_predictions,
        }
        report['eval_batch_size'] = eval_batch_size
        report['best_step'] = global_selection.epoch * epoch_size
        report['metrics'] = dataset.task.calculate_metrics(predictions, 'labels')
        lib.experiment.dump_predictions(exp, predictions)
        for cohort, selection in enumerate(selections):
            np.savez(
                exp / f'cohort_{cohort}_predictions.npz',
                val=selection.val_predictions,
                test=selection.test_predictions,
            )
        oracle_rmse = None
        if test_cohort is not None:
            oracle = {
                'val': stitch_predictions(
                    [x.val_predictions for x in selections], val_cohort
                ),
                'test': stitch_predictions(
                    [x.test_predictions for x in selections], test_cohort
                ),
            }
            np.savez(exp / 'oracle_predictions.npz', **oracle)
            _record_stitched_metrics(
                report, dataset.task.calculate_metrics(oracle, 'labels')
            )
            oracle_rmse = {
                part: float(np.sqrt(np.mean((oracle[part] - target) ** 2)))
                for part, target in [
                    ('val', original_y_val),
                    ('test', original_y_test),
                ]
            }
        np.savez(
            exp / 'cohorts.npz',
            val_target=original_y_val,
            test_target=original_y_test,
            val_index=np.arange(len(original_y_val)),
            test_index=np.arange(len(original_y_test)),
            val_cohort=val_cohort,
            **({} if test_cohort is None else {'test_cohort': test_cohort}),
        )
        extra_signals_path = cohort_dir / 'extra_signals.npz'
        if extra_signals_path.exists():
            extra_signals = np.load(extra_signals_path)
            np.savez(
                exp / 'cohort_signals.npz',
                **{name: extra_signals[name] for name in extra_signals.files},
            )
        assignment_metadata = json.loads((cohort_dir / 'metadata.json').read_text())
        with open(exp / 'cohort_metadata.json', 'w', encoding='utf-8') as output:
            json.dump(
                {
                    'n_cohorts': cohort_controller.n_cohorts,
                    'source_evaluation_config': config.get('source_evaluation_config'),
                    'assignment': assignment_metadata,
                    'global_selected_epoch': global_selection.epoch,
                    'global_stopped_epoch': cohort_controller.global_stopped_epoch,
                    'cohort_selected_epochs': [x.epoch for x in selections],
                    'cohort_stopped_epochs': cohort_controller.cohort_stopped_epochs,
                    'oracle_rmse': oracle_rmse,
                    'history': cohort_controller.history,
                },
                output,
                indent=2,
            )
    else:
        if best_checkpoint is not None:
            model.load_state_dict(best_checkpoint['model'])
        (metrics, predictions), eval_batch_size = evaluate(
            parts=['train', 'val', 'test'], batch_size=eval_batch_size
        )
        report['eval_batch_size'] = eval_batch_size
        report['metrics'] = metrics
        lib.experiment.dump_predictions(exp, predictions)
    if joint_scale_lr is not None:
        report['joint_scale_gain'] = (
            model.output.effective_log_gain().detach().exp().tolist()
            if config.get('rank_scale') is not None
            else model.output.log_gain.detach().exp().item()
        )
    checkpoint = make_checkpoint()
    if cohort_controller is not None:
        checkpoint['cohort_controller'] = cohort_controller
    lib.experiment.dump_checkpoint(exp, checkpoint)

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


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
