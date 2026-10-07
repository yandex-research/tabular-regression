# NOTE[tabular]
import itertools
from typing import Any

import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


# ==================================================================================
# Initialization
# ==================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.no_grad()
def init_empirical_quantile_output_(
    weight: Tensor,
    bias: None | Tensor,
    y_train: Tensor,
    n_quantiles: int,
    *,
    weight_output_dim: int,
    bias_output_dim: int,
) -> Tensor:
    """Initialize the first output channels with centered target quantiles."""
    if y_train.ndim != 1 or not y_train.numel():
        raise ValueError('y_train must be a non-empty one-dimensional tensor')
    if not y_train.is_floating_point() or not torch.isfinite(y_train).all():
        raise ValueError('y_train must contain only finite floating-point values')
    if n_quantiles < 2:
        raise ValueError('n_quantiles must be at least 2')
    if bias is None:
        raise ValueError('Empirical quantile initialization requires an output bias')

    weight_output_dim %= weight.ndim
    bias_output_dim %= bias.ndim
    if weight.shape[weight_output_dim] < n_quantiles:
        raise ValueError('The output weight has fewer channels than n_quantiles')
    if bias.shape[bias_output_dim] < n_quantiles:
        raise ValueError('The output bias has fewer channels than n_quantiles')

    taus = torch.linspace(
        0.001,
        0.999,
        n_quantiles,
        dtype=y_train.dtype,
        device=y_train.device,
    )
    quantile_bias = torch.quantile(y_train, taus)
    quantile_bias += y_train.mean() - quantile_bias.mean()

    weight.narrow(weight_output_dim, 0, n_quantiles).zero_()
    bias_quantiles = bias.narrow(bias_output_dim, 0, n_quantiles)
    broadcast_shape = [1] * bias_quantiles.ndim
    broadcast_shape[bias_output_dim] = n_quantiles
    bias_quantiles.copy_(
        quantile_bias.to(bias_quantiles)
        .reshape(broadcast_shape)
        .expand_as(bias_quantiles)
    )
    return quantile_bias


# ==================================================================================
# Modules
# ==================================================================================
class OneHotEncoding(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


def get_n_parameters(module: nn.Module):
    return sum(x.numel() for x in module.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_statistics(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ==================================================================================
# Optimization
# ==================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(len(group['params']) for group in custom_groups), (
        'Parameters in custom_groups must not intersect'
    )
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, type)(**kwargs)
