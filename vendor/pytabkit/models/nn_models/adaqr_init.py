# Vendored pytabkit snapshot, adapted for the regression benchmark.
# Source: https://github.com/dholzmueller/pytabkit
# License: Apache-2.0; see LICENSE.txt. Local changes: see VENDOR.md.

"""Controlled initialization of the AdaQR sigma output."""

import math

import torch

from pytabkit.models.nn_models.base import BiasLayer, Layer, WeightLayer
from pytabkit.models.nn_models.nn import AdaQRLossLayer


@torch.no_grad()
def initialize_adaqr_sigma(
    model: Layer,
    sigma: float,
    sigma_std: float | None = None,
    reference_tensors: dict[str, torch.Tensor] | None = None,
) -> None:
    """Set constant sigma or retain a narrow, centered random sigma head."""
    if isinstance(sigma, bool) or not isinstance(sigma, int | float):
        raise ValueError("adaqr_sigma_init must be a finite positive number")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("adaqr_sigma_init must be a finite positive number")

    losses = [m for m in model.modules() if isinstance(m, AdaQRLossLayer)]
    if len(losses) != 1:
        raise ValueError("adaqr_sigma_init requires one adaqr loss layer")
    loss = losses[0]
    if not math.isfinite(loss.min_sigma) or not 0 <= loss.min_sigma < sigma:
        raise ValueError("adaqr_sigma_init must be greater than min_sigma >= 0")
    if not math.isfinite(loss.raw_sigma_scale) or loss.raw_sigma_scale <= 0:
        raise ValueError("adaqr_sigma_init requires raw_sigma_scale > 0")

    weights = [
        m
        for m in model.modules()
        if isinstance(m, WeightLayer)
        and "/net/last_layer/" in str(m.weight.context.scope)
    ]
    biases = [
        m
        for m in model.modules()
        if isinstance(m, BiasLayer) and "/net/last_layer/" in str(m.bias.context.scope)
    ]
    if len(weights) != 1 or len(biases) != 1:
        raise ValueError("adaqr_sigma_init requires a linear output head")
    weight_layer, bias_layer = weights[0], biases[0]
    weight, bias = weight_layer.weight, bias_layer.bias
    n_outputs = loss.quantiles.numel() + 1
    if weight.shape[-1] != n_outputs or bias.shape[-1] != n_outputs:
        raise ValueError("The last output must be the adaqr sigma head")

    if sigma_std is not None and (
        isinstance(sigma_std, bool)
        or not isinstance(sigma_std, int | float)
        or not math.isfinite(sigma_std)
        or sigma_std < 0.0
    ):
        raise ValueError("adaqr_sigma_init_std must be a finite non-negative number")
    if sigma_std is None or sigma_std == 0.0:
        # The offset stays outside the optimizer, including weight decay.
        weight[..., -1].zero_()
        bias[..., -1].zero_()
        loss.muls_init = torch.full_like(
            loss.muls_init, (sigma - loss.min_sigma) / loss.raw_sigma_scale
        )
        return

    if reference_tensors is None:
        raise ValueError("adaqr_sigma_init_std requires reference tensors")
    if bias_layer.factor == 0.0:
        raise ValueError("adaqr_sigma_init requires a nonzero output bias factor")

    raw_outputs = []
    handle = loss.register_forward_pre_hook(
        lambda _module, args: raw_outputs.append(args[0]["x_cont"].detach())
    )
    was_training = model.training
    model.eval()
    try:
        model(dict(reference_tensors))
    finally:
        handle.remove()
        model.train(was_training)
    if len(raw_outputs) != 1 or raw_outputs[0].ndim != 3:
        raise ValueError("adaqr_sigma_init_std could not capture packed sigma logits")

    raw_sigma = raw_outputs[0][..., -1]
    raw_mean = raw_sigma.mean(dim=1, keepdim=True)
    raw_std = raw_sigma.std(dim=1, correction=0, keepdim=True)
    if not torch.isfinite(raw_sigma).all() or not torch.all(raw_std > 0.0):
        raise ValueError("adaqr_sigma_init_std requires finite, nonconstant logits")

    raw_scale = (sigma_std / loss.raw_sigma_scale) / raw_std
    pre_abs = (
        (raw_sigma - raw_mean) * raw_scale * loss.raw_sigma_scale
        + sigma
        - loss.min_sigma
    )
    if not torch.isfinite(pre_abs).all() or not torch.all(pre_abs > 0.0):
        raise ValueError(
            "adaqr_sigma_init_std puts initial logits on the abs kink; "
            "choose a smaller standard deviation"
        )
    if weight.shape[0] != raw_scale.shape[0] or bias.shape[0] != raw_scale.shape[0]:
        raise ValueError("adaqr_sigma_init_std has incompatible packed head shapes")

    # This is an initialization, not a persistent output transform. The random
    # head direction is retained, while each member starts centered and narrow.
    weight[..., -1].mul_(raw_scale)
    bias[..., -1].mul_(raw_scale).sub_(raw_scale * raw_mean / bias_layer.factor)
    loss.muls_init = torch.full_like(
        loss.muls_init, (sigma - loss.min_sigma) / loss.raw_sigma_scale
    )
