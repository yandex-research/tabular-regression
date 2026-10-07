# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import functools
import typing
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

import vendor.muon

from .nn import ParameterPack, get_pack_size, make_keep_pack_idx


def _is_valid_lr(value: float) -> bool:
    return value >= 0.0


def _is_valid_momentum(value: float) -> bool:
    return 0.0 <= value < 1.0


def _is_valid_beta(value: float) -> bool:
    return _is_valid_momentum(value)


def _is_valid_weight_decay(value: float) -> bool:
    return value >= 0.0


def _is_valid_eps(value: float) -> bool:
    return value > 0.0


def _is_shared_group_value(value) -> bool:
    """Check if the group value is shared between pack members."""
    return value is None or isinstance(
        value, bool | int | float | str | bytes | tuple | dict
    )


class OptimizerPack(torch.optim.Optimizer):
    """The base class for optimizer packs.

    Analogously to a module pack, an optimizer pack represents a set of optimizers with
    potentially different hyperparameters (learning rate, weight decay, etc.).
    """

    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        super().__init__(
            params,
            {
                # `torch.tensor` is used to ensure that each group
                # receives a tensor with a separate storage.
                k: v if _is_shared_group_value(v) else torch.tensor(v)
                for k, v in defaults.items()
            },
        )

        for group in self.param_groups:
            group_params = group['params']
            if not group_params:
                continue
            for p in group_params:
                assert isinstance(p, ParameterPack), (
                    'For now, only parameter packs are supported'
                )
            device = group_params[0].device
            for key, value in list(group.items()):
                if key != 'params' and isinstance(value, Tensor):
                    group[key] = value.to(device=device)


def _make_weight_decay_multiplier(
    *, lr: float | Tensor, weight_decay: float | Tensor
) -> None | float | Tensor:
    return (
        None
        if (
            (isinstance(lr, float) and lr == 0.0)
            or (isinstance(weight_decay, float) and weight_decay == 0.0)
        )
        else (1 - lr * weight_decay)
    )


@typing.overload
def _maybe_unsqueeze(value: Tensor, *, p: Tensor) -> Tensor: ...


@typing.overload
def _maybe_unsqueeze[T](value: T, *, p: Tensor) -> T: ...


def _maybe_unsqueeze(value, *, p):
    return value[:, *((None,) * (p.ndim - 1))] if isinstance(value, Tensor) else value


class AdamWPack(OptimizerPack):
    """A pack of AdamW optimizers."""

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        beta1: float | list[float] = 0.9,
        beta2: float | list[float] = 0.999,
        eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        pack_size: int,
        shared_step: bool = False,
        follow_pytorch: bool = True,
    ):
        assert pack_size > 0

        defaults = {}
        for key, value, is_valid_fn in [
            ('lr', lr, _is_valid_lr),
            ('beta1', beta1, _is_valid_beta),
            ('beta2', beta2, _is_valid_beta),
            ('eps', eps, _is_valid_eps),
            ('weight_decay', weight_decay, _is_valid_weight_decay),
        ]:
            if isinstance(value, list):
                assert len(value) == pack_size
                assert all(map(is_valid_fn, value))
            else:
                is_valid_fn(value)
            defaults[key] = value

        super().__init__(params, defaults)
        self._shared_step = shared_step
        self._follow_pytorch = follow_pytorch

    @torch.no_grad()
    def step(  # type: ignore
        self,
        closure: None | Callable[[], Tensor] = None,
    ) -> None | Tensor:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']

            weight_decay_multiplier = _make_weight_decay_multiplier(
                lr=lr, weight_decay=group['weight_decay']
            )

            for p in group['params']:
                assert isinstance(p, ParameterPack), (
                    'For now, only parameter packs are supported'
                )
                if p.grad is None:
                    continue

                grad = p.grad.data
                assert not grad.is_sparse, 'Sparse gradients are not supported'

                maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

                state = self.state[p]
                if len(state) == 0:
                    # Initialize the state.
                    state['step'] = (
                        0
                        if self._shared_step
                        else torch.zeros(
                            get_pack_size(p), dtype=torch.int64, device=p.device
                        )
                    )
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                if weight_decay_multiplier is not None:
                    p.mul_(maybe_unsqueeze(weight_decay_multiplier))

                # Update biased first moment estimate.
                exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

                # Update biased second raw moment estimate.
                if self._follow_pytorch and isinstance(beta2, float):
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                else:
                    exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

                # Perform the bias correction.
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1

                if self._follow_pytorch and isinstance(bias_correction2, float):
                    denom = exp_avg_sq.sqrt() / bias_correction2**0.5
                else:
                    denom = exp_avg_sq.sqrt().div_(
                        maybe_unsqueeze(bias_correction2**0.5)
                    )
                denom.add_(maybe_unsqueeze(eps))

                if isinstance(step_size, float):
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p.sub_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)))

        return loss


class MuonAdamWPack(OptimizerPack):
    """A pack of Muon-AdamW optimizers."""

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        beta1: float | list[float] = 0.9,
        beta2: float | list[float] = 0.999,
        eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        muon_lr: float | list[float],
        muon_momentum: float | list[float] = 0.95,
        muon_ns_steps: int = 5,
        muon_nesterov: bool = True,
        pack_size: int,
        shared_step: bool = False,
        follow_pytorch: bool = True,
    ):
        assert pack_size > 0

        defaults: dict[str, Any] = {
            'muon': False,
            'muon_ns_steps': muon_ns_steps,
            'muon_nesterov': muon_nesterov,
            # Per-pack-member spectral correction `max(1, m_i/n_i)**0.5`.
            # Override per param group with a (pack_size,) tensor when the
            # ParameterPack zero-pads matrices of different actual shapes; falls
            # back to the padded global shape when None.
            'muon_update_scale': None,
        }
        for key, value, is_valid_fn in [
            ('lr', lr, _is_valid_lr),
            ('beta1', beta1, _is_valid_beta),
            ('beta2', beta2, _is_valid_beta),
            ('eps', eps, _is_valid_eps),
            ('weight_decay', weight_decay, _is_valid_weight_decay),
            ('muon_lr', muon_lr, _is_valid_lr),
            ('muon_momentum', muon_momentum, _is_valid_beta),
        ]:
            if isinstance(value, list):
                assert len(value) == pack_size
                assert all(map(is_valid_fn, value))
            elif value is not None:
                is_valid_fn(value)
            defaults[key] = value

        super().__init__(params, defaults)
        self._shared_step = shared_step
        self._follow_pytorch = follow_pytorch

    def _step_muon(self, group: dict[str, Any]) -> None:
        lr = group['muon_lr']
        momentum = group['muon_momentum']
        weight_decay = group['weight_decay']
        ns_steps = group['muon_ns_steps']
        nesterov = group['muon_nesterov']
        update_scale = group['muon_update_scale']

        if lr is None:
            lr = group['lr']

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group['params']:
            # 3 = 2 layer dimensions + 1 pack dimension
            assert p.ndim == 3
            assert isinstance(p, ParameterPack), (
                'For now, only parameter packs are supported'
            )
            if p.grad is None:
                continue

            grad = p.grad.data
            assert not grad.is_sparse, 'Sparse gradients are not supported'

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                # Initialize the state.
                state['muon_momentum_buffer'] = torch.zeros_like(p)
            momentum_buffer: Tensor = state['muon_momentum_buffer']

            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            momentum_buffer.lerp_(grad, maybe_unsqueeze(1 - momentum))
            update = (
                grad.lerp_(momentum_buffer, maybe_unsqueeze(momentum))
                if nesterov
                else momentum_buffer
            )
            update = vendor.muon.zeropower_via_newtonschulz5(update, steps=ns_steps)
            if update_scale is None:
                update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
            else:
                update.mul_(update_scale.view(-1, 1, 1))

            assert update.shape == p.shape
            if isinstance(lr, float):
                p.sub_(update, alpha=lr)
            else:
                p.sub_(update.mul_(maybe_unsqueeze(lr)))

    def _step_adamw(self, group: dict[str, Any]) -> None:
        lr = group['lr']
        beta1 = group['beta1']
        beta2 = group['beta2']
        eps = group['eps']
        weight_decay = group['weight_decay']

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group['params']:
            assert isinstance(p, ParameterPack), (
                'For now, only parameter packs are supported'
            )
            if p.grad is None:
                continue

            grad = p.grad.data
            assert not grad.is_sparse, 'Sparse gradients are not supported'

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                # Initialize the state.
                state['step'] = (
                    0
                    if self._shared_step
                    else torch.zeros(
                        get_pack_size(p), dtype=torch.int64, device=p.device
                    )
                )
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            state['step'] += 1

            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            # Update biased first moment estimate.
            exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

            # Update biased second raw moment estimate.
            if self._follow_pytorch and isinstance(beta2, float):
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            else:
                exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

            # Perform the bias correction.
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            step_size = lr / bias_correction1

            if self._follow_pytorch and isinstance(bias_correction2, float):
                denom = exp_avg_sq.sqrt() / bias_correction2**0.5
            else:
                denom = exp_avg_sq.sqrt().div_(maybe_unsqueeze(bias_correction2**0.5))
            denom.add_(maybe_unsqueeze(eps))

            if isinstance(step_size, float):
                p.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                p.sub_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)))

    @torch.no_grad()
    def step(  # type: ignore
        self,
        closure: None | Callable[[], Tensor] = None,
    ) -> None | Tensor:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['muon']:
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss


# >>> Optimizer utilities


def optimizer_pack_remove(
    optimizer: torch.optim.Optimizer,
    pack_idx: Tensor,
    old_to_new: dict[ParameterPack, ParameterPack],
) -> None:
    assert len(pack_idx) > 0
    assert old_to_new

    pack_size = len(next(iter(old_to_new.keys())))
    keep_pack_idx = make_keep_pack_idx(pack_size, pack_idx)

    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if isinstance(value, Tensor) and value.ndim > 0:
                group[key] = value[keep_pack_idx].clone()
            del key, value

        for i, p in list(enumerate(group['params'])):
            if isinstance(p, ParameterPack):
                state = optimizer.state.pop(p, None)
                # `state` can be missing even if optimizer has already been used.
                # For example, in MLPBackbonePack, some of the blocks remain unused
                # (and thus don't have the corresponding optimizer states)
                # if the maximum allowed number of blocks is never used.
                if state is not None:
                    for key, value in list(state.items()):
                        if isinstance(value, Tensor) and value.ndim > 0:
                            state[key] = value[keep_pack_idx].clone()
                        del key, value
                p_new = old_to_new[p]
                group['params'][i] = p_new
                if state is not None:
                    optimizer.state[p_new] = state
