# This is an implementation of SAM and GSAM based on this file:
# https://github.com/juntang-zhuang/GSAM/blob/a770275b3324a3ed4bf4a1cddb976b2782ba852b/gsam/gsam.py
#
# Compared to the above source:
# - Gradient assembly, parameter restoration, and per-group settings follow the
#   retained AdamW-based research adaptation.
# - With GSAM enabled, this snapshot assigns rho to the group's alpha setting.
# - The original SAM is supported by setting alpha=None.
# - The API is closer to standard PyTorch optimizers.
# - Some of the original features (e.g. adaptive=True, multi-GPU training, etc.)
#   are not supported.
#
# NOTE: with AdamWSharpnessAware, the results are not deterministic for unknown reasons.

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

_EPS = 1e-12


def _disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum  # type: ignore
            module.momentum = 0

    model.apply(_disable)


def _enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum  # type: ignore

    model.apply(_enable)


class AdamWSharpnessAware(torch.optim.AdamW):
    def __init__(
        self, params, *, rho: float, alpha: None | float = None, **kwargs
    ) -> None:
        super().__init__(params, **kwargs)

        self.defaults['rho'] = rho
        for group in self.param_groups:
            group['rho'] = rho

        if alpha is not None:
            self._is_gsam = True
            self.defaults['alpha'] = alpha
            for group in self.param_groups:
                group['alpha'] = rho
        else:
            self._is_gsam = False

    def _assemble_gradient(self) -> Tensor:
        return torch.cat(
            [
                p.grad.flatten()
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]
        )

    def _assemble_original_gradient(
        self, original_gradients: dict[torch.nn.Parameter, Tensor]
    ) -> Tensor:
        return torch.cat(
            [
                original_gradients[p].flatten()
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]
        )

    @torch.no_grad()
    def _perturb_parameters(
        self,
    ) -> tuple[
        dict[torch.nn.Parameter, Tensor], None | dict[torch.nn.Parameter, Tensor]
    ]:
        original_parameters = {}
        original_gradients = {} if self._is_gsam else None

        grad_norm = self._assemble_gradient().norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + _EPS)
            for p in group['params']:
                if p.grad is None:
                    continue
                original_parameters[p] = p.data.clone()
                if self._is_gsam:
                    assert original_gradients is not None
                    original_gradients[p] = p.grad.data.clone()
                e_w = p.grad * scale
                # Perturb the parameter (climb to the local maximum).
                p.add_(e_w)

        return original_parameters, original_gradients

    @torch.no_grad()
    def _restore_original_parameters(
        self, original_parameters: dict[torch.nn.Parameter, Tensor]
    ):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = original_parameters[p]

    @torch.no_grad()
    def _apply_gsam(self, original_gradients: dict[torch.nn.Parameter, Tensor]):
        grad = self._assemble_gradient()
        original_grad = self._assemble_original_gradient(original_gradients)

        grad_norm = grad.norm()
        original_grad_norm = original_grad.norm()

        cosine = torch.dot(grad, original_grad) / (
            grad_norm * original_grad_norm + _EPS
        )

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # fmt: off
                vertical = (
                    original_gradients[p]
                    - cosine * original_grad_norm * p.grad / (grad_norm + _EPS)
                )
                # fmt: on
                p.grad.data.sub_(vertical, alpha=group['alpha'])

    def step(self, closure, model: torch.nn.Module) -> Tensor:  # type: ignore
        # Compute the original gradients.
        self.zero_grad()
        with torch.enable_grad():
            loss = closure()
        # Perturb the weights based on the original gradients.
        original_parameters, original_gradients = self._perturb_parameters()

        # Compute the gradients for the perturbed weights.
        _disable_running_stats(model)
        self.zero_grad()
        with torch.enable_grad():
            closure()
        if self._is_gsam:
            assert original_gradients is not None
            self._apply_gsam(original_gradients)

        # Restore the original weights and perform the actual optimization step.
        self._restore_original_parameters(original_parameters)
        super().step()
        _enable_running_stats(model)

        return loss
