import torch
from torch.optim.optimizer import Optimizer


class Signum(Optimizer):
    def __init__(
        self, params, *, lr: float, momentum: float = 0.9, weight_decay: float
    ) -> None:
        assert lr >= 0.0
        assert 0.0 <= momentum <= 1.0
        assert weight_decay >= 0.0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Apply the decoupled weight decay.
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    # Initialize the momentum buffer.
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                momentum = group['momentum']

                exp_avg.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                p.add_(exp_avg.sign(), alpha=-group['lr'])

        return loss
