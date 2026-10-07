# This file is based on:
#
# - https://github.com/kyleliang919/C-Optim/blob/2c0517c4db2e50b56d389ba4803eab1f4b3b8ae3/c_adamw.py
# - https://github.com/kyleliang919/C-Optim/blob/2c0517c4db2e50b56d389ba4803eab1f4b3b8ae3/opt_utils.py
#
# Changes:
# - opt_utils.py is merged to c_adamw.py, i.e. to this file.
# - `torch.compile` is not used anymore
# - `_to_scalar` is not used anymore
#
# fmt: off
# isort: off

# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Generator, cast, Callable, List, Iterable, Tuple, Optional, Union

from itertools import chain
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor import DeviceMesh, DTensor

from torch.optim.optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _stack_if_compiling,
    # _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
)

import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed.tensor import DTensor
from typing import Generator, List, Optional, Union


def to_local(tensor: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
    """
    Convert a single DTensor or list of DTensors to local tensors.
    This is a no-op for regular tensors.
    """
    if isinstance(tensor, Tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]


def dtensor_from_local(
    tensor: Union[Tensor, List[Tensor]], ref: Tensor
) -> Union[DTensor, List[DTensor]]:
    """
    Convert a single local Tensor or list of local Tensors to DTensor.
    The reference tensor's device mesh and placements are used to create the DTensor.
    if the reference tensor is not a DTensor, we return the input unmodified.
    """
    if not isinstance(ref, DTensor):
        assert isinstance(ref, Tensor)
        return tensor

    device_mesh = ref.device_mesh
    placements = ref.placements

    # If we have a single tensor
    if isinstance(tensor, Tensor):
        assert not isinstance(tensor, DTensor)
        return DTensor.from_local(
            tensor, device_mesh=device_mesh, placements=placements
        )

    # We have a list of tensors
    assert not isinstance(tensor[0], DTensor)
    return [
        DTensor.from_local(t, device_mesh=device_mesh, placements=placements)
        for t in tensor
    ]


def create_param_batches(
    params: List[Tensor], batch_size: int
) -> Generator[List[Tensor], None, None]:
    """
    Batch parameters into groups of size `batch_size`.
    Tensors in each batch will have identical shape, sharding, and dtype.
    """
    # Group parameters by shape, sharding, and dtype
    groups = defaultdict(list)
    for p in params:
        sharding = p.placements if isinstance(p, DTensor) else None
        groups[(p.shape, sharding, p.dtype)].append(p)

    # Create batches from grouped parameters
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            yield batch


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """
    Insert dummy tensors so the batch has exactly `batch_size` elements.
    """
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.empty_like(batch[0]))
    return batch


class AsyncTask:
    """
    AsyncTask wraps a Python generator to run until the next yield statement.
    This is used to allow other tasks to run while waiting for distributed operations.
    """

    def __init__(self, generator: Generator[None, None, None]):
        self._generator = generator
        self.run()  # Start running the generator

    def run(self) -> bool:
        # Run the next step of the async task.
        # Returns True if the task is still running and False if completed.
        try:
            next(self._generator)
            return True
        except StopIteration:
            pass
        return False


class AsyncRuntime:
    """
    Event loop for running multiple async tasks concurrently.
    """

    def __init__(
        self, task_gen: Generator["AsyncTask", None, None], max_concurrent_tasks: int
    ):
        # Initialize runtime with a generator that produces AsyncTask objects
        if max_concurrent_tasks <= 0:
            raise ValueError(f"{max_concurrent_tasks=} cannot be <= 0")
        self._task_gen = task_gen
        self._max_concurrent_tasks = max_concurrent_tasks

    def _get_next_task(self) -> Optional["AsyncTask"]:
        try:
            task = next(self._task_gen)
            return task
        except StopIteration:
            return None

    def run(self):
        # Run the event loop until all tasks are completed
        have_new_tasks = True
        previous_tasks: List["AsyncTask"] = []

        while have_new_tasks or previous_tasks:
            # See if we can add another task
            running_tasks = []
            if have_new_tasks and len(previous_tasks) < self._max_concurrent_tasks:
                new_task = self._get_next_task()
                if new_task is not None:
                    # Add new task to the queue
                    running_tasks.append(new_task)
                else:
                    # No more tasks left
                    have_new_tasks = False

            # Run all previous tasks for one step
            for task in previous_tasks:
                still_running = task.run()
                if still_running:
                    running_tasks.append(task)

            # Update task list for next iteration
            previous_tasks = running_tasks


def _multi_tensor_c_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
    cautious: bool = False,  # new flag
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        if not capturable:
            raise RuntimeError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

    if isinstance(beta1, Tensor):
        if not capturable:
            raise ValueError("beta1 as a Tensor is not supported for capturable=False and foreach=True")
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")

    if isinstance(beta2, Tensor):
        if not capturable:
            raise ValueError("beta2 as a Tensor is not supported for capturable=False and foreach=True")
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")

    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(supports_xla=False)
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), (
            f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
        )

    assert grad_scale is None and found_inf is None
    assert not differentiable, "_foreach ops don't support autograd"

    # lr = _to_scalar(lr)
    lr = float(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )

    beta1_dict: Optional[DeviceDict] = (
        {beta1.device: beta1}
        if isinstance(beta1, Tensor) and str(beta1.device) != "cpu"
        else None
    )

    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)

        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            else:
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        torch._foreach_lerp_(device_exp_avgs, device_grads, cast(float, 1 - device_beta1))
        torch._foreach_mul_(device_exp_avg_sqs, beta2)

        if isinstance(beta2, torch.Tensor):
            scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)
            value = 1.0
        else:
            scaled_device_grads = device_grads
            value = 1 - beta2

        torch._foreach_addcmul_(device_exp_avg_sqs, scaled_device_grads, device_grads, value)

        del scaled_device_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction2)

            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)
            torch._foreach_sqrt_(bias_correction2)

            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            if cautious:
                # cautious masking
                mask = torch._foreach_mul(device_exp_avgs, device_grads)
                mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, device_exp_avgs)]
                mean_mask = [m.mean().clamp(min=1e-3) for m in mask]
                mask = [m / mm for m, mm in zip(mask, mean_mask)]
                masked_exp_avg = torch._foreach_mul(device_exp_avgs, mask)
                torch._foreach_addcdiv_(device_params, masked_exp_avg, exp_avg_sq_sqrt)
            else:
                torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)

        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])
            bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)

            if cautious:
                mask = torch._foreach_mul(device_exp_avgs, device_exp_avgs)
                mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, device_exp_avgs)]
                mean_mask = [m.mean().clamp(min=1e-3) for m in mask]
                mask = [m / mm for m, mm in zip(mask, mean_mask)]
                masked_exp_avg = torch._foreach_mul(device_exp_avgs, mask)
                torch._foreach_addcdiv_(
                    device_params, masked_exp_avg, exp_avg_sq_sqrt, step_size
                )
            else:
                torch._foreach_addcdiv_(
                    device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size
                )

# @torch.compile(fullgraph=True)
def c_adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    C AdamW optimizer algorithm (async foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1
    mask = torch._foreach_mul(M, G)
    mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, M)]
    mask_mean = [m.mean().clamp(min=1e-3) for m in mask]
    M = torch._foreach_mul(M, mask)
    M_div = torch._foreach_div(M, denom)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_div_(M_div, mask_mean)
    torch._foreach_sub_(X, M_div)
    yield

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        foreach: bool = True,
        fused: bool = False
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.init_lr = lr
        self._foreach = foreach
        try:
            self._world_size = dist.get_world_size()
        except:
            self._world_size = 1
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (torch.tensor(0.0, dtype=_get_scalar_dtype()))
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])
        return has_complex

    def _get_or_initialize_state(self, param):
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            state["variance"] = torch.zeros_like(param)
        return state

    def _create_tasks(self, param_groups):
        for group in param_groups:
            for params in create_param_batches(group["params"], batch_size = self._world_size):
                params = [p for p in params if p.grad is not None]
                if not params:
                    continue
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p) for p in params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]
                lr = torch.tensor(group["lr"])
                beta1 = torch.tensor(group["betas"][0])
                beta2 = torch.tensor(group["betas"][1])
                weight_decay = torch.tensor(group["weight_decay"])
                epsilon = torch.tensor(group["eps"])
                step = torch.tensor(group["step"])
                yield AsyncTask(
                    c_adamw_update_foreach_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(gradients, self._world_size),
                        M=pad_batch(momentums, self._world_size),
                        V=pad_batch(variances, self._world_size),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon
                    )
                )
    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self._foreach:
            # collect grouped tensors for multi-tensor version
            '''
            params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps = [], [], [], [], []
            max_exp_avg_sqs = []

            for group in self.param_groups:

                has_complex = self._init_group(
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                )

                beta1, beta2 = group["betas"]
                if len(params_with_grad) > 0:
                    _multi_tensor_c_adam(
                        params=params_with_grad,
                        grads=grads,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        max_exp_avg_sqs=max_exp_avg_sqs,
                        state_steps=state_steps,
                        grad_scale=None,
                        found_inf=None,
                        amsgrad=group.get("amsgrad", False),
                        has_complex=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        maximize=False,
                        capturable=False,
                        differentiable=False,
                        decoupled_weight_decay=True,
                        cautious=True,  # <<< enable cautious logic
                    )
            return loss
            '''

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            groups = []
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                group["step"] += 1
                groups.append(group)
            tasks = self._create_tasks(groups)
            all_tasks = chain(tasks)
            runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
            runtime.run()

            return loss

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # apply weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                if type(grad) is torch.distributed.tensor.DTensor:
                    mask = (exp_avg.full_tensor() * grad.full_tensor() > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mask = distribute_tensor(mask, device_mesh = grad.device_mesh, placements = grad.placements)
                else:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    # mask = mask * (mask.numel() / (mask.sum() + 1)) ## original implementation, leaving it here for record
                    mask.div_(mask.mean().clamp_(min=1e-3)) # https://huggingface.co/rwightman/timm-optim-caution found this implementation is more favoarable in many cases
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss

def test_c_adamw():
    import copy
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Linear(10, 1).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
        foreach=True
    )
    # Dummy input and target
    x = torch.randn(16, 10, device=device)
    y = torch.randn(16, 1, device=device)

    # Forward
    y_pred = model(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)

    # Backward
    loss.backward()

    # Save params before step
    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])

    # Step
    optimizer.step()
    optimizer.zero_grad()

    # Save params after step
    after = [p.detach().clone() for p in model.parameters()]

    # Print parameter changes
    for i, (b, a) in enumerate(zip(before, after)):
        diff = (b - a).abs().sum().item()
        print(f"Param {i} changed by: {diff:.6f}")
        assert diff > 0, f"Parameter {i} did not change!"

    print("✅ C-AdamW with foreach ran successfully.")

if __name__ == "__main__":
    test_c_adamw()
