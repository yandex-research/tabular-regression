# NOTE[tabular]
"""Train and evaluate a multilayer perceptron."""

import datetime
import math
import statistics
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, NotRequired, Protocol, TypedDict

import delu
import numpy as np
import tabm
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib
import lib.data
import lib.deep
import lib.env
import lib.experiment
import lib.util
from lib.types import AMPDType, KWArgs, PartKey, PredictionType


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        backbone: KWArgs,
    ) -> None:
        assert n_num_features > 0 or cat_cardinalities
        super().__init__()
        self.cat_module = (
            lib.deep.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        self.backbone = tabm.MLPBackbone(
            d_in=n_num_features + sum(cat_cardinalities), **backbone
        )
        self.output = nn.Linear(
            backbone['d_block'], 1 if n_classes is None or n_classes == 2 else n_classes
        )
        self._n_num_features = n_num_features

    def forward(self, x_num: None | Tensor, x_cat: None | Tensor) -> Tensor:
        x_list: list[Tensor] = []

        if x_num is None:
            assert self._n_num_features == 0
        else:
            assert self._n_num_features > 0
            x_list.append(x_num)

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_list.append(self.cat_module(x_cat).to(torch.get_default_dtype()))

        x = torch.column_stack(x_list)

        x = self.backbone(x)
        x = self.output(x)
        return x


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
) -> EvaluateImplOutput:
    model.eval()

    predictions = {
        part: (
            torch.cat(
                [
                    apply_model(model, dataset, part=part, idx=idx)
                    for idx in torch.arange(dataset.size(part), device=device).split(
                        batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        for part in parts
    }

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        for part in predictions:
            predictions[part] *= regression_label_stats.std
            predictions[part] += regression_label_stats.mean

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
    amp_dtype: NotRequired[AMPDType]
    compile: NotRequired[bool]


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])
    device = lib.util.get_device()
    logger.info(f'Device: {device}')

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    regression_label_stats = dataset.try_standardize_labels_()
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )

    # >>> Model
    model: Model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        **config['model'],
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = prediction_type = (
        'labels' if dataset.task.is_regression else 'logits'
    )
    model.to(device)

    # >>> Training
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    gradient_clipping_norm = config.get('gradient_clipping_norm')
    loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.binary_cross_entropy_with_logits
        if dataset.task.is_binclass
        else nn.functional.cross_entropy
    )

    step = 0
    batch_size = config['batch_size']
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

    # >>> Efficiency
    amp_dtype = config.get('amp_dtype')
    if amp_dtype is not None:
        amp_dtype = lib.util.get_amp_dtype(amp_dtype, device)
    grad_scaler = (
        torch.amp.GradScaler(device.type) if amp_dtype is torch.float16 else None  # type: ignore
    )
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
            prediction_type=prediction_type,
            device=device,
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
    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        model.train()

        batch_losses = []
        epoch_start_time = timer.elapsed()
        for batch_idx in tqdm(
            torch.randperm(
                dataset.size('train'), generator=batch_generator, device=device
            ).split(batch_size),
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
