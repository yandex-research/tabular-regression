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

    logits_predictions = {
        part: (
            # target_transform.inverse_transform(
            torch.cat(
                [
                    apply_model(model, dataset, part=part, idx=idx)
                    for idx in torch.arange(dataset.size(part), device=device).split(
                        batch_size
                    )
                ]
            )
            # .mean(1)
            # .cpu()
            # .numpy()
            #     .reshape(-1, 1)
            # ).squeeze()
        )
        for part in parts
    }

    mixture_size = logits_predictions[parts[0]][0].size(-1) // 3
    for part in parts:
        logits_predictions[part][..., :mixture_size] = nn.functional.softmax(
            logits_predictions[part][..., :mixture_size], dim=-1
        )
        logits_predictions[part] = logits_predictions[part].mean(1).cpu().numpy()

    predictions = {k: get_scalar_prediction(v) for k, v in logits_predictions.items()}
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


class MDNLoss(nn.Module):
    def __init__(self, mixture_size: int):
        super().__init__()
        self.mixture_size = mixture_size

    def forward(
        self, y_true: Tensor, pi_logits: Tensor, mu: Tensor, logsigma: Tensor
    ) -> Tensor:
        """
        y_true: (B), where K = mixture_size
        pi_logits: (B, K) - logits of components weights
        mu: (B, K) - means of gaussians
        logsigma: (B, K) - log std of gaussians
        """
        y_true = y_true.unsqueeze(1)
        log_pi = nn.functional.log_softmax(pi_logits, dim=-1)
        sigma = torch.exp(logsigma)
        log_normal_pdf = -((y_true - mu) ** 2) / 2 / sigma**2 - logsigma

        loss = -torch.logsumexp(log_pi + log_normal_pdf, dim=-1)
        return loss.mean()


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

    mixture_size: int

    target_transform: NotRequired[Literal['power', 'quantile', 'identity']]
    y_scale_method: NotRequired[str]
    quantile_output_distribution: NotRequired[str]


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
    mixture_size = config['mixture_size']

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

    model = tabm_reference.Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=3 * mixture_size,
        **config['model'],
        bins=bin_edges,
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'logits'
    model.to(device)
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

    base_loss_fn = MDNLoss(mixture_size=mixture_size)

    def loss_fn(y_true: Tensor, pi_logits: Tensor, mu: Tensor, logsigma: Tensor):
        pi_logits = pi_logits.flatten(0, 1)
        mu = mu.flatten(0, 1)
        logsigma = logsigma.flatten(0, 1)
        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            k = model.k if model.arch_type != 'plain' else 1
            y_true = y_true.repeat_interleave(k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)
        return base_loss_fn(y_true, pi_logits, mu, logsigma)

    def get_scalar_prediction(logits_prediction: Tensor | np.ndarray):
        if not isinstance(logits_prediction, Tensor):
            logits_prediction = torch.from_numpy(logits_prediction).to(device)

        assert logits_prediction.ndim == 2
        pi = logits_prediction[..., :mixture_size]
        mu = logits_prediction[..., mixture_size : 2 * mixture_size]

        return (pi * mu).sum(-1).cpu().numpy()

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
            outputs = apply_model(model, dataset, part='train', idx=batch_idx)
            pi_logits = outputs[..., :mixture_size]
            mu = outputs[..., mixture_size : 2 * mixture_size]
            logsigma = outputs[..., 2 * mixture_size :]

            loss = loss_fn(
                Y_train[batch_idx], pi_logits=pi_logits, mu=mu, logsigma=logsigma
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
