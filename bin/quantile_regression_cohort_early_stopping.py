import datetime
import json
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
from lib.cohorts import (
    CohortEarlyStoppingController,
    build_balanced_tied_cohorts,
    build_density_cohorts,
    build_kmeans_cohorts,
    build_quantile_cohorts,
    build_random_cohorts,
    build_random_stitched_cohorts,
    stitch_predictions,
)

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


class QuantileLoss(nn.Module):
    def __init__(
        self,
        num_quantiles: int = 999,
        device: torch.device = torch.device('cuda:0'),
        agg_mode: str = 'sum',
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.register_buffer(
            'quantiles', torch.linspace(0.001, 0.999, num_quantiles, device=device)
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
    parameter_statistics: NotRequired[bool]
    # NOTE
    # For models like MLP or TabM only amp_dtype="bfloat16" was tested,
    # so "float16" is supported only for completeness
    # (in theory, it can be a better choice for some models).
    amp_dtype: NotRequired[AMPDType]
    compile: NotRequired[bool]

    n_quantiles: int
    agg_mode: str

    target_transform: NotRequired[Literal['power', 'quantile', 'identity']]
    y_scale_method: NotRequired[str]
    quantile_output_distribution: NotRequired[str]
    verbose: NotRequired[bool]
    source_evaluation_config: NotRequired[str]
    cohort_strategy: str
    n_cohorts: int
    cohort_rng_seed: int
    density_k: int
    cohort_cache_dir: str
    cohort_uncertainty_path: str
    cohort_uncertainty_kind: str
    compute_density_signals: NotRequired[bool]
    balance_ties_on_val: NotRequired[bool]


def _build_cohorts_for_run(config: Config, exp: Path) -> Path:
    """Build and persist a deterministic cohort definition as part of this run."""

    strategy = config['cohort_strategy']
    n_cohorts = config['n_cohorts']
    rng_seed = config['cohort_rng_seed']
    if strategy == 'random_stitched':
        # Keep cohort sampling independent of the model/batch RNG while
        # resampling the random control for each evaluation seed.
        rng_seed += config.get('seed', 0)
    data_dir = Path(config['data']['path'])
    uncertainty_path = Path(config['cohort_uncertainty_path'])
    if config['cohort_uncertainty_kind'] == 'synthetic_gt_log_sigma':
        uncertainty = (
            torch.load(uncertainty_path, weights_only=True).detach().cpu().numpy()
        )
        uncertainty_val, uncertainty_test = (
            uncertainty[-30_000:-15_000],
            uncertainty[-15_000:],
        )
    elif config['cohort_uncertainty_kind'] == 'homoscedastic':
        uncertainty_val = np.zeros(len(np.load(data_dir / 'Y_val.npy')))
        uncertainty_test = np.zeros(len(np.load(data_dir / 'Y_test.npy')))
    else:
        uncertainty = np.load(uncertainty_path)
        uncertainty_val, uncertainty_test = uncertainty['val'], uncertainty['test']

    def load_feature(name: str, part: str) -> np.ndarray | None:
        path = data_dir / f'{name}_{part}.npy'
        return np.load(path) if path.exists() else None

    compute_density = strategy == 'density' or config.get(
        'compute_density_signals', True
    )
    need_features = compute_density or strategy == 'kmeans'
    x_num = (
        {part: load_feature('X_num', part) for part in ('train', 'val', 'test')}
        if need_features
        else None
    )
    x_cat = (
        {part: load_feature('X_cat', part) for part in ('train', 'val', 'test')}
        if need_features
        else None
    )
    density_val = density_test = density_assignment = None
    density_metadata = {}
    if compute_density:
        assert x_num is not None and x_cat is not None
        cache_dir = Path(config['cohort_cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        density_cache = cache_dir / f'density_k{config["density_k"]}.npz'
        density_metadata_cache = cache_dir / f'density_k{config["density_k"]}.json'
        if density_cache.exists() and density_metadata_cache.exists():
            cached_density = np.load(density_cache)
            density_val = cached_density['val']
            density_test = cached_density['test']
            density_metadata = json.loads(density_metadata_cache.read_text())
        else:
            density_assignment = build_density_cohorts(
                x_num['train'],
                x_cat['train'],
                x_num['val'],
                x_cat['val'],
                x_num['test'],
                x_cat['test'],
                n_cohorts=n_cohorts,
                density_k=config['density_k'],
            )
            density_val = density_assignment.val_signal
            density_test = density_assignment.test_signal
            density_metadata = density_assignment.metadata
            np.savez(density_cache, val=density_val, test=density_test)
            density_metadata_cache.write_text(json.dumps(density_metadata, indent=2))
    if strategy == 'uncertainty':
        assignment = (
            build_balanced_tied_cohorts(
                uncertainty_val, uncertainty_test, n_cohorts=n_cohorts
            )
            if config['cohort_uncertainty_kind'] == 'homoscedastic'
            else build_quantile_cohorts(
                uncertainty_val,
                uncertainty_test,
                n_cohorts=n_cohorts,
                balance_ties_on_val=config.get('balance_ties_on_val', False),
            )
        )
    elif strategy == 'random':
        assignment = build_random_cohorts(
            uncertainty_val, n_cohorts=n_cohorts, rng=np.random.default_rng(rng_seed)
        )
    elif strategy == 'random_stitched':
        assignment = build_random_stitched_cohorts(
            uncertainty_val,
            uncertainty_test,
            n_cohorts=n_cohorts,
            rng_seed=rng_seed,
            balance_ties_on_val=config.get('balance_ties_on_val', False),
        )
    elif strategy == 'kmeans':
        assert x_num is not None and x_cat is not None
        assignment = build_kmeans_cohorts(
            x_num['train'],
            x_cat['train'],
            x_num['val'],
            x_cat['val'],
            x_num['test'],
            x_cat['test'],
            n_cohorts=n_cohorts,
            rng_seed=rng_seed,
        )
    elif strategy == 'density':
        assert density_val is not None and density_test is not None
        assignment = (
            density_assignment
            if density_assignment is not None
            else build_quantile_cohorts(
                density_val,
                density_test,
                n_cohorts=n_cohorts,
                strategy='density',
            )
        )
        assignment.metadata.update(density_metadata)
    else:
        raise ValueError(f'unknown cohort_strategy: {strategy}')

    cohort_dir = exp / 'cohort_definition'
    cohort_dir.mkdir(parents=True, exist_ok=True)
    np.save(cohort_dir / 'val_cohorts.npy', assignment.val_ids)
    if assignment.test_ids is not None:
        np.save(cohort_dir / 'test_cohorts.npy', assignment.test_ids)
    np.savez(
        cohort_dir / 'signals.npz',
        val=assignment.val_signal,
        **({} if assignment.test_signal is None else {'test': assignment.test_signal}),
    )
    extra_signals = {
        'uncertainty_val': uncertainty_val,
        'uncertainty_test': uncertainty_test,
    }
    if density_val is not None and density_test is not None:
        extra_signals.update(density_val=density_val, density_test=density_test)
    np.savez(cohort_dir / 'extra_signals.npz', **extra_signals)
    metadata = dict(assignment.metadata)
    for part in ('val', 'test'):
        counts = metadata.get(f'{part}_counts')
        if counts is not None:
            total = sum(counts)
            metadata[f'{part}_fractions'] = [count / total for count in counts]
    metadata.update(
        {'strategy': strategy, 'rng_seed': rng_seed, 'density_k': config['density_k']}
    )
    (cohort_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return cohort_dir


def _record_stitched_metrics(
    report: lib.experiment.Report, stitched_metrics: dict
) -> None:
    report['global_metrics'] = report['metrics']
    report['metrics'] = stitched_metrics
    report['metrics_source'] = 'oracle_predictions.npz'


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

    model = tabm_reference.Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=config['n_quantiles'],
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

    # def base_loss_fn(logits: Tensor, y_true):
    #     mu, logvar = logits[..., 0], logits[..., 1]
    #     loss_vec = logvar / 2 + ((mu - y_true) ** 2) / (2 * (logvar.exp()))
    #     return loss_vec.mean()

    base_loss = QuantileLoss(
        config['n_quantiles'], agg_mode=config.get('agg_mode', 'sum')
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
    cohort_dir = _build_cohorts_for_run(config, exp)
    val_cohort = np.load(cohort_dir / 'val_cohorts.npy')
    test_path = cohort_dir / 'test_cohorts.npy'
    test_cohort = np.load(test_path) if test_path.exists() else None
    n_cohorts = int(np.max(val_cohort)) + 1
    if len(val_cohort) != dataset.size('val'):
        raise ValueError('val cohort IDs do not match the validation split')
    if np.any(np.bincount(val_cohort, minlength=n_cohorts) == 0):
        raise ValueError('at least one validation cohort is empty')
    if test_cohort is not None and len(test_cohort) != dataset.size('test'):
        raise ValueError('test cohort IDs do not match the test split')
    controller = CohortEarlyStoppingController(
        patience=config['patience'], n_cohorts=n_cohorts
    )

    original_y_val = (
        dataset.data['y']['val'].detach().cpu().numpy() * regression_label_stats.std
        + regression_label_stats.mean
    )
    original_y_test = (
        dataset.data['y']['test'].detach().cpu().numpy() * regression_label_stats.std
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
        for batch_idx in tqdm(
            batches,
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

        eval_parts = (
            ['train', 'val', 'test'] if config.get('verbose') else ['val', 'test']
        )
        (metrics, predictions), eval_batch_size = evaluate(
            parts=eval_parts, batch_size=eval_batch_size
        )

        cohort_mse = np.asarray(
            [
                (
                    (
                        predictions['val'][val_cohort == cohort]
                        - original_y_val[val_cohort == cohort]
                    )
                    ** 2
                ).mean()
                for cohort in range(n_cohorts)
            ]
        )
        global_val_mse = ((predictions['val'] - original_y_val) ** 2).mean()
        controller.update(
            step // epoch_size,
            global_val_mse,
            cohort_mse,
            predictions['val'],
            predictions['test'],
        )
        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score'] + 1e-7
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
            # best_checkpoint = make_checkpoint()

        if config.get('verbose'):
            lib.verbose_stats.log_grad_norm_stats(_verbose_grad_norms, writer, step)
            lib.verbose_stats.collect_epoch_eff_rank(
                model, dataset, writer, step, is_clustering=False
            )
            model.train()

        if controller.all_cohorts_stopped or not lib.util.are_valid_predictions(
            predictions
        ):
            break

    report['time'] = timer.elapsed()

    # >>>
    # if best_checkpoint is not None:
    #    model.load_state_dict(best_checkpoint['model'])
    # (metrics, predictions), eval_batch_size = evaluate(
    #    parts=['train', 'val', 'test'], batch_size=eval_batch_size
    # )
    # metrics, predictions =
    global_selection = controller.finalize_global()
    if any(selection is None for selection in controller.cohort_selections):
        raise RuntimeError('cohort early stopping did not freeze every selection')
    predictions = {
        'val': global_selection.val_predictions,
        'test': global_selection.test_predictions,
    }
    global_metrics = dataset.task.calculate_metrics(predictions, 'labels')
    report['eval_batch_size'] = eval_batch_size
    report['best_step'] = global_selection.epoch * epoch_size
    report['metrics'] = global_metrics
    lib.experiment.dump_predictions(exp, predictions)
    selections = [
        selection for selection in controller.cohort_selections if selection is not None
    ]
    for cohort, selection in enumerate(selections):
        np.savez(
            exp / f'cohort_{cohort}_predictions.npz',
            val=selection.val_predictions,
            test=selection.test_predictions,
        )
    if test_cohort is not None:
        np.savez(
            exp / 'oracle_predictions.npz',
            val=stitch_predictions([x.val_predictions for x in selections], val_cohort),
            test=stitch_predictions(
                [x.test_predictions for x in selections], test_cohort
            ),
        )
        if config['cohort_strategy'] in {'uncertainty', 'random_stitched'}:
            oracle = np.load(exp / 'oracle_predictions.npz')
            _record_stitched_metrics(
                report,
                dataset.task.calculate_metrics(
                    {'val': oracle['val'], 'test': oracle['test']}, 'labels'
                ),
            )
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
    oracle_rmse = None
    if test_cohort is not None:
        oracle = np.load(exp / 'oracle_predictions.npz')
        oracle_rmse = {
            part: float(np.sqrt(np.mean((oracle[part] - target) ** 2)))
            for part, target in [('val', original_y_val), ('test', original_y_test)]
        }
    with open(exp / 'cohort_metadata.json', 'w', encoding='utf-8') as output:
        json.dump(
            {
                'n_cohorts': int(n_cohorts),
                'source_evaluation_config': config.get('source_evaluation_config'),
                'assignment': assignment_metadata,
                'global_selected_epoch': global_selection.epoch,
                'global_stopped_epoch': controller.global_stopped_epoch,
                'cohort_selected_epochs': [x.epoch for x in selections],
                'cohort_stopped_epochs': controller.cohort_stopped_epochs,
                'oracle_rmse': oracle_rmse,
                'history': controller.history,
            },
            output,
            indent=2,
        )
    # One self-contained checkpoint holds model/optimizer state, all point
    # prediction snapshots and the cohort/global stopping history.
    lib.experiment.dump_checkpoint(
        exp, {**make_checkpoint(), 'cohort_controller': controller}
    )

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
