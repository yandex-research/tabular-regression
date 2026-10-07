# Copyright 2024 Google LLC.
# Licensed under the Apache License, Version 2.0.
# See vendor/optformer/LICENSE for the complete license.
#
# Modified: PyTorch adaptation of the authors' TensorFlow/Keras attention
# decoder, with benchmark MLP features and vectorized on-device float decoding.
# Source: https://github.com/google-research/optformer/blob/main/optformer/decoding_regression/models.py
# Paper: Song and Bahri, Decoding-based Regression, https://arxiv.org/abs/2501.19383
"""Train and evaluate the benchmark's adapted autoregressive regression decoder.

The local implementation preserves the tokenized regression objective and uses
PyTorch layer conventions. See VENDOR.md for the precise framework adaptations.
"""

import datetime
import math
import statistics
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, NotRequired, Protocol, TypedDict

import attrs
import delu
import numpy as np
import rtdl_num_embeddings
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
import lib.project.optim
import lib.util
import vendor.decoder_vocabs as vocabs
import vendor.tabm_reference as tabm_reference
from lib.types import AMPDType, KWArgs, PartKey, PredictionType

NEG_INF = -1e7

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


def make_num_embeddings(type: str, **kwargs) -> nn.Module:
    return getattr(rtdl_num_embeddings, type)(**kwargs)


@attrs.define
class BatchUnnormalizedVocab(vocabs.UnnormalizedVocab):
    """Torch-friendly implementation for faster inference and decoding"""

    _mantissa_base_powers: torch.Tensor = attrs.field(init=False)
    _exp_base_powers: torch.Tensor = attrs.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # for fast mantissa and exponent calculation
        self._mantissa_base_powers = torch.pow(
            self.base,
            -torch.arange(self.num_mantissa_digits, dtype=torch.float32),
        )
        self._exp_base_powers = torch.pow(
            self.base,
            torch.arange(self.num_exponent_digits - 1, -1, -1, dtype=torch.float32),
        )

    def batch_from_int(self, batch_token_ids: Tensor) -> Tensor:
        """Converts batch of token representatios to the batch of torch.floats

        Args:
            batch_token_ids : Tensor of token_ids (B, L)

        Returns:
            output : Tensor of batch float values (B,)
        """
        B = batch_token_ids.size(0)
        device = batch_token_ids.device
        self._exp_base_powers = self._exp_base_powers.to(device)
        self._mantissa_base_powers = self._mantissa_base_powers.to(device)

        # converting < + > to +1 and < - > to -1
        sign = (-2 * batch_token_ids[:, 0] + 1).float()
        exp_sign = (-2 * batch_token_ids[:, 1] + 1).float()

        # getting exponent and mantissa representation
        exp_digits = (
            batch_token_ids[:, 2 : 2 + self.num_exponent_digits] - 2
        ).float()  # (B, E)
        mantissa_digits = (
            batch_token_ids[:, -self.num_mantissa_digits :] - 2
        ).float()  # (B, M)

        assert exp_digits.size() == (
            B,
            self.num_exponent_digits,
        ), 'Wrong exponent conversion'
        assert mantissa_digits.size() == (
            B,
            self.num_mantissa_digits,
        ), 'Wrong mantissa conversion'

        # calculation mantissa and exponent
        exponent = exp_sign * torch.matmul(exp_digits, self._exp_base_powers)  # (B, )
        mantissa = torch.matmul(mantissa_digits, self._mantissa_base_powers)  # (B, )

        # calculate output and round it
        output = sign * torch.pow(self.base, exponent=exponent) * mantissa
        return output


@attrs.define
class BatchNormalizedVocab(vocabs.NormalizedVocab):
    """Torch-friendly implementation for faster inference and decoding"""

    _base_powers: torch.Tensor = attrs.field(init=False)

    def __attrs_post_init__(self):
        # for fast calculation
        self._base_powers = torch.pow(
            self.base,
            -torch.arange(1, self.length + 1, dtype=torch.float32),
        )

    def batch_from_int(self, batch_token_ids: Tensor) -> Tensor:
        """Converts batch of token representatios to the batch of torch.floats

        Args:
            batch_token_ids : Tensor of token_ids (B, L)

        Returns:
            output : Tensor of batch float values (B,)
        """
        device = batch_token_ids.device
        self._base_powers = self._base_powers.to(device)

        # converting tokens to digits
        digits = batch_token_ids.float()  # (B, L)
        output = torch.matmul(digits, self._base_powers)  # (B, )

        return output


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        backbone: KWArgs,
        decoder: KWArgs,
    ) -> None:
        assert n_num_features > 0 or cat_cardinalities
        super().__init__()
        self.cat_module = (
            lib.deep.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        self.backbone = tabm.MLPBackbone(
            d_in=n_num_features + sum(cat_cardinalities), **backbone
        )
        self._n_num_features = n_num_features

        # preparing vocab and decoder
        self.decoder = decoder
        self.vocab = (
            BatchUnnormalizedVocab(
                base=decoder['vocab']['base'],
                num_exponent_digits=decoder['vocab']['num_exponent_digits'],
                num_mantissa_digits=decoder['vocab']['num_mantissa_digits'],
            )
            if decoder['vocab']['unnormalized']
            else BatchNormalizedVocab(
                base=decoder['vocab']['base'], length=decoder['vocab']['length']
            )
        )

        decoder_dim = decoder['d_block']
        self.enc_proj = nn.Linear(backbone['d_block'], decoder_dim)
        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab.size, embedding_dim=decoder_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=self.vocab.token_length, embedding_dim=decoder_dim
        )

        self.transformer_layers = nn.ModuleList()
        for _ in range(decoder['n_blocks']):
            attn_block = nn.Module()
            attn_block.attn = nn.MultiheadAttention(
                embed_dim=decoder_dim,
                num_heads=decoder['n_heads'],
                dropout=decoder['dropout'],
                batch_first=True,  # work with shape (B, L, D)
            )
            attn_block.ffn = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
            )
            attn_block.layernorm1 = nn.LayerNorm(decoder_dim)
            attn_block.layernorm2 = nn.LayerNorm(decoder_dim)
            self.transformer_layers.append(attn_block)

        self.output = nn.Linear(decoder_dim, self.vocab.size)

    def forward(
        self, x_num: None | Tensor, x_cat: None | Tensor, decoded_ids: Tensor
    ) -> Tensor:
        """Returns probabilities over the next token."""
        # encoding x
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
        encoded = self.enc_proj(x)  # (B, D)

        # decoded part
        device = decoded_ids.device
        decoded = self.token_emb(decoded_ids)  # (B, L, D)

        # adding encoded part
        encoded = encoded.unsqueeze(1)  # (B, 1, D)
        seq = torch.cat([encoded, decoded], dim=1)  # (B, L + 1, D)

        # adding pos embeddings
        positions = torch.arange(0, seq.size(1), device=device)
        pos_embs = self.pos_emb(positions)
        seq += pos_embs

        # generate causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(
            device
        )

        # apply transformer layers
        for attn_block in self.transformer_layers:
            out, _ = attn_block.attn(
                seq, seq, seq, attn_mask=causal_mask, is_causal=True
            )
            out = attn_block.layernorm1(out + seq)
            ffn_out = attn_block.ffn(out)
            seq = attn_block.layernorm2(out + ffn_out)

        # project to logits
        return self.output(seq)  # (B, L + 1, vocab_size)

    def _sample_next_token(
        self,
        logits: Tensor,
        idx: int,
    ):
        """Performs temperature sampling over the next token"""
        device = logits.device
        B = logits.size(0)

        top_k = self.decoder.get('top_k', None)
        top_p = self.decoder.get('top_p', None)
        temperature = self.decoder['temperature']
        assert top_k is None or top_p is None, (
            'Top-K combined with Top-p is not supported'
        )

        # restrict logits by indicies
        mask = torch.from_numpy(self.vocab.logit_mask(idx)).to(device)
        logits[:, ~mask] = NEG_INF  # (B, V)

        # apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))  # prevent index errors
            values, _ = torch.topk(logits, top_k, dim=-1)  # (B, K)
            thresholds = values[:, -1:]  # (B, 1)
            logits = torch.where(
                logits < thresholds,
                NEG_INF,
                logits,
            )

        # apply top-p filtering
        if top_p is not None:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_indicies = torch.argmax(cum_probs > top_p, dim=-1)  # (B, )
            for batch_idx in range(B):
                sorted_indicies = torch.argsort(logits[batch_idx, :])
                top_k_indicies = sorted_indicies[: -cutoff_indicies[batch_idx]]
                logits[batch_idx, top_k_indicies] = NEG_INF

        probs = torch.softmax(logits / temperature, dim=-1)  # (B, V)
        sampled_indicies = torch.multinomial(probs, num_samples=1)  # (B, 1)
        return sampled_indicies.squeeze(1)

    def decode(
        self,
        x_num: None | Tensor,
        x_cat: None | Tensor,
        decoded_ids: None | Tensor = None,  # may not be necessary
    ) -> Tensor:
        """Generate predictions (autoregressively)"""
        assert x_num is not None or x_cat is not None
        device = x_num.device if x_num is not None else x_cat.device
        B = x_num.size(0) if x_num is not None else x_cat.size(0)
        token_ids = -torch.ones(
            (B, self.vocab.token_length), dtype=torch.long, device=device
        )

        # consider already decoded tokens
        if decoded_ids is not None:
            decoded_len = decoded_ids.size(1)
            token_ids[:, :decoded_len] = decoded_ids
        else:
            decoded_len = 0

        was_training = self.training
        self.eval()

        with torch.no_grad():
            for i in range(decoded_len, self.vocab.token_length):
                logits = self(x_num, x_cat, token_ids[:, :i])  # (B, i + 1, V)
                cur_logits = logits[:, -1, :]  # (B, V)

                token_ids[:, i] = self._sample_next_token(cur_logits, idx=i)

        if was_training:
            self.train()

        return self.vocab.batch_from_int(token_ids)


def weighted_crossentropy(
    weights: Tensor | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Weighted sparse categorical cross entropy for sequences.

    Args:
        weights: (L,) - weights for positions in sequence
    """

    def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
        """
        Weighted sparse categorical cross entropy for sequences.

        Args:
            labels: (B, L) - sequences of token indices
            logits: (B, L, V) - logits for each position
        """
        criterion = nn.CrossEntropyLoss(reduction='none')
        per_pos_loss = criterion(logits.transpose(1, 2), labels)  # (B, L)

        if weights is None:
            return per_pos_loss.mean()

        # broadcasting weights to (B, L)
        w = weights.to(logits.device)
        weighted_loss = per_pos_loss * w
        return weighted_loss.mean()

    return loss_fn


# classic forward pass (not generation)
class ApplyModel(Protocol):
    def __call__(
        self,
        model: Model,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        idx: Tensor,
        decoded_ids: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: Model,
    dataset: lib.data.Dataset,
    *,
    part: PartKey,
    idx: Tensor,
    decoded_ids: Tensor,
) -> Tensor:
    """Applying model on dataset (used only in training)"""
    assert model.training
    return (
        model(
            dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
            decoded_ids,
        )
        # .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()  # (B, L, V)
    )


# generation
class GenerateModelPred(Protocol):
    def __call__(
        self,
        model: Model,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        idx: Tensor,
        decoded_ids: Tensor | None = None,
    ) -> Tensor: ...


def generate_model_pred_impl(
    model: Model,
    dataset: lib.data.Dataset,
    *,
    part: PartKey,
    idx: Tensor,
    decoded_ids: Tensor | None = None,
) -> Tensor:
    return (
        model.decode(
            dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
            decoded_ids=decoded_ids,
        )  # (B, )
    )


class EvaluateImplOutput(NamedTuple):
    metrics: dict[PartKey, Any]
    predictions: dict[PartKey, np.ndarray]


@lib.util.adjust_gpu_memory_usage('batch_size')
def evaluate_impl(
    generate_model_pred_impl: GenerateModelPred,
    model: Model,
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
                    generate_model_pred_impl(model, dataset, part=part, idx=idx)
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

    # only regression
    # assert regression_label_stats is not None
    # normalization is not necessary anymore
    if regression_label_stats is not None:
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
    backbone: KWArgs
    decoder: KWArgs  # important!
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
    regression_label_stats = (
        None
        if config['decoder']['vocab']['unnormalized']
        else dataset.try_minmax_scale_labels_()
    )
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )

    # >>> Model
    model: Model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        backbone=config['backbone'],
        decoder=config['decoder'],
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = prediction_type = 'labels'
    model.to(device)

    # >>> Tokenization
    num_train_samples = Y_train.size(0)
    Y_token_ids = []
    for i in range(num_train_samples):
        try:
            Y_token_ids.append(model.vocab.to_int(Y_train[i].item()))
        except:
            print(Y_train[i].item())
            assert False, Y_train[i].item()

    Y_token_ids = torch.tensor(Y_token_ids, device=device, dtype=torch.long)

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
                for x in model.modules()
                if isinstance(
                    x,
                    nn.Linear
                    | tabm.LinearBatchEnsemble
                    | tabm.LinearEnsemble
                    | tabm_reference.LinearEfficientEnsemble,
                )
                and x is not model.output
            ],
            **optimizer_muon_config,
            'use_muon': True,
        }
        _muon_ids = {id(p) for p in muon_parameter_group['params']}
        for _name, _mod in model.named_modules():
            if hasattr(_mod, 'weight') and _mod.weight is not None:
                _tag = 'MUON ' if id(_mod.weight) in _muon_ids else 'adamw'
                logger.info(f'  [{_tag}] {_name} ({type(_mod).__name__})')
        del _muon_ids, _name, _mod, _tag
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
    train_loss_fn = weighted_crossentropy()
    mse_loss_fn = nn.functional.mse_loss  # metric

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
    generate_pred = generate_model_pred_impl
    if autocast is not None:
        apply_model = autocast(apply_model)
        generate_pred = autocast(generate_pred)
    # The following order of `evaluation_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = evaluation_mode()(
        partial(
            evaluate_impl,
            generate_pred,
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
            Y_batch = Y_train[batch_idx]
            Y_batch_tokens = Y_token_ids[batch_idx]
            train_loss = train_loss_fn(
                apply_model(
                    model,
                    dataset,
                    part='train',
                    idx=batch_idx,
                    decoded_ids=Y_batch_tokens[:, :-1],
                ),
                Y_batch_tokens,
            )
            # uncomment for better logging
            # mse_loss = mse_loss_fn(  # MSE
            #     generate_pred(
            #         model,
            #         dataset,
            #         part='train',
            #         idx=batch_idx,
            #         decode_kwargs=decode_kwargs,
            #     ),
            #     Y_batch,
            # )
            if grad_scaler is None:
                train_loss.backward()
            else:
                grad_scaler.scale(train_loss).backward()

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
            batch_losses.append(train_loss.detach())  # now only train losses (CE)
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
            f' [loss] {epoch_loss:.4f}',
            f' [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [it/s] {math.trunc(epoch_size / (epoch_end_time - epoch_start_time)):>3}',
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
