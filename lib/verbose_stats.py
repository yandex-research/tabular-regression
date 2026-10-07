"""Verbose representation-analysis utilities for training scripts.

Collects:
- effective rank of hidden-layer activations (per epoch via TensorBoard, and final)
- linear probing from all hidden layers → NLL on val set
- Jacobian SVD of last hidden layer w.r.t. input (final only)

For synthetic datasets (path starts with 'synth-'), also compares linear probe
predictions to clean_Y (noiseless ground truth).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
from sklearn.linear_model import Ridge
from torch import Tensor

import lib.data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_tabm_model(model: nn.Module, is_clustering: bool) -> nn.Module:
    """Return the tabm_reference.Model (unwrap clustering wrapper if needed)."""
    return model.model if is_clustering else model  # type: ignore[return-value]


def _get_backbone(model: nn.Module, is_clustering: bool) -> nn.Module:
    return _get_tabm_model(model, is_clustering).backbone  # type: ignore[return-value]


@torch.no_grad()
def _preprocess_input(
    tabm_model: nn.Module,
    x_num: Tensor | None,
    x_cat: Tensor | None,
) -> Tensor:
    """Run feature preprocessing (num_module + cat_module) without backbone."""
    x: list[Tensor] = []
    if x_num is not None:
        nm = tabm_model.num_module  # type: ignore[attr-defined]
        x.append(x_num if nm is None else nm(x_num))
    if x_cat is not None:
        x.append(tabm_model.cat_module(x_cat).float())  # type: ignore[attr-defined]
    return torch.column_stack([xi.flatten(1, -1) for xi in x])


@torch.no_grad()
def _collect_layer_acts_gpu(backbone: nn.Module, x_flat: Tensor) -> list[Tensor]:
    """Forward through backbone; collect each block's output as GPU tensors."""
    acts: list[Tensor] = []
    x = x_flat
    for block in backbone.blocks:  # type: ignore[attr-defined]
        x = block(x)
        h = x.detach()
        if h.ndim == 3:
            h = h.reshape(-1, h.shape[-1])
        acts.append(h)
    return acts


@torch.no_grad()
def _collect_layer_acts(backbone: nn.Module, x_flat: Tensor) -> list[np.ndarray]:
    """Forward through backbone blocks; collect output of each block (CPU numpy)."""
    return [h.cpu().numpy() for h in _collect_layer_acts_gpu(backbone, x_flat)]


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def eff_rank(A: np.ndarray) -> float:
    """Effective rank (Roy & Vetterli 2007): exp(H) of normalised singular values."""
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    s = s / (s.sum() + 1e-10)
    return float(np.exp(-np.sum(s * np.log(np.maximum(s, 1e-10)))))


def _eff_rank_gpu(h: Tensor) -> float:
    """GPU-resident eff_rank: avoids CPU transfer and uses cuBLAS SVD."""
    s = torch.linalg.svdvals(h.float())
    s = s / (s.sum() + 1e-10)
    return float(torch.exp(-torch.sum(s * torch.log(s.clamp(min=1e-10)))))


def linear_probe_stats(
    train_acts: np.ndarray,
    train_y: np.ndarray,
    val_acts: np.ndarray,
    val_y: np.ndarray,
    clean_val_y: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit linear probe (mean + variance) and return NLL / RMSE metrics."""
    reg_mean = Ridge(alpha=1.0).fit(train_acts, train_y)
    pred_mean_val = reg_mean.predict(val_acts)

    resid_sq = (reg_mean.predict(train_acts) - train_y) ** 2
    reg_var = Ridge(alpha=1.0).fit(train_acts, resid_sq)
    pred_var_val = np.maximum(reg_var.predict(val_acts), 1e-6)

    nll = float(
        np.mean(
            0.5 * np.log(2 * np.pi * pred_var_val)
            + 0.5 * (val_y - pred_mean_val) ** 2 / pred_var_val
        )
    )
    rmse = float(np.sqrt(np.mean((pred_mean_val - val_y) ** 2)))

    result: dict[str, float] = {'nll': nll, 'rmse': rmse}

    if clean_val_y is not None:
        result['rmse_clean'] = float(
            np.sqrt(np.mean((pred_mean_val - clean_val_y) ** 2))
        )

    return result


def jacobian_sv_spectrum(
    tabm_model: nn.Module,
    backbone: nn.Module,
    x_num: Tensor | None,
    x_cat: Tensor | None,
    *,
    n_samples: int = 64,
) -> list[float] | None:
    """SVD spectrum of ∂h_last/∂x_flat averaged over n_samples val examples.

    Only runs for plain (non-ensemble) models. Returns None for ensemble models.
    """
    if getattr(tabm_model, 'k', None) is not None:
        return None  # skip ensemble

    if x_num is None:
        return None

    idx = torch.randperm(x_num.shape[0])[:n_samples]
    x_sub = x_num[idx]

    tabm_model.eval()

    def last_hidden(x_i: Tensor) -> Tensor:
        x_flat = _preprocess_input_no_grad_ctx(tabm_model, x_i.unsqueeze(0), None)
        for block in backbone.blocks:  # type: ignore[attr-defined]
            x_flat = block(x_flat)
        return x_flat.squeeze(0)

    all_sv: list[np.ndarray] = []
    for i in range(n_samples):
        x_i = x_sub[i]
        try:
            J = torch.autograd.functional.jacobian(
                last_hidden, x_i, vectorize=True, create_graph=False
            )
            sv = np.linalg.svd(
                J.detach().cpu().numpy(), compute_uv=False  # type: ignore[arg-type]
            )
            all_sv.append(sv)
        except Exception:
            continue

    if not all_sv:
        return None
    return np.mean(all_sv, axis=0).tolist()


def _preprocess_input_no_grad_ctx(
    tabm_model: nn.Module,
    x_num: Tensor | None,
    x_cat: Tensor | None,
) -> Tensor:
    """Like _preprocess_input but without @no_grad (needed inside jacobian call)."""
    x: list[Tensor] = []
    if x_num is not None:
        nm = tabm_model.num_module  # type: ignore[attr-defined]
        x.append(x_num if nm is None else nm(x_num))
    if x_cat is not None:
        x.append(tabm_model.cat_module(x_cat).float())  # type: ignore[attr-defined]
    return torch.column_stack([xi.flatten(1, -1) for xi in x])


# ---------------------------------------------------------------------------
# Per-epoch hook (called inside training loop)
# ---------------------------------------------------------------------------


def collect_epoch_eff_rank(
    model: nn.Module,
    dataset: lib.data.Dataset,
    writer: torch.utils.tensorboard.SummaryWriter,
    step: int,
    *,
    is_clustering: bool = False,
    n_subset: int = 512,
) -> None:
    """Compute eff_rank on val subset and log to TensorBoard. No-op on error.

    Stays entirely on GPU: no CPU transfer, uses torch.linalg.svdvals.
    n_subset=512 is sufficient for stable eff_rank estimates.
    """
    try:
        tabm_model = _get_tabm_model(model, is_clustering)
        backbone = _get_backbone(model, is_clustering)

        x_num = (
            dataset.data['x_num']['val'][:n_subset] if 'x_num' in dataset.data else None
        )
        x_cat = (
            dataset.data['x_cat']['val'][:n_subset] if 'x_cat' in dataset.data else None
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            x_flat = _preprocess_input(tabm_model, x_num, x_cat)
            acts = _collect_layer_acts_gpu(backbone, x_flat)
        if was_training:
            model.train()

        for i, h in enumerate(acts):
            er = _eff_rank_gpu(h)
            writer.add_scalar(f'verbose/eff_rank_layer_{i}', er, step)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# End-of-training full verbose stats
# ---------------------------------------------------------------------------


def run_verbose_stats(
    model: nn.Module,
    dataset: lib.data.Dataset,
    exp_dir: Path,
    writer: torch.utils.tensorboard.SummaryWriter,
    *,
    is_clustering: bool = False,
    data_path: Path | None = None,
) -> dict:
    """Run all verbose stats on val set; save to verbose_stats.json."""
    tabm_model = _get_tabm_model(model, is_clustering)
    backbone = _get_backbone(model, is_clustering)

    x_num_val = dataset.data['x_num']['val'] if 'x_num' in dataset.data else None
    x_cat_val = dataset.data['x_cat']['val'] if 'x_cat' in dataset.data else None
    x_num_train = dataset.data['x_num']['train'] if 'x_num' in dataset.data else None
    x_cat_train = dataset.data['x_cat']['train'] if 'x_cat' in dataset.data else None

    y_val: np.ndarray = dataset.data['y']['val'].cpu().numpy()
    y_train: np.ndarray = dataset.data['y']['train'].cpu().numpy()

    # Synthetic dataset: load clean_Y
    clean_val_y: np.ndarray | None = None
    if data_path is not None and data_path.name.startswith('synth-'):
        clean_y_path = data_path / 'clean_Y.npy'
        if clean_y_path.exists():
            clean_y_full = np.load(clean_y_path)
            # val split starts at train_size=12000
            info_path = data_path / 'info.json'
            train_size = 12000
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                train_size = info.get('train_size', 12000)
            val_size = len(y_val)
            clean_val_y = clean_y_full[train_size : train_size + val_size]

    model.eval()
    with torch.no_grad():
        x_flat_val = _preprocess_input(tabm_model, x_num_val, x_cat_val)
        x_flat_train = _preprocess_input(tabm_model, x_num_train, x_cat_train)
        acts_val = _collect_layer_acts(backbone, x_flat_val)
        acts_train = _collect_layer_acts(backbone, x_flat_train)

    # Effective rank
    eff_rank_dict: dict[str, float] = {}
    for i, A in enumerate(acts_val):
        er = eff_rank(A)
        eff_rank_dict[f'layer_{i}'] = er
        writer.add_scalar(f'verbose/eff_rank_layer_{i}_final', er, 0)

    # Linear probing
    probe_dict: dict[str, dict[str, float]] = {}
    synth_probe_dict: dict[str, dict[str, float]] = {}
    for i, (tr_a, va_a) in enumerate(zip(acts_train, acts_val)):
        stats = linear_probe_stats(tr_a, y_train, va_a, y_val, clean_val_y)
        probe_dict[f'layer_{i}'] = {k: v for k, v in stats.items() if k != 'rmse_clean'}
        if 'rmse_clean' in stats:
            synth_probe_dict[f'layer_{i}'] = {'rmse_clean': stats['rmse_clean']}

    # Jacobian SVD
    jac_sv = jacobian_sv_spectrum(
        tabm_model, backbone, x_num_val, x_cat_val, n_samples=64
    )

    result: dict = {
        'eff_rank': eff_rank_dict,
        'linear_probe': probe_dict,
        'jacobian_sv': jac_sv,
    }
    if synth_probe_dict:
        result['synth_linear_probe'] = synth_probe_dict

    out_path = exp_dir / 'verbose_stats.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def collect_batch_grad_norm(model: nn.Module) -> float:
    total_sq = sum(
        p.grad.detach().float().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return float(total_sq**0.5)


def log_grad_norm_stats(
    norms: list[float],
    writer: torch.utils.tensorboard.SummaryWriter,
    step: int,
    *,
    clip_threshold: float | None = None,
) -> None:
    if not norms:
        return
    t = torch.tensor(norms, dtype=torch.float32)
    writer.add_histogram('verbose/grad_norm', t, step)
    writer.add_scalar('verbose/grad_norm_mean', t.mean().item(), step)
    writer.add_scalar('verbose/grad_norm_max', t.max().item(), step)
    writer.add_scalar('verbose/grad_norm_p95', t.quantile(0.95).item(), step)
    if clip_threshold is not None:
        writer.add_scalar(
            'verbose/grad_clip_fraction',
            (t > clip_threshold).float().mean().item(),
            step,
        )
