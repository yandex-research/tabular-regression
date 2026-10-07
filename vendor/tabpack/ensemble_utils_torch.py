# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import abc
import collections
from collections.abc import Callable
from copy import deepcopy
from typing import Any, NamedTuple

import torch
from torch import Tensor
from tqdm import tqdm

from lib.data import Task
from lib.types import PartKey, PredictionType
from vendor.tabpack.utils import numpy_concatenate, numpy_index

from .metrics_torch import calculate_metrics_pack

# >>> Common


type EnsembleScoreFn = Callable[[Tensor], Tensor]


def make_emsemble_score_fn(
    task: Task,
    prediction_type: str | PredictionType,
    *,
    part: PartKey,
    device: torch.device,
) -> EnsembleScoreFn:
    y_true = torch.as_tensor(task.labels[part], device=device)

    def score_fn(y_pred: torch.Tensor) -> torch.Tensor:
        return calculate_metrics_pack(
            y_true=y_true,
            y_pred=y_pred,
            task_type=task.type_,
            prediction_type=prediction_type,
            score=task.score,
        )['score']

    return score_fn


def _validate_predictions(predictions: Tensor) -> None:
    assert predictions.numel() > 0
    assert torch.isfinite(predictions).all()


# >>> Algorithms


type EnsembleFn = (
    # -> ensemble_idx
    Callable[..., Tensor]
    # -> (ensemble_idx, ensemble_weights)
    | Callable[..., tuple[Tensor, Tensor]]
    # -> (ensemble_idx, None | ensemble_weights)
    | Callable[..., tuple[Tensor, None | Tensor]]
)


class EnsembleFnWrapper:
    """A wrapper for converting ensemble function outputs to a single format.

    The wrapper turns an ensemble function output into
    `(ensemble_idx, None | ensemble_weights)`
    """

    def __init__(self, fn: EnsembleFn) -> None:
        self._fn = fn

    def __call__(self, *args, **kwargs) -> tuple[Tensor, None | Tensor]:
        ensemble = self._fn(*args, **kwargs)
        return ensemble if isinstance(ensemble, tuple) else (ensemble, None)  # ty:ignore[invalid-return-type]


def compute_ensemble_prediction(
    predictions: Tensor,
    weights: None | Tensor,
    prediction_type: None | PredictionType = None,
) -> Tensor:
    if weights is None:
        return predictions.mean(0)
    else:
        assert weights.ndim == 1
        prediction = (
            predictions
            * (weights / weights.sum())[
                :, *((None,) * (predictions.ndim - weights.ndim))
            ]
        ).sum(0)
        if prediction_type == PredictionType.PROBS:
            prediction = prediction.clamp_max(1.0)
        return prediction


def topk_ensemble(
    predictions: Tensor,
    *,
    ensemble_size: int,
    score_fn: EnsembleScoreFn,
) -> Tensor:
    _validate_predictions(predictions)
    n_predictions = len(predictions)
    assert 0 < ensemble_size <= n_predictions
    prediction_scores = score_fn(predictions)
    return torch.argsort(prediction_scores, stable=True)[-ensemble_size:]


def autotopk_ensemble(
    predictions: Tensor,
    *,
    score_fn: EnsembleScoreFn,
    prediction_type: PredictionType,
) -> Tensor:
    _validate_predictions(predictions)
    n_predictions = len(predictions)

    prediction_scores = score_fn(predictions)
    sorted_idx = torch.argsort(prediction_scores, stable=True).flip(0)
    sorted_predictions = predictions[sorted_idx]

    topk_ensemble_sizes = torch.arange(1, n_predictions + 1, device=predictions.device)
    topk_ensemble_predictions = (
        sorted_predictions.cumsum(dim=0)
        / topk_ensemble_sizes[:, *((None,) * (predictions.ndim - 1))]
    )
    if prediction_type == PredictionType.PROBS:
        topk_ensemble_predictions = topk_ensemble_predictions.clamp_max(1.0)

    topk_ensemble_prediction_scores = score_fn(topk_ensemble_predictions)
    improved = (
        topk_ensemble_prediction_scores[:-1] < topk_ensemble_prediction_scores[1:]
    )
    # Find the first non-improvement.
    first_nonimprovement_index = improved.float().argmin()
    ensemble_size = first_nonimprovement_index + 1
    return sorted_idx[:ensemble_size]


def bruteforce_ensemble(
    predictions: Tensor, *, ensemble_size: None | int = None, score_fn: EnsembleScoreFn
) -> Tensor:
    _validate_predictions(predictions)
    device = predictions.device

    n_predictions = len(predictions)
    if ensemble_size is None:
        ensemble_size = n_predictions

    # Build all possible index combinations of size `ensemble_size`.
    ensemble_idx = torch.combinations(
        torch.arange(n_predictions, device=device), ensemble_size
    )
    ensemble_predictions = predictions[ensemble_idx].mean(1)
    ensemble_scores = score_fn(ensemble_predictions)
    return ensemble_idx[torch.argmax(ensemble_scores)]


def greedy_ensemble(
    predictions: Tensor,
    *,
    score_fn: EnsembleScoreFn,
    selection_score_fn: None | EnsembleScoreFn = None,
    prediction_scores: None | Tensor = None,
    init_top_k: None | int = None,
    init_autotopk: bool = False,
    init_bruteforce_k: None | int = None,
    init_use_selection_score: bool = False,
    max_ensemble_size: None | int = None,
    with_replacement: bool = False,
    prediction_type: None | PredictionType = None,
    verbose: bool = False,
) -> tuple[Tensor, None | Tensor]:
    if init_use_selection_score:
        assert selection_score_fn is not None
    _validate_predictions(predictions)
    device = predictions.device

    if prediction_scores is None:
        prediction_scores = score_fn(predictions)
    prediction_selection_scores = (
        None if selection_score_fn is None else selection_score_fn(predictions)
    )

    if init_top_k is None and not init_autotopk and init_bruteforce_k is None:
        init_top_k = 1
    if init_top_k is not None:
        init_prediction_scores = (
            prediction_scores
            if prediction_selection_scores is None or not init_use_selection_score
            else prediction_selection_scores
        )
        assert init_bruteforce_k is None
        if init_top_k == 1:
            top_index = torch.argmax(init_prediction_scores)
            init_idx = torch.tensor([top_index], device=device)
        else:
            assert init_top_k > 1
            # The negation and `stable=True` are used
            # to take the top k "earliest" predictions.
            init_idx = torch.argsort(-init_prediction_scores, stable=True)[:init_top_k]

    elif init_autotopk:
        assert prediction_type is not None
        init_idx = autotopk_ensemble(
            predictions, score_fn=score_fn, prediction_type=prediction_type
        )

    else:
        assert init_bruteforce_k is not None
        init_idx = bruteforce_ensemble(
            predictions,
            ensemble_size=init_bruteforce_k,
            score_fn=(
                score_fn
                if selection_score_fn is None or not init_use_selection_score
                else selection_score_fn
            ),
        )

    n_predictions = len(predictions)
    prediction_range_idx = torch.arange(n_predictions, device=device)
    ensemble_mask = torch.zeros(n_predictions, dtype=torch.bool, device=device)
    ensemble_mask[init_idx] = True
    if with_replacement:
        ensemble_weights = torch.zeros(
            n_predictions, dtype=predictions.dtype, device=device
        )
        ensemble_weights[init_idx] = 1.0
    else:
        ensemble_weights = None
    ensemble_prediction = predictions[init_idx].mean(0)
    ensemble_score = score_fn(ensemble_prediction[None])[0]
    ensemble_selection_score = (
        ensemble_score
        if selection_score_fn is None
        else selection_score_fn(ensemble_prediction[None])[0]
    )
    ensemble_size = len(init_idx)

    while ensemble_size < (
        n_predictions if max_ensemble_size is None else max_ensemble_size
    ):
        candidate_idx = (
            prediction_range_idx
            if with_replacement
            else torch.nonzero_static(
                ~ensemble_mask, size=n_predictions - ensemble_size
            )[:, 0]
        )
        candidate_predictions = ensemble_prediction[None] * (
            ensemble_size / (ensemble_size + 1)
        ) + predictions[candidate_idx] * (1 / (ensemble_size + 1))
        candidates_scores = score_fn(candidate_predictions)
        candidate_selection_scores = (
            None
            if selection_score_fn is None
            else selection_score_fn(candidate_predictions)
        )

        if selection_score_fn is not None:
            # In the double scoring setup, the first step is to remove all candidates
            # that make the main score worse.
            candidate_local_idx = torch.nonzero(
                candidates_scores[candidates_scores >= ensemble_score]
            )[:, 0]
            if candidate_local_idx.numel() == 0:
                # If all the candidates make the main score worse, stop the algorithm.
                break

            # Now, filter the candidate data.
            candidate_idx = candidate_idx[candidate_local_idx]
            candidate_predictions = candidate_predictions[candidate_local_idx]
            candidates_scores = candidates_scores[candidate_local_idx]
            if candidate_selection_scores is not None:
                candidate_selection_scores = candidate_selection_scores[
                    candidate_local_idx
                ]
            del candidate_local_idx

        # Set the scores for selecting the best candidate.
        if candidate_selection_scores is None:
            candidate_selection_scores = candidates_scores

        if candidate_selection_scores.numel() == 0:
            break
        best_candidate_selection_score = candidate_selection_scores.max()
        if best_candidate_selection_score <= ensemble_selection_score:
            # If none of the candidates improve the selection score, stop the algorithm.
            break

        # Find all candidates with the best selection score.
        candidate_local_mask = (
            candidate_selection_scores == best_candidate_selection_score
        )
        best_candidate_local_idx = torch.nonzero(candidate_local_mask)[:, 0]
        best_candidate_idx = candidate_idx[candidate_local_mask]
        assert len(best_candidate_idx) > 0

        if len(best_candidate_idx) == 1:
            # If there is only one such candidate, select it.
            best_candidate_local_index = best_candidate_local_idx[0]
            best_candidate_index = best_candidate_idx[0]
        else:
            # Select the candidate with the best individual score.
            best_candidate_local_index = best_candidate_local_idx[
                torch.argmax(
                    (
                        prediction_scores
                        if prediction_selection_scores is None
                        else prediction_selection_scores
                    )[best_candidate_idx]
                )
            ]
            best_candidate_index = candidate_idx[best_candidate_local_index]

        ensemble_mask[best_candidate_index] = True
        if ensemble_weights is not None:
            ensemble_weights[best_candidate_index] += 1.0
        ensemble_score = candidates_scores[best_candidate_local_index]
        ensemble_selection_score = best_candidate_selection_score
        ensemble_prediction = candidate_predictions[best_candidate_local_index]
        if (
            ensemble_weights is None
            or ensemble_weights[best_candidate_index].item() == 1.0
        ):
            ensemble_size += 1

        if verbose:
            print(f'{ensemble_size:>2}: {ensemble_score:.4f}')

    ensemble_idx = torch.nonzero_static(ensemble_mask, size=ensemble_size)[:, 0]
    return ensemble_idx, (
        None if ensemble_weights is None else ensemble_weights[ensemble_idx]
    )


def beam_ensemble(
    predictions: Tensor,
    *,
    beam_size: int,
    max_ensemble_size: None | int = None,
    max_n_iterations: None | int = None,
    prefer_minimal_updates: bool = False,
    score_fn: Callable[[Tensor], Tensor],
    batch_size: None | int = None,
    verbose: bool = False,
) -> Tensor:
    assert predictions.numel() > 0
    assert torch.isfinite(predictions).all()

    n_predictions = len(predictions)
    if max_ensemble_size is None:
        max_ensemble_size = n_predictions
    device = predictions.device

    ensemble_size = 0
    ensemble_mask = torch.zeros(n_predictions, dtype=torch.bool, device=device)
    ensemble_prediction = torch.zeros_like(predictions[0])
    ensemble_score = None

    n_iterations = 0
    while ensemble_size < max_ensemble_size and (
        max_n_iterations is None or n_iterations < max_n_iterations
    ):
        candidate_idx = torch.nonzero_static(
            ~ensemble_mask, size=n_predictions - ensemble_size
        )[:, 0]
        best_candidate_ensemble_score = None
        best_candidate_ensemble_idx = None

        for candidate_ensemble_size in range(
            1, 1 + min(beam_size, max_ensemble_size - ensemble_size)
        ):
            new_ensemble_size = ensemble_size + candidate_ensemble_size
            adjusted_ensemble_prediction = ensemble_prediction * (
                ensemble_size / new_ensemble_size
            )
            candidate_ensemble_idx = torch.combinations(
                candidate_idx, candidate_ensemble_size
            )
            if batch_size is None:
                candidate_ensemble_scores = score_fn(
                    adjusted_ensemble_prediction
                    + predictions[candidate_ensemble_idx].sum(1) / new_ensemble_size
                )
            else:
                candidate_ensemble_scores = torch.cat(
                    [
                        score_fn(
                            adjusted_ensemble_prediction
                            + predictions[batch_candidate_ensemble_idx].sum(1)
                            / new_ensemble_size
                        )
                        for batch_candidate_ensemble_idx in tqdm(
                            candidate_ensemble_idx.split(batch_size),
                            leave=False,
                            disable=not verbose,
                        )
                    ]
                )
            promising_candidate_local_index = candidate_ensemble_scores.argmax()
            promising_candidate_score = candidate_ensemble_scores[
                promising_candidate_local_index
            ]

            if (
                best_candidate_ensemble_score is None
                or promising_candidate_score > best_candidate_ensemble_score
            ):
                best_candidate_ensemble_score = promising_candidate_score
                best_candidate_ensemble_idx = candidate_ensemble_idx[
                    promising_candidate_local_index
                ]
                if prefer_minimal_updates:
                    break

        assert best_candidate_ensemble_score is not None
        assert best_candidate_ensemble_idx is not None
        if ensemble_score is None or best_candidate_ensemble_score > ensemble_score:
            ensemble_size += len(best_candidate_ensemble_idx)
            ensemble_mask[best_candidate_ensemble_idx] = True
            ensemble_prediction = predictions[ensemble_mask].mean(0)
            ensemble_score = best_candidate_ensemble_score
        else:
            break

        n_iterations += 1

    return torch.nonzero_static(ensemble_mask, size=ensemble_size)[:, 0]


def get_ensemble_fn(type: str) -> EnsembleFn:
    return {
        'topk': topk_ensemble,
        'autotopk': autotopk_ensemble,
        'bruteforce': bruteforce_ensemble,
        'greedy': greedy_ensemble,
        'beam': beam_ensemble,
    }[type]


# >>> Online ensembles


class _OnlineEnsembleWindowItem(NamedTuple):
    predictions: Tensor
    data: dict[str, Any]


class _OnlineEnsembleState(NamedTuple):
    predictions: Tensor
    weights: None | Tensor
    score: float
    data: dict[str, Any]


class _OnlineEnsemble:
    def __init__(self, *, score_fn: EnsembleScoreFn, window_size: int = 1) -> None:
        self._score_fn = score_fn
        self._window: collections.deque[_OnlineEnsembleWindowItem] = collections.deque(
            maxlen=window_size
        )
        self._state: None | _OnlineEnsembleState = None

    def _evaluate_one_prediction(self, prediction: Tensor) -> float:
        return self._score_fn(prediction[None]).item()

    @property
    def weights(self) -> None | Tensor:
        assert self._state is not None
        return self._state.weights

    @property
    def data(self) -> dict[str, Any]:
        assert self._state is not None
        return self._state.data

    @abc.abstractmethod
    def _update_state(self) -> bool: ...

    def update(self, predictions: Tensor, data: dict[str, Any]) -> bool:
        self._window.append(deepcopy(_OnlineEnsembleWindowItem(predictions, data)))
        return self._update_state()


class SimpleOnlineEnsemble(_OnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        assert kwargs.get('window_size', 1) == 1
        super().__init__(**kwargs)

    def _update_state(self) -> bool:
        predictions, data = self._window[0]
        ensemble_score = self._evaluate_one_prediction(predictions.mean(0))
        improved = self._state is None or ensemble_score > self._state.score
        if improved:
            self._state = _OnlineEnsembleState(predictions, None, ensemble_score, data)
        return improved


class _SelectiveOnlineEnsemble(_OnlineEnsemble):
    def __init__(
        self,
        ensemble_fn: EnsembleFn,
        *,
        score_fn: EnsembleScoreFn,
        ensemble_score_fn: None | EnsembleScoreFn = None,
        window_size: int = 1,
        reuse_previous: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(score_fn=score_fn, window_size=window_size)
        self._ensemble_fn = EnsembleFnWrapper(ensemble_fn)
        self._ensemble_score_fn = ensemble_score_fn
        self._reuse_previous = reuse_previous
        self._kwargs = kwargs

    @property
    def reuse_previous(self) -> bool:
        return self._reuse_previous

    def _update_state(self) -> bool:
        window_predictions = [x.predictions for x in self._window]
        prediction_pool = torch.cat(
            [self._state.predictions, *window_predictions]
            if self._reuse_previous and self._state is not None
            else window_predictions
        )

        ensemble_idx, ensemble_weights = self._ensemble_fn(
            prediction_pool,
            score_fn=(
                self._score_fn
                if self._ensemble_score_fn is None
                else self._ensemble_score_fn
            ),
            **self._kwargs,
        )
        ensemble_predictions = prediction_pool[ensemble_idx]
        ensemble_score = self._evaluate_one_prediction(
            compute_ensemble_prediction(ensemble_predictions, ensemble_weights)
        )

        improved = self._state is None or ensemble_score > self._state.score
        if improved:
            window_data = [x.data for x in self._window]
            all_data = numpy_concatenate(
                [self._state.data, *window_data]
                if self._reuse_previous and self._state is not None
                else window_data
            )
            ensemble_data = numpy_index(all_data, ensemble_idx.cpu().numpy())
            self._state = _OnlineEnsembleState(
                ensemble_predictions, ensemble_weights, ensemble_score, ensemble_data
            )

        return improved


class TopKOnlineEnsemble(_SelectiveOnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        return super().__init__(topk_ensemble, **kwargs)


class BruteForceOnlineEnsemble(_SelectiveOnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        return super().__init__(bruteforce_ensemble, **kwargs)


class GreedyOnlineEnsemble(_SelectiveOnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        return super().__init__(greedy_ensemble, **kwargs)


class BeamOnlineEnsemble(_SelectiveOnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        return super().__init__(beam_ensemble, **kwargs)


def make_online_ensemble(type: str, **kwargs) -> _OnlineEnsemble:
    ensemble_classes = {
        cls.__name__: cls
        for cls in [
            SimpleOnlineEnsemble,
            TopKOnlineEnsemble,
            BruteForceOnlineEnsemble,
            GreedyOnlineEnsemble,
            BeamOnlineEnsemble,
        ]
    }
    return ensemble_classes[type](**kwargs)
