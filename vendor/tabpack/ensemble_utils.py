# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import abc
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from lib.data import Task
from lib.types import PartKey, PredictionType
from vendor.tabpack.metrics import calculate_metrics_pack
from vendor.tabpack.utils import numpy_concatenate, numpy_index

# >>> Common


type EnsembleScoreFn = Callable[[np.ndarray], np.ndarray]


def make_ensemble_score_fn(
    task: Task, prediction_type: str | PredictionType, *, part: PartKey
) -> EnsembleScoreFn:
    def score_fn(y_pred: np.ndarray) -> np.ndarray:
        return calculate_metrics_pack(
            y_true=task.labels[part],
            y_pred=y_pred,
            task_type=task.type_,
            prediction_type=prediction_type,
            score=task.score,
        )['score']

    return score_fn


def _validate_predictions(predictions: np.ndarray) -> None:
    assert predictions.size > 0
    assert np.isfinite(predictions).all()


def compute_ensemble_prediction(
    predictions: np.ndarray,
    weights: None | np.ndarray,
    prediction_type: None | PredictionType = None,
) -> np.ndarray:
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
            prediction = np.clip(prediction, max=1.0)  # type: ignore
        return prediction


# >>> Algorithms


type EnsembleFn = (
    Callable[..., np.ndarray]
    | Callable[..., tuple[np.ndarray, None | np.ndarray, float]]
)


def topk_ensemble(
    predictions: np.ndarray, *, ensemble_size: int, score_fn: EnsembleScoreFn
) -> np.ndarray:
    _validate_predictions(predictions)
    n_predictions = len(predictions)
    assert 0 < ensemble_size <= n_predictions
    return np.argsort(score_fn(predictions), stable=True)[-ensemble_size:]


def bruteforce_ensemble(
    predictions: np.ndarray,
    *,
    ensemble_size: None | int = None,
    score_fn: EnsembleScoreFn,
) -> np.ndarray:
    _validate_predictions(predictions)

    n_predictions = len(predictions)
    if ensemble_size is None:
        ensemble_size = n_predictions

    # Build all possible index combinations of size `ensemble_size`.
    index_combinations = np.stack(
        np.meshgrid(*((np.arange(n_predictions),) * ensemble_size), indexing='ij'),
        axis=-1,
    ).reshape(-1, ensemble_size)
    # Keep only those index combinations where all indices are different and sorted.
    ensemble_idx = index_combinations[
        (index_combinations[:, :-1] < index_combinations[:, 1:]).all(1)
    ]
    ensemble_predictions = predictions[ensemble_idx].mean(1)
    ensemble_scores = score_fn(ensemble_predictions)
    return ensemble_idx[np.argmax(ensemble_scores)]


def greedy_ensemble(
    predictions: np.ndarray,
    *,
    score_fn: EnsembleScoreFn,
    prediction_scores: None | np.ndarray = None,
    init_top_k: None | int = None,
    init_bruteforce_k: None | int = None,
    init_idx: None | np.ndarray = None,
    max_ensemble_size: None | int = None,
    with_replacement: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, None | np.ndarray, float]:
    _validate_predictions(predictions)

    if prediction_scores is None:
        prediction_scores = score_fn(predictions)

    if init_idx is None:
        if init_top_k is None and init_bruteforce_k is None:
            init_top_k = 1
        if init_top_k is not None:
            assert init_bruteforce_k is None
            if init_top_k == 1:
                top_index = np.argmax(prediction_scores)
                init_idx = np.array([top_index])
            else:
                assert init_top_k > 1
                # The negation and `stable=True` are used
                # to take the top k "earliest" predictions.
                init_idx = np.argsort(-prediction_scores, stable=True)[:init_top_k]
        else:
            assert init_bruteforce_k is not None
            init_idx = bruteforce_ensemble(
                predictions, ensemble_size=init_bruteforce_k, score_fn=score_fn
            )
    else:
        assert init_top_k is None
        assert init_bruteforce_k is None

    n_predictions = len(predictions)
    ensemble_weights = np.zeros(n_predictions, dtype=predictions.dtype)
    ensemble_weights[init_idx] = 1.0
    ensemble_prediction = predictions[init_idx].mean(0)
    ensemble_score = float(score_fn(ensemble_prediction[None])[0])
    ensemble_size = len(init_idx)

    while ensemble_size < (
        n_predictions if max_ensemble_size is None else max_ensemble_size
    ):
        candidate_idx = (
            np.arange(n_predictions)
            if with_replacement
            else np.nonzero(ensemble_weights == 0.0)[0]
        )
        candidate_predictions = ensemble_prediction[None] * (
            ensemble_size / (ensemble_size + 1)
        ) + predictions[candidate_idx] * (1 / (ensemble_size + 1))
        candidates_scores = score_fn(candidate_predictions)

        best_candidate_score = float(np.max(candidates_scores))
        if best_candidate_score <= ensemble_score:
            break

        # Find all candidates with the best score.
        candidate_local_mask = candidates_scores == best_candidate_score
        best_candidate_local_idx = np.nonzero(candidate_local_mask)[0]
        best_candidate_idx = candidate_idx[candidate_local_mask]
        assert len(best_candidate_idx) > 0

        if len(best_candidate_idx) == 1:
            # If there is only one such candidate, select it.
            best_candidate_local_index = best_candidate_local_idx[0]
            best_candidate_index = best_candidate_idx[0]
        else:
            # Select the candidate with the best individual score.
            best_candidate_local_index = best_candidate_local_idx[
                np.argmax(prediction_scores[best_candidate_idx])
            ]
            best_candidate_index = candidate_idx[best_candidate_local_index]

        ensemble_weights[best_candidate_index] += 1.0
        ensemble_score = best_candidate_score
        ensemble_prediction = candidate_predictions[best_candidate_local_index]
        ensemble_size += 1

        if verbose:
            print(f'{ensemble_size:>2}: {ensemble_score:.4f}')

    ensemble_idx = np.nonzero(ensemble_weights > 0)[0]
    return (
        ensemble_idx,
        (ensemble_weights[ensemble_idx] if with_replacement else None),
        ensemble_score,
    )


def greedy_remove_ensemble(
    predictions: np.ndarray,
    *,
    score_fn: EnsembleScoreFn,
    prediction_scores: None | np.ndarray = None,
    max_ensemble_size: None | int = None,
    verbose: bool = False,
) -> np.ndarray:
    _validate_predictions(predictions)

    if prediction_scores is None:
        prediction_scores = score_fn(predictions)

    n_predictions = len(predictions)
    ensemble_mask = np.ones(n_predictions, dtype=np.bool)
    ensemble_prediction = predictions.mean(0)
    ensemble_score = float(score_fn(ensemble_prediction[None])[0])
    ensemble_size = n_predictions

    while ensemble_size > 1:
        # NOTE
        # Recall that here, "candidate" means a candidate for _removal_.
        candidate_idx = np.nonzero(ensemble_mask)[0]
        candidate_predictions = (
            ensemble_prediction[None] - predictions[candidate_idx] / ensemble_size
        ) * (ensemble_size / (ensemble_size - 1))
        candidates_scores = score_fn(candidate_predictions)

        best_candidate_score = float(np.max(candidates_scores))
        if best_candidate_score <= ensemble_score and (
            max_ensemble_size is None or ensemble_size <= max_ensemble_size
        ):
            break

        # Find all candidates with the best score.
        candidate_local_mask = candidates_scores == best_candidate_score
        best_candidate_local_idx = np.nonzero(candidate_local_mask)[0]
        best_candidate_idx = candidate_idx[candidate_local_mask]
        assert len(best_candidate_idx) > 0

        if len(best_candidate_idx) == 1:
            # If there is only one such candidate, select it.
            best_candidate_local_index = best_candidate_local_idx[0]
            best_candidate_index = best_candidate_idx[0]
        else:
            # Select the candidate with the _worst_ individual score.
            best_candidate_local_index = best_candidate_local_idx[
                np.argmin(prediction_scores[best_candidate_idx])
            ]
            best_candidate_index = candidate_idx[best_candidate_local_index]

        ensemble_mask[best_candidate_index] = False
        ensemble_score = best_candidate_score
        ensemble_prediction = candidate_predictions[best_candidate_local_index]
        ensemble_size -= 1

        if verbose:
            print(f'{ensemble_size:>2}: {ensemble_score:.4f}')

    ensemble_idx = np.nonzero(ensemble_mask > 0)[0]
    return ensemble_idx


# >>> Online ensembles


class _OnlineEnsembleState(NamedTuple):
    predictions: np.ndarray
    score: float
    data: dict[str, Any]


class _OnlineEnsemble:
    def __init__(self, *, score_fn: EnsembleScoreFn) -> None:
        self._score_fn = score_fn
        self._state: None | _OnlineEnsembleState = None

    def _evaluate_one_prediction(self, prediction: np.ndarray) -> float:
        return float(self._score_fn(prediction[None])[0])

    @property
    def data(self) -> dict[str, Any]:
        assert self._state is not None
        return self._state.data

    @abc.abstractmethod
    def update(self, predictions: np.ndarray, data: dict[str, Any]) -> bool:
        raise NotImplementedError()


class SimpleOnlineEnsemble(_OnlineEnsemble):
    def update(self, predictions: np.ndarray, data: dict[str, Any]) -> bool:
        ensemble_score = self._evaluate_one_prediction(predictions.mean(0))
        improved = self._state is None or ensemble_score > self._state.score
        if improved:
            self._state = _OnlineEnsembleState(predictions, ensemble_score, data)
        return improved


class _SelectiveOnlineEnsemble(_OnlineEnsemble):
    def __init__(self, ensemble_fn, *, score_fn: EnsembleScoreFn, **kwargs) -> None:
        super().__init__(score_fn=score_fn)
        self._ensemble_fn = ensemble_fn
        self._kwargs = kwargs

    def update(self, predictions: np.ndarray, data: dict[str, Any]) -> bool:
        if self._state is not None:
            predictions = np.concatenate([self._state.predictions, predictions])

        ensemble_idx = self._ensemble_fn(
            predictions, score_fn=self._score_fn, **self._kwargs
        )
        ensemble_predictions = predictions[ensemble_idx]
        ensemble_score = self._evaluate_one_prediction(ensemble_predictions.mean(0))

        improved = self._state is None or ensemble_score > self._state.score
        if improved:
            all_data = (
                data
                if self._state is None
                else numpy_concatenate([self._state.data, data])
            )
            ensemble_data = numpy_index(all_data, ensemble_idx)
            self._state = _OnlineEnsembleState(
                ensemble_predictions, ensemble_score, ensemble_data
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
        assert not kwargs.get('with_replacement', False)

        def greedy_ensemble_without_replacement(*args, **kwargs_) -> np.ndarray:
            idx, *_ = greedy_ensemble(*args, **kwargs_)
            return idx

        return super().__init__(greedy_ensemble_without_replacement, **kwargs)


class GreedyRemoveOnlineEnsemble(_SelectiveOnlineEnsemble):
    def __init__(self, **kwargs) -> None:
        return super().__init__(greedy_remove_ensemble, **kwargs)


def make_online_ensemble(type: str, **kwargs) -> _OnlineEnsemble:
    ensemble_classes = {
        cls.__name__: cls
        for cls in [
            SimpleOnlineEnsemble,
            TopKOnlineEnsemble,
            BruteForceOnlineEnsemble,
            GreedyOnlineEnsemble,
            GreedyRemoveOnlineEnsemble,
        ]
    }
    return ensemble_classes[type](**kwargs)
