# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import numpy as np
import sklearn.metrics

import lib.data
from lib.types import PredictionType, TaskType

from .nn import BATCH_DIM, PACK_DIM

type Metrics = dict[str, np.ndarray]


def calculate_metrics_pack(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
    score: lib.data.Score,
) -> Metrics:
    assert y_true.ndim == 1
    task_type = TaskType(task_type)
    prediction_type = PredictionType(prediction_type)

    pack_size = y_pred.shape[PACK_DIM]
    y_true = np.repeat(np.expand_dims(y_true, PACK_DIM), pack_size, axis=PACK_DIM)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        assert y_true.shape == y_pred.shape
        y_true_T = y_true.T
        y_pred_T = y_pred.T
        result = {
            'rmse': (
                sklearn.metrics.mean_squared_error(
                    y_true_T, y_pred_T, multioutput='raw_values'
                )
                ** 0.5
            ),
            'r2': sklearn.metrics.r2_score(
                y_true_T, y_pred_T, multioutput='raw_values'
            ),
        }

    elif task_type == TaskType.BINCLASS:
        assert prediction_type == PredictionType.PROBS
        assert y_true.shape == y_pred.shape
        # Always compute accuracy.
        result = {
            'accuracy': (
                (y_true == np.round(y_pred).astype(np.int64)).sum(BATCH_DIM)
                / y_true.shape[BATCH_DIM]
            ),
        }
        if score == lib.data.Score.ROC_AUC:
            roc_auc_values = sklearn.metrics.roc_auc_score(
                y_true.T, y_pred.T, average=None
            )
            if isinstance(roc_auc_values, float):
                # This happens when the pack size equals 1.
                roc_auc_values = np.array([roc_auc_values])
            result['roc-auc'] = roc_auc_values
        elif score == lib.data.Score.CROSS_ENTROPY:
            result['cross-entropy'] = np.mean(
                -np.log(np.where(y_true, y_pred, 1 - y_pred)), axis=-1
            )
    else:
        assert task_type == TaskType.MULTICLASS
        assert prediction_type == PredictionType.PROBS
        assert y_true.shape == y_pred.shape[:2]
        # Always compute accuracy.
        result = {
            'accuracy': (
                (y_true == np.argmax(y_pred, axis=-1)).sum(BATCH_DIM)
                / y_true.shape[BATCH_DIM]
            )
        }
        if score == lib.data.Score.CROSS_ENTROPY:
            result['cross-entropy'] = np.mean(
                -np.log(np.take_along_axis(y_pred, y_true[..., None], axis=-1))[..., 0],
                axis=-1,
            )

    result['score'] = (
        result[score.value]
        if lib.data._SCORE_HIGHER_IS_BETTER[score]
        else -result[score.value]
    )

    return result
