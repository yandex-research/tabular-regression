# CatBoost

import os
import shutil
from pathlib import Path

import delu
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRegressor
from loguru import logger
from typing_extensions import TypedDict

import lib
import lib.data
import lib.deep
import lib.env
import lib.experiment
import lib.util
from lib.types import KWArgs


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    fit: KWArgs


def _check_config(config: Config):
    assert 'random_state' not in config['model']
    assert config['data'].get('cat_policy') in [None, 'ordinal']
    assert (
        config['model'].get('l2_leaf_reg', 3.0) > 0
    )  # CatBoost fails on multiclass problems with l2_leaf_reg=0

    assert 'early_stopping_rounds' in config['model'], 'We require early-stopping'
    assert 'task_type' in config['model'], (
        'task_type significantly affects performance, so must be set explicitly'
    )
    if config['model']['task_type'] == 'GPU':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')
    else:
        assert not os.environ.get('CUDA_VISIBLE_DEVICES')


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

    if 'x_num' in dataset.data:
        X = {
            part: pd.DataFrame(dataset.data['x_num'][part]) for part in dataset.parts()
        }
    else:
        X = {part: pd.DataFrame() for part in dataset.parts()}

    if 'x_bin' in dataset.data:
        # Merge binary features to continuous features.
        X = {
            part: pd.concat(
                [X[part], pd.DataFrame(dataset.data['x_bin'][part])], axis=1
            )
            for part in dataset.parts()
        }
        # Rename columns
        for part in X:
            X[part].columns = range(X[part].shape[1])

    if 'x_cat' in dataset.data:
        # Merge one-hot-encoded categorical features
        categorical_features = range(
            dataset.n_num_features + dataset.n_bin_features, dataset.n_features
        )

        if 'x_num' not in dataset.data:
            # does not consider existing binary features, but we do not care now
            X = {k: pd.DataFrame(v) for k, v in dataset.data.pop('x_cat').items()}
        else:
            X = {
                part: pd.concat(
                    [
                        X[part],
                        pd.DataFrame(
                            dataset.data['x_cat'][part], columns=categorical_features
                        ),
                    ],
                    axis=1,
                )
                for part in dataset.parts()
            }
            dataset.data.pop('x_cat')
    else:
        categorical_features = None

    # >>> model
    model_kwargs = config['model'] | {
        'random_seed': config['seed'],
        'train_dir': exp / 'catboost_info',
        'cat_features': categorical_features,
    }
    if config['model']['task_type'] == 'GPU':
        config['model']['devices'] = '0'
    if dataset.task.is_regression:
        model = CatBoostRegressor(loss_function='RMSEWithUncertainty', **model_kwargs)
        predict = model.predict
    else:
        model = CatBoostClassifier(
            **model_kwargs,
            eval_metric='AUC'
            if dataset.task.score == lib.data.Score.ROC_AUC
            else 'Accuracy',
        )
        predict = (
            model.predict_proba
            if dataset.task.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
        )

    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'

    # >>> training
    logger.info('training...')
    did_fail = 0
    timer = delu.tools.Timer()
    timer.run()
    try:
        model.fit(
            X['train'],
            dataset.data['y']['train'],
            eval_set=(X['val'], dataset.data['y']['val']),
            **config['fit'],
        )
    except:
        predictions = {'train': np.nan, 'val': np.nan, 'test': np.nan}
        did_fail = 1

    shutil.rmtree(model_kwargs['train_dir'])
    report['time'] = timer.elapsed()
    if did_fail == 0:
        report['best_iteration'] = model.get_best_iteration()

        # >>> finish
        # model.save_model(output / 'model.cbm')
        np.save(
            exp / 'feature_interactions.npy',
            model.get_feature_importance(type='Interaction'),
        )
        np.save(exp / 'feature_importances.npy', model.get_feature_importance())
        predictions = {k: predict(v) for k, v in X.items()}

    # if regression_label_stats is not None:
    #    predictions_for_val = {
    #        k: v[..., 0] * regression_label_stats.std + regression_label_stats.mean
    #        for k, v in predictions.items()
    #    }
    class UncAwareLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, y_pred, y_true):
            mu, var = y_pred[..., 0], y_pred[..., 1]
            # print(mu, "MU")
            # print(y_true, "Y_TRUE", flush=True)
            if isinstance(mu, torch.Tensor):
                logvar = torch.log(var)
                loss = (logvar / 2 + (y_true - mu) ** 2 / torch.exp(logvar) / 2).mean()
            else:
                logvar = np.log(var)
                loss = (logvar / 2 + (y_true - mu) ** 2 / np.exp(logvar) / 2).mean()
            return loss

    loss_fn = UncAwareLoss()
    metrics = dict()
    if lib.util.are_valid_predictions(predictions):
        for part in predictions:
            metrics[part] = dict()
            if not (isinstance(dataset.data['y'][part], torch.Tensor)):
                metrics[part]['score'] = float(
                    -loss_fn(predictions[part], dataset.data['y'][part])
                )
            else:
                metrics[part]['score'] = float(
                    -loss_fn(predictions[part], dataset.data['y'][part].cpu().numpy())
                )
    else:
        metrics = {x: {'score': -99999999999.0} for x in predictions}

    report['metrics'] = metrics  # dataset.task.calculate_metrics(
    #        predictions_for_val, report['prediction_type']  # type: ignore[code]
    # )
    # lib.experiment.dump_checkpoint(exp, make_checkpoint())
    lib.experiment.dump_predictions(exp, predictions)

    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
