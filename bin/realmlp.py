"""Train and evaluate RealMLP using pytabkit's sklearn interface.

Accepts model parameters from config, enabling integration with bin/tune.py
for Optuna-based hyperparameter optimization.
"""

from pathlib import Path
from typing import TypedDict

import delu
import numpy as np
import torch
from loguru import logger
from pytabkit import RealMLP_TD_Regressor

import lib
import lib.data
import lib.experiment
import lib.util
from lib.types import KWArgs


def _get_device() -> str:
    """Get the best available device for pytabkit (returns 'cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)
    delu.random.seed(config['seed'])

    # >>> Data
    logger.info('Loading dataset...')
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.task.is_regression, 'Only regression supported'

    # >>> Prepare X, Y
    X: dict[str, np.ndarray] = {}
    for part in dataset.parts():
        arrays = []
        if 'x_num' in dataset.data:
            arrays.append(dataset.data['x_num'][part])
        if 'x_bin' in dataset.data:
            arrays.append(dataset.data['x_bin'][part])
        if 'x_cat' in dataset.data:
            arrays.append(dataset.data['x_cat'][part].astype('float64'))
        X[part] = (
            np.concatenate(arrays, axis=1)
            if arrays
            else np.empty((dataset.size(part), 0))
        )

    Y = {k: v.astype('float32') for k, v in dataset.data['y'].items()}

    n_train = len(Y['train'])
    n_val = len(Y['val'])
    X_trainval = np.concatenate([X['train'], X['val']], axis=0)
    Y_trainval = np.concatenate([Y['train'], Y['val']], axis=0)
    val_idxs = np.arange(n_train, n_train + n_val)

    logger.info(f'Train indices: 0..{n_train - 1}')
    logger.info(f'Val indices: {n_train}..{n_train + n_val - 1}')
    logger.info(f'Fitting model (n_train={n_train}, n_val={n_val})...')

    model = RealMLP_TD_Regressor(
        random_state=config['seed'],
        n_cv=1,
        n_refit=0,
        verbosity=0,
        device=_get_device(),
        **config['model'],
    )

    timer = delu.tools.Timer()
    timer.run()
    model.fit(X_trainval, Y_trainval, val_idxs=val_idxs)
    train_time = timer.elapsed()

    predictions = {
        'train': model.predict(X['train']),
        'val': model.predict(X['val']),
        'test': model.predict(X['test']),
    }

    report['prediction_type'] = 'labels'
    report['time'] = train_time
    report['metrics'] = dataset.task.calculate_metrics(
        predictions, report['prediction_type']
    )

    lib.experiment.dump_predictions(exp, predictions)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
