import shutil
from pathlib import Path
from typing import Any, TypedDict

import delu
import numpy as np
import scipy.special
from loguru import logger

import lib
import lib.data
import lib.experiment
import lib.util


class Config(TypedDict):
    ensemble_size: int


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=False)

    evaluation_exp = exp.with_name('evaluation')
    assert lib.experiment.is_done(evaluation_exp), (
        'Cannot evaluate ensembles, because the evaluation of single models'
        ' is either missing or is not done.'
        f' The expected evaluation experiment: {evaluation_exp}'
    )
    expected_exp_name = f'ensemble-{config["ensemble_size"]}'
    assert exp.name == f'ensemble-{config["ensemble_size"]}', (
        f'The experiment name must be "{expected_exp_name}".'
        f' The actual name is {exp.name}'
    )

    single_experiments = lib.experiment.load_report(evaluation_exp)['experiments']
    first_single_exp = single_experiments[0]
    first_single_report = first_single_exp['report']
    task = lib.data.Task.from_dir(first_single_exp['config']['data']['path'])
    timer = delu.tools.Timer()

    report.setdefault('experiments', [])
    timer.run()
    for ensemble_id in range(len(single_experiments) // config['ensemble_size']):
        if ensemble_id < len(report['experiments']):
            # This is possible only if the experiment is resumed
            # and this ensemble is already processed.
            continue

        next_exp = exp / str(ensemble_id)
        if next_exp.exists():
            logger.warning(f'Removing the incomplete experiment {exp}')
            shutil.rmtree(exp)

        next_config = {'data': {'path': first_single_exp['config']['data']['path']}}
        next_report: dict[str, Any] = {
            'single-model-info': {'function': first_single_report['function']},
            'seeds': list(
                range(
                    ensemble_id * config['ensemble_size'],
                    (ensemble_id + 1) * config['ensemble_size'],
                )
            ),
            'prediction_type': 'labels' if task.is_regression else 'probs',
        }

        single_predictions = [
            lib.experiment.load_predictions(evaluation_exp / str(x))
            for x in next_report['seeds']
        ]
        predictions = {}
        for part in ['train', 'val', 'test']:
            stacked_predictions = np.stack([x[part] for x in single_predictions])
            if task.is_binclass:
                # Predictions for binary classifications are expected to contain
                # only the probability of the positive label.
                assert stacked_predictions.ndim == 2
                if first_single_report['prediction_type'] == 'logits':
                    stacked_predictions = scipy.special.expit(stacked_predictions)
            elif task.is_multiclass:
                assert stacked_predictions.ndim == 3
                if first_single_report['prediction_type'] == 'logits':
                    stacked_predictions = scipy.special.softmax(stacked_predictions, -1)
            else:
                assert task.is_regression
                assert stacked_predictions.ndim == 2
            predictions[part] = stacked_predictions.mean(0)

        next_report['metrics'] = task.calculate_metrics(
            predictions, next_report['prediction_type']
        )
        next_exp.mkdir()
        # Only the artifacts not tracked by the VCS are saved,
        # and the rest goes into report['experiments'].
        lib.experiment.dump_predictions(exp, predictions)
        lib.experiment.dump_summary(next_exp, lib.experiment.summarize(next_report))

        report['experiments'].append({'config': next_config, 'report': next_report})
        lib.experiment.dump_report(exp, report)

    report['time'] = timer.elapsed()
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main, resumable=True)
