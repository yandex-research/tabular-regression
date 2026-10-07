"""RealMLP Muon with a fixed, HPO-selected global AdaQR smoothing scale."""

import math
from pathlib import Path
from unittest.mock import patch

import torch
from pytabkit.models.nn_models import models
from pytabkit.models.nn_models.base import Fitter, FitterFactory, Layer
from pytabkit.models.nn_models.base import TrainContext

import bin.realmlp as realmlp
import lib.experiment
from bin.realmlp import Config


class GlobalAdaQRLayer(Layer):
    def __init__(self, y: torch.Tensor, config: dict):
        super().__init__()
        self.sigma = float(config['loss_sigma'])
        if not math.isfinite(self.sigma) or self.sigma < 1e-4:
            raise ValueError('loss_sigma must be finite and at least 1e-4')
        self.mean = y.mean()
        self.std = y.std()
        self.quantiles = torch.linspace(
            0.001, 0.999, config['loss_n_quantiles'], device=y.device
        )

    def forward_tensors(self, tensors):
        preds = tensors['x_cont'].float()
        if self.training:
            y = (tensors['y'].float() - self.mean) / self.std
            errors = preds - y
            sigma = self.sigma
            external = math.sqrt(2 / math.pi) * sigma * torch.exp(
                -0.5 * (errors / sigma).square()
            ) + errors * torch.erf(errors / (sigma * math.sqrt(2)))
            tensors['loss'] = (
                external.mean()
                + preds.mean()
                - 2 * (self.quantiles * preds).mean()
                - sigma / math.sqrt(math.pi)
            )
        else:
            tensors['x_cont'] = preds.mean(-1, keepdim=True) * self.std + self.mean
        return tensors


class GlobalAdaQRFactory(Fitter, FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def _fit(self, ds):
        y = (
            TrainContext.get_global_context()
            .hp_manager.get_more_info_dict()['trainval_ds']
            .tensors['y']
        )
        return GlobalAdaQRLayer(y, self.config)


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    # Reuse the quantile output shape and forward-loss training dispatch.
    if config['model']['train_metric_name'] != 'quantile':
        raise ValueError('Global adaqr requires the quantile-sized output head')
    create_report = lib.experiment.create_report

    def global_report(function, **kwargs):
        return create_report(main if function is realmlp.main else function, **kwargs)

    with (
        patch.object(models, 'QuantileLossFactory', GlobalAdaQRFactory),
        patch.object(lib.experiment, 'create_report', global_report),
    ):
        return realmlp.main(config, exp)


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
