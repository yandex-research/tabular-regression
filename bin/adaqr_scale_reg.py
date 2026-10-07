"""Train AdaQR with quantile/smoothing scale regularization."""

from pathlib import Path

import lib
import lib.util
from bin import adaqr as base


class Config(base.Config):
    pass


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    if config.get('scale_reg_weight', 0.0) <= 0.0:
        raise ValueError('scale_reg_weight must be positive')
    return lib.experiment.relabel_report(main, exp, base.main(config, exp))


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
