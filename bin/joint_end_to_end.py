"""Joint End-to-End experiments using the shared binning trainer."""

from pathlib import Path

import lib.experiment
import lib.util
from bin import regression

Config = regression.Config


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    return lib.experiment.relabel_report(main, exp, regression.main(config, exp))


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
