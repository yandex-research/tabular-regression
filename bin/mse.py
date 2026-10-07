"""MSE entrypoints for the Muon and AdamW experiments."""

from pathlib import Path

import lib.experiment
import lib.util
from bin import regression, regression_adamw

Config = regression.Config


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    return lib.experiment.relabel_report(main, exp, regression.main(config, exp))


def main_adamw(
    config: regression_adamw.Config, exp: str | Path
) -> lib.experiment.Report:
    return lib.experiment.relabel_report(
        main_adamw, exp, regression_adamw.main(config, exp)
    )


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
