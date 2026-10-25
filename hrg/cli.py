"""Config loading for the ``run_*.py`` processes and scripts.

    python run_controller.py --config configs/controller.yaml --hrg.field linear

``--config`` takes one or more YAML files, merged in order (later ones
override); ``--a.b value`` then overrides ``a.b``. Paths in configs are
relative to the repository root, which becomes the working directory::

    python run_controller.py --config configs/controller.yaml configs/robot/controller.yaml
"""

from __future__ import annotations

import argparse
import os

from omegaconf import DictConfig, OmegaConf

from fr3.utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args
from hrg.paths import REPO


def load_config(description: str = "", default: str | None = None, quiet: bool = False) -> DictConfig:
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, nargs="+", required=default is None,
                        default=None if default is None else [default])
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in args.config], parse_nested_args(parse_unknown_args(unknown)))
    if not quiet:
        print(OmegaConf.to_yaml(cfg))
    return cfg
