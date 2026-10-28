"""Config loading for the ``run_*.py`` processes and scripts.

    python run_controller.py --config configs/controller.yaml --hrg.field linear

``--config`` takes one or more YAML files, merged in order (later ones
override); ``--a.b value`` then overrides ``a.b``. Paths in configs are
relative to the repository root, which becomes the working directory. Every
process's terminal output is also written to
``outputs/logs/<script>_<time>.log`` (errors and robot reflexes included)::

    python run_controller.py --config configs/controller.yaml configs/robot/controller.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from omegaconf import DictConfig, OmegaConf

from fr3.utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args
from hrg.paths import REPO


class _Tee:
    """A stream that also writes to a log file."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, data):
        self._stream.write(data)
        try:
            self._log.write(data)
        except ValueError:  # log closed at exit
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._log.flush()
        except ValueError:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _log_output() -> None:
    """Copy stdout / stderr to outputs/logs/<script>_<time>.log (``HRG_LOG=0``:
    not; tests and the simulation studies set it, they keep their own logs)."""
    if isinstance(sys.stdout, _Tee) or os.environ.get("HRG_LOG", "1") == "0":
        return
    os.makedirs(os.path.join(REPO, "outputs", "logs"), exist_ok=True)
    name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "process"
    log = open(os.path.join(REPO, "outputs", "logs", f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"), "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)


def load_config(description: str = "", default: str | None = None, quiet: bool = False) -> DictConfig:
    os.chdir(REPO)
    _log_output()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, nargs="+", required=default is None,
                        default=None if default is None else [default])
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in args.config], parse_nested_args(parse_unknown_args(unknown)))
    if not quiet:
        print(OmegaConf.to_yaml(cfg))
    return cfg
