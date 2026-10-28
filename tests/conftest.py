"""Test setup: repo root on sys.path / as cwd, an isolated LCM bus.

    pytest                      # everything
    pytest -m "not integration" # fast unit tests
    pytest -m integration       # starts the sim / controller / fake hardware processes

Integration tests run processes in real time; on an overloaded machine they
can time out, so rerun a failure on its own before suspecting the code.
Process logs of each test: /tmp/pytest-of-$USER/pytest-current/<test>/.
"""

import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

# Isolated LCM bus for this test session (processes started by the tests
# inherit it): tests never talk to a sim / robot you have running.
os.environ.setdefault("LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(20000, 29999)}?ttl=0")
os.environ["HRG_LOG"] = "0"  # the processes' outputs/logs copies (hrg.cli): tests keep their own logs


def pytest_report_header(config):
    try:
        load, n = os.getloadavg()[0], os.cpu_count() or 1
    except OSError:
        return None
    msg = f"load average {load:.0f} on {n} CPUs"
    if load > n:
        msg += "  -- OVERLOADED: real-time integration tests may time out"
    return msg


def pytest_collection_modifyitems(items):
    import pytest

    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
