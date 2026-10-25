import pytest

from hrg.processes import Procs


@pytest.fixture
def procs(tmp_path):
    with Procs(tmp_path) as p:
        yield p
