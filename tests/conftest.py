import logging

import pytest

from olmoearth_projects.utils.mp import init_mp

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_init_mp() -> None:
    init_mp()
