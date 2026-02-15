from __future__ import annotations

import pytest

from tests.numba_layers import NUMBA_TEST_CACHE, NUMBA_TEST_LAYER, reset_nbux_numba_cache


def pytest_report_header() -> str:
    """Show active numba testing mode in pytest output."""
    return f"NUMBA_TEST_LAYER={NUMBA_TEST_LAYER}, NUMBA_TEST_CACHE={'true' if NUMBA_TEST_CACHE else 'false'}"


@pytest.fixture(scope="session", autouse=True)
def maybe_reset_numba_cache() -> None:
    """Optionally clear dispatcher caches once per test session."""
    if NUMBA_TEST_CACHE:
        reset_nbux_numba_cache()
