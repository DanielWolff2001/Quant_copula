"""Shared test fixtures."""
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_main_module():
    """Undo what Streamlit's ``AppTest`` leaves behind: it installs the dashboard script as ``__main__``.

    Worker processes started with the ``spawn`` method re-run ``__main__`` from its file, so every later
    test that uses a process pool would execute the dashboard script in its workers and crash. Putting the
    original module object back after each test keeps the tests independent of their order.
    """
    original = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = original
