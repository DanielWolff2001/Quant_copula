"""Shared test fixtures."""
import sys

import pytest

_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_main_module():
    """Undo what Streamlit's ``AppTest`` leaves behind in ``__main__``.

    ``AppTest`` runs ``dashboard/app.py`` in this process and leaves ``__main__.__file__`` pointing at it.
    Worker processes started with the ``spawn`` method re-run ``__main__`` from that path, so every later
    test that uses a process pool would execute the dashboard script in its workers and crash. Restoring
    the attributes after each test keeps the tests independent of their order.
    """
    main = sys.modules["__main__"]
    saved = {k: getattr(main, k, _MISSING) for k in ("__file__", "__spec__")}
    yield
    for k, v in saved.items():
        if v is _MISSING:
            main.__dict__.pop(k, None)
        else:
            setattr(main, k, v)
