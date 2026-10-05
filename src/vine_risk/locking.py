"""A lock for run folders, so two jobs (for example overlapping scheduled updates) cannot write at once."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class RunLockError(RuntimeError):
    """Raised when another process is using the run folder."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def run_lock(run_dir: str | Path) -> Iterator[Path]:
    """Hold ``<run_dir>/.lock`` (containing our process id) for the duration of the block.

    A lock left behind by a process that no longer exists (crash, power loss) is taken over.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or 0)
            except (ValueError, OSError):
                pid = 0
            if pid and pid != os.getpid() and _alive(pid):
                raise RunLockError(f"{run_dir} is in use by process {pid}. Wait for it to finish "
                                   f"(or delete {path} if you are sure that process is gone).") from None
            path.unlink(missing_ok=True)  # stale: the owner is gone
    else:  # pragma: no cover
        raise RunLockError(f"could not lock {run_dir}")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
