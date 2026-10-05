"""The ``vine-risk`` command line.

::

    vine-risk run [--steps rolling,metrics,changes,risk] [--last N] ...   # the whole pipeline
    vine-risk update [--dry-run] ...                                      # daily update with the latest prices
    vine-risk schedule --kind cron|launchd|systemd                        # print a ready-made scheduler entry
    vine-risk replay [--days 120] ...                                     # simulated live monitoring
    vine-risk info [RUN_DIR]                                              # what produced a run folder
    vine-risk dashboard [--run-dir ...]                                   # open the Streamlit dashboard

``python -m vine_risk`` works as well. Every command takes ``--config`` (default
``configs/default.yaml``); ``--help`` on a command lists its options.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence, TextIO

import pandas as pd

from vine_risk import __version__
from vine_risk.config import load_config
from vine_risk.locking import RunLockError, run_lock
from vine_risk.manifest import ManifestMismatch, lock_hash, read_manifest
from vine_risk.pipeline import load_prices_and_returns
from vine_risk.runner import STEPS, default_run_dir, run_pipeline, with_rolling

DEFAULT_CONFIG = "configs/default.yaml"


class TextProgress:
    """A tiny progress bar for ``progress(done, total)`` callbacks.

    On a terminal it redraws one line with percentage and an estimate of the remaining time; when output
    is redirected to a file it prints a line at every 10 %.
    """

    def __init__(self, label: str, stream: TextIO | None = None, min_interval: float = 0.5) -> None:
        self.label, self.stream, self.min_interval = label, stream or sys.stderr, min_interval
        self._t0 = self._last = time.time()
        self._decile = -1

    def __call__(self, done: int, total: int) -> None:
        now = time.time()
        pct = 100 * done / max(total, 1)
        eta = (now - self._t0) / max(done, 1) * (total - done)
        text = f"{self.label}: {done}/{total} ({pct:.0f}%), about {int(eta // 60)}:{int(eta % 60):02d} left"
        if self.stream.isatty():
            if done == total or now - self._last >= self.min_interval:
                bar = "#" * int(pct / 5)
                self.stream.write(f"\r{self.label} [{bar:<20}] {done}/{total} {pct:3.0f}%  {int(eta // 60)}:{int(eta % 60):02d} left ")
                self.stream.write("\n" if done == total else "")
                self.stream.flush()
                self._last = now
        elif int(pct // 10) > self._decile or done == total:
            self._decile = int(pct // 10)
            print(text, file=self.stream, flush=True)


# ------------------------------------------------------------------ commands
def cmd_run(a: argparse.Namespace) -> int:
    cfg = with_rolling(load_config(a.config), window=a.window, refit_frequency=a.refit_frequency, n_jobs=a.n_jobs)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    _, returns = load_prices_and_returns(cfg, refresh=a.refresh)
    if a.last:
        returns = returns.iloc[-a.last:]
    run_dir = Path(a.run_dir) if a.run_dir else default_run_dir(cfg)
    print(f"run folder {run_dir} | {len(cfg.assets)} assets | window {cfg.rolling.window}, refit every "
          f"{cfg.rolling.refit_frequency} day(s) | data {returns.index[0].date()} to {returns.index[-1].date()}")
    with run_lock(run_dir):
        timings = run_pipeline(cfg, returns, run_dir, steps, n_jobs=a.n_jobs, scan_step=a.scan_step, n_perm=a.n_perm,
                               risk_step=a.risk_step, progress=lambda name: TextProgress(name), force=a.force,
                               command=["vine-risk", *sys.argv[1:]])
    for name, sec in timings.items():
        print(f"  {name:<8} {sec:7.1f} s")
    print(f"done -> {run_dir}   (see `vine-risk info {run_dir}`; dashboard: `vine-risk dashboard --run-dir {run_dir}`)")
    return 0


def cmd_update(a: argparse.Namespace) -> int:
    from vine_risk.update import update_run

    cfg = with_rolling(load_config(a.config), window=a.window, refit_frequency=a.refit_frequency)
    overrides = {k: v for k, v in dict(alpha=a.alpha, enter_ratio=a.enter_ratio, exit_ratio=a.exit_ratio).items()
                 if v is not None}
    report = update_run(cfg, a.run_dir or default_run_dir(cfg), threads=a.threads, n_jobs=a.n_jobs,
                        scan_step=a.scan_step, n_perm=a.n_perm, risk_step=a.risk_step,
                        monitor_overrides=overrides, dry_run=a.dry_run, force=a.force,
                        today=pd.Timestamp.today().normalize())
    for alert in report.alerts:
        print(f"ALERT {alert['date']}: {alert['message']}")
    return 0


def render_schedule(kind: str, command: str, workdir: str | Path, at: str = "23:30", log: str = "data/update.log") -> str:
    """Text of a scheduler entry that runs ``command`` on weekdays at ``at`` (HH:MM, local time).

    Nothing is installed; copy the text where it belongs (instructions are printed with it).
    """
    hh, mm = (int(x) for x in at.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"bad time {at!r}; use HH:MM")
    wd = str(workdir)
    if kind == "cron":
        return (f"# add with `crontab -e` (times are the machine's local time; adjust to when your data vendor has the day's close)\n"
                f"{mm} {hh} * * 1-5 cd {wd} && {command} >> {log} 2>&1")
    if kind == "launchd":
        days = "".join(f"    <dict><key>Weekday</key><integer>{d}</integer><key>Hour</key><integer>{hh}</integer>"
                       f"<key>Minute</key><integer>{mm}</integer></dict>\n" for d in range(1, 6))
        prog = "".join(f"    <string>{x}</string>\n" for x in command.split())
        return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                "<plist version=\"1.0\"><dict>\n  <key>Label</key><string>com.vinerisk.update</string>\n"
                f"  <key>ProgramArguments</key><array>\n{prog}  </array>\n"
                f"  <key>WorkingDirectory</key><string>{wd}</string>\n"
                f"  <key>StartCalendarInterval</key><array>\n{days}  </array>\n"
                f"  <key>StandardOutPath</key><string>{wd}/{log}</string>\n"
                f"  <key>StandardErrorPath</key><string>{wd}/{log}</string>\n</dict></plist>\n"
                "<!-- save as ~/Library/LaunchAgents/com.vinerisk.update.plist, then: launchctl load ~/Library/LaunchAgents/com.vinerisk.update.plist -->")
    if kind == "systemd":
        return (f"# /etc/systemd/system/vine-risk-update.service\n[Unit]\nDescription=vine-risk daily update\n\n[Service]\n"
                f"Type=oneshot\nWorkingDirectory={wd}\nExecStart={command}\n\n"
                f"# /etc/systemd/system/vine-risk-update.timer\n[Unit]\nDescription=vine-risk daily update\n\n[Timer]\n"
                f"OnCalendar=Mon..Fri {hh:02d}:{mm:02d}\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n"
                "# enable with: systemctl enable --now vine-risk-update.timer")
    raise ValueError(f"unknown scheduler {kind!r}; choose cron, launchd or systemd")


def cmd_schedule(a: argparse.Namespace) -> int:
    import shutil

    exe = shutil.which("vine-risk") or f"{sys.executable} -m vine_risk"
    cmd = (f"{exe} update --config {Path(a.config).resolve()}" + (f" --run-dir {Path(a.run_dir).resolve()}" if a.run_dir else "")
           + (f" --window {a.window}" if a.window else "") + (f" --refit-frequency {a.refit_frequency}" if a.refit_frequency else ""))
    print(render_schedule(a.kind, cmd, Path.cwd(), a.at))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from vine_risk.replay import run_replay

    cfg = with_rolling(load_config(a.config), window=a.window, refit_frequency=a.refit_frequency)
    run_replay(cfg, a.run_dir or default_run_dir(cfg), days=a.days, start=a.start, end=a.end, delay=a.delay,
               scan_step=a.scan_step, n_perm=a.n_perm, risk_every=a.risk_every, threads=a.threads,
               verify=not a.no_verify, out_dir=a.out)
    return 0


def format_info(run_dir: str | Path) -> str:
    """Human-readable description of a run folder: what made it, and what it contains."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} does not exist.")
    m = read_manifest(run_dir)
    lines = [f"run folder: {run_dir}"]
    if m is None:
        lines.append("no manifest.json (made before manifests existed, or not a run folder)")
    else:
        git = m.get("git")
        lines += [
            f"created {m['created']}, last updated {m['updated']}",
            f"code: vine-risk {m['package_version']}" + (f", git {git['commit'][:10]} on {git['branch']}"
                                                          + (" (uncommitted changes!)" if git["dirty"] else "") if git else ""),
            f"python {m['python']} on {m['platform']}",
            "libraries: " + ", ".join(f"{k} {v}" for k, v in m["libraries"].items() if k in ("numpy", "pandas", "scipy", "pyvinecopulib")),
            f"data: {m['data']['n_obs']} observations, {m['data']['first_date']} to {m['data']['last_date']}, "
            f"{len(m['data']['assets'])} assets ({', '.join(m['data']['assets'])})",
            f"fit settings (hash {m['fit_hash'][:10]}): window {m['fit_fingerprint']['window']}, refit every "
            f"{m['fit_fingerprint']['refit_frequency']}, truncation {m['fit_fingerprint']['truncation_level']}, "
            f"{m['fit_fingerprint']['selection_criterion']}, seed {m['fit_fingerprint']['seed']}",
            "steps:",
        ]
        for name in STEPS:
            s = m.get("steps", {}).get(name)
            if s:
                detail = ", ".join(f"{k} {v}" for k, v in s.items() if k not in ("settings", "finished", "seconds"))
                lines.append(f"  {name:<8} {s['finished']}  {s.get('seconds', '?')} s  {detail}")
            else:
                lines.append(f"  {name:<8} not run")
        notes = []
        for k, v in m["libraries"].items():
            try:
                now = metadata.version(k)
                if now != v:
                    notes.append(f"{k}: run used {v}, installed {now}")
            except metadata.PackageNotFoundError:
                pass
        if m.get("lock_sha256") and lock_hash() and m["lock_sha256"] != lock_hash():
            notes.append("requirements-lock.txt has changed since this run")
        lines += ["differences from the current environment: " + ("none" if not notes else "")] + [f"  {n}" for n in notes]
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    lines.append(f"files ({len(files)}): " + ", ".join(f"{p.relative_to(run_dir)} ({p.stat().st_size / 1e6:.1f} MB)"
                                                       for p in files[:12]) + (" ..." if len(files) > 12 else ""))
    return "\n".join(lines)


def cmd_info(a: argparse.Namespace) -> int:
    run_dir = a.run_dir or default_run_dir(load_config(a.config))
    print(format_info(run_dir))
    return 0


def find_dashboard_app() -> Path:
    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    if not app.is_file():
        raise FileNotFoundError("dashboard/app.py not found; the dashboard ships with the repository checkout, "
                                "not with an installed wheel. Run `streamlit run dashboard/app.py` from the repo.")
    return app


def cmd_dashboard(a: argparse.Namespace) -> int:
    import os

    cfg = load_config(a.config)
    env = {**os.environ, "VINE_RISK_RUN_DIR": str(a.run_dir or default_run_dir(cfg)), "VINE_RISK_CONFIG": a.config}
    cmd = [sys.executable, "-m", "streamlit", "run", str(find_dashboard_app()), "--server.port", str(a.port)]
    print("starting:", " ".join(cmd))
    return subprocess.call(cmd, env=env)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vine-risk", description="Dynamic vine copula risk monitoring.")
    p.add_argument("--version", action="version", version=f"vine-risk {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, func: Callable, help: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help, description=help)
        sp.add_argument("--config", default=DEFAULT_CONFIG, help="settings file (default %(default)s)")
        sp.set_defaults(func=func)
        return sp

    r = add("run", cmd_run, "run (or extend) the pipeline: rolling fits, metrics, change detection, risk")
    r.add_argument("--run-dir", help="results folder (default data/results/w<window>)")
    r.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: %(default)s")
    r.add_argument("--last", type=int, help="use only the last N return observations (quick test)")
    r.add_argument("--window", type=int, help="override rolling.window")
    r.add_argument("--refit-frequency", type=int, help="override rolling.refit_frequency")
    r.add_argument("--n-jobs", type=int, help="worker processes (default from the config)")
    r.add_argument("--scan-step", type=int, default=5, help="permutation test every N observations")
    r.add_argument("--n-perm", type=int, default=499, help="permutations per test")
    r.add_argument("--risk-step", type=int, default=5, help="VaR/ES for every N-th fit")
    r.add_argument("--force", action="store_true", help="skip the safety checks on settings and data (not recommended)")
    r.add_argument("--refresh", action="store_true", help="download the prices again instead of using the cache")

    u = add("update", cmd_update, "daily update: fetch new prices, run the live monitor on them, extend the run folder")
    u.add_argument("--run-dir", help="results folder to update (default data/results/w<window>)")
    u.add_argument("--dry-run", action="store_true", help="fetch and check the prices, report what would be done, change nothing")
    u.add_argument("--force", action="store_true", help="skip the safety checks on settings and data (not recommended)")
    u.add_argument("--window", type=int, help="override rolling.window (must match the run)")
    u.add_argument("--refit-frequency", type=int, help="override rolling.refit_frequency (must match the run)")
    u.add_argument("--threads", type=int, default=4, help="threads of the vine fitter")
    u.add_argument("--n-jobs", type=int, help="worker processes for the table updates")
    u.add_argument("--scan-step", type=int, default=5)
    u.add_argument("--n-perm", type=int, default=499)
    u.add_argument("--risk-step", type=int, default=5)
    u.add_argument("--alpha", type=float, help="significance level for alerts (default 0.01)")
    u.add_argument("--enter-ratio", type=float, help="alert starts above this multiple of the no-change level (default 2.0)")
    u.add_argument("--exit-ratio", type=float, help="alert ends below this multiple (default 1.5)")

    sc = add("schedule", cmd_schedule, "print a cron / launchd / systemd entry that runs the daily update (installs nothing)")
    sc.add_argument("--kind", choices=["cron", "launchd", "systemd"], required=True)
    sc.add_argument("--at", default="23:30", help="local time, HH:MM (default %(default)s)")
    sc.add_argument("--run-dir", help="results folder to update")
    sc.add_argument("--window", type=int, help="add --window to the scheduled command")
    sc.add_argument("--refit-frequency", type=int, help="add --refit-frequency to the scheduled command")

    pl = add("replay", cmd_replay, "simulated live monitoring: replay recent history day by day")
    pl.add_argument("--run-dir", help="results folder to resume from")
    pl.add_argument("--window", type=int, help="override rolling.window (must match the run)")
    pl.add_argument("--refit-frequency", type=int, help="override rolling.refit_frequency (must match the run)")
    pl.add_argument("--days", type=int, default=120, help="replay the last N observations")
    pl.add_argument("--start", help="first replayed date (overrides --days)")
    pl.add_argument("--end", help="last replayed date")
    pl.add_argument("--delay", type=float, default=0.0, help="seconds to wait between observations")
    pl.add_argument("--scan-step", type=int, default=5)
    pl.add_argument("--n-perm", type=int, default=499)
    pl.add_argument("--risk-every", type=int, default=5, help="VaR/ES at every N-th fit (0 = never)")
    pl.add_argument("--threads", type=int, default=4, help="threads of the vine fitter")
    pl.add_argument("--no-verify", action="store_true", help="skip the comparison with the stored batch run")
    pl.add_argument("--out", help="output folder (default <run>/live)")

    i = add("info", cmd_info, "show what produced a run folder (versions, settings, data, steps)")
    i.add_argument("run_dir", nargs="?", help="results folder (default data/results/w<window>)")

    d = add("dashboard", cmd_dashboard, "open the Streamlit dashboard")
    d.add_argument("--run-dir", help="results folder to show")
    d.add_argument("--port", type=int, default=8501)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ManifestMismatch, RunLockError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted (progress is saved; run the same command again to resume)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
