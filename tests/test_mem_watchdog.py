"""
Tests for tools/mem_watchdog.py -- the guard on an oversubscribed model load.

This is the code that stands between a 22 GB model and the desktop's OOM killer
(which already took one run in this repo, see docs/expert_offloading.md). Its
decision logic and its actual kill path are both pinned here, offline.
"""

import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import mem_watchdog as wd  # noqa: E402

GIB = 2 ** 30


def test_a_single_low_sample_is_not_enough_to_act():
    """Page cache is reclaimable and dips recover, so a transient low reading must
    not kill a multi-hour run."""
    assert not wd.should_act(1 * GIB, 2 * GIB, consecutive_below=1, grace_samples=3)
    assert not wd.should_act(1 * GIB, 2 * GIB, consecutive_below=2, grace_samples=3)


def test_a_sustained_breach_acts():
    assert wd.should_act(1 * GIB, 2 * GIB, consecutive_below=3, grace_samples=3)
    assert wd.should_act(0, 2 * GIB, consecutive_below=99, grace_samples=3)


def test_above_the_floor_never_acts():
    assert not wd.should_act(8 * GIB, 2 * GIB, consecutive_below=99, grace_samples=3)


def test_target_alive_and_rss_read_a_real_process():
    assert wd.target_alive(os.getpid())
    assert wd.target_rss(os.getpid()) > 0
    # A pid that has certainly exited (kernel pids wrap far above this).
    assert not wd.target_alive(4_000_000)


def test_a_zombie_target_counts_as_finished():
    """Regression: os.kill(pid, 0) succeeds on a zombie, so the watchdog used to
    keep sampling a run that had already exited and never wrote its summary."""
    target = subprocess.Popen(["sleep", "0.05"])
    try:
        deadline = time.time() + 10
        while time.time() < deadline and wd.process_state(target.pid) != "Z":
            time.sleep(0.05)  # do NOT call poll(): it reaps the child
        assert wd.process_state(target.pid) == "Z"
        assert not wd.target_alive(target.pid)
    finally:
        target.wait(timeout=10)


def test_watchdog_terminates_a_target_that_breaches_the_floor(tmp_path):
    """End-to-end: the guard must actually end the process it was handed, and say
    so in its exit code, log and summary. The floor is set above any real machine
    so the breach is deterministic."""
    target = subprocess.Popen(["sleep", "60"])
    log = tmp_path / "watch.log"
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                          "mem_watchdog.py"),
             "--pid", str(target.pid), "--min-ram-gib", "999999",
             "--grace-samples", "2", "--interval", "0.2", "--log", str(log)],
            capture_output=True, text=True, timeout=60)
        assert p.returncode == 2, p.stdout + p.stderr
        assert "GUARD TERMINATED" in p.stdout
        text = log.read_text()
        assert f"watching pid {target.pid}" in text
        assert "GUARD: MemAvailable" in text
        # The target must be gone, not merely asked to leave.
        target.wait(timeout=10)
        assert not wd.target_alive(target.pid)
    finally:
        if target.poll() is None:
            target.kill()
            target.wait(timeout=10)


def test_watchdog_logs_samples_and_leaves_a_healthy_target_alone(tmp_path):
    target = subprocess.Popen(["sleep", "3"])
    log = tmp_path / "watch.log"
    p = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                      "mem_watchdog.py"),
         "--pid", str(target.pid), "--min-ram-gib", "0.01",
         "--grace-samples", "2", "--interval", "0.2", "--log", str(log)],
        capture_output=True, text=True, timeout=60)
    target.wait(timeout=10)
    assert p.returncode == 0 and "GUARD TERMINATED" not in p.stdout
    text = log.read_text()
    assert "target exited on its own" in text
    # Every sample carries the numbers the run has to be able to answer for.
    sample = [ln for ln in text.splitlines() if ln[:2].isdigit()]
    assert sample and len(sample[0].split()) >= 5


def test_watchdog_reads_the_target_pid_from_a_pidfile(tmp_path):
    """A long run writes its own PID; the guard must arm on that process and on
    nothing else, which is the whole reason --pidfile exists."""
    target = subprocess.Popen(["sleep", "30"])
    pidfile = tmp_path / "target.pid"
    pidfile.write_text(str(target.pid))
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                          "mem_watchdog.py"),
             "--pidfile", str(pidfile), "--min-ram-gib", "999999",
             "--grace-samples", "2", "--interval", "0.2",
             "--log", str(tmp_path / "w.log")],
            capture_output=True, text=True, timeout=60)
        assert p.returncode == 2
        assert f"watching pid {target.pid}" in (tmp_path / "w.log").read_text()
        target.wait(timeout=10)
    finally:
        if target.poll() is None:
            target.kill()
            target.wait(timeout=10)


def test_watchdog_waits_for_a_pidfile_that_appears_late(tmp_path):
    """The eval writes its PIDfile at startup, but a racing launch must not make
    the guard bail out and leave the run unguarded."""
    target = subprocess.Popen(["sleep", "30"])
    pidfile = tmp_path / "late.pid"
    writer = subprocess.Popen(
        ["bash", "-c", f"sleep 1; echo {target.pid} > {pidfile}"])
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                          "mem_watchdog.py"),
             "--pidfile", str(pidfile), "--wait-pidfile", "20",
             "--min-ram-gib", "999999", "--grace-samples", "2",
             "--interval", "0.2", "--log", str(tmp_path / "w.log")],
            capture_output=True, text=True, timeout=60)
        assert p.returncode == 2, p.stdout + p.stderr
        target.wait(timeout=10)
    finally:
        writer.wait(timeout=10)
        if target.poll() is None:
            target.kill()
            target.wait(timeout=10)


def test_watchdog_requires_a_target():
    p = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                      "mem_watchdog.py"),
         "--min-ram-gib", "1"], capture_output=True, text=True, timeout=30)
    assert p.returncode != 0 and "--pid or --pidfile" in (p.stdout + p.stderr)


def test_watchdog_stops_at_max_minutes_without_killing(tmp_path):
    """A watcher must not outlive its usefulness, and must not kill on the way out."""
    target = subprocess.Popen(["sleep", "30"])
    log = tmp_path / "watch.log"
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools",
                                          "mem_watchdog.py"),
             "--pid", str(target.pid), "--min-ram-gib", "0.01",
             "--interval", "0.2", "--max-minutes", "0.02", "--log", str(log)],
            capture_output=True, text=True, timeout=60)
        assert p.returncode == 0
        assert target.poll() is None  # left running, as asked
        assert "max-minutes reached" in log.read_text()
    finally:
        target.send_signal(signal.SIGTERM)
        target.wait(timeout=10)
