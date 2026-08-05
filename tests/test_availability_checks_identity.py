"""An "available" backend must be the backend, not merely a file that exists.

A setup audit found `FOURC_BINARY=/bin/true` reporting
`available — 4C at /bin/true`. The check confirmed a path existed and was
executable, and nothing more.

That is the worst place in the system to be wrong. An agent calls `discover`,
is told the backend is available, and proceeds; every run then fails for a cause
the availability report has already ruled out. A stale path, a half-built tree,
or a same-named program earlier on PATH were all indistinguishable from a
working install — and the community will hit exactly those, because they are what
a machine that is not ours looks like.

The check now runs the program and looks for 4C's own vocabulary. It fails OPEN
when it cannot look (timeout, permission, OSError), because a check that cannot
run must not condemn a working install; it fails CLOSED only when the program
ran and said something that is not 4C.
"""
from __future__ import annotations

import os

import pytest

from core.backend import BackendStatus
from core.registry import get_backend, load_all_backends


@pytest.fixture(scope="module", autouse=True)
def _backends():
    load_all_backends()


@pytest.fixture
def fourc(monkeypatch):
    import backends.fourc.backend as M
    monkeypatch.setattr(M, "_FOURC_IDENT_CACHE", {})   # per-test isolation
    return get_backend("fourc")


# Every one of these was measured passing an earlier version of the check.
# `4c` as a token is two characters, so `pwd`/`ls` pass when the working
# directory contains it and `env` passes when any variable does — which is the
# normal state for a 4C user, since LD_LIBRARY_PATH=/opt/4C-dependencies/lib
# contains it. `input file` passes gcc, whose no-argument output is "no input
# files". And gzip's non-UTF-8 output raised UnicodeDecodeError, a ValueError,
# which the fail-open branch caught and ACCEPTED.
@pytest.mark.parametrize("path", ["/bin/true", "/bin/echo", "/bin/pwd",
                                  "/bin/ls", "/usr/bin/env", "/usr/bin/gcc",
                                  "/usr/bin/gzip", "/bin/cat"])
def test_an_executable_that_is_not_4c_is_not_available(fourc, monkeypatch, path):
    """The regression. `/bin/true` exits 0, says nothing, and is not a solver."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    monkeypatch.setenv("FOURC_BINARY", path)
    status, message = fourc.check_availability()
    assert status is not BackendStatus.AVAILABLE, message
    assert "does not identify itself as 4C" in message


def test_the_real_binary_is_still_available(fourc, monkeypatch):
    """The other half: the fix must not condemn a working install."""
    monkeypatch.delenv("FOURC_BINARY", raising=False)
    status, message = fourc.check_availability()
    if status is BackendStatus.NOT_INSTALLED:
        pytest.skip("no 4C build on this machine")
    assert status is BackendStatus.AVAILABLE, message


def test_identity_check_fails_open_when_it_cannot_look(monkeypatch):
    """A check that cannot run must not accuse.

    On a machine where the binary is present but unrunnable — a sandbox, a
    permission-restricted CI box — refusing to report the backend would be a
    false negative that no user could diagnose.
    """
    import subprocess

    import backends.fourc.backend as M

    monkeypatch.setattr(M, "_FOURC_IDENT_CACHE", {})

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="4C", timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    ok, why = M._identifies_as_fourc("/anything")
    assert ok is True
    assert "not checked" in why


def test_the_verdict_is_cached_so_discover_stays_cheap(monkeypatch):
    """`check_availability` is called by every knowledge surface; spawning a
    process each time would make the whole tool layer slow enough that someone
    removes the check."""
    import subprocess

    import backends.fourc.backend as M

    monkeypatch.setattr(M, "_FOURC_IDENT_CACHE", {})
    calls = []
    real = subprocess.run

    def _count(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(subprocess, "run", _count)
    M._identifies_as_fourc("/bin/true")
    M._identifies_as_fourc("/bin/true")
    M._identifies_as_fourc("/bin/true")
    assert len(calls) == 1, f"ran the binary {len(calls)} times, expected 1"


def test_the_probe_does_not_read_the_parents_stdin(monkeypatch):
    """The most dangerous defect found in this check.

    `subprocess.run([binary])` inherits the parent's stdin. An audit pointed the
    probe at `/bin/cat`, which consumed the parent's ENTIRE stdin and made the
    verdict a function of that text; pointed at a real solver it hung the full
    timeout and then failed open. Under an MCP stdio server the parent's stdin is
    the JSON-RPC stream, so an availability probe could eat the protocol.
    """
    import subprocess

    import backends.fourc.backend as M

    monkeypatch.setattr(M, "_FOURC_IDENT_CACHE", {})
    seen = {}
    real = subprocess.run

    def _capture(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(subprocess, "run", _capture)
    M._identifies_as_fourc("/bin/true")
    assert seen.get("stdin") is subprocess.DEVNULL, (
        "the identity probe must close stdin; inheriting it lets the probed "
        "program consume the MCP JSON-RPC stream")


def test_a_non_utf8_binary_is_not_accepted_by_the_fail_open_path(monkeypatch):
    """UnicodeDecodeError is a ValueError. Catching ValueError in the
    cannot-look branch meant every binary with non-text output was accepted."""
    import backends.fourc.backend as M

    monkeypatch.setattr(M, "_FOURC_IDENT_CACHE", {})
    import os
    if not os.path.exists("/usr/bin/gzip"):
        pytest.skip("no gzip to probe")
    ok, why = M._identifies_as_fourc("/usr/bin/gzip")
    assert ok is False, why
