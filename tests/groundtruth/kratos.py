"""Ground-truth probes for Kratos Multiphysics.

Each sub-application of Kratos lives under ``applications/<Name>Application/``
in the ``KratosMultiphysics/Kratos`` repository.  The catalog refers to
them with a ``Kratos`` prefix (``KratosFluidDynamicsApplication``,
``KratosStructuralMechanicsApplication``, ...).  Verification therefore
maps each catalog mention to the corresponding directory entry.

This is the source-enumeration variant of the source-grep family --
no Kratos install needed.  Directory listing is fetched via the
GitHub contents API (using ``gh auth token`` when available to avoid
the unauthenticated 60-req/h rate limit) and cached locally for 24h.

Returns ``None`` when the listing is unobtainable so the test skips
rather than fails in offline / rate-limited environments.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "KratosMultiphysics/Kratos"
_BRANCH = os.environ.get("KRATOS_BRANCH", "master")
_API_BASE = f"https://api.github.com/repos/{_REPO}"

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "open-fem-agent" / "kratos-source"
_CACHE_TTL_SECONDS = 24 * 3600


def _list_applications_local() -> list[str] | None:
    """List ``applications/`` subdirectories of ``$KRATOS_ROOT``."""
    root = os.environ.get("KRATOS_ROOT")
    if not root:
        return None
    apps_dir = Path(root) / "applications"
    if not apps_dir.is_dir():
        return None
    return sorted(
        p.name for p in apps_dir.iterdir()
        if p.is_dir() and p.name.endswith("Application")
    )


def _list_applications_cached() -> list[str] | None:
    cached = _CACHE_DIR / _BRANCH / "_listing_applications.json"
    if not cached.is_file():
        return None
    if (time.time() - cached.stat().st_mtime) > _CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(cached.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _list_applications_network() -> list[str] | None:
    """Fetch the ``applications/`` directory listing via the GitHub
    contents API.  Falls back to unauthenticated if ``gh`` is not
    available."""
    url = f"{_API_BASE}/contents/applications"
    token = ""
    try:
        token = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    if not isinstance(data, list):
        return None
    return sorted(
        e["name"] for e in data
        if isinstance(e, dict)
        and e.get("type") == "dir"
        and e.get("name", "").endswith("Application")
    )


def application_names() -> set[str] | None:
    """Set of valid ``Kratos<Name>Application`` strings as the catalog
    references them.  Returns ``None`` when the directory listing is
    unobtainable (no $KRATOS_ROOT and no network).

    Mapping: each ``<Name>Application`` directory in the upstream
    repo becomes ``Kratos<Name>Application`` in the catalog; the
    set returned here is already pre-prefixed for direct subset
    comparison with the catalog scan.
    """
    local = _list_applications_local()
    if local is not None:
        return {f"Kratos{name}" for name in local}
    cached = _list_applications_cached()
    if cached is not None:
        return {f"Kratos{name}" for name in cached}
    fetched = _list_applications_network()
    if fetched is None:
        return None
    cached_path = _CACHE_DIR / _BRANCH / "_listing_applications.json"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(json.dumps(fetched))
    return {f"Kratos{name}" for name in fetched}
