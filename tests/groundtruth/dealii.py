"""Ground-truth probes for deal.II.

deal.II is a C++ FEM framework whose canonical finite-element types
live as ``class FE_<Name>`` declarations in headers under
``include/deal.II/fe/`` of the dealii/dealii repository.  Each
header declares one concrete element type; the catalog (see
``src/backends/dealii/generators/``) references these names directly
in Python template strings the agent will execute.

This module is a second instance of the *source-grep* probe family
(``fourc.py`` was the first).  Resolution order: local checkout
(``$DEALII_ROOT``) -> on-disk cache -> network.  Returns ``None`` if
both local and network are unavailable so the test skips gracefully.

The cache lives at ``$XDG_CACHE_HOME/open-fem-agent/dealii-source/``
with a 24h TTL.  Headers are tiny so the full enumeration of
``FE_*`` classes is roughly half a megabyte over the wire on a cold
cache; subsequent runs within 24h are essentially free.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Default upstream when no local checkout is configured.
_REPO = "dealii/dealii"
_BRANCH = os.environ.get("DEALII_BRANCH", "master")
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}"
_API_BASE = f"https://api.github.com/repos/{_REPO}"

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "open-fem-agent" / "dealii-source"
_CACHE_TTL_SECONDS = 24 * 3600

# The MCP catalog only references concrete finite-element classes
# (``FE_Q``, ``FE_DGQ``, ...).  The base classes ``FE_Base`` and
# ``FE_Data`` are infrastructure and never appear in the catalog;
# excluding them makes the "no unknown class" check clearer.
_INFRASTRUCTURE_HEADERS = {
    "fe_base.h",
    "fe_data.h",
    "fe_coupling_values.h",
}


def _read_local(rel_path: str) -> str | None:
    root = os.environ.get("DEALII_ROOT")
    if not root:
        return None
    candidate = Path(root) / rel_path
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace")


def _read_cached(rel_path: str) -> str | None:
    cached = _CACHE_DIR / _BRANCH / rel_path
    if not cached.is_file():
        return None
    if (time.time() - cached.stat().st_mtime) > _CACHE_TTL_SECONDS:
        return None
    return cached.read_text(encoding="utf-8", errors="replace")


def _write_cache(rel_path: str, content: str) -> None:
    cached = _CACHE_DIR / _BRANCH / rel_path
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content, encoding="utf-8")


def _read_network(rel_path: str) -> str | None:
    url = f"{_RAW_BASE}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def fetch_source(rel_path: str) -> str | None:
    """Return the contents of a deal.II source file, or ``None`` if
    unavailable.

    Resolution order: local checkout (``$DEALII_ROOT``) -> on-disk
    cache -> network.  Network responses are written to the cache for
    subsequent calls; the value returned on the network branch is the
    freshly-fetched content.
    """
    local = _read_local(rel_path)
    if local is not None:
        return local
    cached = _read_cached(rel_path)
    if cached is not None:
        return cached
    fetched = _read_network(rel_path)
    if fetched is not None:
        _write_cache(rel_path, fetched)
    return fetched


def _list_fe_headers_local() -> list[str] | None:
    """List ``fe_*.h`` filenames under ``$DEALII_ROOT/include/deal.II/fe``."""
    root = os.environ.get("DEALII_ROOT")
    if not root:
        return None
    fe_dir = Path(root) / "include" / "deal.II" / "fe"
    if not fe_dir.is_dir():
        return None
    return sorted(
        p.name for p in fe_dir.glob("fe_*.h")
        if ".templates" not in p.name
    )


def _list_fe_headers_cached() -> list[str] | None:
    """Cached directory listing -- saved as JSON so we do not refetch
    the listing on every test invocation."""
    cached = _CACHE_DIR / _BRANCH / "_listing_fe.json"
    if not cached.is_file():
        return None
    if (time.time() - cached.stat().st_mtime) > _CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(cached.read_text())
    except json.JSONDecodeError:
        return None


def _list_fe_headers_network() -> list[str] | None:
    """Fetch the ``include/deal.II/fe/`` directory listing via the
    GitHub contents API.  Uses ``gh auth token`` when available to
    avoid the 60 req/hr unauthenticated limit; falls back to
    unauthenticated otherwise.
    """
    url = f"{_API_BASE}/contents/include/deal.II/fe"
    token = ""
    try:
        token = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
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
        and e.get("name", "").startswith("fe_")
        and e.get("name", "").endswith(".h")
        and ".templates" not in e.get("name", "")
    )


def list_fe_headers() -> list[str] | None:
    """Return ``fe_*.h`` filenames under ``include/deal.II/fe/`` (concrete
    finite-element headers only -- excludes templates and infra)."""
    local = _list_fe_headers_local()
    if local is not None:
        return [h for h in local if h not in _INFRASTRUCTURE_HEADERS]
    cached = _list_fe_headers_cached()
    if cached is not None:
        return [h for h in cached if h not in _INFRASTRUCTURE_HEADERS]
    fetched = _list_fe_headers_network()
    if fetched is None:
        return None
    cached_path = _CACHE_DIR / _BRANCH / "_listing_fe.json"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(json.dumps(fetched))
    return [h for h in fetched if h not in _INFRASTRUCTURE_HEADERS]


_CLASS_RE = re.compile(r"class\s+FE_([A-Z][A-Za-z0-9_]*)\b")


def fe_class_names() -> set[str] | None:
    """Set of concrete ``FE_*`` class names declared under
    ``include/deal.II/fe/fe_*.h``.

    Strategy: list the directory, fetch each header (cached for 24h),
    grep for ``class FE_<Name>``.  Returns ``None`` if the listing
    cannot be obtained (no $DEALII_ROOT and no network); returns an
    empty set if the listing works but no headers parsed (probable
    code change, the test will surface it).
    """
    headers = list_fe_headers()
    if headers is None:
        return None
    classes: set[str] = set()
    for h in headers:
        content = fetch_source(f"include/deal.II/fe/{h}")
        if content is None:
            continue
        for m in _CLASS_RE.finditer(content):
            classes.add(f"FE_{m.group(1)}")
    return classes
