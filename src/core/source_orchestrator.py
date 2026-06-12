"""ensure_source orchestrator — single entry point for the MCP / cron.

For a given backend, returns the path to a usable source tree (cloning
if missing) and optionally triggers a build (compiling if no binary).

Pipeline:
  1. Run discover()
  2. If status == SOURCE_TREE or BOTH → already have source, return.
  3. If status == INSTALLED_BINARY or MISSING → fetch from canonical git.
  4. If require_binary=True and discovery shows no working binary →
     trigger build (background by default).

The caller (cron, MCP tool) gets back a structured result so it can
decide whether to proceed reading source, kick off a wait, or fail.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .source_discovery import (  # type: ignore
        discover, SourceStatus, _CACHE_PATH
    )
    from .source_fetch import fetch  # type: ignore
    from .source_build import build as _build  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.source_discovery import (  # type: ignore
        discover, SourceStatus, _CACHE_PATH
    )
    from core.source_fetch import fetch  # type: ignore
    from core.source_build import build as _build  # type: ignore


def ensure_source(backend: str,
                  fetch_if_missing: bool = True,
                  build_if_no_binary: bool = False,
                  background: bool = True,
                  wait_for_clone: bool = False,
                  ) -> dict:
    """Make sure `backend` has a usable source tree (and optionally binary).

    Returns:
      {
        "backend": str,
        "status": str,        # final status after orchestration
        "source_path": str|None,
        "binary_info": str|None,
        "actions": [           # what the orchestrator did
            {"step": "discover", "result": "..."},
            {"step": "fetch", "dest": "...", "pids": [...]},
            {"step": "build",  "log": "...",  "pid": ...},
        ],
      }
    """
    actions: list[dict] = []

    initial = discover(use_cache=True)
    info = initial.get(backend)
    if info is None:
        return {"backend": backend, "status": "unknown",
                "error": f"no spec for backend {backend!r}"}
    actions.append({"step": "discover", "result": info["status"],
                    "source_path": info["source_path"]})

    # Already have source — fast path.
    if info["status"] in (SourceStatus.SOURCE_TREE.value,
                          SourceStatus.BOTH.value):
        if (info["status"] == SourceStatus.BOTH.value
                or not build_if_no_binary):
            return {"backend": backend, "status": info["status"],
                    "source_path": info["source_path"],
                    "binary_info": info["binary_info"],
                    "actions": actions}
        # Have source but no binary, caller asked to build.
        log, proc = _build(backend, Path(info["source_path"]),
                           background=background)
        actions.append({"step": "build", "log": str(log),
                        "pid": proc.pid if proc else None})
        return {"backend": backend, "status": "building",
                "source_path": info["source_path"],
                "binary_info": None, "actions": actions}

    # Status is INSTALLED_BINARY or MISSING — no source on disk.
    if not fetch_if_missing:
        return {"backend": backend, "status": info["status"],
                "source_path": None,
                "binary_info": info["binary_info"], "actions": actions}

    dest, procs = fetch(backend, background=background,
                        shallow=True)
    actions.append({"step": "fetch", "dest": str(dest),
                    "pids": [p.pid for p in procs]})

    if wait_for_clone and procs:
        for p in procs:
            p.wait()
        actions.append({"step": "wait_clone", "result": "all done"})
        # Force a cache-bypassing rescan.
        if _CACHE_PATH.exists():
            try:
                _CACHE_PATH.unlink()
            except Exception:
                pass
        final = discover(use_cache=False)
        info2 = final.get(backend, {})
        actions.append({"step": "rediscover",
                        "result": info2.get("status")})
        if (info2.get("status") in (SourceStatus.SOURCE_TREE.value,
                                    SourceStatus.BOTH.value)
                and build_if_no_binary
                and info2.get("status") != SourceStatus.BOTH.value):
            log, proc = _build(backend, Path(info2["source_path"]),
                               background=background)
            actions.append({"step": "build", "log": str(log),
                            "pid": proc.pid if proc else None})
        return {"backend": backend, "status": info2.get("status", "?"),
                "source_path": info2.get("source_path"),
                "binary_info": info2.get("binary_info"),
                "actions": actions}

    return {"backend": backend, "status": "fetching",
            "source_path": str(dest),
            "binary_info": info.get("binary_info"),
            "actions": actions}


def ensure_all(fetch_if_missing: bool = True,
               build_if_no_binary: bool = False,
               background: bool = True) -> dict:
    """ensure_source for every registered backend."""
    out = {}
    for be in ("skfem", "fenics", "ngsolve", "kratos", "dealii",
               "fourc", "dune", "febio"):
        out[be] = ensure_source(be, fetch_if_missing=fetch_if_missing,
                                build_if_no_binary=build_if_no_binary,
                                background=background)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Source orchestrator")
    ap.add_argument("--backend", help="single backend, or omit for all")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--build", action="store_true",
                    help="Trigger build if no binary present")
    ap.add_argument("--foreground", action="store_true",
                    help="Block while clones/builds run (default: background)")
    ap.add_argument("--wait-clone", action="store_true",
                    help="Wait for clone before reporting result")
    args = ap.parse_args()

    if args.backend:
        r = ensure_source(args.backend,
                          fetch_if_missing=not args.no_fetch,
                          build_if_no_binary=args.build,
                          background=not args.foreground,
                          wait_for_clone=args.wait_clone)
        print(json.dumps(r, indent=2))
    else:
        r = ensure_all(fetch_if_missing=not args.no_fetch,
                       build_if_no_binary=args.build,
                       background=not args.foreground)
        print(json.dumps(r, indent=2))
