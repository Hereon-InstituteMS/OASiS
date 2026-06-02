"""Seed data/source_file_progress.json — the queue the cron walks.

For each backend with a source tree on disk (per discovery cache), walks
the tree and emits one entry per source file (.py / .cpp / .hh / .cc / .h
/ .feb / .yaml / .json / .prm / .cmake), sorted (subdir, name) within
backend.

Run once; subsequent ticks update entries in place. Re-running is safe
(merges by path) — preserves status of already-processed files.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from core.source_discovery import discover  # noqa: E402


_PROGRESS = _REPO / "data" / "source_file_progress.json"

# Extensions to walk per backend (the cron prompt's list).
_EXTS = {".py", ".cpp", ".hh", ".cc", ".h", ".hpp",
         ".feb", ".yaml", ".yml", ".json", ".prm", ".cmake"}

# Subdirs we always skip inside any backend source tree.
_PRUNE = {".git", "build", "build_logs", "node_modules", "__pycache__",
          ".venv", ".tox", "external_dependencies", "third_party",
          "dist", "bin", ".cache", ".mypy_cache",
          # cloned-but-not-source vendor dirs commonly present:
          "tinyxml", "Eigen", "boost", "metis",
          # large generated trees we shouldn't process:
          "build-cmake", "cmbuild", "Release", "Debug",
          "doc",  # docs aren't source
          "docs",
          "examples",  # we cover demos via upstream_demo_audit
          "tests",  # we cover tests separately
          # CI / packaging / version-control metadata:
          ".github", ".gitlab", ".circleci", ".vscode", ".idea",
          "bundled",  # vendored deps in some repos
          }


# Top-level filenames to auto-skip (CI / docs / packaging).
_SKIP_FILENAMES = {
    ".readthedocs.yaml", ".zenodo.json", "CITATION.cff",
    "environment.yml", "environment.yaml", "pyproject.toml",
    "setup.cfg", "MANIFEST.in", "tox.ini", "noxfile.py",
    ".pre-commit-config.yaml", ".clang-format", ".clang-tidy",
    "codecov.yml", ".gitignore", ".gitattributes",
}


def _walk_tree(root: Path, backend: str) -> list[dict]:
    out: list[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in _EXTS:
            continue
        parts_lower = {p.lower() for p in path.parts}
        if parts_lower & {p.lower() for p in _PRUNE}:
            continue
        if path.name in _SKIP_FILENAMES:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        try:
            lines = sum(1 for _ in path.open("rb"))
        except Exception:
            lines = 0
        out.append({
            "backend": backend,
            "path": str(rel),
            "abs_path": str(path),
            "ext": path.suffix,
            "lines": lines,
            "status": "pending",
            "lines_processed": 0,
            "symbols_found": 0,
            "symbols_documented_after": 0,
            "commit_sha": None,
            "processed_at": None,
        })
    # Sort (subdir, name) for deterministic rotation.
    out.sort(key=lambda e: (str(Path(e["path"]).parent), Path(e["path"]).name))
    return out


def main():
    res = discover(use_cache=True)
    existing: dict = {}
    if _PROGRESS.exists():
        try:
            existing_data = json.loads(_PROGRESS.read_text())
            for entry in existing_data.get("files", []):
                key = (entry["backend"], entry["path"])
                existing[key] = entry
        except Exception:
            pass

    all_files: list[dict] = []
    summary: dict[str, int] = {}
    for backend in ("skfem", "fenics", "ngsolve", "kratos", "dealii",
                    "fourc", "dune", "febio"):
        info = res.get(backend)
        if not info or not info.get("source_path"):
            print(f"  {backend}: no source on disk (skipping)")
            summary[backend] = 0
            continue
        root = Path(info["source_path"])
        entries = _walk_tree(root, backend)
        # Merge with existing (preserve processed/skip status).
        for entry in entries:
            key = (backend, entry["path"])
            if key in existing:
                old = existing[key]
                entry["status"] = old.get("status", "pending")
                entry["lines_processed"] = old.get("lines_processed", 0)
                entry["symbols_found"] = old.get("symbols_found", 0)
                entry["symbols_documented_after"] = old.get(
                    "symbols_documented_after", 0)
                entry["commit_sha"] = old.get("commit_sha")
                entry["processed_at"] = old.get("processed_at")
        all_files.extend(entries)
        summary[backend] = len(entries)
        print(f"  {backend}: {len(entries)} files (root: "
              f"{root.relative_to(Path.home()) if str(root).startswith(str(Path.home())) else root})")

    # Aggregate stats.
    processed = sum(1 for e in all_files if e["status"] == "processed")
    skipped = sum(1 for e in all_files if e["status"] == "skip")
    pending = sum(1 for e in all_files if e["status"] == "pending")
    in_progress = sum(1 for e in all_files
                      if e["status"] == "in_progress")

    out = {
        "summary": {
            "total_files": len(all_files),
            "processed": processed,
            "skipped": skipped,
            "pending": pending,
            "in_progress": in_progress,
            "by_backend": summary,
        },
        "files": all_files,
    }
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS.write_text(json.dumps(out, indent=1))
    print(f"\nTotal: {len(all_files)} files; "
          f"{processed} processed, {skipped} skipped, "
          f"{pending} pending")
    print(f"Wrote {_PROGRESS.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
