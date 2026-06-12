#!/usr/bin/env python3
"""
Comprehensive enumeration of every (backend, physics, variant) tuple the
MCP catalog advertises, and a single-row pass/fail/run-result per tuple.

Where the hand-curated `sweep_layer3.py` matrix exercises a few canonical
cells per physics row with care (analytic refs, cross-backend consistency
checks), this script *exhaustively* probes every template the backends
expose.  The goal is a master "what works / what doesn't" table that
guides the multi-week fix-and-encode pipeline.

Three stages per cell:

  1. `generate_input(physics, variant, params)` — does the catalog
     even contain a working template at this key?
  2. `validate_input(content)` — does the produced text satisfy the
     backend's own validation rule (e.g. Kratos requires the script
     to import KratosMultiphysics)?
  3. `backend.run(content, work_dir, np=1, timeout=…)` — does it
     actually execute, and does it write a .vtu?

Outcomes are written to `benchmarks/probe_results/templates.json` and
a rendered Markdown table at `benchmarks/probe_results/templates.md`.

Usage:
    python benchmarks/probe_all_templates.py [--stage 1|2|3] [--backend NAME] [--timeout SEC]

`--stage 1` only walks generate_input (fastest, ~1 s).
`--stage 2` adds validate_input.
`--stage 3` (default) adds backend.run() with a short timeout per cell.

The cell timeout (`--timeout`, default 120 s) caps any single backend
invocation so one hanging template cannot block the whole sweep.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))


# ── env discovery ────────────────────────────────────────────────────────
def _discover_env():
    home = Path.home()
    # Resolve deal.II via the backend's own version-aware resolver
    # instead of a hardcoded env name. The old hardcoded
    # ~/miniconda3/envs/ofa-dealii kept pinning every compile to a
    # stale 9.1.1 env even after a newer 9.3.2 env existed and the
    # backend itself had moved on (probe 2026-06-12: 10 templates
    # "failed" purely from compiling against the old headers the
    # probe forced via DEAL_II_DIR/CMAKE_PREFIX_PATH).
    dealii_env_root = None
    try:
        from backends.dealii.backend import _find_dealii
        dealii_env_root = _find_dealii()
    except ImportError:
        pass
    if dealii_env_root is None:
        fallback = home / "miniconda3/envs/ofa-dealii"
        dealii_env_root = fallback if fallback.is_dir() else None
    if "DEAL_II_DIR" not in os.environ and dealii_env_root is not None:
        for c in [
            dealii_env_root / "lib/cmake/deal.II",
            Path("/usr/lib/x86_64-linux-gnu/cmake/deal.II"),
        ]:
            if c.is_dir():
                os.environ["DEAL_II_DIR"] = str(c)
                break
    if dealii_env_root is not None and dealii_env_root.is_dir():
        existing = os.environ.get("CMAKE_PREFIX_PATH", "")
        prefix_parts = [p for p in existing.split(os.pathsep) if p]
        if str(dealii_env_root) not in prefix_parts:
            prefix_parts.append(str(dealii_env_root))
            os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefix_parts)
    import shutil as _sh
    if "CXX" not in os.environ:
        for cxx in ("g++", "c++", "clang++"):
            if _sh.which(cxx):
                os.environ["CXX"] = cxx
                break
    if "CC" not in os.environ:
        for cc in ("gcc", "cc", "clang"):
            if _sh.which(cc):
                os.environ["CC"] = cc
                break
    if "FOURC_BINARY" not in os.environ:
        for c in [
            home / "Schreibtisch/4C-src/4C/build/4C",
            home / "4C/build/4C",
        ]:
            if c.is_file():
                os.environ["FOURC_BINARY"] = str(c)
                os.environ.setdefault("FOURC_ROOT", str(c.parent.parent))
                break


# ─────────────────────────────────────────────────────────────────────────
#   TEMPLATE ENUMERATION
# ─────────────────────────────────────────────────────────────────────────
# Backends expose their (physics, variant) templates in heterogeneous
# ways:
#   - skfem, ngsolve, kratos, dune    have a `GENERATORS: dict[str,callable]`
#                                     at backends.<name>.generators with
#                                     keys like "poisson_2d",
#                                     "linear_elasticity_2d_nonlinear".
#   - fourc                            has `list_generators()` in the same
#                                     module.
#   - fenics, dealii                   do not export a uniform registry;
#                                     we recover the key list by probing
#                                     `generate_input(physics, '__bogus__', {})`
#                                     for every physics name the backend
#                                     advertises and parsing the
#                                     `Available: [...]` portion of the
#                                     resulting error.
#
# In every case the key format is roughly `<physics>_<variant>` though
# the boundary between the two is fuzzy — for the purposes of this
# probe we keep the raw key and an explicit (physics, variant) parsing
# only when the backend itself uses that split in `generate_input`.


@dataclass
class Template:
    backend: str
    key: str  # the registry key (e.g. "poisson_2d")
    physics: str  # best-effort parse
    variant: str
    default_params: dict = field(default_factory=dict)


def _split_key(key: str, physics_candidates: list[str]) -> tuple[str, str]:
    """Best-effort split of a `<physics>_<variant>` key.

    `physics_candidates` is the list of physics names the backend
    advertises (via `b.supported_physics()`).  Longest match wins so
    `linear_elasticity_2d_nonlinear` splits as
    `('linear_elasticity', '2d_nonlinear')`, not
    `('linear', 'elasticity_2d_nonlinear')`.
    """
    for p in sorted(physics_candidates, key=len, reverse=True):
        if key == p:
            return p, ""
        prefix = p + "_"
        if key.startswith(prefix):
            return p, key[len(prefix):]
    # Fallback: split on last underscore-then-suffix-of-digits/letters.
    m = re.match(r"^(.+?)_([0-9].*)$", key)
    if m:
        return m.group(1), m.group(2)
    return key, ""


def enumerate_templates() -> list[Template]:
    from core.registry import load_all_backends, get_backend, available_backends
    load_all_backends()

    out: list[Template] = []

    for be in available_backends():
        name = be.name()
        phys_names = [p.name for p in be.supported_physics()]
        keys: list[str] = []

        # 1. uniform GENERATORS dict
        try:
            mod = __import__(f"backends.{name}.generators", fromlist=["GENERATORS"])
            if hasattr(mod, "GENERATORS") and isinstance(mod.GENERATORS, dict):
                keys.extend(mod.GENERATORS.keys())
        except ModuleNotFoundError:
            pass

        # 2. 4C uses a *two-level* schema (module → variants).  Each
        #    module returned by list_generators() has its own
        #    list_variants() that yields per-variant metadata dicts.
        #    Build (module, variant_name) Templates directly here and
        #    skip the generic key-split path below — but only if the
        #    expected pair of helpers (list_generators + get_generator)
        #    is actually present.  If the import succeeds but either
        #    helper is missing (e.g. an upstream rename), fall through
        #    to the generic key-split paths below rather than silently
        #    enumerating zero templates.
        if not keys and name == "fourc":
            try:
                mod = __import__(
                    f"backends.{name}.generators",
                    fromlist=["list_generators", "get_generator"],
                )
            except ModuleNotFoundError:
                mod = None
            if mod is not None and (
                hasattr(mod, "list_generators")
                and hasattr(mod, "get_generator")
            ):
                for module_key in mod.list_generators():
                    gen = mod.get_generator(module_key)
                    if not hasattr(gen, "list_variants"):
                        continue
                    for v in gen.list_variants():
                        vname = v["name"] if isinstance(v, dict) else str(v)
                        out.append(Template(
                            backend=name,
                            key=f"{module_key}/{vname}",
                            physics=module_key,
                            variant=vname,
                        ))
                continue  # 4C done — don't fall through

        # 3. probe-by-error for backends without a uniform registry.
        #    Two distinct error formats exist:
        #      (a) fenics-style "per-physics" — variants are physics-local
        #          (e.g. ['2d', '3d', 'l_domain']); prepend `physics_`.
        #      (b) dealii-style "global" — every variant string is already
        #          a full <physics>_<variant> key (e.g.
        #          'linear_elasticity_2d', 'poisson_2d', 'stokes_2d');
        #          do NOT prepend — use as-is.
        #    Detect (b) by checking whether any returned variant string
        #    starts with any *other* physics name from `phys_names`.
        if not keys:
            for p in phys_names:
                try:
                    be.generate_input(p, "__OFA_PROBE_BOGUS__", {})
                except Exception as e:
                    msg = str(e)
                    m = re.search(r"Available:\s*\[([^\]]+)\]", msg)
                    if m:
                        variants = re.findall(r"'([^']+)'", m.group(1))
                        # Heuristic: if a returned variant starts with some
                        # OTHER physics name from this backend's catalog,
                        # then the error format is global (full keys, no
                        # prepend).  Otherwise it is per-physics.
                        other_phys = [q for q in phys_names if q != p]
                        is_global = any(
                            any(v == q or v.startswith(q + "_")
                                for q in other_phys)
                            for v in variants
                        )
                        if is_global:
                            keys.extend(variants)
                            break  # one probe is enough — every physics
                                   # returns the same global list
                        else:
                            for v in variants:
                                keys.append(f"{p}_{v}" if v else p)

        # de-dup keys
        keys = sorted(set(keys))

        for k in keys:
            phys, var = _split_key(k, phys_names)
            out.append(Template(backend=name, key=k, physics=phys, variant=var))

    return out


# ─────────────────────────────────────────────────────────────────────────
#   CELL RUNNER
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    backend: str
    key: str
    physics: str
    variant: str
    gen_ok: bool = False
    validate_ok: bool | None = None  # None = stage <2
    run_status: str | None = None  # None = stage <3
    has_vtu: bool | None = None
    vtu_fields: list[str] | None = None
    elapsed_s: float | None = None
    error: str | None = None


_DEFAULT_PARAMS = {
    # generic fall-backs covering the most common kw-args
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0,
    "Re": 100, "mu": 1.0,
    "refinements": 4,
    "T_end": 0.01, "dt": 0.001,
}


async def probe_one(template: Template, stage: int, timeout: int,
                    work_root: Path) -> Result:
    from core.registry import get_backend
    r = Result(backend=template.backend, key=template.key,
               physics=template.physics, variant=template.variant)
    b = get_backend(template.backend)
    if b is None:
        r.error = "no backend"
        return r

    # ── stage 1: generate
    try:
        content = b.generate_input(template.physics, template.variant, _DEFAULT_PARAMS)
        r.gen_ok = True
    except Exception as e:
        r.error = f"{type(e).__name__}: {e!s:.200}"
        return r

    if stage < 2:
        return r

    # ── stage 2: validate
    try:
        errs = b.validate_input(content)
        r.validate_ok = not errs
        if errs:
            r.error = ("; ".join(errs))[:300]
    except Exception as e:
        r.validate_ok = False
        r.error = f"validate {type(e).__name__}: {e!s:.200}"
        return r

    if stage < 3:
        return r
    if not r.validate_ok:
        return r

    # ── stage 3: run
    wd = work_root / template.backend / template.key.replace("/", "_")
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        job = await b.run(content, wd, np=1, timeout=timeout)
    except Exception as e:
        r.run_status = "exception"
        r.elapsed_s = time.time() - t0
        r.error = f"run {type(e).__name__}: {e!s:.200}"
        return r
    r.elapsed_s = time.time() - t0
    r.run_status = job.status
    if job.status == "completed":
        vtus = [f for f in b.get_result_files(job) if f.suffix == ".vtu"]
        r.has_vtu = bool(vtus)
        if vtus:
            try:
                import pyvista as pv
                m = pv.read(str(sorted(vtus)[-1]))
                r.vtu_fields = list(m.point_data.keys())
            except Exception as e:
                r.error = f"vtu-read {type(e).__name__}: {e!s:.100}"
    else:
        r.error = (job.error or "")[:300]
    return r


# ─────────────────────────────────────────────────────────────────────────
#   DRIVER + REPORTING
# ─────────────────────────────────────────────────────────────────────────


def render_markdown(results: list[Result]) -> str:
    by_backend: dict[str, list[Result]] = {}
    for r in results:
        by_backend.setdefault(r.backend, []).append(r)

    lines: list[str] = []
    lines.append("# Probe of every catalog template\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on this machine._\n")
    # Count rules: each stage's column is `True` strictly when the
    # check actually ran AND passed.  A cell whose stage was skipped
    # (e.g. validate_ok is None on `--stage 1`) is NOT counted as a
    # success — that would let the summary line claim "0 validate-OK"
    # for a stage-1 run and look like every template just failed
    # validation, when in fact validation was never performed.  We
    # therefore also report the per-stage *skipped* count when nonzero
    # so the reader can tell "not run" from "ran and failed".
    total = len(results)
    g = sum(1 for r in results if r.gen_ok)
    v_ok = sum(1 for r in results if r.validate_ok is True)
    v_skipped = sum(1 for r in results if r.validate_ok is None)
    c = sum(1 for r in results if r.run_status == "completed")
    r_skipped = sum(1 for r in results if r.run_status is None)
    vtu = sum(1 for r in results if r.has_vtu)
    vtu_skipped = sum(1 for r in results if r.has_vtu is None)

    def _fmt(label, ok, skipped):
        if skipped:
            return f"{ok} {label} ({skipped} not probed at this stage)"
        return f"{ok} {label}"

    lines.append(
        "**Summary** — "
        + f"{total} templates probed: "
        + f"{g} generate-OK, "
        + _fmt("validate-OK", v_ok, v_skipped)
        + ", "
        + _fmt("run-completed", c, r_skipped)
        + ", "
        + _fmt("produced a VTU", vtu, vtu_skipped)
        + ".\n"
    )
    for backend, rs in sorted(by_backend.items()):
        bg = sum(1 for r in rs if r.gen_ok)
        bv = sum(1 for r in rs if r.validate_ok is True)
        bc = sum(1 for r in rs if r.run_status == "completed")
        bvtu = sum(1 for r in rs if r.has_vtu)
        lines.append(f"\n## {backend} — {len(rs)} templates "
                     f"({bg} gen, {bv} val, {bc} run, {bvtu} vtu)\n")
        lines.append("| key | physics | variant | gen | val | run | vtu | fields | time | error |")
        lines.append("|-----|---------|---------|-----|-----|-----|-----|--------|------|-------|")
        for r in sorted(rs, key=lambda x: x.key):
            gen = "✓" if r.gen_ok else "✗"
            val = "—" if r.validate_ok is None else ("✓" if r.validate_ok else "✗")
            run = "—" if r.run_status is None else r.run_status
            vt = "—" if r.has_vtu is None else ("✓" if r.has_vtu else "✗")
            fields = ", ".join((r.vtu_fields or [])[:4])
            tm = "" if r.elapsed_s is None else f"{r.elapsed_s:.1f}s"
            err = (r.error or "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| `{r.key}` | {r.physics} | {r.variant} | {gen} | {val} | "
                f"{run} | {vt} | {fields} | {tm} | {err} |"
            )
    return "\n".join(lines)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=3,
                   help="1=generate only, 2=+validate, 3=+run (default 3)")
    p.add_argument("--backend", default="",
                   help="restrict to one backend name (e.g. skfem)")
    p.add_argument("--timeout", type=int, default=120,
                   help="per-cell run timeout in seconds (default 120)")
    args = p.parse_args()

    _discover_env()

    results_dir = REPO_ROOT / "benchmarks" / "probe_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_root = results_dir / "work"

    print(f"stage={args.stage}, backend={args.backend or 'ALL'}, timeout={args.timeout}s")
    templates = enumerate_templates()
    if args.backend:
        templates = [t for t in templates if t.backend == args.backend]
    print(f"enumerated {len(templates)} templates")

    results: list[Result] = []
    t_start = time.time()
    for i, tmpl in enumerate(templates, 1):
        r = await probe_one(tmpl, args.stage, args.timeout, work_root)
        results.append(r)
        gen = "g" if r.gen_ok else "."
        val = "v" if r.validate_ok else ("." if r.validate_ok is False else "?")
        run = (r.run_status or "?")[:8]
        vtu = "V" if r.has_vtu else ("." if r.has_vtu is False else "?")
        print(f"  [{i:3d}/{len(templates)}] {tmpl.backend:8s} {tmpl.key:40s} "
              f"{gen}{val} {run:8s} {vtu}")
    elapsed = time.time() - t_start
    print(f"\ntotal elapsed: {elapsed:.1f}s")

    # Merge with prior rows instead of clobbering: a per-backend run
    # (--backend fourc) used to overwrite the WHOLE templates.json,
    # silently destroying every other backend's most recent probe
    # results (bitten twice 2026-06-12: a fourc run erased the dealii
    # rows that had just taken 6.5 minutes to compile). Rows are keyed
    # by (backend, key); rows re-probed in THIS run replace their old
    # version, all other backends' rows survive.
    json_path = results_dir / "templates.json"
    new_rows = [asdict(r) for r in results]
    merged: dict[tuple, dict] = {}
    if json_path.exists():
        try:
            for row in json.loads(json_path.read_text()):
                merged[(row.get("backend"), row.get("key"))] = row
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt prior file — start fresh
    for row in new_rows:
        merged[(row["backend"], row["key"])] = row
    all_rows = sorted(merged.values(),
                      key=lambda r: (r["backend"], r["key"]))
    json_path.write_text(json.dumps(all_rows, indent=2, default=str))
    # Markdown table renders the full merged picture, not just this
    # run's slice.
    from dataclasses import fields as dc_fields
    field_names = {f.name for f in dc_fields(Result)}
    merged_results = [
        Result(**{k: v for k, v in row.items() if k in field_names})
        for row in all_rows
    ]
    md = render_markdown(merged_results)
    (results_dir / "templates.md").write_text(md)
    print(f"JSON: {json_path}  ({len(new_rows)} probed this run, "
          f"{len(all_rows)} total rows after merge)")
    print(f"  MD: {results_dir / 'templates.md'}")

    # ── final one-line summary
    total = len(results)
    # Same "True strictly, not falsy" rule as the markdown renderer
    # so `--stage 1` doesn't appear to report "0 validated" when in
    # fact validation was simply not performed.
    g = sum(1 for r in results if r.gen_ok)
    v = sum(1 for r in results if r.validate_ok is True)
    c = sum(1 for r in results if r.run_status == "completed")
    vtu = sum(1 for r in results if r.has_vtu)
    print(f"\nSUMMARY  total={total}  gen={g}  val={v}  run={c}  vtu={vtu}")


if __name__ == "__main__":
    asyncio.run(main())
