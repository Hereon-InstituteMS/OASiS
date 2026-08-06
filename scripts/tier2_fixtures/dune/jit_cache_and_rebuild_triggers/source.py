"""Tier-2: where the JIT cache is, what re-triggers a C++ build, and
what does not.

  poisson#1                 a first run prints 'DUNE-INFO: Compiling
                            <X> (new)' and then goes quiet for tens of
                            seconds; the cache is
                            <sys.prefix>/.cache/dune-py, NOT ~/.dune.
  poisson#2                 changing a bare float literal inside the
                            form costs a build and adds a .so;
                            re-running the identical form and changing
                            a dune.ufl.Constant's .value are free.
  poisson#6                 dune.ufl.Constant is the runtime-updatable
                            one; plain ufl.Constant needs a domain.
  poisson#15                the cache path again, plus the location of
                            the compiled modules.
  poisson_mms#4             each distinct (grid dimension, order, form)
                            compiles its own module; the grid RESOLUTION
                            is not part of the generated code, so inside
                            one refinement loop only the first level
                            pays.
  heat#2                    first step slow, later steps fast; if every
                            step is slow the form is regenerating.
  time_dependent_heat#1     re-creating the scheme every step instead of
                            reusing it is far slower per step.

Cost: observing a COLD compile needs a form the cache has never seen,
so the subprocess below mints one per run from the wall clock. That is
one C++ build (~22 s measured) and one extra .so in the cache per run.
Everything else is warm.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.fem.scheme import galerkin                           # noqa: E402
from dune.ufl import Constant, DirichletBC                     # noqa: E402
from dune.common.module import getDunePyDir                    # noqa: E402
from ufl import (TrialFunction, TestFunction,                   # noqa: E402
                 dot, grad, dx)

_GENERATED = "python/dune/generated"

# Runs a form the cache has never seen at three resolutions. The FORM
# text is identical across the three, so if the grid resolution is
# really absent from the generated code, only the first level compiles.
_COLD_SCRIPT = """
import time, warnings
warnings.filterwarnings("ignore")
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
for nx in (4, 8, 16):
    gv = structuredGrid([0, 0], [1, 1], [nx, nx])
    sp = lagrange(gv, order=1)
    u, v = TrialFunction(sp), TestFunction(sp)
    t = time.time()
    s = galerkin([dot(grad(u), grad(v)) * dx == {lit!r} * v * dx,
                  DirichletBC(sp, 0)], solver="cg")
    uh = sp.interpolate(0, name="uh")
    info = s.solve(target=uh)
    print("cold_level_%d_seconds=%.3f" % (nx, time.time() - t),
          flush=True)
    assert info["converged"]
print("COLD_RUN_FINISHED")
"""


def _n_so(cache: Path) -> int:
    d = cache / _GENERATED
    return len(list(d.glob("*.so"))) if d.is_dir() else -1


def main() -> int:
    fail: list[str] = []

    # ── 1. Where the cache is (poisson#1, poisson#15) ────────────────
    cache = Path(getDunePyDir())
    print(f"cache_dir_ends_with={'/'.join(cache.parts[-2:])}")
    print(f"cache_is_under_sys_prefix="
          f"{str(cache).startswith(sys.prefix)}")
    print(f"generated_so_dir_exists={(cache / _GENERATED).is_dir()}")
    print(f"home_dot_dune_exists={(Path.home() / '.dune').exists()}")
    if cache.name != "dune-py" or cache.parent.name != ".cache":
        fail.append(f"getDunePyDir() returned {cache}, which is not a "
                    f".cache/dune-py path; the documented locations are "
                    f"<sys.prefix>/.cache/dune-py or ~/.cache/dune-py")
    if not (cache / _GENERATED).is_dir():
        fail.append(f"the compiled modules are not under "
                    f"{cache / _GENERATED}")
    if (Path.home() / ".dune").exists():
        fail.append("~/.dune exists on this host, so the fixture cannot "
                    "support the claim that it is not the cache")

    # ── 2. Cold build: announced, costly, and paid ONCE for the whole
    #      refinement loop (poisson#1, poisson_mms#4, heat#2) ─────────
    n_before = _n_so(cache)
    lit = 1.0 + (time.time() % 1000.0) / 7.0        # never seen before
    env = dict(os.environ)
    env.setdefault("CONDA_DEFAULT_ENV", "dune-fem-env")
    proc = subprocess.run(
        [sys.executable, "-c", _COLD_SCRIPT.format(lit=lit)],
        capture_output=True, text=True, timeout=1200, env=env,
        cwd=str(Path(__file__).parent))
    blob = (proc.stdout or "") + (proc.stderr or "")
    n_after = _n_so(cache)
    levels = {int(m.group(1)): float(m.group(2)) for m in
              re.finditer(r"cold_level_(\d+)_seconds=([0-9.]+)", blob)}
    print(f"cold_run_finished={'COLD_RUN_FINISHED' in blob}")
    print(f"cold_run_announced_a_compile="
          f"{'Compiling' in blob and '(new)' in blob}")
    print(f"cold_level_times={sorted(levels.items())}")
    print(f"cold_run_added_so_files={n_after - n_before}")
    if "COLD_RUN_FINISHED" not in blob:
        fail.append(f"the cold subprocess did not finish: "
                    f"{blob[-400:]!r}")
    if not ("Compiling" in blob and "(new)" in blob):
        fail.append(f"a never-before-seen form did NOT print a "
                    f"'Compiling ... (new)' line; captured: "
                    f"{blob[:400]!r}")
    if sorted(levels) != [4, 8, 16]:
        fail.append(f"the cold run reported levels {sorted(levels)}, "
                    f"expected [4, 8, 16]")
    else:
        first, rest = levels[4], [levels[8], levels[16]]
        print(f"first_level_pays_the_compile={first > 5.0}")
        print(f"later_levels_are_free={max(rest) < 1.0}")
        if first <= 5.0:
            fail.append(f"the first level of a fresh form took only "
                        f"{first:.3f} s; the claim is tens of seconds "
                        f"of C++ build")
        if max(rest) >= 1.0:
            fail.append(f"later refinement levels of the SAME form took "
                        f"{rest} s; the claim is that grid resolution is "
                        f"not part of the generated code, so they are "
                        f"free")
    if n_after - n_before < 1:
        fail.append(f"the cold run added {n_after - n_before} .so files; "
                    f"a never-before-seen float literal inside the form "
                    f"is supposed to cost a new compiled module")

    # ── 3. What is free in-process (poisson#2, poisson#6) ────────────
    # Everything below is built ONCE and reused; see the fixture
    # _comment for why re-creating spaces in a loop is a trap of its own.
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    c = Constant(1.0, name="src")
    a = dot(grad(u), grad(v)) * dx
    dbc = DirichletBC(space, 0)

    n0 = _n_so(cache)
    scheme = galerkin([a == c * v * dx, dbc], solver="cg")
    uh = space.interpolate(0, name="uh")
    scheme.solve(target=uh)
    n1 = _n_so(cache)

    c.value = 5.0
    t = time.time()
    info2 = scheme.solve(target=uh)
    constant_update = time.time() - t
    n2 = _n_so(cache)
    print(f"constant_value_change_seconds={constant_update:.4f}")
    print(f"constant_value_change_added_so={n2 - n1}")
    print(f"constant_value_change_is_free="
          f"{n2 == n1 and constant_update < 1.0}")
    if n2 != n1:
        fail.append("changing dune.ufl.Constant.value added a compiled "
                    "module; the claim is that it reuses the existing "
                    "one")
    if not info2["converged"]:
        fail.append("the re-solve after a Constant update did not "
                    "converge")

    t = time.time()
    scheme_again = galerkin([a == c * v * dx, dbc], solver="cg")
    scheme_again.solve(target=uh)
    identical_rebuild = time.time() - t
    n3 = _n_so(cache)
    print(f"identical_form_rebuild_seconds={identical_rebuild:.4f}")
    print(f"identical_form_added_so={n3 - n2}")
    print(f"identical_form_is_free="
          f"{n3 == n2 and identical_rebuild < 1.0}")
    if n3 != n2:
        fail.append("re-building the identical form added a compiled "
                    "module")

    # A bare float literal, by contrast, is part of the generated code.
    # Proven from the cold run above: the same three levels shared ONE
    # module, and that module existed only because the literal was new.
    print(f"float_literal_is_part_of_the_code={n_after - n_before >= 1}")

    # ── 4. Constant vs ufl.Constant (poisson#6) ─────────────────────
    import ufl
    try:
        ufl.Constant(1.0)
        print("plain_ufl_constant_rejected=False")
        fail.append("ufl.Constant(1.0) with no domain was accepted; the "
                    "claim is that it needs a domain and that "
                    "dune.ufl.Constant is the one taking a bare value")
    except Exception as exc:                                 # noqa: BLE001
        print(f"plain_ufl_constant_rejected={type(exc).__name__}")
    print(f"dune_constant_takes_a_bare_value={float(c.value) == 5.0}")

    # ── 5. Reuse the scheme across steps, or pay per step
    #      (time_dependent_heat#1, heat#2) ───────────────────────────
    steps = 20
    t = time.time()
    for k in range(steps):
        c.value = 1.0 + k * 0.01
        scheme.solve(target=uh)
    reuse = time.time() - t
    t = time.time()
    for k in range(steps):
        c.value = 1.0 + k * 0.01
        galerkin([a == c * v * dx, dbc], solver="cg").solve(target=uh)
    rebuild = time.time() - t
    ratio = rebuild / reuse if reuse > 0 else float("inf")
    print(f"reuse_scheme_seconds_for_{steps}_steps={reuse:.4f}")
    print(f"rebuild_scheme_seconds_for_{steps}_steps={rebuild:.4f}")
    print(f"rebuild_over_reuse_ratio={ratio:.1f}")
    print(f"rebuilding_the_scheme_costs_more={ratio > 5.0}")
    if ratio <= 5.0:
        fail.append(f"re-creating the scheme each step cost only "
                    f"{ratio:.1f}x reusing it; the claim is an "
                    f"order-of-magnitude penalty. Note this is measured "
                    f"WITHOUT a re-JIT — the form text never changes — "
                    f"so it is the floor of the penalty, not the "
                    f"ceiling")

    if not fail:
        print("dune_jit_cache_and_rebuild_triggers_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
