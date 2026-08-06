"""Shared machinery for the COUPLING tier-2 fixtures.

Why these fixtures exist
------------------------
The tier-2 suite had 108 fixtures and not one of them exercised coupling. Every
one was single-code. Coupling is the capability the project leans on hardest —
it is the thing a model without OASiS cannot do at all — and its knowledge was
~35 kB across 18 per-backend payloads with no fixture-level verification behind
any of it.

What a coupling fixture must do, and what it must not
-----------------------------------------------------
It runs a REAL two-code coupling through the REGISTERED `couple` tool (the same
one an agent calls, reached through the same FastMCP tool manager the server
uses — not `run_coupling` directly, because `couple`'s own argument handling and
validation block are part of what the knowledge claims) and then checks the
PHYSICS against a closed form.

Checking `converged` is NOT enough and that is the whole point: a partitioned
fixed-point scheme converges to a fixed point, which is only the solution if the
two participants exchange the right quantity with the right sign in the right
units. A unit mismatch converges beautifully, balances to roundoff, and passes
every check the driver has. So each fixture asserts the interface VALUE, the
interface FLUX and the flux BALANCE against the analytic answer for the
placeholder problem.

No measured number from these runs is written into the knowledge text, and no
fixture pins one: every threshold here is a tolerance the physics must beat, and
the errors are PRINTED so the run reports its own numbers.

Absent backends
---------------
A fixture must not pass on a machine that cannot run it. Every entry point here
raises `Absent` when a required backend is missing, and `main()` turns that into
a `FAIL:` line plus a non-zero exit. `FAIL:` is in every coupling fixture's
`forbid_in_output`, and the expectations name substantive values (an error
magnitude, an iteration count) that are printed only on the path that really
ran. So a host without FEniCSx reports a FAILURE, never a pass.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

PARTICIPANT_DIR = REPO_ROOT / "data" / "coupling_participants"


class Absent(Exception):
    """A backend this fixture needs is not installed on this host."""


# ── the placeholder problem, and its closed form ───────────────────────────
#
# Steady conduction with no source on a rectangle split by a straight interface
# at x = XI. No y-variation, so the exact solution is piecewise linear in x and
# the interface unknowns are available in closed form. `conductance` is the
# subdomain's stiffness as seen from the interface, k / (distance from the
# interface to that subdomain's own Dirichlet boundary) — the same quantity the
# served knowledge calls "interface conductance" and builds rho out of.

@dataclass(frozen=True)
class Problem:
    """One split conduction problem. Lengths in m, k in W/(m K), T in K."""
    xl: float = 0.0        # left subdomain's outer boundary
    xi: float = 0.6        # the interface
    xr: float = 1.1        # right subdomain's outer boundary
    y0: float = 0.0
    y1: float = 0.4
    kl: float = 0.8        # left conductivity
    kr: float = 1.5        # right conductivity
    tl: float = 320.0      # Dirichlet at x = xl
    tr: float = 300.0      # Dirichlet at x = xr

    @property
    def cl(self) -> float:
        return self.kl / (self.xi - self.xl)

    @property
    def cr(self) -> float:
        return self.kr / (self.xr - self.xi)

    @property
    def t_iface(self) -> float:
        return (self.cl * self.tl + self.cr * self.tr) / (self.cl + self.cr)

    @property
    def q(self) -> float:
        """Flux density in the +x direction at the interface, W/m^2."""
        return self.cl * (self.tl - self.t_iface)

    def rho(self, dirichlet_side: str) -> float:
        """Conductance ratio the knowledge's theta rule is written in:
        Dirichlet-side conductance over Neumann-side conductance."""
        if dirichlet_side == "left":
            return self.cl / self.cr
        return self.cr / self.cl

    def theta_opt(self, dirichlet_side: str) -> float:
        return 1.0 / (1.0 + self.rho(dirichlet_side))

    def amplification(self, dirichlet_side: str, theta: float) -> float:
        """sqrt((1-theta)^2 + rho*theta^2) — the knowledge's own formula."""
        r = self.rho(dirichlet_side)
        return ((1.0 - theta) ** 2 + r * theta ** 2) ** 0.5


DEFAULT = Problem()


# ── the elastic analogue FEBio solves (FEBio 4 has no heat module) ──────────

@dataclass(frozen=True)
class ElasticProblem:
    """Uniaxial-strain bar: displacement plays T, the P-wave modulus plays k."""
    xl: float = 0.0
    xi: float = 0.5
    xr: float = 1.0
    el: float = 1000.0
    er: float = 2250.0
    nu: float = 0.3
    ul: float = 0.0        # prescribed u_x at x = xl
    ur: float = 1.0e-4     # prescribed u_x at x = xr

    def modulus(self, e: float) -> float:
        n = self.nu
        return e * (1.0 - n) / ((1.0 + n) * (1.0 - 2.0 * n))

    @property
    def cl(self) -> float:
        return self.modulus(self.el) / (self.xi - self.xl)

    @property
    def cr(self) -> float:
        return self.modulus(self.er) / (self.xr - self.xi)

    @property
    def u_iface(self) -> float:
        return (self.cl * self.ul + self.cr * self.ur) / (self.cl + self.cr)

    @property
    def q(self) -> float:
        """The LEFT subdomain's outward normal traction export,
        q_out = -(sigma . n_own)_x with n_own = +e_x."""
        return -self.cl * (self.u_iface - self.ul)


# ── which interpreter / binary this install actually uses ───────────────────
#
# Resolved from the live registry, never hard-coded: the served knowledge tells
# agents to take the interpreter from `discover(query='list')`, and that is the
# same resolution path.

def _availability(name: str) -> tuple[bool, str]:
    from core.registry import get_backend, load_all_backends
    load_all_backends()
    b = get_backend(name)
    if not b:
        return False, f"no backend named {name!r} in the registry"
    st, msg = b.check_availability()
    return st.value == "available", str(msg)


def require_available(*names: str) -> None:
    for n in names:
        ok, msg = _availability(n)
        if not ok:
            raise Absent(f"backend {n!r} is not available on this install "
                         f"({msg[:160]})")


def interpreter(backend: str) -> str:
    """The command that RUNS this backend's participant script.

    For the four wrapper backends (4C, deal.II, FEBio, SPARTA) that is a plain
    Python with numpy, and the solver binary goes into a constant inside the
    script — exactly the distinction the served launch text has to make.
    """
    if backend == "fenics":
        from backends.fenics.backend import _find_fenics_python
        p = _find_fenics_python()
        if not p:
            raise Absent("no FEniCSx interpreter resolved")
        return str(p)
    if backend == "dune":
        from backends.dune.backend import _find_dune_python
        p = _find_dune_python()
        if not p:
            raise Absent("no DUNE-fem interpreter resolved")
        return str(p)
    if backend in ("skfem", "ngsolve", "fourc", "dealii", "febio", "kratos"):
        return sys.executable
    raise Absent(f"no interpreter rule for backend {backend!r}")


def _fourc_edits() -> dict:
    from backends.fourc.backend import _find_fourc_binary
    b = _find_fourc_binary()
    if not b:
        raise Absent("no 4C binary resolved")
    e = {"FOURC_BIN": json.dumps(str(b))}
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld:
        e["FOURC_LD"] = json.dumps(ld.split(os.pathsep)[0])
    else:
        # The registry found the binary; 4C still needs its dependency libs.
        for cand in ("/opt/4C-dependencies/lib",):
            if Path(cand).is_dir():
                e["FOURC_LD"] = json.dumps(cand)
                break
    return e


def _febio_edits() -> dict:
    from backends.febio.backend import _find_febio_binary
    b = _find_febio_binary()
    if not b:
        raise Absent("no FEBio binary resolved")
    return {"FEBIO": json.dumps(str(b))}


def _dealii_edits() -> dict:
    exe = dealii_exe()
    return {"DEALII_EXE": json.dumps(str(exe))}


def dealii_exe() -> Path:
    """Build the shipped deal.II participant solver once, and cache it.

    The deal.II participant is TWO files — a compiled C++ solver plus a thin
    Python wrapper — so unlike every other backend there is a build step before
    a coupling can run at all. That is a claim in the served payload and this is
    where it gets exercised.
    """
    if not shutil.which("cmake"):
        raise Absent("cmake is not on PATH; cannot build the deal.II solver")
    cand = [os.environ.get("DEAL_II_DIR", ""),
            str(Path.home() / "dealii" / "build"),
            str(Path.home() / "dealii"), "/usr/local", "/usr"]
    root = next((c for c in cand if c and
                 (Path(c) / "lib/cmake/deal.II/deal.IIConfig.cmake").is_file()),
                None)
    if not root:
        raise Absent("no deal.II install tree with "
                     "lib/cmake/deal.II/deal.IIConfig.cmake")
    build = Path(os.environ.get("TMPDIR", "/tmp")) / "oasis_dealii_participant_build"
    exe = build / "heat_iface_dealii"
    if exe.is_file():
        return exe
    build.mkdir(parents=True, exist_ok=True)
    for cmd in (["cmake", "-S", str(PARTICIPANT_DIR), "-B", str(build),
                 f"-DDEAL_II_DIR={root}", "-DCMAKE_BUILD_TYPE=Release"],
                ["make", "-C", str(build), "-j4"]):
        if subprocess.run(cmd, capture_output=True, text=True,
                          timeout=2400).returncode != 0:
            raise Absent("the shipped deal.II participant solver did not build")
    if not exe.is_file():
        raise Absent("the deal.II build produced no heat_iface_dealii")
    return exe


BINARY_EDITS = {"fourc": _fourc_edits, "febio": _febio_edits,
                "dealii": _dealii_edits}


# ── editing the shipped participant script ─────────────────────────────────

def edit(text: str, edits: dict) -> str:
    """Replace top-level `NAME = value` assignments, the way the served launch
    text tells an agent to. A key that is not in the script is a hard error:
    four `.replace()` calls in the knowledge module were silent no-ops for
    exactly this reason, and a silent no-op here would leave the fixture
    verifying a script it did not configure."""
    for k, v in edits.items():
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        if not pat.search(text):
            raise AssertionError(
                f"edit key {k!r} is not a top-level assignment in the shipped "
                f"script — the edit would have been a silent no-op")
        text = pat.sub(lambda m: m.group(1) + v + m.group(3), text, count=1)
    return text


def shipped(backend: str) -> Path:
    p = PARTICIPANT_DIR / f"participant_{backend}.py"
    if not p.is_file():
        raise Absent(f"no shipped participant script for {backend!r}")
    return p


def heat_edits(problem: Problem, position: str, role: str,
               partner: str, mesh: tuple[int, int]) -> dict:
    """The edit block for one side of the conduction problem.

    `position` is which subdomain ("left"/"right"); `role` is which side of the
    Dirichlet-Neumann split it takes. The two are INDEPENDENT — that is the
    "all four role/position combinations" the sides table claims.
    """
    if position == "left":
        x0, x1, k, t_outer = problem.xl, problem.xi, problem.kl, problem.tl
    else:
        x0, x1, k, t_outer = problem.xi, problem.xr, problem.kr, problem.tr
    return {"SIDE": json.dumps(role), "PARTNER": json.dumps(partner),
            "X0, X1": f"{x0}, {x1}", "Y0, Y1": f"{problem.y0}, {problem.y1}",
            "IFACE_X": f"{problem.xi}", "K": f"{k}", "F_SRC": "0.0",
            "T_OUTER": f"{t_outer}",
            "NX, NY": f"{mesh[0]}, {mesh[1]}",
            "T_INIT": "310.0", "Q_INIT": "0.0"}


def elastic_edits(problem: ElasticProblem, position: str, role: str,
                  partner: str, mesh: tuple[int, int]) -> dict:
    if position == "left":
        x0, x1, e, u_outer = problem.xl, problem.xi, problem.el, problem.ul
    else:
        x0, x1, e, u_outer = problem.xi, problem.xr, problem.er, problem.ur
    return {"SIDE": json.dumps(role), "PARTNER": json.dumps(partner),
            "X0, X1": f"{x0}, {x1}", "IFACE_X": f"{problem.xi}",
            "E_MOD": f"{e}", "NU": f"{problem.nu}", "U_OUTER": f"{u_outer}",
            "NX, NY": f"{mesh[0]}, {mesh[1]}",
            "U_INIT": "5.0e-5", "Q_INIT": "0.0"}


def stage(root: Path, name: str, backend: str, edits: dict) -> dict:
    """Write one configured participant into its own work_dir and return the
    spec `couple` takes. The driver does NOT copy the script — staging it is
    the agent's job, and that is what this reproduces."""
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    full = dict(edits)
    if backend in BINARY_EDITS:
        full.update(BINARY_EDITS[backend]())
    script = wd / f"participant_{name}.py"
    script.write_text(edit(shipped(backend).read_text(), full))
    interp = interpreter(backend)
    # Print the resolved interpreter so the run's own output evidences WHICH
    # code took each side. Without it, "backend=skfem" in a log is an intention,
    # not a fact.
    print(f"participant[{name}]={backend} interpreter={interp}")
    return {"name": name, "command": [interp, script.name],
            "work_dir": str(wd), "timeout": 900}


# ── the real `couple` tool, reached the way the server reaches it ───────────

_TOOL_MANAGER = None


def _tool_manager():
    global _TOOL_MANAGER
    if _TOOL_MANAGER is None:
        from mcp.server.fastmcp import FastMCP
        from core.registry import load_all_backends
        from tools.consolidated import register_consolidated_tools
        m = FastMCP("tier2-coupling")
        register_consolidated_tools(m)
        load_all_backends()
        _TOOL_MANAGER = m._tool_manager
    return _TOOL_MANAGER


def _text(res) -> str:
    if isinstance(res, tuple) and len(res) >= 1:
        res = res[0]
    if isinstance(res, list):
        return "\n".join(getattr(b, "text", str(b)) for b in res)
    return getattr(res, "text", str(res))


def call_tool(name: str, args: dict) -> str:
    return _text(asyncio.run(_tool_manager().call_tool(name, args)))


def couple(specs: list[dict], max_iter: int = 120, tol: float = 1e-8,
           accelerator: str = "constant", theta: float = 0.5) -> dict:
    """Run the REGISTERED `couple` tool and return its parsed JSON."""
    raw = call_tool("couple", {
        "participants": json.dumps(specs), "max_iter": max_iter, "tol": tol,
        "accelerator": accelerator, "theta": theta, "critic_approved": True})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"couple returned non-JSON: {raw[:400]}")


def pair(specs: list[dict], **kw) -> dict:
    """Wire two staged participants into each other and couple them."""
    assert len(specs) == 2
    a, b = specs
    a["imports_from"] = [b["name"]]
    b["imports_from"] = [a["name"]]
    return couple([a, b], **kw)


def workroot(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"t2cpl_{tag}_"))


# ── measuring the physics out of a coupling result ─────────────────────────

def net_flux(export: dict) -> float:
    """Net normal flux leaving through the interface, integrated over
    arclength — the same quantity `check_interface_balance` forms, recomputed
    here so the fixture does not depend on that function being right."""
    co = export.get("coordinates") or []
    fl = export.get("normal_fluxes")
    if fl is None or not co:
        raise AssertionError("export carries no normal_fluxes / coordinates")
    ys = [c[1] for c in co]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    tot = 0.0
    for i, j in zip(order, order[1:]):
        tot += 0.5 * (float(fl[i]) + float(fl[j])) * (ys[j] - ys[i])
    return tot


def span(values) -> tuple[float, float]:
    v = [float(x) for x in values]
    return min(v), max(v)


def check_balance(res: dict) -> list[str]:
    """`check_interface_balance` on the two exports, as `couple` runs it."""
    from core.quality_checks import check_interface_balance
    names = list(res["exports"])
    assert len(names) == 2
    return check_interface_balance(res["exports"][names[0]],
                                   res["exports"][names[1]], *names)


# ── assertion plumbing: every failure is loud and named ────────────────────

_FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    if not ok:
        _FAILS.append(f"{label}: {detail}" if detail else label)
        print(f"FAIL: {label} {detail}".rstrip())
    return ok


def close(got: float, want: float, atol: float, label: str) -> bool:
    err = abs(float(got) - float(want))
    print(f"{label}={err:.3e}")
    return check(err <= atol, f"{label}_over_tolerance",
                 f"|{got!r} - {want!r}| = {err:.6e} > {atol:.1e}")


def main(fn) -> None:
    """Run a fixture body. Absent backend -> FAIL + non-zero exit, never a
    quiet pass; any exception -> the same, with the traceback."""
    try:
        fn()
    except Absent as e:
        print(f"FAIL: this fixture cannot run here — {e}. A fixture must not "
              f"report a pass on a host that cannot run it.")
        sys.exit(2)
    except Exception:
        import traceback
        print("FAIL: fixture raised")
        traceback.print_exc()
        sys.exit(3)
    if _FAILS:
        print(f"failures_count={len(_FAILS)}")
        sys.exit(1)
    print("failures_count=0")
