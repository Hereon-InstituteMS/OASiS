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

def _looks_like_the_checkout(d: Path) -> bool:
    return ((d / "src" / "tools" / "coupling_knowledge.py").is_file()
            and (d / "data" / "coupling_participants").is_dir())


def _find_checkout(start: Path | None) -> Path | None:
    if start is None:
        return None
    try:
        start = start.resolve()
    except OSError:
        return None
    for d in (start, *start.parents):
        try:
            if _looks_like_the_checkout(d):
                return d
        except OSError:
            continue
    return None


# WHERE THE CHECKOUT IS, and why this is not just `parents[4]`.
#
# Normally the answer is "walk up from this file", and that is exact: it finds
# the checkout this fixture belongs to, branch and all. But
# scripts/mutate_tier2_fixtures.py copies a fixture into a scratch tree under
# /tmp before mutating it, and from there walking up finds nothing — the fixture
# would die on `import core.registry` and every mutation verdict would be
# VACUOUS_BASELINE, which says only that nothing ran.
#
# So there are fallbacks, and the resolved root is PRINTED, because two of them
# can answer with a DIFFERENT checkout of the same repo and a reader comparing
# runs needs to see that rather than infer it.
#
#   OASIS_REPO   an explicit pin, for anyone who wants one.
#   $PWD         the directory the RUNNER was invoked from. `subprocess.run`
#                sets the child's working directory but leaves the inherited
#                PWD alone, so this survives into a staged fixture and names
#                the checkout the harness itself is working in — which is the
#                one whose mutation is being tested.
#   sys.executable  last, and only useful when the interpreter lives inside a
#                checkout.
_ENV_ROOT = os.environ.get("OASIS_REPO")
_PWD = os.environ.get("PWD")
REPO_ROOT = (_find_checkout(Path(__file__))
             or _find_checkout(Path(_ENV_ROOT) if _ENV_ROOT else None)
             or _find_checkout(Path(_PWD) if _PWD else None)
             or _find_checkout(Path(sys.executable)))
if REPO_ROOT is None:
    print("FAIL: cannot locate an OASiS checkout from this fixture's own path, "
          "from $OASIS_REPO, or from the running interpreter. A coupling "
          "fixture needs the checkout for src/ and for the shipped participant "
          "scripts; it cannot run without one.")
    raise SystemExit(2)
print(f"checkout={REPO_ROOT}")
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


# ── the VECTOR problem: plane-strain elasticity, displacement AND traction ──
#
# Everything above exchanges ONE SCALAR across the interface. This exchanges a
# two-component displacement and a two-component traction, in both directions,
# which is the primitive every multiphysics coupling (TSI, FSI, contact) is
# built on and the thing a scalar fixture cannot establish.
#
# The problem is a plane-strain bimaterial strip [xl, xr] x [y0, y1] split at
# x = xi, with a PRESCRIBED DISPLACEMENT on the whole outer boundary (the two
# outer x-faces and both y-faces) taken from a field that is piecewise linear
# in (x, y):
#
#     u_x = a_i (x - x_i0) + c_i + p y        a_i, b_i per block
#     u_y = b_i (x - x_i0) + d_i + r y        p, r global
#
# Strains are constant in each block, so sigma is constant in each block, so
# div sigma = 0 and the field is an exact solution. Continuity of u and of
# sigma . e_x across x = xi fixes a_i, b_i from the totals (dux, duy) and the
# two global gradients (p, r). P1/Q1 reproduce it EXACTLY — it is a patch test
# — so the only error left in a converged coupling is the coupling itself, and
# the tolerances can be tight enough that a wrong Poisson ratio cannot hide.
#
# THREE PROPERTIES THIS IS BUILT FOR, none of which the scalar problem has:
#
#   * BOTH COMPONENTS ARE ALIVE. u_x, u_y, t_x and t_y are all non-zero at the
#     interface, so a coupling that drops or scrambles a component is visible.
#   * THE INTERFACE STATE VARIES ALONG THE INTERFACE. p and r make u_x and u_y
#     linear in y on the interface, so a mapping between two non-matching
#     interface meshes has something to get wrong. With a constant profile,
#     ANY mapping — including one that interleaves the components — reproduces
#     it, and the non-matching-mesh claim would be vacuous.
#   * THE Y-FACES ARE PRESCRIBED, NOT TRACTION-FREE, AND THAT IS DELIBERATE.
#     With traction-free y-faces each block can BEND, the bending compliance
#     scales as L^3 against L^1 for the axial one, and the Steklov spectrum of
#     the two-subdomain iteration opens up to [0.049, 2.07] — measured. The
#     single theta the driver applies then converges the x component while
#     DIVERGING the y one: measured here, theta = 1/(1+rho_x) with the y-faces
#     free left u_x settling on the exact answer while u_y oscillated with
#     growing amplitude, and the run reported only "did not converge". Clamping
#     the y-faces removes the bending modes; the spectrum tightens to
#     [0.25, 0.64], mesh-independently, and one theta serves both components.
#     This is a real property of vector coupling, not a detail: see
#     vector_relaxation_needs_the_worst_component.

@dataclass(frozen=True)
class VectorProblem:
    """Plane-strain bimaterial strip, split at `xi`, prescribed displacement on
    the whole outer boundary. Lengths in m, E in Pa, u in m."""
    xl: float = 0.0
    xi: float = 0.55       # the interface — equal halves by default, see rho
    xr: float = 1.1
    y0: float = 0.0
    y1: float = 0.4
    el: float = 1000.0     # left Young's modulus
    er: float = 2500.0     # right Young's modulus
    nul: float = 0.3       # left Poisson ratio (plane strain)
    nur: float = 0.3       # right Poisson ratio
    dux: float = 1.0e-3    # total u_x across the strip
    duy: float = 4.0e-4    # total u_y across the strip
    p: float = 3.0e-4      # du_x/dy, global — makes u_x vary ALONG the interface
    r: float = 5.0e-4      # du_y/dy, global

    # ── material ──
    @staticmethod
    def _lame(e: float, nu: float) -> tuple[float, float]:
        return e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu))

    @property
    def lam_l(self) -> float:
        return self._lame(self.el, self.nul)[0]

    @property
    def mu_l(self) -> float:
        return self._lame(self.el, self.nul)[1]

    @property
    def lam_r(self) -> float:
        return self._lame(self.er, self.nur)[0]

    @property
    def mu_r(self) -> float:
        return self._lame(self.er, self.nur)[1]

    @property
    def ml(self) -> float:
        """P-wave modulus lambda + 2 mu of the left block."""
        return self.lam_l + 2.0 * self.mu_l

    @property
    def mr(self) -> float:
        return self.lam_r + 2.0 * self.mu_r

    @property
    def ll(self) -> float:
        return self.xi - self.xl

    @property
    def lr(self) -> float:
        return self.xr - self.xi

    # ── the closed form ──
    @property
    def sxx(self) -> float:
        """sigma_xx, constant over the whole strip."""
        return ((self.dux + self.r * (self.lam_l * self.ll / self.ml
                                      + self.lam_r * self.lr / self.mr))
                / (self.ll / self.ml + self.lr / self.mr))

    @property
    def sxy(self) -> float:
        """sigma_xy, constant over the whole strip."""
        return ((self.duy + self.p * (self.xr - self.xl))
                / (self.ll / self.mu_l + self.lr / self.mu_r))

    @property
    def al(self) -> float:
        return (self.sxx - self.lam_l * self.r) / self.ml

    @property
    def ar(self) -> float:
        return (self.sxx - self.lam_r * self.r) / self.mr

    @property
    def bl(self) -> float:
        return self.sxy / self.mu_l - self.p

    @property
    def br(self) -> float:
        return self.sxy / self.mu_r - self.p

    def u_exact(self, x, y):
        """The exact displacement (u_x, u_y) at (x, y). Accepts arrays."""
        import numpy as np
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        left = x <= self.xi + 1e-12
        ux = np.where(left, self.al * (x - self.xl),
                      self.al * self.ll + self.ar * (x - self.xi)) + self.p * y
        uy = np.where(left, self.bl * (x - self.xl),
                      self.bl * self.ll + self.br * (x - self.xi)) + self.r * y
        return ux, uy

    def u_iface(self, y):
        return self.u_exact(np.full_like(np.asarray(y, float), self.xi), y)

    def t_export(self, side: str) -> tuple[float, float]:
        """What `side` ("left"/"right") must export as `normal_fluxes`.

        The convention is q_out = -(sigma . n_own), the SAME one the scalar
        participants use for heat, so the two sides' exports cancel and the
        Neumann side applies the partner's numbers unchanged. n_own = +e_x on
        the left, -e_x on the right.
        """
        s = 1.0 if side == "left" else -1.0
        return (-s * self.sxx, -s * self.sxy)

    def poly(self, position: str) -> tuple[tuple, tuple]:
        """The outer-boundary Dirichlet polynomial (UDX, UDY) for one block:
        u = (c0 + c1 x + c2 y + c3 y^2). c3 is zero for this problem — it is
        there so the same participant script can be driven off a field that is
        NOT an exact solution, which is how the genuinely-2-D arrangement is
        made (see vector_bar_edits)."""
        if position == "left":
            return ((0.0, self.al, self.p, 0.0), (0.0, self.bl, self.r, 0.0))
        return ((self.al * self.ll - self.ar * self.xi, self.ar, self.p, 0.0),
                (self.bl * self.ll - self.br * self.xi, self.br, self.r, 0.0))

    # ── the relaxation the driver needs ──
    @property
    def rho_x(self) -> float:
        """Interface conductance ratio for the x component, left over right."""
        return (self.ml / self.ll) / (self.mr / self.lr)

    @property
    def rho_y(self) -> float:
        """...and for the y component. EQUAL to rho_x only when the two blocks
        share a Poisson ratio: M/mu = 2(1-nu)/(1-2nu) is independent of E, so
        equal nu makes the two ratios collapse to one number and one theta
        serves both components. Different nu splits them, and the single theta
        the driver applies must then be chosen for the WORST of the two."""
        return (self.mu_l / self.ll) / (self.mu_r / self.lr)

    def rho(self, dirichlet_side: str) -> tuple[float, float]:
        if dirichlet_side == "left":
            return self.rho_x, self.rho_y
        return 1.0 / self.rho_x, 1.0 / self.rho_y

    def theta_opt(self, dirichlet_side: str) -> float:
        """theta = 1/(1 + max_c rho_c).

        The driver's iteration is JACOBI, so its amplification per component is
        sqrt((1-theta)^2 + rho_c theta^2) — the same expression the scalar
        knowledge uses — and it is below one only while theta < 2/(1+rho_c).
        The binding component is the LARGEST rho, so the max is not a
        conservative choice but the only one that converges: with rho_y above
        1 + 2 rho_x, theta = 1/(1+rho_x) diverges on y while x settles."""
        return 1.0 / (1.0 + max(self.rho(dirichlet_side)))


VECTOR_DEFAULT = VectorProblem()


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
           accelerator: str = "constant", theta: float = 0.5,
           noise_replicates: int = 0, noise_floor: float = 0.0,
           noise_block: int = 3) -> dict:
    """Run the REGISTERED `couple` tool and return its parsed JSON.

    The three `noise_*` arguments are the tool's stochastic branch and are
    passed straight through, so a fixture exercises them exactly as an agent
    would rather than reaching into the driver.
    """
    args = {"participants": json.dumps(specs), "max_iter": max_iter, "tol": tol,
            "accelerator": accelerator, "theta": theta, "critic_approved": True}
    if noise_replicates:
        args["noise_replicates"] = noise_replicates
    if noise_floor:
        args["noise_floor"] = noise_floor
    if noise_block != 3:
        args["noise_block"] = noise_block
    raw = call_tool("couple", args)
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


# ── one arrangement of one pair, checked against the closed form ───────────
#
# Every pair fixture is the same body with different backends in the two slots,
# so it lives here once. `dirichlet` is which SUBDOMAIN takes the Dirichlet
# role; the two backends' POSITIONS are given by which slot they sit in. Those
# are independent, which is what the sides table means by "all four
# role/position combinations".

def heat_arrangement(tag: str, backend_left: str, backend_right: str,
                     dirichlet: str, mesh_l=(16, 16), mesh_r=(14, 12),
                     theta: float | None = None, problem: Problem | None = None,
                     max_iter: int = 200, tol: float = 1e-8,
                     t_atol: float = 1e-3, q_atol: float = 5e-3) -> dict:
    """Couple two backends on the split conduction problem and check the
    physics against the closed form. Returns the `couple` result."""
    p = problem or DEFAULT
    if theta is None:
        theta = p.theta_opt(dirichlet)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    print(f"--- {tag}: left={backend_left}/{roles['left']} "
          f"right={backend_right}/{roles['right']} theta={theta:.6f}")
    root = workroot(tag)
    specs = [
        stage(root, "left", backend_left,
              heat_edits(p, "left", roles["left"], "right", mesh_l)),
        stage(root, "right", backend_right,
              heat_edits(p, "right", roles["right"], "left", mesh_r)),
    ]
    res = pair(specs, max_iter=max_iter, tol=tol,
               accelerator="constant", theta=theta)
    if not check(bool(res.get("converged")), f"{tag}_did_not_converge",
                 str(res.get("error"))[:300]):
        return res
    print(f"{tag}_iterations={res['iterations']}")
    print(f"{tag}_residual={res['residual']:.3e}")
    print(f"{tag}_converged=True")
    _check_interface_physics(tag, res, p.t_iface, p.q, t_atol, q_atol)
    # And against a SECOND, independently computed reference: the same problem
    # solved un-split in one code. See monolithic_interface_state.
    ex = res["exports"]
    t_got = 0.5 * sum(span(ex["left"]["values"]))
    q_got = 0.5 * sum(span(ex["left"]["normal_fluxes"]))
    assert_against_monolithic(tag, t_got, q_got, p)
    return res


def elastic_arrangement(tag: str, backend_left: str, backend_right: str,
                        dirichlet: str, mesh_l=(16, 8), mesh_r=(12, 6),
                        theta: float | None = None, max_iter: int = 200,
                        tol: float = 1e-8, u_atol: float = 1e-7,
                        q_atol: float = 1e-3) -> dict:
    """The same, for the elastic analogue FEBio solves — FEBio 4 has no heat
    module, so a conduction participant is impossible there and the shipped
    script is a uniaxial-strain bar instead.

    The tolerances are looser than the conduction ones in RELATIVE terms
    (~1e-3 of the problem's own scale, against ~1e-8 for conduction) and that
    is a property of the model, not slack. The P1 conduction participants are
    exact for a piecewise-linear solution, so their only error is the coupling
    tolerance. FEBio solves a three-dimensional hex model of a bar that is
    uniaxial-strain only in the continuum limit — one element through the
    thickness, a lateral constraint, its own nonlinear solver tolerance — so a
    small model error survives however tightly the coupling converges. Measured
    here it is a few parts in 1e5, and these thresholds sit an order of
    magnitude above that while staying orders of magnitude below any of the
    pathologies the fixture exists to catch: a sign error is O(1) and a wrong
    modulus O(0.1).
    """
    p = ElasticProblem()
    if theta is None:
        rho = (p.cl / p.cr) if dirichlet == "left" else (p.cr / p.cl)
        theta = 1.0 / (1.0 + rho)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    print(f"--- {tag}: left={backend_left}/{roles['left']} "
          f"right={backend_right}/{roles['right']} theta={theta:.6f}")
    root = workroot(tag)
    specs = [
        stage(root, "left", backend_left,
              elastic_edits(p, "left", roles["left"], "right", mesh_l)),
        stage(root, "right", backend_right,
              elastic_edits(p, "right", roles["right"], "left", mesh_r)),
    ]
    res = pair(specs, max_iter=max_iter, tol=tol,
               accelerator="constant", theta=theta)
    if not check(bool(res.get("converged")), f"{tag}_did_not_converge",
                 str(res.get("error"))[:300]):
        return res
    print(f"{tag}_iterations={res['iterations']}")
    print(f"{tag}_residual={res['residual']:.3e}")
    print(f"{tag}_converged=True")
    # No monolithic counterpart here: the un-split solve in the harness is the
    # CONDUCTION problem, and FEBio's participant is the elastic analogue. The
    # closed form for the bar is the only independent answer this arrangement
    # has, and saying so is better than reusing a reference for a different
    # physics and calling it a check.
    _check_interface_physics(tag, res, p.u_iface, p.q, u_atol, q_atol,
                             value="u")
    return res


# ── the relaxation study: rho, theta, and what the iteration does ──────────

SWEEP_T_INIT = 290.0    # see probe_theta: away from every ratio's answer


def problem_with_rho(rho: float) -> Problem:
    """A split conduction problem whose Dirichlet-side conductance ratio is
    exactly `rho`, built by moving ONE constant.

    rho is (Dirichlet-side interface conductance) / (Neumann-side one), with
    "interface conductance" = k / (distance from the interface to that
    subdomain's own outer boundary). Holding the geometry and the right
    conductivity fixed and solving for the left one keeps everything else about
    the problem — including its closed form — comparable across ratios.
    """
    right = DEFAULT.kr / (DEFAULT.xr - DEFAULT.xi)
    return Problem(kl=rho * right * (DEFAULT.xi - DEFAULT.xl))


def probe_theta(tag: str, rho: float, theta: float, max_iter: int,
                tol: float = 1e-8, dirichlet: str = "left",
                backend: str = "skfem", mesh_l=(16, 16), mesh_r=(14, 12),
                quiet: bool = False) -> dict:
    """Run one (rho, theta) with a CONSTANT accelerator and report what the
    iteration did — without asserting any physics, because a diverging run has
    none to assert.

    Returns `converged`, `iterations`, `residual`, and `deviation`: how far the
    interface value ended from the exact answer, relative to it.

    `deviation` rather than the raw magnitude, and that choice matters. The
    interface temperature sits near 306 K and starts a few K wrong, so an
    iteration can amplify its ERROR fifty-fold and still leave the VALUE inside
    a factor of two of the right answer — measured here, a genuinely diverging
    run at rho=4 registered 0.2 orders of magnitude on the raw values after
    thirty iterations, indistinguishable from a converging one. The deviation
    from the exact answer grows as the amplification factor to the power of the
    iteration count with nothing to hide behind, which is what separates a slow
    iteration from a runaway one at a bearable iteration budget.

    The residual is not that discriminator either: it is normalised by the raw
    export magnitude, so a diverging run SATURATES it near a constant of order
    one rather than sending it to infinity.
    """
    import math
    p = problem_with_rho(rho)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    root = workroot(tag)
    specs = []
    for pos, mesh in (("left", mesh_l), ("right", mesh_r)):
        partner = "right" if pos == "left" else "left"
        edits = heat_edits(p, pos, roles[pos], partner, mesh)
        # The shipped script's iteration-1 fallback is 310 K, which happens to
        # BE the exact answer at rho=1. Starting a stability study from the
        # solution would measure nothing, so the sweep starts well away from
        # every ratio's answer and each run's initial deviation is comparable.
        edits["T_INIT"] = f"{SWEEP_T_INIT}"
        specs.append(_quiet_stage(root, pos, backend, edits, quiet))
    res = pair(specs, max_iter=max_iter, tol=tol,
               accelerator="constant", theta=theta)
    worst = 0.0
    for ex in res.get("exports", {}).values():
        for v in ex.get("values", []) or []:
            d = abs(float(v) - p.t_iface)
            if math.isfinite(d):
                worst = max(worst, d)
            else:
                worst = float("inf")
    deviation = worst / abs(p.t_iface) if math.isfinite(worst) else float("inf")
    out = {"converged": bool(res.get("converged")),
           "iterations": int(res.get("iterations", 0)),
           "residual": float(res.get("residual", float("nan"))),
           "deviation": deviation, "theta": theta, "rho": rho,
           "amplification": p.amplification(dirichlet, theta),
           "result": res}
    return out


def _quiet_stage(root: Path, name: str, backend: str, edits: dict,
                 quiet: bool) -> dict:
    """`stage` prints the interpreter it resolved, which is the right thing for
    a pair fixture and far too noisy for a sweep of thirty runs."""
    if not quiet:
        return stage(root, name, backend, edits)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec = stage(root, name, backend, edits)
    return spec


# ── the MONOLITHIC reference: the same problem solved un-split, in one code ─
#
# Every pair fixture already checks the converged interface state against the
# closed form. That is an independent answer, but it is an independent FORMULA:
# it shares an author with the fixture, and if the formula were wrong every pair
# fixture would agree with it in unison. So the harness also solves the WHOLE
# rectangle as ONE problem — no interface, no partitioning, no driver — and
# compares against that.
#
# Two properties this deliberately has:
#   * it is COMPUTED HERE, in the fixture harness, never served. An agent cannot
#     reach it through any tool, so it cannot be quoted back as an answer;
#   * it is a different KIND of evidence. The closed form assumes the solution is
#     piecewise linear in x with no y-variation; the monolithic solve assumes
#     nothing and would disagree if that were wrong.
#
# The monolithic flux is taken from the LEFT DIRICHLET BOUNDARY REACTION rather
# than from a gradient at the interface. In steady conduction with no source the
# heat crossing every vertical section is the same, so the boundary reaction IS
# the interface flux — and a reaction needs no differentiation of the discrete
# solution, which is what makes it accurate enough to be a reference at all.

_MONO_CACHE: dict = {}


def monolithic_interface_state(problem: Problem | None = None,
                               nxl: int = 40, nxr: int = 36,
                               ny: int = 24) -> tuple[float, float]:
    """Solve the un-split problem in one code; return (T at the interface,
    flux density in +x through it). Cached per (problem, mesh)."""
    import numpy as np
    p = problem or DEFAULT
    key = (p, nxl, nxr, ny)
    if key in _MONO_CACHE:
        return _MONO_CACHE[key]
    try:
        from skfem import (Basis, BilinearForm, ElementTriP1, MeshTri, asm,
                           condense, solve)
        from skfem.helpers import dot, grad
    except ImportError as e:
        raise Absent(f"the monolithic reference needs scikit-fem: {e}")

    # A mesh whose x-lines include the interface, so "the interface nodes" is
    # exact rather than a tolerance search.
    x = np.concatenate([np.linspace(p.xl, p.xi, nxl + 1),
                        np.linspace(p.xi, p.xr, nxr + 1)[1:]])
    y = np.linspace(p.y0, p.y1, ny + 1)
    m = MeshTri.init_tensor(x, y)
    basis = Basis(m, ElementTriP1())
    xc = m.p[0][m.t].mean(axis=0)                     # element centroid x
    kel = np.where(xc < p.xi, p.kl, p.kr)

    @BilinearForm
    def a(u, v, w):
        return w["k"] * dot(grad(u), grad(v))

    K = asm(a, basis, k=kel[:, None])                 # (nelems, 1) broadcasts
    f = basis.zeros()
    left = np.flatnonzero(np.isclose(m.p[0], p.xl))
    right = np.flatnonzero(np.isclose(m.p[0], p.xr))
    u = basis.zeros()
    u[left] = p.tl
    u[right] = p.tr
    D = np.concatenate([left, right])
    u = solve(*condense(K, f, x=u, D=D))

    iface = np.flatnonzero(np.isclose(m.p[0], p.xi))
    t_iface = float(np.mean(u[iface]))
    # Reaction at the left Dirichlet boundary = heat entering there; in steady
    # conduction with no source that is the heat crossing the interface too.
    react = K @ u - f
    q = float(np.sum(react[left])) / (p.y1 - p.y0)
    _MONO_CACHE[key] = (t_iface, q)
    return t_iface, q


def assert_against_monolithic(tag: str, t_got: float, q_got: float,
                              problem: Problem | None = None,
                              rtol: float = 0.02) -> bool:
    """Compare a converged coupled interface state against the un-split solve.

    Uses `core.quality_checks.check_monolithic_consistency` — the same detector
    an agent gets — so the fixture proves the detector as well as the coupling.
    Prints a discrete verdict, because an expectation that names an assertion is
    worth more than one that names a variable.
    """
    from core.quality_checks import check_monolithic_consistency
    p = problem or DEFAULT
    t_mono, q_mono = monolithic_interface_state(p)
    print(f"{tag}_monolithic_T={t_mono:.9g} monolithic_q={q_mono:.9g}")
    w = (check_monolithic_consistency(t_got, t_mono, rtol=rtol, qoi=f"{tag}_T")
         + check_monolithic_consistency(q_got, q_mono, rtol=rtol, qoi=f"{tag}_q"))
    ok = check(not w, f"{tag}_disagrees_with_monolithic", "; ".join(w)[:300])
    print(f"{tag}_matches_monolithic={bool(ok)}")
    return bool(ok)


def _check_interface_physics(tag: str, res: dict, exact_value: float,
                             exact_q: float, v_atol: float, q_atol: float,
                             value: str = "T") -> None:
    """The four things a converged coupling has to get right."""
    ex = res["exports"]

    # Non-matching interface discretisation is the normal case for a
    # partitioned coupling, and every pair claim states it explicitly.
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    print(f"{tag}_meshes_nonmatching="
          f"{bool(check(nl != nr, f'{tag}_matching_meshes', f'both sides used {nl} interface points, so the claim about NON-matching interface meshes was not exercised'))}")

    # (1) the interface value, from BOTH sides, against the closed form
    ok_v = True
    for side in ("left", "right"):
        lo, hi = span(ex[side]["values"])
        print(f"{tag}_{side}_{value}_span=[{lo:.10g},{hi:.10g}]")
        ok_v &= close(0.5 * (lo + hi), exact_value, v_atol,
                      f"{tag}_{side}_{value}_err")
        ok_v &= check(hi - lo < max(v_atol, 1e-12),
                      f"{tag}_{side}_{value}_not_uniform",
                      f"the exact interface {value} has no y-variation, got a "
                      f"spread of {hi - lo:.3e}")
    print(f"{tag}_interface_{value}_matches_reference={bool(ok_v)}")

    # (2) the interface flux. The two sides export with respect to their OWN
    # outward normals, which are anti-parallel, so the signs differ.
    ok_q = True
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.10g},{hi:.10g}]")
        ok_q &= close(0.5 * (lo + hi), sign * exact_q, q_atol,
                      f"{tag}_{side}_q_err")
    print(f"{tag}_interface_q_matches_reference={bool(ok_q)}")
    print(f"{tag}_flux_signs_are_opposite="
          f"{bool(span(ex['left']['normal_fluxes'])[0] * span(ex['right']['normal_fluxes'])[0] < 0)}")

    # (3) conservation: what leaves one subdomain enters the other.
    net_l, net_r = net_flux(ex["left"]), net_flux(ex["right"])
    scale = max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={abs(net_l + net_r) / scale:.3e}")
    ok_b = check(abs(net_l + net_r) / scale < 1e-4, f"{tag}_flux_not_balanced",
                 f"net(left)={net_l:.6e} net(right)={net_r:.6e}")
    bal = check_balance(res)
    ok_b &= check(not bal, f"{tag}_balance_check_complained",
                  "; ".join(bal)[:300])
    print(f"{tag}_flux_balanced={bool(ok_b)}")

    # (4) an empty `validation` block is what an agent is told to expect from a
    # coupling that is right.
    ok_val = check(not res["validation"], f"{tag}_validation_not_empty",
                   "; ".join(res["validation"])[:300])
    print(f"{tag}_validation_empty={bool(ok_val)}")


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
