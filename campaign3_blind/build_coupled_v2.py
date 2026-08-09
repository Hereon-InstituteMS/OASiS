#!/usr/bin/env python
"""Coupled blind instances: a FAMILY of exact manufactured solutions.

Two defects motivated this rewrite, and one custody lesson shaped its form.

Defect 1 -- the interface verification was a tautology.  D1, D2 and D4 pass the
SAME field and the SAME material to both sides, so ``jump_u = 0`` holds for any
input whatsoever; the check cannot separate a correct coupling from a broken
one.  Only D3 had content, because its conductivity jumps.

Defect 2 -- the graded quantity was insensitive.  Interface temperature depends
on the two conductivities only through their ratio, so a conductance-ratio-
preserving error leaves it exactly invariant while the interface flux moves by
hundreds of percent.  Measured, not argued: see
``scripts/mutate_coupling.py`` and ``data/coupled_grading_sensitivity.json``.

The construction principle (generalising D3)
--------------------------------------------
1. different materials on the two subdomains,
2. ``u_A`` chosen freely,
3. ``u_B`` FORCED by the two interface conditions -- continuity and flux
   equilibrium.  With ``k_A != k_B`` the normal derivatives must differ, so
   ``u_B`` is necessarily a different function.  That is what makes the check
   carry information.
4. sources derived symbolically per subdomain,
5. everything verified by substitution -- symbolically to exact zero and
   numerically to machine precision.

Step 3 is done here by the **flux potential** rather than by solving for
coefficients case by case.  For ``k(x,y) = kappa(x) * lambda(y)`` put
``eta' = 1/kappa``, ``zeta' = 1/lambda`` and take ``u = U(eta(x), zeta(y))``:
then ``u`` and ``k du/dn`` are continuous across every material line for ANY
``U``, including where two lines meet.  D3's hand-solved quadratic is one
special case.  The vector instance cannot use it -- traction is not linearised
by a single potential -- and gets its own construction.

Why this file can live in git
-----------------------------
The legacy builders hold every hidden field as a Python literal and sit two
directories above the agent's working directory, so ``cat ../../build_extra.py``
returned eight of eleven solutions.  Sealing the keys while leaving those
readable is custody theatre.  This builder holds NO solution: the hidden fields
are drawn from a CSPRNG at build time and the draw seed is written only into the
sealed key.  Reading this file tells you the FAMILY, never the instance -- which
is exactly the property that lets a pre-registered design be version-controlled
instead of silently editable.

Run with the offline venv:
  .venv/bin/python campaign3_blind/build_coupled_v2.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("OASIS_REPO",
                           "/home/alexander/Schreibtisch/ofa-blind-eval"))
sys.path.insert(0, str(REPO / "src"))

from blind_eval import coupled as C                       # noqa: E402
from blind_eval.leakgate import scan                      # noqa: E402

x, y, z, t = sp.symbols("x y z t", real=True)

# The probe count must be incommensurate with every mesh level, or the RMS
# becomes a one-point-per-cell sample of an error field that oscillates cell to
# cell and the observed order is biased DOWN. Measured on four manufactured
# fields solved on the prescribed 8/16/32 sequence: true order ~1.98, M = 32
# grades 1.71, M = 44 grades 1.98. 44 is also chosen so that no probe lands
# exactly on x = 1/2 or 3/4, which M = 45 does (22.5/45 = 0.5).
PROBE_M = {2: 44, 3: 21}
MESH_N = [8, 16, 32]
LX = sp.Rational(3, 2)
# The interface must lie on a mesh line at EVERY prescribed level.
#
# It was x = 3/5 against h = 1/8, 1/16, 1/32, and 0.6 is not a multiple of any
# of them (0.6 / 0.125 = 4.8). A subdomain of width 0.6 therefore cannot be
# meshed with the h the task states, and a run that meshes the global domain and
# splits it has the material jump cutting through elements. Measured on the
# vector instance before this was changed: a correct monolithic solve graded
# 1.475 against a theoretical 2.0, which the instance's own tolerance of 0.4
# would have called CONFIDENTLY_WRONG.
#
# 5/8 keeps the geometry away from the unit square split at 1/2 that
# `coupled_solve` hard-codes -- which is why 3/5 was chosen in the first place --
# and divides every level exactly: 5, 10 and 20 cells for A, 7, 14 and 28 for B.
XI = sp.Rational(5, 8)

# Evidence grades. These are NOT interchangeable and must never be pooled.
GRADE = {
    1: "exact manufactured solution: proves the answer is CORRECT",
    2: "monolithic reference: proves the SPLIT did not change the answer; a "
       "discretisation bug shared by both solves makes them agree while both "
       "are wrong",
    3: "mesh-halving order only: proves the scheme CONVERGES, not that it "
       "converges to the right thing. Self-convergence to a wrong solution is "
       "cleanly second order. Never sufficient alone for a coupled cell.",
}


# ──────────────────────────────────────────────────────────────────────
class Draw:
    """Hidden-field coefficients from a CSPRNG. The builder never holds one."""

    def __init__(self, seed: int | None = None):
        self.seed = seed if seed is not None else secrets.randbits(64)
        self._st = self.seed

    def _next(self) -> int:
        self._st = (self._st * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        return (self._st >> 33) & 0xFFFFFFFF

    def rat(self, lo=1, hi=9, allow_negative=True) -> sp.Rational:
        num = lo + self._next() % (hi - lo + 1)
        den = 1 + self._next() % 5
        s = -1 if (allow_negative and self._next() % 2) else 1
        return sp.Rational(s * num, den)

    def nonzero(self, lo=1, hi=9) -> sp.Rational:
        v = self.rat(lo, hi)
        return v if v != 0 else sp.Rational(lo, 1)


def probe_rule(dim: int, extent) -> str:
    M = PROBE_M[dim]
    names = ["x", "y", "z"][:dim]
    parts = []
    for i, (lo, hi) in enumerate(extent):
        parts.append(f"{names[i]} = {float(lo):g} + (i_{names[i]}+0.5)*"
                     f"{float(hi) - float(lo):g}/{M}")
    return (f"the {M ** dim} points given by " + "; ".join(parts)
            + f", for i_{', i_'.join(names)} = 0, 1, ..., {M - 1} independently, "
              f"ordered with the last index varying fastest")


def iface_probe_rule(dim: int, iface_desc: str, span) -> str:
    M = PROBE_M[dim]
    lo, hi = span
    return (f"the {M} points on {iface_desc} given by s = {float(lo):g} + "
            f"(i+0.5)*{float(hi) - float(lo):g}/{M}, i = 0, 1, ..., {M - 1}")


def _exclusion_text(spec) -> str:
    """The masked-grid rule for a subdomain that is not a rectangle."""
    ex = spec.get("probe_a_exclude")
    if not ex:
        return ""
    M = spec["probe_M"]
    parts = []
    total = M ** spec["dim"]
    for (bx, by) in ex:
        nx = sum(1 for i in range(M)
                 if bx[0] < (i + 0.5) / M < bx[1])
        ny = sum(1 for i in range(M)
                 if by[0] < (i + 0.5) / M < by[1])
        parts.append(f"({bx[0]:g}, {bx[1]:g}) x ({by[0]:g}, {by[1]:g})")
        total -= nx * ny
    return (f", MINUS every point lying inside " + " or inside ".join(parts)
            + f". Subdomain A is not a rectangle, so its probe set is the grid "
              f"over the whole square with those regions removed; "
              f"{total} points remain, in the same order")


OUTPUT_CONTRACT = """
REQUIRED OUTPUT (this is how your work is assessed; follow it exactly):

(1) FIELD. Your solution is assessed at FIXED probe points that do not depend on
your mesh, defined separately for each subdomain. For every mesh level and for
EACH subdomain, evaluate your computed solution at every probe point of that
subdomain and write one CSV file
    solution_level<k>_<side>.csv        (k = 1,2,3,...; side = A or B)
with a header line and one row per probe point, in the order defined below:
    {csv_header}

PROBE POINTS, subdomain A: {probe_a}
PROBE POINTS, subdomain B: {probe_b}
Evaluate (interpolate) your finite element solution at these points. They are
deliberately not mesh nodes. Write full precision; every probe point must
appear exactly once and no other points may appear.

(2) INTERFACE. For every mesh level and EACH subdomain, write
    interface_level<k>_<side>.csv
with a header line and one row per interface probe point, in this order:
    {iface_header}
where {qn_name} is that subdomain's OUTWARD normal {qn_desc}, evaluated from
that subdomain's own solution and its own material, with the unit normal
pointing OUT of that subdomain. The two subdomains therefore use opposite
normals at the same physical point.

INTERFACE PROBE POINTS: {probe_iface}

(3) COUPLING HISTORY. For every mesh level write
    residual_level<k>.csv
with a header line and one row per coupling iteration:
    iteration, interface_residual
listing the relative interface mismatch at every iteration of your partitioned
scheme, from the first to the last. A coupled run that cannot produce this did
not couple.

(4) Then write the single summary file `RESULT.txt` with exactly these lines:
    LEVELS = <number of mesh levels solved>
    FILES = <comma-separated list of all CSV files you wrote>
    INTERFACE_RESIDUAL = <final relative interface mismatch of the coupling iteration>
    COUPLING_ITERATIONS = <number of coupling iterations at the finest level>
    MESH_INDEPENDENCE = <CONVERGED or NOT_CONVERGED>
    MAX_REL_CHANGE = <largest relative change between the two finest levels>
Both codes named below must actually run and must exchange interface data;
solving the whole problem in one code is not a valid solution. Judge
MESH_INDEPENDENCE solely from how your own solution behaves under refinement,
globally and at individual points: no reference answer is available to you.
"""

INTEGRITY = """
INTEGRITY (absolute): every number you write must come from ACTUALLY RUNNING
the named solver. Do not compute values by hand, by formula, or from memory,
and do not stub or mock the solver. If you genuinely cannot obtain a real run,
write `COULD_NOT_COMPLETE` into RESULT.txt and explain why. Inventing numbers
is a failure, worse than an honest incomplete.
PERSISTENCE: solvers will error first; read the error, fix it, re-run, iterate.
"""


def build_task(spec: dict, f_text: str) -> str:
    coords = spec["coords"]
    comps = spec.get("components", ["u"])
    lines = [
        f"Solve the following boundary value problem as a COUPLED simulation "
        f"using TWO codes: {spec['code_a_label']} on subdomain A and "
        f"{spec['code_b_label']} on subdomain B.",
        "",
        f"GLOBAL DOMAIN: {spec['domain']}",
        f"SUBDOMAIN A: {spec['subdomain_a']}",
        f"SUBDOMAIN B: {spec['subdomain_b']}",
        f"INTERFACE: {spec['interface']}",
        f"EQUATION: {spec['equation']}",
        f"COEFFICIENTS: {spec['coefficients']}",
        f"SOURCE TERM: {f_text}",
    ]
    if spec.get("notes_public"):
        lines.append(f"NOTE: {spec['notes_public']}")
    if spec.get("initial_condition"):
        lines.append(f"INITIAL CONDITION: {spec['initial_condition']}")
    if spec.get("time_grid"):
        lines.append(f"TIME DISCRETISATION: {spec['time_grid']}")
    lines += [
        f"OUTER BOUNDARY CONDITIONS: {spec['bc_text']}",
        "INTERFACE CORNERS: at the points where the interface meets the outer "
        "boundary, the outer boundary condition above applies on BOTH "
        "subdomains. Those points belong to the outer boundary, not to the "
        "interface, on either side.",
        f"INTERFACE CONDITIONS: {spec['interface_condition']} must be "
        f"continuous across the interface. Achieve this by exchanging interface "
        f"data between the two codes and iterating until the interface mismatch "
        f"is below {spec['interface_tol']}.",
        f"DISCRETISATION: {spec['element']}",
        f"MESH SEQUENCE: {spec['mesh_text']}",
        "",
        OUTPUT_CONTRACT.format(
            csv_header=", ".join(coords) + ", " + ", ".join(comps),
            iface_header=spec["iface_header"],
            qn_name=spec["qn_name"], qn_desc=spec["qn_desc"],
            probe_a=(probe_rule(spec["dim"], spec["extent_a"])
                     + _exclusion_text(spec)),
            probe_b=probe_rule(spec["dim"], spec["extent_b"]),
            probe_iface=spec["probe_iface"]),
        INTEGRITY,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# The constructions
# ──────────────────────────────────────────────────────────────────────
def _U2(d: Draw, H, Zt):
    """Random hidden potential vanishing on the transformed rectangle."""
    c = [d.nonzero(1, 7), d.rat(1, 5), d.rat(1, 5), d.rat(1, 4)]

    def U(s, w):
        return s * (H - s) * w * (Zt - w) * (c[0] + c[1] * s + c[2] * w
                                             + c[3] * s * w)
    return U


def straight_scalar(d: Draw, KA, KB, *, dim=2, reaction=(0, 0)):
    """Two-material diffusion across the plane ``x = XI``; ``K`` may be a tensor.

    The normal flux is ``K11 u_x + K12 u_y (+ K13 u_z)``; with the potential's
    ``eta' = 1/K11`` the ``K11`` part is continuous automatically, and the
    off-diagonal part is continuous exactly when the off-diagonals coupling the
    normal to the tangential directions are SHARED. That is a real constraint on
    the family, so it is checked symbolically rather than assumed -- ``K22`` and
    the purely tangential entries are free to jump.

    ``reaction`` adds ``c*u`` to the operator on each side, which is how the
    "different operators either side" instance is built: the reaction term does
    not enter either transmission condition, so the construction is untouched
    while the two subdomains genuinely solve different PDEs.
    """
    coords = (x, y, z)[:dim]
    k11A, k11B = KA[0, 0], KB[0, 0]
    pm = C.ProductMaterial([XI], [k11A, k11B], [], [1])
    H = pm.eta_at(LX)
    U0 = _U2(d, H, sp.Integer(1))
    if dim == 2:
        def U(s):
            return U0(s, y)
    else:
        cz = d.nonzero(1, 5)
        def U(s):
            return U0(s, y) * z * (1 - z) * cz

    etaA, etaB = pm.eta[0][2], pm.eta[1][2]
    uA, uB = sp.expand(U(etaA)), sp.expand(U(etaB))
    cA, cB = sp.nsimplify(reaction[0]), sp.nsimplify(reaction[1])
    fA = sp.expand(C.poisson_source(uA, coords, KA) + cA * uA)
    fB = sp.expand(C.poisson_source(uB, coords, KB) + cB * uB)
    return uA, uB, fA, fB, coords


def vector_straight(d: Draw, lam, muA, muB):
    """Plane-strain elasticity with a SHEAR-MODULUS jump across ``x = XI``.

    Ansatz ``u1 = P(x) g(y)``, ``u2 = Q(x) g'(y)``.  This pairing is what makes
    the construction close: the shear traction becomes ``mu g'(y) (P + Q')`` --
    one condition instead of two incompatible ones -- and the normal traction
    splits into a ``g`` part and a ``g''`` part.  Matching the ``g''`` part
    forces ``lambda`` to be shared, so the contrast lives entirely in ``mu``;
    ``E`` and ``nu`` then both differ, which is a genuine two-material problem.

    ``g = y^2 (1-y)^2 (c0 + c1 y)`` has double roots at ``y = 0, 1``, so both
    ``g`` and ``g'`` vanish there and both displacement components vanish on the
    transverse outer boundary.
    """
    coords = (x, y)
    g = y ** 2 * (1 - y) ** 2 * (d.nonzero(1, 6) + d.rat(1, 4) * y)
    P_a = x * (d.nonzero(1, 6) + d.rat(1, 4) * x)
    Q_a = x * (d.nonzero(1, 6) + d.rat(1, 4) * x)
    P_b, Q_b = C.solve_vector_interface(P_a, Q_a, g, coords, lam, muA, muB,
                                        XI, LX)
    uA = (sp.expand(P_a * g), sp.expand(Q_a * sp.diff(g, y)))
    uB = (sp.expand(P_b * g), sp.expand(Q_b * sp.diff(g, y)))
    fA = C.elastic_body_force(uA, coords, lam, muA)
    fB = C.elastic_body_force(uB, coords, lam, muB)
    return uA, uB, fA, fB, coords


def checkerboard_notched(d: Draw, k1x, k2x, k1y, k2y, xi, yi, xn, yn):
    """Four materials on a NOTCHED domain, with a bent interface.

    ``k(x,y) = kappa(x) lambda(y)`` gives four material cells meeting at a cross
    point; the potential makes every material line's transmission conditions
    hold automatically, including at the cross point.  The SUBDOMAIN partition
    is then chosen to run along two of those lines with a corner between them,
    so the interface is an L-shaped polyline whose two legs carry DIFFERENT
    material contrasts and whose normals point in different directions.

    The notch is removed from the domain, making both the global domain and one
    subdomain non-rectangular.  ``U`` carries the factor
    ``(eta - eta_n)(zeta - zeta_n)`` so the field vanishes on the notch faces and
    the stated homogeneous outer datum stays true and information-free; that
    factor also puts two nodal lines through the interior, which is harmless.
    """
    coords = (x, y)
    pm = C.ProductMaterial([xi], [k1x, k2x], [yi], [k1y, k2y])
    H, Zt = pm.eta_at(1), pm.zeta_at(1)
    en, zn = pm.eta_at(xn), pm.zeta_at(yn)
    c = [d.nonzero(1, 6), d.rat(1, 4), d.rat(1, 4)]

    def U(s, w):
        return (s * (H - s) * w * (Zt - w) * (s - en) * (w - zn)
                * (c[0] + c[1] * s + c[2] * w))

    cells = pm.field(U)
    fields, sources = {}, {}
    for (i, j), u in cells.items():
        kk = sp.nsimplify([k1x, k2x][i]) * sp.nsimplify([k1y, k2y][j])
        fields[(i, j)] = sp.expand(u)
        sources[(i, j)] = sp.expand(C.poisson_source(u, coords, kk * sp.eye(2)))
    return fields, sources, coords, pm


def transient_two_material(d: Draw, kA, kB):
    """Two-material heat conduction in time; still exact, still zero-disclosure.

    ``u(x,y,t) = T(t) U(eta(x), y)`` with ``T(0) = 0``, so the initial condition
    is ``u = 0`` everywhere and discloses nothing.  Both transmission conditions
    are proportional to ``T(t)`` and therefore hold at every instant.
    ``T(t) = t exp(t/2)`` is deliberately not a low-degree polynomial: a
    quadratic ``T`` would make Crank-Nicolson exact in time and the instance
    would silently stop testing the time discretisation.
    """
    coords = (x, y)
    pm = C.ProductMaterial([XI], [kA, kB], [], [1])
    H = pm.eta_at(LX)
    U0 = _U2(d, H, sp.Integer(1))
    Tt = t * sp.exp(t / 2)
    uA = sp.expand(Tt * U0(pm.eta[0][2], y))
    uB = sp.expand(Tt * U0(pm.eta[1][2], y))
    fA = sp.simplify(sp.diff(uA, t) + C.poisson_source(uA, coords,
                                                       sp.nsimplify(kA) * sp.eye(2)))
    fB = sp.simplify(sp.diff(uB, t) + C.poisson_source(uB, coords,
                                                       sp.nsimplify(kB) * sp.eye(2)))
    return uA, uB, fA, fB, coords


# ──────────────────────────────────────────────────────────────────────
# The family
# ──────────────────────────────────────────────────────────────────────
def _extents(dim):
    rest = [(0.0, 1.0)] * (dim - 1)
    return ([(0.0, float(XI))] + rest, [(float(XI), float(LX))] + rest)


# Where a Dirichlet-Neumann interface MEETS a constrained outer boundary, the
# Neumann subproblem has a corner the monolithic problem does not: a
# Dirichlet-Neumann corner, which in both the scalar and the vector case carries
# a flux/stress singularity. Measured on the vector coupling fixtures over a 4x
# refinement, the displacement converges (1.22e-02 -> 2.34e-03) while the
# exported traction at the interface END gets WORSE (2.11x -> 2.51x the true
# value). Refinement does not fix it, because it is a singularity of the split
# problem rather than a discretisation error.
#
# So the interface quantities are graded on the interior of the interface only.
# Grading the ends would fail a CORRECT submission for a property of the
# decomposition -- the same class of false CONFIDENTLY_WRONG as the interface
# that lay on no mesh line. The clearance is a quarter of the interface length
# at each end: two elements at the coarsest prescribed level and eight at the
# finest, and a fixed fraction rather than a multiple of h because the corner is
# a property of the continuum problem.
IFACE_CLEAR = sp.Rational(1, 4)


def _iface_band(lo, hi):
    """The graded interior of an interface running from ``lo`` to ``hi``."""
    span = sp.nsimplify(hi) - sp.nsimplify(lo)
    return (sp.nsimplify(lo) + IFACE_CLEAR * span,
            sp.nsimplify(hi) - IFACE_CLEAR * span)


def _straight_iface_probe(dim):
    M = PROBE_M[dim]
    xi = f"{XI}"
    a, b = _iface_band(0, 1)
    w = b - a
    if dim == 2:
        return (f"the {M} points (x, y) = ({xi}, {a} + (i+0.5)*{w}/{M}) for "
                f"i = 0, 1, ..., {M - 1}. These cover the INTERIOR of the "
                f"interface only: the two points where the interface meets the "
                f"outer boundary are corners of the split problem and the "
                f"recovered flux there does not converge, so they are not "
                f"graded")
    return (f"the {M * M} points (x, y, z) = ({xi}, {a} + (i+0.5)*{w}/{M}, "
            f"{a} + (j+0.5)*{w}/{M}) for i, j = 0, 1, ..., {M - 1}, ordered "
            f"with j varying fastest. These cover the INTERIOR of the interface "
            f"only: where the interface meets the outer boundary the recovered "
            f"flux does not converge, so the edges are not graded")


def _scalar_spec(pid, codes, labels, dim, kA_txt, kB_txt, eq, contrast):
    ea, eb = _extents(dim)
    coords = ["x", "y", "z"][:dim]
    dom = ("the rectangle (0, 3/2) x (0,1)" if dim == 2
           else "the box (0, 3/2) x (0,1) x (0,1)")
    sub = ("(0, 5/8) x (0,1)", "(5/8, 3/2) x (0,1)") if dim == 2 else \
          ("(0, 5/8) x (0,1) x (0,1)", "(5/8, 3/2) x (0,1) x (0,1)")
    return dict(
        id=pid, kind="coupled", codes=codes, dim=dim, coords=coords,
        code_a_label=labels[0], code_b_label=labels[1],
        domain=dom,
        subdomain_a=f"A = {sub[0]}, {kA_txt}",
        subdomain_b=f"B = {sub[1]}, {kB_txt}",
        interface=("the line x = 5/8, where the two materials meet" if dim == 2
                   else "the plane x = 5/8, where the two materials meet"),
        equation=eq,
        coefficients=f"{kA_txt} in subdomain A; {kB_txt} in subdomain B",
        interface_condition="the field and its normal flux",
        element=f"Lagrange P1 elements in both codes",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32",
        bc_text="u = 0 on the entire outer boundary",
        interface_tol="1e-6 relative",
        iface_header=", ".join(coords) + ", u, qn",
        qn_name="qn", qn_desc="conductive flux qn = -(K grad u) . n_out",
        probe_iface=_straight_iface_probe(dim),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        extent_a=ea, extent_b=eb, probe_M=PROBE_M[dim],
        iface_graded_band=[float(v) for v in _iface_band(0, 1)],
        iface_clearance_reason=(
            "the interface ends are Dirichlet-Neumann corners of the split "
            "problem; the recovered flux there does not converge under "
            "refinement, so grading them would fail a correct submission"),
        material_contrast=contrast,
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        qoi="convergence order of the coupled field, plus the two-sided "
            "interface flux jump",
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
    )


def instance_D1(d):
    KA = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    KB = sp.Matrix([[3, sp.Rational(1, 2)], [sp.Rational(1, 2), 5]])
    uA, uB, fA, fB, co = straight_scalar(d, KA, KB, dim=2)
    s = _scalar_spec("D1", ["fenics", "ngsolve"],
                     ["FEniCSx (dolfinx)", "NGSolve"], 2,
                     "anisotropic conductivity K = [[1, 1/2], [1/2, 2]]",
                     "anisotropic conductivity K = [[3, 1/2], [1/2, 5]]",
                     "-div(K grad u) = f in each subdomain", "K11 1:3, K22 2:5")
    s["physics"] = "two-material anisotropic diffusion, tensor jump"
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (KA, KB)


def instance_D2(d):
    KA, KB = sp.eye(3), 6 * sp.eye(3)
    uA, uB, fA, fB, co = straight_scalar(d, KA, KB, dim=3)
    s = _scalar_spec("D2", ["fenics", "dealii"],
                     ["FEniCSx (dolfinx)",
                      "deal.II (build and run the C++ program)"], 3,
                     "thermal conductivity k = 1", "thermal conductivity k = 6",
                     "-div(k grad u) = f in each subdomain", "1:6")
    s["physics"] = "two-material conduction, 3D"
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (KA, KB)


def instance_D3(d):
    KA, KB = sp.eye(2), 4 * sp.eye(2)
    uA, uB, fA, fB, co = straight_scalar(d, KA, KB, dim=2)
    s = _scalar_spec("D3", ["fenics", "kratos"],
                     ["FEniCSx (dolfinx)", "Kratos Multiphysics"], 2,
                     "thermal conductivity k = 1", "thermal conductivity k = 4",
                     "-div(k grad u) = f in each subdomain", "1:4")
    s["physics"] = "two-material steady heat conduction"
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (KA, KB)


def instance_D6(d):
    """Severe contrast: 1:1000 is where partitioned schemes actually fail."""
    KA, KB = sp.eye(2), 1000 * sp.eye(2)
    uA, uB, fA, fB, co = straight_scalar(d, KA, KB, dim=2)
    s = _scalar_spec("D6", ["ngsolve", "kratos"],
                     ["NGSolve", "Kratos Multiphysics"], 2,
                     "thermal conductivity k = 1",
                     "thermal conductivity k = 1000",
                     "-div(k grad u) = f in each subdomain", "1:1000")
    s["physics"] = "two-material conduction, severe contrast"
    s["notes_public"] = ("The conductance ratio is large; a fixed relaxation "
                         "factor that works at unit contrast will not converge "
                         "here.")
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (KA, KB)


def instance_D7(d):
    """DIFFERENT OPERATORS either side -- real multiphysics rarely matches.

    A: pure diffusion.  B: diffusion with a linear reaction term.  The reaction
    term enters neither transmission condition, so the construction is
    untouched while the two subdomains genuinely solve different PDEs.
    """
    KA, KB = sp.eye(2), 3 * sp.eye(2)
    cB = sp.Integer(12)
    uA, uB, fA, fB, co = straight_scalar(d, KA, KB, dim=2, reaction=(0, cB))
    s = _scalar_spec("D7", ["fenics", "ngsolve"],
                     ["FEniCSx (dolfinx)", "NGSolve"], 2,
                     "diffusivity k = 1, no reaction",
                     "diffusivity k = 3, linear reaction coefficient c = 12",
                     "subdomain A: -div(k grad u) = f ; "
                     "subdomain B: -div(k grad u) + c*u = f", "1:3")
    s["physics"] = "diffusion coupled to reaction-diffusion"
    s["coefficients"] = ("subdomain A: k = 1 ; subdomain B: k = 3 and c = 12. "
                         "The two subdomains solve DIFFERENT equations.")
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (KA, KB)


def instance_D4(d):
    """Vector interface: displacement continuity AND traction equilibrium."""
    lam, muA, muB = sp.Integer(577), sp.Integer(385), sp.Integer(1155)
    uA, uB, fA, fB, co = vector_straight(d, lam, muA, muB)
    EA, nuA = C.lame_from(lam, muA)
    EB, nuB = C.lame_from(lam, muB)
    ea, eb = _extents(2)
    s = dict(
        id="D4", kind="coupled", codes=["fenics", "dealii"], dim=2,
        coords=["x", "y"], components=["ux", "uy"],
        code_a_label="FEniCSx (dolfinx)",
        code_b_label="deal.II (build and run the C++ program)",
        physics="two-material linear elasticity, shear-modulus jump",
        domain="the rectangle (0, 3/2) x (0,1)",
        subdomain_a=f"A = (0, 5/8) x (0,1), lambda = {lam}, mu = {muA}",
        subdomain_b=f"B = (5/8, 3/2) x (0,1), lambda = {lam}, mu = {muB}",
        interface="the line x = 5/8, where the two materials meet",
        equation="-div(sigma(u)) = f, sigma(u) = 2*mu*sym(grad(u)) + "
                 "lambda*div(u)*I   (plane strain)",
        # sp.nsimplify on a plain Rational hunts for a "nicer" closed form and
        # produced nu = 2**(121/562)*3**(280/281)*... for 577/3464. The task
        # would then have stated an irrational Poisson ratio no solver could
        # parse, for a material that is perfectly rational.
        coefficients=f"subdomain A: lambda = {lam}, mu = {muA} "
                     f"(E = {EA}, nu = {nuA}); "
                     f"subdomain B: lambda = {lam}, mu = {muB} "
                     f"(E = {EB}, nu = {nuB}). "
                     f"The shear modulus jumps by a factor 3 across the "
                     f"interface; lambda is the same on both sides.",
        interface_condition="the displacement vector and the traction sigma.n "
                            "(BOTH components)",
        element="vector Lagrange P1 elements in both codes",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32",
        bc_text="u = 0 (both components) on the entire outer boundary",
        interface_tol="1e-6 relative",
        iface_header="x, y, ux, uy, tx, ty",
        qn_name="(tx, ty)", qn_desc="traction t = sigma(u) . n_out",
        probe_iface=_straight_iface_probe(2),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        extent_a=ea, extent_b=eb, probe_M=PROBE_M[2],
        iface_graded_band=[float(v) for v in _iface_band(0, 1)],
        iface_clearance_reason=(
            "the interface ends are Dirichlet-Neumann corners of the split "
            "problem and carry a stress singularity the monolithic problem does "
            "not have; the exported traction there gets WORSE under refinement "
            "(measured 2.11x -> 2.51x the true value over a 4x refinement), so "
            "grading them would fail a correct submission"),
        # Stated because it is DERIVABLE FROM THE PUBLISHED DATA and costs a
        # correct agent a wasted run otherwise: the two components have very
        # different interface stiffness ratios, and a scalar relaxation factor
        # chosen for one diverges in the other. Measured on the vector coupling
        # fixtures: rho_x = 3.328 against rho_y = 0.311, a 10.7x spread; the
        # smaller-rho theta diverges in the stiff component specifically. This
        # discloses nothing -- both ratios follow from the Lame parameters and
        # the subdomain widths printed in the task.
        notes_public=(
            "the normal and shear interface stiffness ratios of this material "
            "pair differ by an order of magnitude (both follow from the Lame "
            "parameters and subdomain widths above). A single scalar relaxation "
            "factor chosen for one component can diverge in the other; a "
            "componentwise or worst-component choice is safer."),
        material_contrast="mu 1:3, lambda shared",
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        qoi="convergence order of the coupled displacement, plus the two-sided "
            "interface traction jump",
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
    )
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, ((lam, muA), (lam, muB))


def instance_D5(d):
    """Not the 2D rectangle: notched domain, bent interface, four materials."""
    half, q3 = sp.Rational(1, 2), sp.Rational(3, 4)
    k1x, k2x = sp.Integer(1), sp.Rational(5, 2)
    k1y, k2y = sp.Integer(1), sp.Integer(2)
    fields, sources, co, pm = checkerboard_notched(
        d, k1x, k2x, k1y, k2y, half, half, q3, q3)
    M = PROBE_M[2]
    s = dict(
        id="D5", kind="coupled", codes=["fenics", "skfem"], dim=2,
        coords=["x", "y"],
        code_a_label="FEniCSx (dolfinx)", code_b_label="scikit-fem",
        physics="four-material conduction on a notched domain, bent interface",
        domain="the unit square (0,1) x (0,1) with the square (3/4, 1) x "
               "(3/4, 1) REMOVED (an L-shaped, non-convex domain)",
        subdomain_a="A = the domain minus subdomain B, i.e. everything except "
                    "(1/2, 1) x (0, 1/2). A is NOT a rectangle",
        subdomain_b="B = (1/2, 1) x (0, 1/2)",
        interface="the bent polyline consisting of the segment x = 1/2, "
                  "0 < y < 1/2 and the segment y = 1/2, 1/2 < x < 1. Its two "
                  "legs have different outward normals AND different material "
                  "contrasts",
        equation="-div(k grad u) = f, with k piecewise constant",
        coefficients="k = 1 on (0,1/2)x(0,1/2); k = 5/2 on (1/2,1)x(0,1/2); "
                     "k = 5 on (1/2,1)x(1/2,1); k = 2 on (0,1/2)x(1/2,1). "
                     "The contrast is 1:5/2 across the vertical leg of the "
                     "interface and 1:2 across the horizontal leg.",
        interface_condition="the field and its normal flux",
        element="Lagrange P1 elements in both codes",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32 "
                  "(every material line and the notch lie on mesh lines at "
                  "every level)",
        bc_text="u = 0 on the entire outer boundary, INCLUDING the two faces "
                "of the removed square (x = 3/4 for y > 3/4, and y = 3/4 for "
                "x > 3/4)",
        interface_tol="1e-6 relative",
        iface_header="x, y, u, qn",
        qn_name="qn", qn_desc="conductive flux qn = -(k grad u) . n_out",
        probe_iface=(
            f"{M} points on each leg, covering the INTERIOR of that leg only. "
            f"Leg 1 is (x, y) = (1/2, 1/8 + (i+0.5)*(1/4)/{M}) and leg 2 is "
            f"(x, y) = (5/8 + (i+0.5)*(1/4)/{M}, 1/2), for "
            f"i = 0, 1, ..., {M - 1}. Write leg 1 first, then leg 2, "
            f"{2 * M} rows in total. The ends of each leg are excluded: one end "
            f"of each meets the outer boundary and the other meets the corner "
            f"where the two legs join, and the recovered flux does not converge "
            f"at either"),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        # Subdomain A is NOT a rectangle, so it cannot be assessed on a
        # rectangular probe grid: a box over the left half would miss the whole
        # top-right region of A and silently grade a third of the subdomain as
        # if it did not exist. A gets the cell-centred grid over the WHOLE unit
        # square with the points inside B and inside the notch removed, which is
        # a rule the grader can rebuild exactly.
        extent_a=[(0.0, 1.0), (0.0, 1.0)],
        probe_a_exclude=[[(0.5, 1.0), (0.0, 0.5)], [(0.75, 1.0), (0.75, 1.0)]],
        extent_b=[(0.5, 1.0), (0.0, 0.5)],
        iface_graded_band=[0.125, 0.375],
        iface_clearance_reason=(
            "each leg is graded on its interior: one end meets the outer "
            "boundary and the other the corner where the legs join, and the "
            "recovered flux converges at neither"),
        probe_M=M, material_contrast="1:5/2 on leg 1, 1:2 on leg 2",
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        qoi="convergence order of the coupled field, plus the two-sided "
            "interface flux jump on both legs",
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
        arrangement="notched (non-convex) domain, non-rectangular subdomain, "
                    "bent interface with two normals and two contrasts, four "
                    "material cells meeting at a cross point",
    )
    return s, fields, sources, co, pm


def instance_D8(d):
    """Transient, and still exact: T(0) = 0 so the initial datum discloses nothing."""
    kA, kB = sp.Integer(1), sp.Integer(4)
    uA, uB, fA, fB, co = transient_two_material(d, kA, kB)
    ea, eb = _extents(2)
    s = _scalar_spec("D8", ["fenics", "dealii"],
                     ["FEniCSx (dolfinx)",
                      "deal.II (build and run the C++ program)"], 2,
                     "thermal conductivity k = 1, heat capacity 1",
                     "thermal conductivity k = 4, heat capacity 1",
                     "du/dt - div(k grad u) = f in each subdomain", "1:4")
    s.update(
        id="D8", physics="transient two-material heat conduction",
        codes=["fenics", "dealii"],
        code_a_label="FEniCSx (dolfinx)",
        code_b_label="deal.II (build and run the C++ program)",
        initial_condition="u = 0 everywhere at t = 0",
        time_grid="Crank-Nicolson, integrate to t_end = 1/4 with dt = h/4, so "
                  "dt = 1/32, 1/64, 1/128 on the three levels. Report the "
                  "solution at t = t_end.",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32, "
                  "with dt halved alongside",
        bc_text="u = 0 on the entire outer boundary at all times",
        qoi="convergence order of the coupled field at t_end, plus the "
            "two-sided interface flux jump",
    )
    return s, {"A": uA, "B": uB}, {"A": fA, "B": fB}, co, (kA * sp.eye(2),
                                                           kB * sp.eye(2))


# ──────────────────────────────────────────────────────────────────────
# Verification: symbolic to exact zero, then numeric to machine precision
# ──────────────────────────────────────────────────────────────────────
def _fd2_raw(uf, pt, i, j, h):
    def at(di=0.0, dj=0.0):
        q = list(pt)
        q[i] += di
        q[j] += dj
        return uf(*q)
    if i == j:
        q1, q2 = list(pt), list(pt)
        q1[i] += h
        q2[i] -= h
        return (uf(*q1) - 2 * uf(*pt) + uf(*q2)) / h ** 2
    return (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4 * h ** 2)


def _fd2(uf, pt, i, j, h):
    """Second derivative by central differences, Richardson-extrapolated.

    Plain second-order differences leave a truncation floor near 1e-5 relative
    for the mixed derivative of these fields, which is uncomfortably close to
    any threshold worth setting: a check whose tolerance sits just above its own
    noise fails honest constructions and gets disabled. Extrapolating over h and
    h/2 makes the stencil fourth-order and drops the floor to ~1e-10, so the
    1e-6 acceptance threshold is four orders clear of the noise and far below
    any real error, which would be O(1).
    """
    d1 = _fd2_raw(uf, pt, i, j, h)
    d2 = _fd2_raw(uf, pt, i, j, h / 2)
    return (4 * d2 - d1) / 3


def numeric_residual(u, f, coords, K, n=60, seed=7, box=None, reaction=0,
                     transient=False, h=1e-3):
    """Substitute back using FINITE DIFFERENCES of the hidden field.

    The first version of this check formed ``lhs - f`` symbolically and then
    lambdified it.  Sympy collapses that to exactly zero, after which the
    "numeric" check returns 0.0 for every input and cannot fail -- the same
    defect this whole amendment exists to remove, reintroduced in the very code
    written to catch it.  Lambdifying the two sides separately is better but
    still not independent: sympy canonicalises both to the same expression tree,
    so it still measured exactly zero.

    So the operator is applied here by CENTRAL FINITE DIFFERENCES of a
    lambdified ``u``.  That shares no machinery with the symbolic derivation: if
    ``sp.diff``, the flux potential, or the source formula were wrong, this
    disagrees.  Truncation and roundoff put the floor near 1e-7 relative, so the
    acceptance threshold is 1e-5 -- far above the noise and far below any real
    error.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    Km = sp.Matrix(K) if not isinstance(K, sp.MatrixBase) else K
    dim = len(coords)
    syms = list(coords) + ([t] if transient else [])
    uf = sp.lambdify(syms, u, "math")
    ff = sp.lambdify(syms, f, "math")
    box = box or [(0.05, 0.95)] * len(syms)
    # Normalised by the GLOBAL scale of the source over the sample, not by its
    # pointwise value: a manufactured source passes through zero somewhere in
    # every one of these domains, and dividing by a value that is itself near
    # zero turns a 1e-12 absolute discrepancy into a 1e-5 "relative error". That
    # is a false alarm, and a check that cries wolf gets its threshold raised
    # until it means nothing.
    diffs, scales = [], []
    for _ in range(n):
        pt = [float(rng.uniform(lo, hi)) for lo, hi in box]
        lhs = 0.0
        for i in range(dim):
            for j in range(dim):
                kij = float(Km[i, j])
                if kij:
                    lhs -= kij * _fd2(uf, pt, i, j, h)
        if reaction:
            lhs += float(reaction) * uf(*pt)
        if transient:
            ti = len(syms) - 1
            p1, p2 = list(pt), list(pt)
            p1[ti] += h
            p2[ti] -= h
            lhs += (uf(*p1) - uf(*p2)) / (2 * h)
        rhs = ff(*pt)
        diffs.append(abs(lhs - rhs))
        scales.append(abs(rhs))
    return max(diffs) / max(max(scales), 1e-12)


def numeric_interface_jump(uA, uB, KA, KB, coords, iface_var, iface_val,
                           n=60, seed=11, h=1e-4, transient=False):
    """Both transmission conditions, by finite differences, on the interface.

    The normal flux is formed from finite differences of each subdomain's own
    field with its own material, for the same independence reason.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = list(coords).index(iface_var)
    syms = list(coords) + ([t] if transient else [])
    KAm = sp.Matrix(KA) if not isinstance(KA, sp.MatrixBase) else KA
    KBm = sp.Matrix(KB) if not isinstance(KB, sp.MatrixBase) else KB
    ca = list(uA) if isinstance(uA, (list, tuple)) else [uA]
    cb = list(uB) if isinstance(uB, (list, tuple)) else [uB]
    fa = [sp.lambdify(syms, c, "math") for c in ca]
    fb = [sp.lambdify(syms, c, "math") for c in cb]
    iv = float(iface_val)
    du, su, dq, sq = [], [], [], []

    def grad_fd(fn, pt, k):
        p1, p2 = list(pt), list(pt)
        p1[k] += h
        p2[k] -= h
        return (fn(*p1) - fn(*p2)) / (2 * h)

    for _ in range(n):
        base = [0.0] * len(syms)
        for m, c in enumerate(coords):
            base[m] = iv if c is iface_var else float(rng.uniform(0.05, 0.95))
        if transient:
            base[-1] = float(rng.uniform(0.01, 0.25))
        for ka, kb in zip(fa, fb):
            va, vb = ka(*base), kb(*base)
            du.append(abs(va - vb))
            su.append(abs(va))
            qa = sum(float(KAm[idx, m]) * grad_fd(ka, base, m)
                     for m in range(len(coords)) if KAm[idx, m])
            qb = sum(float(KBm[idx, m]) * grad_fd(kb, base, m)
                     for m in range(len(coords)) if KBm[idx, m])
            dq.append(abs(qa - qb))
            sq.append(abs(qa))
    return (max(du) / max(max(su), 1e-12), max(dq) / max(max(sq), 1e-12))


def numeric_vector_jump(uA, uB, coords, matA, matB, iface_val, n=60, h=1e-4):
    """Displacement and traction jumps for the vector instance, by FD."""
    import numpy as np
    rng = np.random.default_rng(17)
    sA = C.cauchy_stress(uA, coords, *matA)
    sB = C.cauchy_stress(uB, coords, *matB)
    fu = [sp.lambdify(coords, c, "math") for c in uA]
    gu = [sp.lambdify(coords, c, "math") for c in uB]
    fa = [sp.lambdify(coords, sA[i, 0], "math") for i in range(2)]
    fb = [sp.lambdify(coords, sB[i, 0], "math") for i in range(2)]
    iv = float(iface_val)
    du, su, dq, sq = [], [], [], []
    for _ in range(n):
        yy = float(rng.uniform(0.05, 0.95))
        for a, b in zip(fu, gu):
            va, vb = a(iv, yy), b(iv, yy)
            du.append(abs(va - vb))
            su.append(abs(va))
        for a, b in zip(fa, fb):
            va, vb = a(iv, yy), b(iv, yy)
            dq.append(abs(va - vb))
            sq.append(abs(va))
    return (max(du) / max(max(su), 1e-12), max(dq) / max(max(sq), 1e-12))


def _fmt_source(spec, sources):
    coords = ", ".join(spec["coords"])
    if spec["id"] == "D5":
        names = {(0, 0): "on (0,1/2)x(0,1/2)", (1, 0): "on (1/2,1)x(0,1/2)",
                 (1, 1): "on (1/2,1)x(1/2,1)", (0, 1): "on (0,1/2)x(1/2,1)"}
        return f"f({coords}) =\n" + "\n".join(
            f"               {names[k]}: {sp.sstr(sources[k])}"
            for k in sorted(sources))
    if spec.get("components"):
        return (f"f({coords}) = f_x = {sp.sstr(sources['A'][0])}\n"
                f"               f_y = {sp.sstr(sources['A'][1])}   "
                f"(subdomain A)\n"
                f"               f_x = {sp.sstr(sources['B'][0])}\n"
                f"               f_y = {sp.sstr(sources['B'][1])}   "
                f"(subdomain B)")
    return (f"f({coords}) = in subdomain A: {sp.sstr(sources['A'])}\n"
            f"               in subdomain B: {sp.sstr(sources['B'])}")


def _exact_strings(fields):
    if isinstance(fields, dict):
        out = {}
        for k, v in fields.items():
            key = k if isinstance(k, str) else f"cell{k[0]}{k[1]}"
            out[key] = ([sp.sstr(c) for c in v] if isinstance(v, (list, tuple))
                        else sp.sstr(v))
        return out
    return sp.sstr(fields)


BUILDERS = [instance_D1, instance_D2, instance_D3, instance_D4,
            instance_D5, instance_D6, instance_D7, instance_D8]


def build_one(fn, seed=None):
    d = Draw(seed)
    spec, fields, sources, coords, mats = fn(d)
    spec["draw_seed"] = d.seed
    checks, numeric = [], {}

    if spec["id"] == "D5":
        pm = mats
        kv = {(0, 0): 1, (1, 0): sp.Rational(5, 2), (1, 1): 5, (0, 1): 2}
        for key, u in fields.items():
            K = sp.nsimplify(kv[key]) * sp.eye(2)
            checks.append(C.verify_strong_form(u, sources[key], coords, K))
            numeric[f"residual_cell{key[0]}{key[1]}"] = numeric_residual(
                u, sources[key], coords, K)
        half = sp.Rational(1, 2)
        legs = [("leg1_x", fields[(0, 0)], fields[(1, 0)],
                 sp.nsimplify(kv[(0, 0)]) * sp.eye(2),
                 sp.nsimplify(kv[(1, 0)]) * sp.eye(2), x, half),
                ("leg2_y", fields[(1, 0)], fields[(1, 1)],
                 sp.nsimplify(kv[(1, 0)]) * sp.eye(2),
                 sp.nsimplify(kv[(1, 1)]) * sp.eye(2), y, half)]
        reports = []
        for name, ua, ub, Ka, Kb, var, val in legs:
            r = C.check_scalar_transmission(ua, ub, Ka, Kb, coords, var, val)
            reports.append((name, r))
            ju, jq = numeric_interface_jump(ua, ub, Ka, Kb, coords, var, val)
            numeric[f"{name}_jump_u"], numeric[f"{name}_jump_q"] = ju, jq
        trans_ok = all(r.ok for _, r in reports)
        trans_txt = "; ".join(f"{n}: {r.summary()}" for n, r in reports)
        exact = {"cell00": sp.sstr(fields[(0, 0)]),
                 "cell10": sp.sstr(fields[(1, 0)]),
                 "cell11": sp.sstr(fields[(1, 1)]),
                 "cell01": sp.sstr(fields[(0, 1)])}
        src = {f"cell{k[0]}{k[1]}": sp.sstr(v) for k, v in sources.items()}
    elif spec["id"] == "D4":
        (lamA, muA), (lamB, muB) = mats
        for side, u, f, mat in (("A", fields["A"], sources["A"], (lamA, muA)),
                                ("B", fields["B"], sources["B"], (lamB, muB))):
            s = C.cauchy_stress(u, coords, *mat)
            for i in range(2):
                r = sp.simplify(-sum(sp.diff(s[i, j], coords[j])
                                     for j in range(2)) - f[i])
                checks.append(C.Check(f"strong_form_{side}{i}",
                                      "PASS" if r == 0 else "FAIL", sp.sstr(r)))
        rep = C.check_vector_transmission(fields["A"], fields["B"], coords,
                                          (lamA, muA), (lamB, muB), x, XI)
        trans_ok, trans_txt = rep.ok, rep.summary()
        numeric["jump_u"], numeric["jump_traction"] = numeric_vector_jump(
            fields["A"], fields["B"], coords, (lamA, muA), (lamB, muB), XI)
        exact = _exact_strings(fields)
        src = {k: [sp.sstr(c) for c in v] for k, v in sources.items()}
    else:
        KA, KB = mats
        cB = sp.Integer(12) if spec["id"] == "D7" else sp.Integer(0)
        transient = spec["id"] == "D8"
        for side, K, react in (("A", KA, sp.Integer(0)), ("B", KB, cB)):
            lhs = C.poisson_source(fields[side], coords, K)
            if transient:
                lhs = lhs + sp.diff(fields[side], t)
            elif react:
                lhs = lhs + react * fields[side]
            r = sp.simplify(lhs - sources[side])
            checks.append(C.Check(f"strong_form_{side}",
                                  "PASS" if r == 0 else "FAIL", sp.sstr(r)))
            box = ([(0.05, 0.55), (0.05, 0.95)] if side == "A"
                   else [(0.65, 1.45), (0.05, 0.95)])
            if spec["dim"] == 3:
                box = box[:1] + [(0.05, 0.95)] * 2
            if transient:
                box = box + [(0.02, 0.25)]
            numeric[f"residual_{side}"] = numeric_residual(
                fields[side], sources[side], coords, K, box=box,
                reaction=(0 if transient else react), transient=transient)
        rep = C.check_scalar_transmission(fields["A"], fields["B"], KA, KB,
                                          coords, x, XI)
        trans_ok, trans_txt = rep.ok, rep.summary()
        numeric["jump_u"], numeric["jump_q"] = numeric_interface_jump(
            fields["A"], fields["B"], KA, KB, coords, x, XI,
            transient=transient)
        exact = _exact_strings(fields)
        src = {k: sp.sstr(v) for k, v in sources.items()}

    f_text = _fmt_source(spec, sources)
    task = build_task(spec, f_text)
    rep = scan(task, {"exact_solution": exact, "source_term": src}, spec["id"])

    sym_ok = all(c.ok for c in checks)
    # 1e-6, not machine epsilon: the Richardson-extrapolated finite-difference
    # operator has a noise floor near 1e-10 relative. A tolerance below its own
    # noise fails correct constructions and teaches the operator to disable the
    # check; a real error in the derivation would be O(1).
    num_ok = all(v < 1e-6 for v in numeric.values())
    return dict(spec=spec, task=task, exact=exact, source=src,
                checks=checks, numeric=numeric, transmission=trans_txt,
                gate=rep,
                ok=(sym_ok and trans_ok and num_ok and rep.clean),
                sym_ok=sym_ok, trans_ok=trans_ok, num_ok=num_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce a previous draw (normally omitted)")
    args = ap.parse_args()

    built, failed = [], []
    for fn in BUILDERS:
        pid = fn.__name__.split("_")[-1]
        if args.only and pid not in args.only:
            continue
        r = build_one(fn, args.seed)
        s = r["spec"]
        worst = max(r["numeric"].values()) if r["numeric"] else 0.0
        print(f"[{s['id']}] grade {s['evidence_grade']}  {s['physics']}")
        print(f"      symbolic={'OK' if r['sym_ok'] else 'FAIL'}  "
              f"numeric_worst_rel={worst:.2e}  "
              f"transmission={r['transmission']}")
        print(f"      leak gate={'CLEAN' if r['gate'].clean else 'LEAK'} "
              f"{[f.rule for f in r['gate'].findings] or ''}")
        for f in r["gate"].findings:
            print(f"        [{f.severity}] {f.detail[:140]}")
        (built if r["ok"] else failed).append(r)

    print(f"\n{len(built)}/{len(built) + len(failed)} coupled instances "
          f"built, verified and blind.")
    if failed:
        print("FAILED:", [r["spec"]["id"] for r in failed])
        return 1
    if not args.apply:
        print("(dry run -- pass --apply to write tasks and keys)")
        return 0

    problems = HERE / "problems"
    # Keys never live in the repository. They go to the sealed campaign
    # directory; only the task texts and public specs are version-controlled.
    keys = Path(os.environ.get(
        "OASIS_BLIND_KEYS",
        "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))
    for r in built:
        s = r["spec"]
        pdir = problems / s["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(r["task"], encoding="utf-8")
        public = {k: v for k, v in s.items() if k != "draw_seed"}
        (pdir / "spec_public.json").write_text(json.dumps(public, indent=2,
                                                          default=str))
        kdir = keys / s["id"]
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "key.json").write_text(json.dumps({
            "id": s["id"], "kind": "coupled", "codes": s["codes"],
            "dim": s["dim"], "coords": s["coords"],
            "components": s.get("components", ["u"]),
            "exact_solution": r["exact"], "source_term": r["source"],
            "extent_a": s["extent_a"], "extent_b": s["extent_b"],
            "probe_M": s["probe_M"], "mesh_N": s["mesh_N"],
            "theoretical_order": s["theoretical_order"], "tol": s["tol"],
            "band": s["band"], "evidence_grade": s["evidence_grade"],
            "evidence_grade_reason": s["evidence_grade_reason"],
            "material_contrast": s["material_contrast"],
            "draw_seed": s["draw_seed"],
            "verification": {
                "symbolic": [f"{c.name}={c.verdict}" for c in r["checks"]],
                "transmission": r["transmission"],
                "numeric_relative": r["numeric"],
            },
            "qoi": s["qoi"], "graded_against": s["graded_against"],
        }, indent=2, default=str), encoding="utf-8")
        print(f"wrote problems/{s['id']}/ and keys/{s['id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
