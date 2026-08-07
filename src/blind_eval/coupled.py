"""Coupled manufactured solutions whose transmission conditions have CONTENT.

Why this module exists
----------------------
The pre-existing coupled instances D1, D2 and D4 pass the **same field** and the
**same material** to both sides of the interface, so

    jump_u = 0 ;  jump_flux = 0

holds for any input whatsoever.  The check cannot separate a correct coupling
from a broken one, from a constant, from noise.  It is not a weak check; it is
not a check.  D3 is the one instance with content, because its conductivity
jumps across the interface and its field is genuinely piecewise.

Two things follow, and both are implemented here.

1. **The construction must make the interface conditions binding.**  Different
   materials either side, and a field that is genuinely different either side,
   so that continuity is a constraint the construction has to *satisfy* rather
   than a statement about one function evaluated twice.

2. **The verification must refuse to certify a tautology.**  Every check in
   here returns a report that is ``VACUOUS`` — not ``PASS`` — when the two sides
   are the same expression or the two materials are the same.  A harness that
   cannot notice it is checking nothing is how the defect survived review.

The generalisation of D3
------------------------
D3 works by solving the interface coefficients symbolically from continuity,
flux match and the far-wall condition.  That is correct but bespoke: it fixes a
quadratic profile for one interface, one direction, one isotropic pair.

The generalisation used here is the **flux potential**.  For a coefficient that
factorises as ``k(x, y) = kappa(x) * lambda(y)`` define

    eta(x) = int_0^x dt / kappa(t)      zeta(y) = int_0^y dt / lambda(t)

and take ``u(x, y) = U(eta(x), zeta(y))`` for ANY smooth ``U``.  Then across a
vertical material line

    u                 is continuous because eta is continuous, and
    k du/dx = kappa*lambda * U_eta * (1/kappa) = lambda * U_eta

is continuous because ``lambda`` does not jump at fixed ``y``.  Both
transmission conditions hold **identically, for every U**, at every material
line, including where several meet.  D3's hand-solved quadratic is the special
case ``U`` polynomial, one interface, ``lambda = 1``.

This buys three things the bespoke construction cannot: arbitrary hidden fields
(so the builder can draw them at random and hold no answer), several interfaces
at once, and interfaces in both coordinate directions meeting at a point.

The vector case is genuinely different and gets its own construction, because
the transmission condition is a traction and no single potential linearises it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

X, Y, Z = sp.symbols("x y z", real=True)


# ──────────────────────────────────────────────────────────────────────
# Verification reports that can say "this proved nothing"
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Check:
    name: str
    verdict: str                    # PASS | FAIL | VACUOUS
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class TransmissionReport:
    checks: list = field(default_factory=list)

    def add(self, name, verdict, detail=""):
        self.checks.append(Check(name, verdict, detail))
        return self

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.verdict == "PASS" for c in self.checks)

    @property
    def vacuous(self) -> list:
        return [c.name for c in self.checks if c.verdict == "VACUOUS"]

    @property
    def failed(self) -> list:
        return [c.name for c in self.checks if c.verdict == "FAIL"]

    def summary(self) -> str:
        if self.vacuous:
            return f"VACUOUS ({', '.join(self.vacuous)}): the check cannot fail"
        if self.failed:
            return f"FAIL ({', '.join(self.failed)})"
        return f"PASS ({len(self.checks)} binding checks)"


def _zero(e) -> bool:
    return sp.simplify(sp.expand(e)) == 0


def _differ(a, b) -> bool:
    """Are these two genuinely different expressions?"""
    return not _zero(sp.sympify(a) - sp.sympify(b))


# ──────────────────────────────────────────────────────────────────────
# Scalar transmission across an axis-aligned interface
# ──────────────────────────────────────────────────────────────────────
def check_scalar_transmission(u_a, u_b, K_a, K_b, coords, iface_var,
                              iface_val) -> TransmissionReport:
    """Field and normal-flux continuity, with a vacuity guard in front.

    ``K_a``/``K_b`` are constant matrices (``sp.Matrix``) or scalars.  The guard
    is the whole point: if the caller hands the same field and the same material
    to both sides, the two jumps are zero for a reason that has nothing to do
    with the construction being right, and this reports VACUOUS.
    """
    rep = TransmissionReport()
    Ka = sp.Matrix(K_a) if not isinstance(K_a, sp.MatrixBase) else K_a
    Kb = sp.Matrix(K_b) if not isinstance(K_b, sp.MatrixBase) else K_b
    same_field = not _differ(u_a, u_b)
    same_mat = _zero(sp.Matrix(Ka - Kb))

    if same_field and same_mat:
        return rep.add("non_tautological", "VACUOUS",
                       "the same field and the same material were passed to "
                       "both sides: jump_u = jump_flux = 0 holds for ANY input, "
                       "so neither continuity check can fail")
    if same_mat:
        rep.add("materials_differ", "VACUOUS",
                "the two subdomains carry the same material, so a coupling "
                "that transmits du/dn instead of k du/dn is indistinguishable "
                "from a correct one")
    else:
        rep.add("materials_differ", "PASS", f"K_a - K_b = {sp.sstr(Ka - Kb)}")
    if same_field:
        rep.add("fields_differ", "VACUOUS",
                "both subdomains carry the same expression; field continuity "
                "is an identity, not a constraint")
    else:
        rep.add("fields_differ", "PASS", "the two subdomain fields are "
                                         "different expressions")

    jump_u = sp.simplify(u_a.subs(iface_var, iface_val)
                         - u_b.subs(iface_var, iface_val))
    rep.add("field_continuity", "PASS" if jump_u == 0 else "FAIL",
            f"jump_u = {sp.sstr(jump_u)}")

    idx = list(coords).index(iface_var)
    ga = sp.Matrix([sp.diff(u_a, c) for c in coords])
    gb = sp.Matrix([sp.diff(u_b, c) for c in coords])
    qa = (Ka * ga)[idx].subs(iface_var, iface_val)
    qb = (Kb * gb)[idx].subs(iface_var, iface_val)
    jump_q = sp.simplify(qa - qb)
    rep.add("flux_continuity", "PASS" if jump_q == 0 else "FAIL",
            f"jump_flux = {sp.sstr(jump_q)}")

    # A continuity condition satisfied by both sides being ZERO on the
    # interface is satisfied for a degenerate reason: the graded quantity has
    # no signal there and an agent reporting zeros would look perfect.
    val = sp.simplify(u_a.subs(iface_var, iface_val))
    rep.add("interface_value_nonzero", "PASS" if val != 0 else "VACUOUS",
            f"u|iface = {sp.sstr(val)}")
    rep.add("interface_flux_nonzero", "PASS" if sp.simplify(qa) != 0 else "VACUOUS",
            f"q|iface = {sp.sstr(sp.simplify(qa))}")
    return rep


def check_transmission_is_falsifiable(u_a, u_b, K_a, K_b, coords, iface_var,
                                      iface_val, perturbation=None) -> Check:
    """The antidote: perturb the construction and require the check to FAIL.

    A verification that has never been seen to fail is not known to be able to.
    This mutates the B-side field by a term that vanishes nowhere on the
    interface and asserts that continuity breaks.
    """
    eps = perturbation if perturbation is not None else sp.Rational(1, 7)
    u_b_bad = u_b + eps
    rep = check_scalar_transmission(u_a, u_b_bad, K_a, K_b, coords,
                                    iface_var, iface_val)
    broke = "field_continuity" in rep.failed
    return Check("transmission_is_falsifiable", "PASS" if broke else "FAIL",
                 f"perturbing the B-side field by {sp.sstr(eps)} "
                 f"{'breaks' if broke else 'DOES NOT BREAK'} the continuity check")


# ──────────────────────────────────────────────────────────────────────
# The flux potential: piecewise materials with automatic transmission
# ──────────────────────────────────────────────────────────────────────
def flux_potential(var, breaks: list, values: list):
    """Piecewise-linear ``eta`` with slope ``1/values[i]`` on segment ``i``.

    ``breaks`` are the interior break coordinates, ascending; ``values`` has one
    more entry than ``breaks``.  Returns a list of ``(lo, hi, expr)`` covering
    the axis, with ``eta`` continuous by construction.
    """
    if len(values) != len(breaks) + 1:
        raise ValueError("values must have exactly one more entry than breaks")
    segs, acc, lo = [], sp.Integer(0), None
    edges = [None] + list(breaks) + [None]
    for i, v in enumerate(values):
        a, b = edges[i], edges[i + 1]
        expr = acc + (var - (a if a is not None else 0)) / sp.nsimplify(v)
        segs.append((a, b, sp.simplify(expr)))
        if b is not None:
            acc = sp.simplify(expr.subs(var, b))
    return segs


def potential_value(segs, at):
    """Evaluate the piecewise potential at a coordinate value."""
    for a, b, e in segs:
        if (a is None or at >= a) and (b is None or at <= b):
            return sp.simplify(e.subs_dict if False else e.subs(list(e.free_symbols)[0], at)) \
                if len(e.free_symbols) == 1 else sp.simplify(e)
    raise ValueError(f"{at} is outside the potential's range")


@dataclass
class ProductMaterial:
    """``k(x, y) = kappa(x) * lambda(y)`` — the class the potential linearises.

    A jump in ``kappa`` at ``x = xi`` is a material interface with normal ``e_x``
    across which ``lambda`` is unchanged; a jump in ``lambda`` is the same in
    ``y``.  Every such interface, and every point where two of them meet, gets
    both transmission conditions for free.
    """

    x_breaks: list
    x_values: list
    y_breaks: list
    y_values: list

    def __post_init__(self):
        self.eta = flux_potential(X, self.x_breaks, self.x_values)
        self.zeta = flux_potential(Y, self.y_breaks, self.y_values)

    def eta_at(self, xv):
        return _seg_eval(self.eta, X, xv)

    def zeta_at(self, yv):
        return _seg_eval(self.zeta, Y, yv)

    def k(self, xv, yv):
        return sp.nsimplify(self._pick(self.x_values, self.x_breaks, xv)) * \
            sp.nsimplify(self._pick(self.y_values, self.y_breaks, yv))

    @staticmethod
    def _pick(values, breaks, at):
        for i, b in enumerate(breaks):
            if at < b:
                return values[i]
        return values[-1]

    def field(self, U):
        """The hidden field on each material cell, as an explicit expression."""
        cells = {}
        xs = [None] + list(self.x_breaks) + [None]
        ys = [None] + list(self.y_breaks) + [None]
        for i, (a, b, ex) in enumerate(self.eta):
            for j, (c, d, ez) in enumerate(self.zeta):
                cells[(i, j)] = sp.simplify(U(ex, ez))
        return cells


def _seg_eval(segs, var, at):
    for a, b, e in segs:
        if (a is None or at >= a) and (b is None or at <= b):
            return sp.simplify(e.subs(var, at))
    raise ValueError(f"{at} outside range")


def poisson_source(u, coords, K):
    """``f = -div(K grad u)`` for a constant matrix (or scalar) ``K``."""
    Km = sp.Matrix(K) if not isinstance(K, sp.MatrixBase) else K
    g = sp.Matrix([sp.diff(u, c) for c in coords])
    q = Km * g
    return sp.simplify(-sum(sp.diff(q[i], coords[i]) for i in range(len(coords))))


def verify_strong_form(u, f, coords, K) -> Check:
    res = sp.simplify(poisson_source(u, coords, K) - f)
    return Check("strong_form", "PASS" if res == 0 else "FAIL",
                 f"residual = {sp.sstr(res)}")


# ──────────────────────────────────────────────────────────────────────
# Vector transmission: displacement AND traction, with a material jump
# ──────────────────────────────────────────────────────────────────────
def cauchy_stress(u_vec, coords, lam, mu):
    n = len(coords)
    eps = sp.Matrix(n, n, lambda i, j:
                    sp.Rational(1, 2) * (sp.diff(u_vec[i], coords[j])
                                         + sp.diff(u_vec[j], coords[i])))
    tr = sum(eps[i, i] for i in range(n))
    return sp.Matrix(n, n, lambda i, j:
                     2 * mu * eps[i, j] + (lam * tr if i == j else 0))


def elastic_body_force(u_vec, coords, lam, mu):
    n = len(coords)
    s = cauchy_stress(u_vec, coords, lam, mu)
    return [sp.simplify(-sum(sp.diff(s[i, j], coords[j]) for j in range(n)))
            for i in range(n)]


def check_vector_transmission(uA, uB, coords, matA, matB,
                              iface_var, iface_val) -> TransmissionReport:
    """Displacement continuity and traction continuity, both components.

    Same vacuity guard.  ``matA``/``matB`` are ``(lambda, mu)`` pairs.  A
    construction continuous in the normal traction but not in the shear would
    make a partitioned scheme fail for a reason the problem statement never
    mentions, so both components are checked, and the shear traction is required
    to be NONZERO — otherwise the vector instance is two scalar problems wearing
    a coat.
    """
    rep = TransmissionReport()
    same_field = not any(_differ(a, b) for a, b in zip(uA, uB))
    same_mat = (sp.simplify(matA[0] - matB[0]) == 0
                and sp.simplify(matA[1] - matB[1]) == 0)
    if same_field and same_mat:
        return rep.add("non_tautological", "VACUOUS",
                       "same displacement field and same material on both "
                       "sides: displacement and traction continuity are "
                       "identities that hold for any input")
    rep.add("materials_differ", "PASS" if not same_mat else "VACUOUS",
            f"(lambda,mu)_A - (lambda,mu)_B = "
            f"({sp.sstr(sp.simplify(matA[0] - matB[0]))}, "
            f"{sp.sstr(sp.simplify(matA[1] - matB[1]))})")
    rep.add("fields_differ", "PASS" if not same_field else "VACUOUS",
            "the two subdomain displacement fields are different expressions")

    for i, (a, b) in enumerate(zip(uA, uB)):
        j = sp.simplify(a.subs(iface_var, iface_val) - b.subs(iface_var, iface_val))
        rep.add(f"displacement_continuity_{i}", "PASS" if j == 0 else "FAIL",
                f"jump = {sp.sstr(j)}")

    idx = list(coords).index(iface_var)
    sA = cauchy_stress(uA, coords, *matA)
    sB = cauchy_stress(uB, coords, *matB)
    for i in range(len(coords)):
        j = sp.simplify(sA[i, idx].subs(iface_var, iface_val)
                        - sB[i, idx].subs(iface_var, iface_val))
        rep.add(f"traction_continuity_{i}", "PASS" if j == 0 else "FAIL",
                f"jump = {sp.sstr(j)}")

    shear = sp.simplify(sA[1 - idx, idx].subs(iface_var, iface_val))
    rep.add("shear_traction_nonzero", "PASS" if shear != 0 else "VACUOUS",
            f"sigma_shear|iface = {sp.sstr(shear)}; a zero shear traction "
            f"would let the instance be passed as two scalar problems")
    normal = sp.simplify(sA[idx, idx].subs(iface_var, iface_val))
    rep.add("normal_traction_nonzero", "PASS" if normal != 0 else "VACUOUS",
            f"sigma_nn|iface = {sp.sstr(normal)}")
    return rep


def solve_vector_interface(P_a, Q_a, g, coords, lam, mu_a, mu_b,
                           xi, x_far) -> tuple:
    """B-side profiles from displacement + traction continuity at ``x = xi``.

    Ansatz ``u1 = P(x) g(y)``, ``u2 = Q(x) g'(y)``.  With this pairing the shear
    traction is ``mu g'(y) (P + Q')`` — a single condition rather than two
    incompatible ones — and the normal traction splits into a ``g`` part and a
    ``g''`` part.  Matching the ``g''`` part forces ``lambda`` to be common
    across the interface, so the material contrast lives entirely in ``mu``.
    That is a real two-material problem: ``E`` and ``nu`` both differ.

    Returns ``(P_b, Q_b)``, each a quadratic in ``(x - xi)``.
    """
    x = coords[0]
    c = sp.symbols("c0:3")
    d = sp.symbols("d0:3")
    P_b = c[0] + c[1] * (x - xi) + c[2] * (x - xi) ** 2
    Q_b = d[0] + d[1] * (x - xi) + d[2] * (x - xi) ** 2

    solP = sp.solve([
        sp.Eq(P_b.subs(x, xi), P_a.subs(x, xi)),
        sp.Eq((lam + 2 * mu_b) * sp.diff(P_b, x).subs(x, xi),
              (lam + 2 * mu_a) * sp.diff(P_a, x).subs(x, xi)),
        sp.Eq(P_b.subs(x, x_far), 0),
    ], list(c), dict=True)
    if len(solP) != 1:
        raise ValueError(f"P_b under/over-determined: {solP}")
    P_b = sp.simplify(P_b.subs(solP[0]))

    shear_a = mu_a * (P_a.subs(x, xi) + sp.diff(Q_a, x).subs(x, xi))
    solQ = sp.solve([
        sp.Eq(Q_b.subs(x, xi), Q_a.subs(x, xi)),
        sp.Eq(mu_b * (P_b.subs(x, xi) + sp.diff(Q_b, x).subs(x, xi)), shear_a),
        sp.Eq(Q_b.subs(x, x_far), 0),
    ], list(d), dict=True)
    if len(solQ) != 1:
        raise ValueError(f"Q_b under/over-determined: {solQ}")
    return P_b, sp.simplify(Q_b.subs(solQ[0]))


def lame_from(lam, mu):
    """Young's modulus and Poisson ratio, so the task can state real materials."""
    lam, mu = sp.sympify(lam), sp.sympify(mu)
    E = sp.simplify(mu * (3 * lam + 2 * mu) / (lam + mu))
    nu = sp.simplify(lam / (2 * (lam + mu)))
    return E, nu


# ──────────────────────────────────────────────────────────────────────
def boundary_trace_vanishes(u, coords, faces) -> Check:
    """``faces`` is a list of ``(var, value)``; every trace must be zero."""
    bad = []
    comps = list(u) if isinstance(u, (list, tuple)) else [u]
    for comp in comps:
        for var, val in faces:
            t = sp.simplify(comp.subs(var, sp.nsimplify(val)))
            if t != 0:
                bad.append(f"{var}={val} -> {sp.sstr(t)}")
    return Check("outer_boundary_trace_zero", "PASS" if not bad else "FAIL",
                 "; ".join(bad) or "all stated outer traces vanish")
