"""Step 1 of the blind pipeline: hidden solution in, source term out.

This module is the *only* place ``u_exact`` is handled.  It takes a secret task
definition, derives the source term symbolically with sympy, proves the
derivation by substituting back into the PDE, runs the blindness **design**
checks, and then splits its output in two:

* a **blind payload** — domain, PDE, coefficients, source term, boundary data,
  mesh sequence, probes.  Never the solution.
* a **key** — ``u_exact``, the true probe values, the theoretical order.

Design checks (why they exist)
------------------------------
A manufactured source term is built *from* the hidden solution, so it always
carries its fingerprint.  Two failure modes matter:

``PROPORTIONAL``  ``f = c * u_exact`` exactly.  This happens whenever
    ``u_exact`` is an eigenfunction of the operator — the classic
    ``u = sin(pi x) sin(pi y)`` / ``f = 2 pi^2 sin(pi x) sin(pi y)`` pair.
    Publishing ``f`` then *is* publishing ``u_exact`` up to one division.
    **Hard failure.**

``TERM_RECONSTRUCTS``  some single additive term of ``expand(f)`` is
    proportional to ``u_exact``.  Recovery needs the reader to spot which
    summand it is, which is not mechanical, but it is a real weakness.
    **Warning**, surfaced in the audit report.

Both are unavoidable-in-general properties of MMS, which is why they are
checked rather than assumed away.
"""

from __future__ import annotations

import sympy as sp

# Canonical coordinate + time symbols.  Everything downstream uses these.
X, Y, Z, T = sp.symbols("x y z t", real=True)
COORDS = {2: (X, Y), 3: (X, Y, Z)}


# ──────────────────────────────────────────────────────────────────────
# Differential operators
# ──────────────────────────────────────────────────────────────────────
def grad(u, vars_):
    return sp.Matrix([sp.diff(u, v) for v in vars_])


def laplacian(u, vars_):
    return sum(sp.diff(u, v, 2) for v in vars_)


def divergence(vec, vars_):
    return sum(sp.diff(vec[i], vars_[i]) for i in range(len(vars_)))


def sym_grad(u_vec, vars_):
    n = len(vars_)
    return sp.Matrix(n, n, lambda i, j:
                     sp.Rational(1, 2) * (sp.diff(u_vec[i], vars_[j])
                                          + sp.diff(u_vec[j], vars_[i])))


def cauchy_stress(u_vec, vars_, lam, mu):
    n = len(vars_)
    eps = sym_grad(u_vec, vars_)
    tr = sum(eps[i, i] for i in range(n))
    return sp.Matrix(n, n, lambda i, j:
                     2 * mu * eps[i, j] + (lam * tr if i == j else 0))


def div_tensor(sig, vars_):
    n = len(vars_)
    return sp.Matrix(n, 1, lambda i, _:
                     sum(sp.diff(sig[i, j], vars_[j]) for j in range(n)))


def lame(E, nu):
    """Lame parameters from Young's modulus and Poisson ratio (3D / plane strain)."""
    E, nu = sp.nsimplify(E), sp.nsimplify(nu)
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# ──────────────────────────────────────────────────────────────────────
# Source-term derivation, per PDE family
# ──────────────────────────────────────────────────────────────────────
def derive_source(family: str, u_exact, coeffs: dict, dim: int) -> dict:
    """Return {component_name: sympy expression} for the manufactured source."""
    v = COORDS[dim]

    if family == "poisson":
        kappa = sp.nsimplify(coeffs.get("kappa", 1))
        return {"f": sp.simplify(-divergence([kappa * g for g in grad(u_exact, v)], v))}

    if family == "elasticity":
        lam, mu = lame(coeffs["E"], coeffs["nu"])
        sig = cauchy_stress(list(u_exact), v, lam, mu)
        b = -div_tensor(sig, v)
        return {f"f{i + 1}": sp.simplify(b[i]) for i in range(dim)}

    if family == "heat":
        kappa = sp.nsimplify(coeffs.get("kappa", 1))
        return {"f": sp.simplify(sp.diff(u_exact, T) - kappa * laplacian(u_exact, v))}

    if family == "stokes":
        mu = sp.nsimplify(coeffs.get("mu", 1))
        u_vec, p = u_exact["u"], u_exact["p"]
        gp = grad(p, v)
        f = [sp.simplify(-mu * laplacian(u_vec[i], v) + gp[i]) for i in range(dim)]
        return {f"f{i + 1}": f[i] for i in range(dim)}

    if family == "reaction_diffusion":
        du, dv = sp.nsimplify(coeffs["D_u"]), sp.nsimplify(coeffs["D_v"])
        u, w = u_exact["u"], u_exact["v"]
        ru, rv = coeffs["R_u"](u, w), coeffs["R_v"](u, w)
        return {
            "f_u": sp.simplify(sp.diff(u, T) - du * laplacian(u, v) - ru),
            "f_v": sp.simplify(sp.diff(w, T) - dv * laplacian(w, v) - rv),
        }

    if family == "convection_diffusion":
        eps = sp.nsimplify(coeffs["eps"])
        b = [sp.nsimplify(c) for c in coeffs["b"]]
        conv = sum(b[i] * sp.diff(u_exact, v[i]) for i in range(dim))
        return {"f": sp.simplify(-eps * laplacian(u_exact, v) + conv)}

    if family == "helmholtz_reaction":          # -Lap u + u = f  (deal.II step-7)
        return {"f": sp.simplify(-laplacian(u_exact, v) + u_exact)}

    raise ValueError(f"unknown PDE family: {family}")


def verify_residual(family: str, u_exact, source: dict, coeffs: dict, dim: int):
    """Substitute (u_exact, f) back into the PDE; every residual must simplify to 0.

    This is the derivation's own proof.  A silent sign error here would make an
    otherwise perfect blind harness measure the wrong problem.
    """
    v = COORDS[dim]
    res = {}

    if family == "poisson":
        kappa = sp.nsimplify(coeffs.get("kappa", 1))
        res["f"] = -divergence([kappa * g for g in grad(u_exact, v)], v) - source["f"]
    elif family == "elasticity":
        lam, mu = lame(coeffs["E"], coeffs["nu"])
        b = -div_tensor(cauchy_stress(list(u_exact), v, lam, mu), v)
        for i in range(dim):
            res[f"f{i + 1}"] = b[i] - source[f"f{i + 1}"]
    elif family == "heat":
        kappa = sp.nsimplify(coeffs.get("kappa", 1))
        res["f"] = sp.diff(u_exact, T) - kappa * laplacian(u_exact, v) - source["f"]
    elif family == "stokes":
        mu = sp.nsimplify(coeffs.get("mu", 1))
        u_vec, p = u_exact["u"], u_exact["p"]
        gp = grad(p, v)
        for i in range(dim):
            res[f"f{i + 1}"] = (-mu * laplacian(u_vec[i], v) + gp[i]
                                - source[f"f{i + 1}"])
        # incompressibility is part of the problem, so it is part of the proof
        res["div_u"] = divergence(list(u_vec), v)
    elif family == "reaction_diffusion":
        du, dv = sp.nsimplify(coeffs["D_u"]), sp.nsimplify(coeffs["D_v"])
        u, w = u_exact["u"], u_exact["v"]
        res["f_u"] = (sp.diff(u, T) - du * laplacian(u, v)
                      - coeffs["R_u"](u, w) - source["f_u"])
        res["f_v"] = (sp.diff(w, T) - dv * laplacian(w, v)
                      - coeffs["R_v"](u, w) - source["f_v"])
    elif family == "convection_diffusion":
        eps = sp.nsimplify(coeffs["eps"])
        b = [sp.nsimplify(c) for c in coeffs["b"]]
        conv = sum(b[i] * sp.diff(u_exact, v[i]) for i in range(dim))
        res["f"] = -eps * laplacian(u_exact, v) + conv - source["f"]
    elif family == "helmholtz_reaction":
        res["f"] = -laplacian(u_exact, v) + u_exact - source["f"]
    else:
        raise ValueError(f"unknown PDE family: {family}")

    bad = {k: sp.simplify(r) for k, r in res.items()}
    bad = {k: r for k, r in bad.items() if sp.simplify(r) != 0}
    return bad


# ──────────────────────────────────────────────────────────────────────
# Blindness design checks
# ──────────────────────────────────────────────────────────────────────
def is_proportional(a, b) -> tuple[bool, object]:
    """True iff ``a = c * b`` for some nonzero constant ``c``."""
    a, b = sp.simplify(a), sp.simplify(b)
    if a == 0 or b == 0:
        return False, None
    ratio = sp.simplify(sp.cancel(sp.together(a / b)))
    if ratio.free_symbols:
        ratio = sp.simplify(sp.expand(ratio))
    if not ratio.free_symbols and ratio != 0:
        return True, ratio
    return False, None


def _n_terms(expr) -> int:
    return len(sp.Add.make_args(sp.expand(expr)))


def embedded_multiple(f, u):
    """Detect ``f = c*u + remainder`` where ``c*u``'s terms survive intact in ``f``.

    This is the check that matters in practice, and it is strictly stronger than
    both "is ``f`` proportional to ``u``" and "is one term of ``f`` a multiple of
    ``u``".  Two real leaks in the pre-existing campaign motivated it:

    * ``B1``  ``f``'s leading printed term is ``12*pi**2 * u`` — one division.
    * ``D3``  ``f_A = pi**2*u_A - 2*sin(pi*y)``: no *single* term is a multiple
      of ``u_A`` (``u_A`` is itself a two-monomial sum), but a **sub-sum** of
      ``f``'s terms is.  A single-term rule reports this as clean.

    Enumerating subsets is exponential, so instead: for every pair (term of ``f``,
    term of ``u``) whose ratio is a nonzero constant ``c``, subtract ``c*u`` and
    see whether terms *cancel*.  Cancellation is exactly the statement that
    ``c*u``'s terms were present.  Both the as-printed and the expanded form are
    scanned, because the two catch different cases: ``B1`` hides in the printed
    form (expansion scatters it) and ``D3`` shows up only after expansion.

    Returns ``(coefficient, fraction_of_u_cancelled)`` or ``(None, 0.0)``.
    """
    best = (None, 0.0)
    for F, U in ((f, u), (sp.expand(f), sp.expand(u))):
        tf, tu = sp.Add.make_args(F), sp.Add.make_args(U)
        n_f, n_u = _n_terms(F), _n_terms(U)
        if n_f == 0 or n_u == 0:
            continue
        for a in tf:
            for b in tu:
                if b == 0:
                    continue
                c = sp.simplify(sp.cancel(sp.together(a / b)))
                if c.free_symbols or c == 0:
                    continue
                removed = n_f - _n_terms(sp.expand(F - c * U))
                if removed <= 0:
                    continue
                frac = min(1.0, removed / n_u)
                if frac > best[1]:
                    best = (c, frac)
    return best


def design_checks(u_exact, source: dict, dim: int, family: str) -> dict:
    """Hard failures and warnings about how much of ``u_exact`` ``f`` gives away."""
    if family == "elasticity":
        comps = {f"f{i + 1}": u_exact[i] for i in range(dim)}
    elif family == "stokes":
        comps = {f"f{i + 1}": u_exact["u"][i] for i in range(dim)}
    elif family == "reaction_diffusion":
        comps = {"f_u": u_exact["u"], "f_v": u_exact["v"]}
    else:
        comps = {"f": u_exact}

    failures, warnings = [], []
    for key, u_comp in comps.items():
        f = source[key]
        prop, c = is_proportional(f, u_comp)
        if prop:
            failures.append(
                f"PROPORTIONAL: {key} = ({c}) * u_exact — the hidden solution "
                f"is an eigenfunction of this operator, so publishing {key} "
                f"publishes the solution up to one division. Choose a "
                f"non-eigenfunction manufactured field.")
            continue
        coeff, frac = embedded_multiple(f, u_comp)
        if coeff is not None and frac >= 0.999:
            failures.append(
                f"EMBEDDED_FULL: {key} = ({coeff}) * u_exact + remainder — every "
                f"term of the hidden solution survives intact in the published "
                f"source term, so subtracting the remainder recovers it. Choose "
                f"a manufactured field whose terms do not survive into the source.")
        elif coeff is not None and frac > 0:
            warnings.append(
                f"EMBEDDED_PARTIAL: {frac:.0%} of u_exact's terms appear in {key} "
                f"scaled by ({coeff}). Recovery is not mechanical but the source "
                f"term carries a visible fingerprint of the solution.")
    return {"failures": failures, "warnings": warnings}


def boundary_trace_is_zero(u_exact, dim: int, box: tuple) -> bool:
    """True iff ``u_exact`` vanishes on every face of the axis-aligned ``box``.

    A vanishing trace is the cleanest possible Dirichlet datum: the prompt says
    ``u = 0 on the boundary`` and discloses nothing whatever about the interior.
    """
    v = COORDS[dim]
    comps = list(u_exact) if isinstance(u_exact, (list, tuple, sp.Matrix)) else [u_exact]
    for comp in comps:
        for i, var in enumerate(v):
            for value in (box[i][0], box[i][1]):
                if sp.simplify(comp.subs(var, sp.nsimplify(value))) != 0:
                    return False
    return True
