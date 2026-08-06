"""Tier-2: this template's pressure pin is optional, and the warning it
tells you to catch fires for a different fault entirely.

Claim: skfem hydraulic_resistance#2 -- corrected 2026-08-06.  The entry
previously said Stokes saddle points have a 1-D constant-pressure null space,
that "without pinning one pressure DOF the solver fails -- direct LU gets
SingularMatrix", and that the signal is
`MatrixRankWarning: Matrix is exactly singular` or an all-NaN solve.

Measured on skfem 12.0.1 / scipy 1.15.3 with the shipped channel geometry
(Taylor-Hood ElementVector(P2)/P1, MeshTri.init_tensor, no-slip walls, inlet
pressure traction as a natural BC, traction-free outlet):

  * THE PREMISE IS WRONG FOR THIS TEMPLATE.  The inlet traction sets the
    pressure level, so the condensed block has nullity ZERO whether or not
    the outlet pressure DOF is pinned.  Dropping the pin changes neither the
    flow rate nor the computed resistance nor the pressure range.  The pin
    is a convenience that puts p = 0 at the outlet, not a well-posedness
    requirement.
  * THE NULL SPACE IS REAL, but only for ENCLOSED flow -- velocity
    prescribed on the whole boundary.  The same assembly with the inlet and
    outlet velocities also constrained has nullity exactly one.
  * THE SIGNAL IS WRONG.  On that enclosed, genuinely singular system
    scipy emits NOTHING: the catch_warnings record is empty, no exception is
    raised, and the returned velocity is finite and matches the pinned
    solution.  A `catch MatrixRankWarning` guard cannot fire, and an
    isfinite assertion passes too.
  * THE WARNING IS REAL, FOR A DIFFERENT FAULT.  The fixture reproduces it
    twice: on a matrix with a structurally zero pivot, and on the
    equal-order P1/P1 pair that violates inf-sup, on the structured
    MeshTri().refined() mesh this backend's own hydraulic_resistance#5
    names.  In both the solution is non-finite, which is why isfinite is the
    right partner THERE and useless for the null-space case.
  * the inf-sup warning was probed on two cavity meshes, refined() and
    tensor-product, and fires on both, so this fixture does NOT reproduce a
    mesh dependence there.  It did not fire in the channel configuration
    above, but that differs in boundary conditions as well as in mesh, so
    the cause is not isolated here and no claim is made about it.  What the
    fixture does establish is one-directional: the warning is a SUFFICIENT
    signal of a rank-deficient matrix and never a necessary one.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    Functional,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad

L_CHAN, H_CHAN, MU, P_IN = 2.0, 0.1, 1.0, 1.0


@BilinearForm
def a_form(u, v, w):
    return MU * ddot(grad(u), grad(v))


@BilinearForm
def b_form(u, q, w):
    return -div(u) * q


@LinearForm
def inlet_traction(v, w):
    return P_IN * v[0]


@Functional
def outflow(w):
    return w["uh"].value[0]


def channel(velocity_element=None):
    m = MeshTri.init_tensor(np.linspace(0.0, L_CHAN, 41),
                            np.linspace(0.0, H_CHAN, 7))
    m = m.with_boundaries({
        "inlet": lambda x: x[0] < 1e-10,
        "outlet": lambda x: x[0] > L_CHAN - 1e-10,
        "wall": lambda x: (x[1] < 1e-10) | (x[1] > H_CHAN - 1e-10),
    })
    ev = velocity_element or ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    fb = FacetBasis(m, ev, facets=m.boundaries["inlet"], intorder=4)
    F = np.zeros(K.shape[0])
    F[:bu.N] = inlet_traction.assemble(fb)
    return m, ev, bu, bp, K, F


def run(K, F, D, bu, bp, m, ev):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Kc, fc, xc, I = condense(K, F, D=D)
        x = solve(Kc, fc)
        msgs = sorted({str(c.message) for c in caught})
    full = xc.copy()
    full[I] = x
    fbo = FacetBasis(m, ev, facets=m.boundaries["outlet"], intorder=4)
    q = float(outflow.assemble(fbo, uh=fbo.interpolate(full[:bu.N])))
    return full, msgs, q, Kc


def nullity(Kc):
    return int((np.abs(np.linalg.eigvalsh(Kc.toarray())) < 1e-10).sum())


def main() -> int:
    ok = True
    m, ev, bu, bp, K, F = channel()
    wall = bu.get_dofs("wall").all()
    pin = bu.N + int(bp.get_dofs("outlet").all()[0])
    print(f"velocity_dofs={bu.N} pressure_dofs={bp.N} block={K.shape}")

    # --- the template as shipped, and the same thing without the pin ------
    res = {}
    for tag, D in (("pinned", np.concatenate([wall, [pin]])),
                   ("unpinned", wall)):
        full, msgs, q, Kc = run(K, F, D, bu, bp, m, ev)
        res[tag] = (full, msgs, q, nullity(Kc))
        print(f"{tag}_nullity={res[tag][3]}")
        print(f"{tag}_warnings={msgs!r}")
        print(f"{tag}_flow_rate={q:.6e}")
        print(f"{tag}_resistance={P_IN / q:.6e}")
        print(f"{tag}_finite={bool(np.isfinite(full).all())}")

    print(f"open_channel_has_no_nullspace="
          f"{res['pinned'][3] == 0 and res['unpinned'][3] == 0}")
    dq = abs(res["pinned"][2] - res["unpinned"][2]) / abs(res["pinned"][2])
    dv = float(np.abs(res["pinned"][0][:bu.N]
                      - res["unpinned"][0][:bu.N]).max())
    print(f"flow_rate_relative_change_without_pin={dq:.3e}")
    print(f"velocity_change_without_pin={dv:.3e}")
    print(f"pin_is_not_required_for_wellposedness="
          f"{res['unpinned'][3] == 0 and not res['unpinned'][1]
             and dq < 1e-9}")
    if res["pinned"][3] != 0 or res["unpinned"][3] != 0:
        print("FAIL: the open channel DID carry a pressure null space",
              file=sys.stderr)
        ok = False
    if dq >= 1e-9 or dv >= 1e-9:
        print("FAIL: removing the pin changed the answer", file=sys.stderr)
        ok = False

    # --- enclosed flow: the null space is real ----------------------------
    D_enc = np.concatenate([bu.get_dofs("wall").all(),
                            bu.get_dofs("inlet").all(),
                            bu.get_dofs("outlet").all()])
    full_e, msgs_e, _, Kc_e = run(K, F, D_enc, bu, bp, m, ev)
    print(f"enclosed_nullity={nullity(Kc_e)}")
    print(f"enclosed_has_one_dimensional_nullspace={nullity(Kc_e) == 1}")
    print(f"enclosed_warnings={msgs_e!r}")
    print(f"enclosed_emits_matrixrankwarning="
          f"{any('Matrix is exactly singular' in x for x in msgs_e)}")
    print(f"enclosed_solution_finite={bool(np.isfinite(full_e).all())}")
    print(f"enclosed_solution_all_nan={bool(np.isnan(full_e).all())}")
    print(f"catch_matrixrankwarning_guard_would_fire={bool(msgs_e)}")
    print(f"isfinite_guard_would_fire="
          f"{not bool(np.isfinite(full_e).all())}")
    if nullity(Kc_e) != 1:
        print("FAIL: the enclosed system did not have a 1-D null space",
              file=sys.stderr)
        ok = False
    if msgs_e:
        print(f"FAIL: the singular enclosed solve DID warn {msgs_e!r}",
              file=sys.stderr)
        ok = False
    if not np.isfinite(full_e).all():
        print("FAIL: the enclosed solve was non-finite, so isfinite WOULD "
              "have caught it", file=sys.stderr)
        ok = False

    # --- where the warning really comes from -------------------------------
    A = sp.eye(6, format="csr").tolil()
    A[3, 3] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z = spl.spsolve(A.tocsr(), np.ones(6))
        zmsgs = sorted({str(c.message) for c in caught})
    print(f"zero_pivot_warnings={zmsgs!r}")
    print(f"zero_pivot_emits_matrixrankwarning="
          f"{any('Matrix is exactly singular' in x for x in zmsgs)}")
    print(f"zero_pivot_result_all_nan={bool(np.isnan(z).all())}")
    if not any("Matrix is exactly singular" in x for x in zmsgs):
        print("FAIL: a structurally singular matrix did not warn, so the "
              "warning is not real at all", file=sys.stderr)
        ok = False

    # equal-order P1/P1 violates inf-sup.  Per this backend's own
    # hydraulic_resistance#5, the resulting matrix is EXACTLY singular on the
    # structured MeshTri().refined() mesh; on a less symmetric mesh you get a
    # checkerboard instead.  Both branches are measured here, because the
    # warning only exists on one of them.
    def equal_order(mesh):
        e1 = ElementVector(ElementTriP1())
        b1 = Basis(mesh, e1, intorder=4)
        b2 = Basis(mesh, ElementTriP1(), intorder=4)
        Kx = sp.bmat([[a_form.assemble(b1), b_form.assemble(b1, b2).T],
                      [b_form.assemble(b1, b2), None]], format="csr")
        lid = b1.get_dofs(lambda x: x[1] > 1.0 - 1e-10)
        xb = np.zeros(Kx.shape[0])
        xb[lid.nodal["u^1"]] = 1.0
        Dx = np.concatenate([b1.get_dofs().all(), [b1.N]])
        with warnings.catch_warnings(record=True) as cc:
            warnings.simplefilter("always")
            sol = solve(*condense(Kx, np.zeros(Kx.shape[0]), x=xb, D=Dx))
            mm = sorted({str(c.message) for c in cc})
        return sol, mm

    x2, msgs2 = equal_order(MeshTri().refined(3))
    x3, msgs3 = equal_order(MeshTri.init_tensor(np.linspace(0.0, 1.0, 9),
                                                np.linspace(0.0, 1.0, 9)))
    print(f"equal_order_unstructured_warnings={msgs3!r}")
    print(f"equal_order_unstructured_finite={bool(np.isfinite(x3).all())}")
    print(f"infsup_warning_is_mesh_dependent="
          f"{bool(msgs2) and not bool(msgs3)}")
    print(f"equal_order_p1p1_warnings={msgs2!r}")
    print(f"equal_order_p1p1_emits_matrixrankwarning="
          f"{any('Matrix is exactly singular' in x for x in msgs2)}")
    print(f"equal_order_p1p1_finite={bool(np.isfinite(x2).all())}")
    print(f"warning_discriminates_infsup_not_nullspace="
          f"{any('Matrix is exactly singular' in x for x in msgs2)
             and not msgs_e}")
    if not any("Matrix is exactly singular" in x for x in msgs2):
        print("FAIL: the inf-sup-violating pair did not warn either",
              file=sys.stderr)
        ok = False
    if np.isfinite(x2).all():
        print("FAIL: the inf-sup-violating solve stayed finite, so isfinite "
              "does not partner the warning after all", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
