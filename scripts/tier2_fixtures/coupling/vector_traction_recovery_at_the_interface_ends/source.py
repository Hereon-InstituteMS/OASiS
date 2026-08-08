"""Where a Dirichlet-Neumann interface MEETS a Dirichlet outer boundary, the
exported traction does not converge — and the displacement does.

WHY THIS IS HERE AT ALL. The vector pair fixtures verify the exchange on a
patch test, where the exact stress is constant, every recovery is exact and the
two sides' tractions cancel to 1e-9. That is a real verification of the
mechanism and it is also the friendliest possible case for the traction
channel. This fixture measures the case the pair fixtures do not cover, on the
SAME participants and the SAME geometry, driven by boundary data that is not a
solution of the Navier equations so that the answer is genuinely
two-dimensional and the interface traction varies along the interface.

WHAT IT FINDS, and it is a property of the SPLIT, not a bug in any participant.
Splitting the domain gives the Neumann subproblem two corners where its
traction-loaded interface meets a displacement-prescribed y-face. A
Dirichlet-Neumann corner in elasticity carries a stress singularity. The
monolithic problem has no such corner — its interface is interior — so:

  * the coupled DISPLACEMENT converges to the monolithic one, and does so
    cleanly (measured below across three meshes);
  * the recovered TRACTION the Neumann side exports does NOT converge near the
    interface ends. Refining makes the spike WORSE, which is the signature of a
    singularity rather than of a discretisation error: measured, the exported
    t_x reaches roughly twice the true traction at one interface end on a
    13x11 mesh and grows further on 52x44.

CONSEQUENCE, stated plainly: on a vector interface that meets a constrained
boundary, the displacement channel of a partitioned Dirichlet-Neumann coupling
is trustworthy and the exported-traction channel is NOT trustworthy near the
interface ends. Nothing here is tuned to make that go away; it is measured,
asserted in the direction it actually goes, and reported.

The measurement is ONE SOLVE per configuration — the Neumann side is handed the
monolithic interface traction directly — so no iteration or relaxation is in
the way of the effect.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def _run(root, p, poly, pos, role, mesh, ym, um, qm):
    partner = "left" if pos == "right" else "right"
    spec = L.stage(root / f"{pos}_{mesh[0]}x{mesh[1]}", pos, "skfem",
                   L.vector_edits(p, pos, role, partner, mesh, poly=poly),
                   kind="vector")
    wd = Path(spec["work_dir"])
    # The partner's export, in the shipped convention: the two sides' exports
    # are negatives of each other.
    qp = qm if pos == "right" else -qm
    (wd / "imports.json").write_text(json.dumps({partner: {
        "field_name": "displacement", "n_points": len(ym),
        "coordinates": [[p.xi, float(y)] for y in ym],
        "values": um.tolist(), "normal_fluxes": qp.tolist()}}))
    r = subprocess.run(spec["command"], cwd=wd, capture_output=True, text=True,
                       timeout=900)
    if r.returncode != 0:
        raise AssertionError(f"{pos}/{role} {mesh} failed: {r.stderr[-400:]}")
    e = json.loads((wd / "exports.json").read_text())
    c = np.asarray(e["coordinates"], float)
    v = np.asarray(e["values"], float)
    q = np.asarray(e["normal_fluxes"], float)
    keep = (c[:, 1] >= ym.min()) & (c[:, 1] <= ym.max())
    uref = L._map_to(ym, um, c[keep, 1])
    qref = L._map_to(ym, qm, c[keep, 1]) * (-1.0 if pos == "right" else 1.0)
    du = float(np.max(np.abs(v[keep] - uref)) / np.max(np.abs(uref)))
    dq = float(np.max(np.abs(q[keep] - qref)) / np.max(np.abs(qref)))
    peak = float(np.max(np.abs(q[:, 0])) / np.max(np.abs(qref[:, 0])))
    return du, dq, peak


def body() -> None:
    L.require_available("skfem")
    p = L.VECTOR_DEFAULT
    poly = L.VECTOR_BAR_POLY
    # THE MUTATION: use the patch-test boundary data instead, whose exact stress
    # is constant. The corner stops being singular, every recovery becomes
    # exact, and the non-convergence this fixture asserts disappears — which is
    # the point: the effect is a property of the DATA at the corner, and a
    # fixture that would report it either way would be measuring nothing.
    if MUTATE:
        poly = None
    ym, um, qm = L.monolithic_vector_interface_state(p, None if poly is None
                                                     else (poly, poly))
    root = L.workroot("trec")
    meshes = [(13, 11), (26, 22), (52, 44)]
    us, qs, peaks = [], [], []
    for mesh in meshes:
        du, dq, peak = _run(root, p, poly, "right", "neumann", mesh,
                            ym, um, qm)
        us.append(du)
        qs.append(dq)
        peaks.append(peak)
        print(f"neumann_{mesh[0]}x{mesh[1]}_u_rel={du:.4e} "
              f"t_rel={dq:.4e} t_peak_over_true={peak:.4f}")
    # The displacement channel: it must improve, or already sit at roundoff.
    conv = L.check(us[-1] < 0.5 * us[0] or us[-1] < 1e-10,
                   "displacement_did_not_converge",
                   f"{us[0]:.3e} -> {us[-1]:.3e} over a 4x refinement")
    print(f"displacement_converges={bool(conv)}")
    print(f"u_rel_first={us[0]:.4e}")
    print(f"u_rel_last={us[-1]:.4e}")
    print(f"t_peak_first={peaks[0]:.4f}")
    print(f"t_peak_last={peaks[-1]:.4f}")
    print(f"t_rel_last={qs[-1]:.4e}")
    # THE FINDING, asserted in the direction it actually goes: refining makes
    # the exported traction WORSE at the interface end, which no discretisation
    # error does. Asserted in BOTH branches, so the mutation is caught by the
    # assertion that carries the claim rather than by a side effect.
    grows = L.check(peaks[-1] > peaks[0] * 1.05,
                    "traction_peak_did_not_grow",
                    f"{peaks[0]:.4f} -> {peaks[-1]:.4f}: the interface-end "
                    "traction did not worsen under refinement, so the "
                    "singularity this fixture reports is not what is being "
                    "measured here")
    print(f"exported_traction_peak_grows_with_refinement={bool(grows)}")
    print(f"traction_channel_trustworthy_near_interface_ends={not bool(grows)}")
    print(f"mutated={MUTATE}")


L.main(body)
