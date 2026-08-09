"""Tier-2 for fenics mixed_poisson#2: no direct solver is robust across every
(cell type, RT degree, mesh size) combination of this saddle point, the
failures are SPORADIC rather than monotone in degree or mesh size, and the
worst case is a solver that reports success and returns a wrong answer. That is
why a mass-balance check is mandatory and why superlu_dist must not be used as
a blind fallback.

Wrong variant: pick one direct solver, trust KSPConvergedReason. Right variant,
and the mutation: run every configuration through the sequential umfpack
factoriser, which was clean on all of them, and check the mass balance anyway.

Each configuration runs in its OWN process, re-executed from this file, because
a failed factorisation leaves the PETSc objects in a state you do not want the
next configuration to inherit. The mass-balance residual is the l2 norm of the
assembled residual of (div(sigma_h) + f) * q * dx over the DG pressure space,
which is exactly zero for a correct discrete solve.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: superlu_dist returns -11 on
triangles at RT1 with N=24 where MUMPS is clean; MUMPS returns -11 on triangles
at RT2 with N=16 where superlu_dist is clean, and setting mat_mumps_icntl_14 to
200 fixes exactly that case and reproduces umfpack to 1e-15; and on
quadrilaterals at RT2 with N=8 superlu_dist reports KSPConvergedReason 4 with a
mass-balance residual of 1.2e-01 and an algebraic residual of 4.2e-02, while
MUMPS and umfpack on the identical matrix give 2.8e-07 and 2.6e-15. A silent
wrong answer that no solver status reveals.

Mutation control: T2_MUTATE=1 forces umfpack everywhere, nothing diverges,
nothing comes back mass-imbalanced, and the fixture loses its own expectations.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import basix.ufl  # noqa: E402
from petsc4py import PETSc  # noqa: E402

# (label, N, RT degree, quadrilateral?, factoriser, mat_mumps_icntl_14)
CONFIGS = [
    ("tri_rt1_n24_superlu", 24, 1, False, "superlu_dist", ""),
    ("tri_rt1_n24_mumps", 24, 1, False, "mumps", ""),
    ("tri_rt1_n24_umfpack", 24, 1, False, "umfpack", ""),
    ("tri_rt2_n16_mumps", 16, 2, False, "mumps", ""),
    ("tri_rt2_n16_mumps_icntl14", 16, 2, False, "mumps", "200"),
    ("tri_rt2_n16_superlu", 16, 2, False, "superlu_dist", ""),
    ("tri_rt2_n16_umfpack", 16, 2, False, "umfpack", ""),
    ("quad_rt2_n8_superlu", 8, 2, True, "superlu_dist", ""),
    ("quad_rt2_n8_umfpack", 8, 2, True, "umfpack", ""),
]
MASS_TOL = 1.0e-3


def child(n, degree, quad, factoriser, icntl14):
    cell = (dolfinx.mesh.CellType.quadrilateral if quad
            else dolfinx.mesh.CellType.triangle)
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=cell)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    RT = basix.ufl.element("RT", msh.basix_cell(), degree)
    DG = basix.ufl.element("DG", msh.basix_cell(), degree - 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([RT, DG]))
    (sig, u) = ufl.TrialFunctions(W)
    (tau, v) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    nrm = ufl.FacetNormal(msh)
    a = (ufl.inner(sig, tau) + ufl.div(tau) * u + ufl.div(sig) * v) * ufl.dx
    natural = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.isclose(X[1], 0.0) | np.isclose(X[1], 1.0))
    flux = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0))
    tags = dolfinx.mesh.meshtags(msh, fdim, np.sort(natural),
                                 np.full(len(natural), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    L = -f * v * ufl.dx - ufl.sin(5.0 * x[0]) * ufl.dot(tau, nrm) * ds(1)
    V0, _ = W.sub(0).collapse()
    g = dolfinx.fem.Function(V0)
    g.x.array[:] = 0.0
    bcs = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, flux),
        W.sub(0))]
    af, Lf = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType(factoriser)
    if icntl14:
        PETSc.Options()["mat_mumps_icntl_14"] = icntl14
        ksp.setFromOptions()
        pc.setFromOptions()
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    reason = int(ksp.getConvergedReason())

    Q, _ = W.sub(1).collapse()
    q = ufl.TestFunction(Q)
    res = dolfinx.fem.assemble_vector(
        dolfinx.fem.form((ufl.div(w.sub(0)) + f) * q * ufl.dx))
    mass = (float(np.linalg.norm(res.array))
            if np.all(np.isfinite(res.array)) else float("inf"))
    r = A.createVecLeft()
    A.mult(w.x.petsc_vec, r)
    r.axpy(-1.0, b)
    try:
        alg = float(r.norm() / b.norm())
    except Exception:
        alg = float("inf")
    p = np.array(w.sub(1).collapse().x.array)
    print("T2RESULT " + json.dumps(dict(
        reason=reason, mass=mass, alg=alg,
        pmin=float(np.min(p)) if np.all(np.isfinite(p)) else float("nan"),
        pmax=float(np.max(p)) if np.all(np.isfinite(p)) else float("nan"))))


def run_config(label, n, degree, quad, factoriser, icntl14):
    if MUTATE:
        factoriser, icntl14 = "umfpack", ""
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           str(n), str(degree), "1" if quad else "0", factoriser, icntl14]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines():
        if line.startswith("T2RESULT "):
            rec = json.loads(line[len("T2RESULT "):])
            rec["label"] = label
            rec["factoriser"] = factoriser
            return rec
    return dict(label=label, factoriser=factoriser, reason=-999,
                mass=float("inf"), alg=float("inf"),
                pmin=float("nan"), pmax=float("nan"))


def main() -> int:
    results = {}
    for cfg in CONFIGS:
        rec = run_config(*cfg)
        results[cfg[0]] = rec
        print(f"{rec['label']:28s} factoriser={rec['factoriser']:13s} "
              f"reason={rec['reason']:4d} mass_balance={rec['mass']:.3e} "
              f"algebraic_residual={rec['alg']:.3e} "
              f"pressure_range=[{rec['pmin']:.3f}, {rec['pmax']:.3f}]")

    diverged = sorted(k for k, r in results.items() if r["reason"] < 0)
    silent = sorted(k for k, r in results.items()
                    if r["reason"] > 0 and r["mass"] > MASS_TOL)
    print(f"diverged_configurations={diverged}")
    print(f"converged_but_mass_imbalanced_configurations={silent}")

    any_diverged = bool(diverged)
    non_overlapping = (results["tri_rt1_n24_superlu"]["reason"] < 0
                       and results["tri_rt1_n24_mumps"]["reason"] > 0
                       and results["tri_rt2_n16_mumps"]["reason"] < 0
                       and results["tri_rt2_n16_superlu"]["reason"] > 0)
    icntl_fixed = (results["tri_rt2_n16_mumps"]["reason"] < 0
                   and results["tri_rt2_n16_mumps_icntl14"]["reason"] > 0
                   and results["tri_rt2_n16_mumps_icntl14"]["mass"] < MASS_TOL)
    silent_wrong = (results["quad_rt2_n8_superlu"]["reason"] > 0
                    and results["quad_rt2_n8_superlu"]["mass"] > MASS_TOL
                    and results["quad_rt2_n8_umfpack"]["mass"] < MASS_TOL)
    umfpack_clean = all(r["reason"] > 0 and r["mass"] < MASS_TOL
                        for k, r in results.items()
                        if r["factoriser"] == "umfpack")
    print(f"some_configuration_returned_a_negative_converged_reason={any_diverged}")
    print(f"mumps_and_superlu_dist_failure_sets_do_not_overlap={non_overlapping}")
    print(f"mumps_workspace_option_fixed_exactly_the_failing_case={icntl_fixed}")
    print(f"a_converged_solve_returned_a_broken_mass_balance={silent_wrong}")
    print(f"umfpack_was_clean_on_every_configuration_it_ran={umfpack_clean}")
    if (any_diverged and non_overlapping and icntl_fixed and silent_wrong
            and umfpack_clean):
        print("VERDICT=direct_solver_failures_are_sporadic_so_the_mass_balance_check_is_mandatory")
        return 0
    print("VERDICT=every_direct_solver_was_robust_here")
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1",
              sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else "")
        raise SystemExit(0)
    raise SystemExit(main())
