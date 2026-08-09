"""Tier-2 Layer-C: FEBio numerical-correctness gate (hex8 patch test).

This is the FEBio equivalent of the manufactured-solution
fixtures the other backends carry (fenics / skfem / ngsolve
elasticity_mms_convergence, dune poisson3d_mms, fourc
thermo_transient_mms). It does not check that a .feb file
parses — it checks that FEBio SOLVES CORRECTLY, against a
closed form that is exact for the discretisation.

THE TEST
--------
A homogeneous deformation x = F.X is exactly representable by
trilinear hex8 shape functions, so a conforming FE solution
must reproduce it to round-off. That is the classical patch
test:

  * mesh a 2x2x2 hex8 grid on [0,1]^3 (27 nodes, 8 elements,
    exactly ONE interior node at (0.5,0.5,0.5));
  * prescribe u = (F - I).X on all 26 boundary nodes with the
    `prescribed deformation` BC (FEBCPrescribedDeformation:
    u = scale*(F*X - X));
  * the interior node is FREE — the solver has to find it.

Two independent numeric assertions:

  1. the free interior node must land on the exact linear
     field, |u_fe - (F-I).X_c| <= 1e-12;
  2. the Cauchy stress in every element must equal the
     closed-form St.Venant-Kirchhoff pushforward
        S     = lambda*tr(E)*I + 2*mu*E,  E = (F^T F - I)/2
        sigma = F S F^T / J,              J = det F
     to 1e-8 absolute, and be uniform across all 8 elements.

Assertion 2 is also what PINS the constitutive law: FEBio's
`isotropic elastic` is NOT small-strain linear elasticity, it
is SVK evaluated on the Green-Lagrange strain
(FEIsotropicElastic::Stress computes the algebraically
identical b*(lam*trE - mu)/J + b^2*mu/J). If a future FEBio
release changed that material — or if someone "fixed" the
catalog to describe it as linear elasticity — this fixture
fails with a concrete number rather than a style complaint.

Verified 2026-08-03 on FEBio 4.12.0.86045466d (source build,
USE_MKL=OFF, skyline): interior-node error 1.39e-17, max
stress error 3.46e-11, element-to-element spread 1.00e-13.

Deliberately NOT using numpy: the tier-2 runner invokes febio
fixtures with plain `sys.executable`, which is whatever python
happens to be driving the harness.

MUTATION CONTROL. This fixture is a numerical-correctness gate, so the
control shows the gate can fail rather than that a pathology can be
removed. T2_MUTATE=1 builds the expected Cauchy stress from SMALL-STRAIN
linear elasticity instead of the St.Venant-Kirchhoff pushforward — i.e.
it asserts the very constitutive law this fixture exists to rule out.
The stress error then leaves the 1e-8 band and 'febio_patch_test=passed'
is no longer printed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATE = os.environ.get("T2_MUTATE") == "1"

E_MOD = 1000.0
POISSON = 0.3
# A general homogeneous deformation: stretch in all three axes
# plus two shears, so the test exercises off-diagonal stress.
F = [[1.02, 0.01, 0.0],
     [0.0, 0.99, 0.005],
     [0.0, 0.0, 1.01]]

TOL_DISP = 1.0e-12
TOL_STRESS = 1.0e-8
TOL_UNIFORM = 1.0e-8


# ── tiny 3x3 linear algebra (stdlib only) ──────────────────
def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3))
             for j in range(3)] for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def det3(A):
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def find_febio() -> Path | None:
    env = os.environ.get("FEBIO_BINARY")
    if env and Path(env).is_file():
        return Path(env)
    for c in (Path.home() / "FEBio" / "bin" / "febio4",
              Path.home() / "FEBioStudio" / "bin" / "febio4",
              Path("/opt/febio/bin/febio4"),
              Path("/usr/local/bin/febio4")):
        if c.is_file():
            return c
    w = shutil.which("febio4") or shutil.which("febio")
    return Path(w) if w else None


def build_deck(n: int = 2, L: float = 1.0) -> tuple[str, int]:
    """Return (.feb text, global id of the single interior node)."""
    nid: dict[tuple[int, int, int], int] = {}
    coords: dict[int, tuple[float, float, float]] = {}
    k = 1
    for kk in range(n + 1):
        for jj in range(n + 1):
            for ii in range(n + 1):
                nid[(ii, jj, kk)] = k
                coords[k] = (ii * L / n, jj * L / n, kk * L / n)
                k += 1
    node_lines = "\n".join(
        f'      <node id="{i}">{c[0]:.16g},{c[1]:.16g},{c[2]:.16g}</node>'
        for i, c in sorted(coords.items()))
    elem_lines = []
    e = 1
    for kk in range(n):
        for jj in range(n):
            for ii in range(n):
                conn = [nid[(ii, jj, kk)], nid[(ii + 1, jj, kk)],
                        nid[(ii + 1, jj + 1, kk)], nid[(ii, jj + 1, kk)],
                        nid[(ii, jj, kk + 1)], nid[(ii + 1, jj, kk + 1)],
                        nid[(ii + 1, jj + 1, kk + 1)],
                        nid[(ii, jj + 1, kk + 1)]]
                elem_lines.append(
                    f'      <elem id="{e}">'
                    f'{",".join(str(c) for c in conn)}</elem>')
                e += 1
    on_bnd = []
    interior = []
    for i, c in sorted(coords.items()):
        if any(abs(c[d]) < 1e-12 or abs(c[d] - L) < 1e-12
               for d in range(3)):
            on_bnd.append(i)
        else:
            interior.append(i)
    if len(interior) != 1:
        raise RuntimeError(
            f"expected exactly 1 interior node, got {len(interior)}")
    fstr = ",".join(f"{v:.16g}" for row in F for v in row)
    deck = f'''<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="M1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E_MOD}</E>
      <v>{POISSON}</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="N">
{node_lines}
    </Nodes>
    <Elements type="hex8" name="P1">
{chr(10).join(elem_lines)}
    </Elements>
    <NodeSet name="bnd">{",".join(str(i) for i in on_bnd)}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="P1" mat="M1"/>
  </MeshDomains>
  <Boundary>
    <bc name="patch" type="prescribed deformation" node_set="bnd">
      <scale lc="1">1.0</scale>
      <F>{fstr}</F>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="ux;uy;uz" delim="," file="nd.csv"/>
      <element_data data="sx;sy;sz;sxy;syz;sxz;J" delim=","
                    file="ed.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''
    return deck, interior[0]


def parse_last_block(path: Path) -> list[list[float]]:
    """FEBio logfile CSVs are '*Step/*Time/*Data' blocks; the
    last block is the final state. Step 0 is always written, so
    a file with a single block means nothing was solved."""
    blocks: list[list[list[float]]] = []
    cur: list[list[float]] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("*Step"):
            cur = []
            blocks.append(cur)
        elif line.startswith("*"):
            continue
        elif cur is not None and line.strip():
            cur.append([float(v) for v in line.split(",")])
    if len(blocks) < 2:
        raise RuntimeError(
            "logfile CSV has fewer than 2 blocks — FEBio wrote the "
            "step-0 state only, i.e. nothing was solved")
    return blocks[-1]


def main() -> int:
    binary = find_febio()
    if binary is None:
        # A missing binary is a FAILURE, not a skip. This fixture
        # exists to verify solver behaviour; with no solver it
        # verifies nothing, and returning 0 would let the tier-2
        # floor certify an unrun check. "FAIL:" is in this
        # fixture's forbid_in_output, so the runner records a
        # failure even if the exit status were ignored.
        print("FAIL: FEBio binary not found. This fixture RUNS the "
              "solver and cannot be satisfied without it. Set "
              "FEBIO_BINARY or symlink ~/FEBio/bin/febio4.")
        return 1

    deck, interior_id = build_deck()
    work = Path(tempfile.mkdtemp(prefix="febio_patch_"))
    try:
        (work / "in.feb").write_text(deck)
        proc = subprocess.run(
            [str(binary), "-i", "in.feb", "-nosplash"],
            cwd=str(work), capture_output=True, text=True, timeout=180)
        out = (proc.stdout or "") + (proc.stderr or "")
        normal = "N O R M A L   T E R M I N A T I O N" in out
        print(f"febio_exit_code={proc.returncode}")
        print(f"febio_normal_termination={int(normal)}")
        if proc.returncode != 0 or not normal:
            print("FAIL: FEBio did not reach normal termination")
            print(out[-1500:])
            return 1

        # ── assertion 1: free interior node on the exact field ──
        rows = parse_last_block(work / "nd.csv")
        u_fe = None
        for r in rows:
            if int(r[0]) == interior_id:
                u_fe = r[1:4]
        if u_fe is None:
            print(f"FAIL: interior node {interior_id} missing from "
                  f"nd.csv")
            return 1
        Xc = [0.5, 0.5, 0.5]
        u_ex = [a - b for a, b in zip(mat_vec(F, Xc), Xc)]
        disp_err = max(abs(a - b) for a, b in zip(u_fe, u_ex))
        print(f"interior_node_disp_err={disp_err:.3e}")

        # ── assertion 2: analytic SVK Cauchy stress ─────────────
        Ft = transpose(F)
        C = mat_mul(Ft, F)
        Egl = [[0.5 * (C[i][j] - (1.0 if i == j else 0.0))
                for j in range(3)] for i in range(3)]
        lam = POISSON * E_MOD / ((1 + POISSON) * (1 - 2 * POISSON))
        mu = 0.5 * E_MOD / (1 + POISSON)
        trE = Egl[0][0] + Egl[1][1] + Egl[2][2]
        S = [[lam * trE * (1.0 if i == j else 0.0) + 2 * mu * Egl[i][j]
              for j in range(3)] for i in range(3)]
        J = det3(F)
        FS = mat_mul(F, S)
        sig = [[v / J for v in row] for row in mat_mul(FS, Ft)]
        if MUTATE:
            print("mutation=the_expected_stress_uses_small_"
                  "strain_linear_elasticity")
            eps = [[0.5 * (F[i][j] + F[j][i])
                    - (1.0 if i == j else 0.0)
                    for j in range(3)] for i in range(3)]
            tre = eps[0][0] + eps[1][1] + eps[2][2]
            sig = [[lam * tre * (1.0 if i == j else 0.0)
                    + 2 * mu * eps[i][j]
                    for j in range(3)] for i in range(3)]
        expect = [sig[0][0], sig[1][1], sig[2][2],
                  sig[0][1], sig[1][2], sig[0][2], J]

        erows = parse_last_block(work / "ed.csv")
        if len(erows) != 8:
            print(f"FAIL: expected 8 elements in ed.csv, got "
                  f"{len(erows)}")
            return 1
        stress_err = 0.0
        for r in erows:
            for got, want in zip(r[1:], expect):
                stress_err = max(stress_err, abs(got - want))
        spread = 0.0
        for col in range(1, 8):
            vals = [r[col] for r in erows]
            spread = max(spread, max(vals) - min(vals))
        print(f"analytic_J={J:.12f}")
        print(f"analytic_sigma_xx={sig[0][0]:.9f}")
        print(f"max_cauchy_stress_err={stress_err:.3e}")
        print(f"elementwise_stress_spread={spread:.3e}")

        ok = True
        if disp_err > TOL_DISP:
            print(f"FAIL: interior node off the exact linear field "
                  f"({disp_err:.3e} > {TOL_DISP:.0e}) — hex8 does "
                  f"not pass the patch test")
            ok = False
        if stress_err > TOL_STRESS:
            print(f"FAIL: Cauchy stress off the SVK closed form "
                  f"({stress_err:.3e} > {TOL_STRESS:.0e}) — either "
                  f"the `isotropic elastic` constitutive law "
                  f"changed or the log-variable order sx;sy;sz;"
                  f"sxy;syz;sxz changed")
            ok = False
        if spread > TOL_UNIFORM:
            print(f"FAIL: stress not uniform across the patch "
                  f"({spread:.3e} > {TOL_UNIFORM:.0e})")
            ok = False
        if not ok:
            return 1
        print("febio_patch_test=passed")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
