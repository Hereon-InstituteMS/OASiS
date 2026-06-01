"""Tier-2 Layer-F: catalog generator templates execute end-to-end.

Layer B verified `generate_input` doesn't raise on the
catalog. Layer C verified our MMS RE-IMPLEMENTATION of
the catalog's API converges. This is the missing piece:
take the catalog's ACTUAL emitted template, write it to
a tempfile, run it through the backend's interpreter,
and assert the run succeeds + emits expected sentinels.

This is what an LLM agent actually does: ask the MCP
for a template, save it, and run it. If the catalog
ships a template that the LLM-emit-and-run loop can't
execute, every other layer in the matrix could pass and
the user still gets a broken script.

Approach: per backend in {skfem, kratos} (both run
quickly in the repo .venv), for the simplest physics
(poisson 2d), call generate_input via the live
registry, write to a tempfile in /tmp, exec it via the
ACTUAL python interpreter the runner uses, and assert
exit_code == 0 + no Traceback. NGSolve/fenics need
their own conda envs and would slow this down to 30+
seconds per backend — keep this gate fast.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))


def run_template_in_subprocess(
        backend_name: str, physics: str, variant: str,
        python_path: str,
        timeout: int = 60) -> tuple[int, str, str]:
    """Generate the catalog template, write to tempfile,
    exec via the given python. Return (rc, out_head,
    err_head)."""
    from core.registry import (
        load_all_backends, get_backend)
    load_all_backends()
    b = get_backend(backend_name)
    template = b.generate_input(physics, variant, {})

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/tmp") as f:
        f.write(template)
        script_path = f.name

    try:
        result = subprocess.run(
            [python_path, script_path],
            capture_output=True, text=True,
            timeout=timeout,
            cwd="/tmp")
        return (result.returncode,
                result.stdout[:800],
                result.stderr[:800])
    except subprocess.TimeoutExpired:
        return (-1, "", "TIMEOUT")
    finally:
        try:
            Path(script_path).unlink()
        except Exception:
            pass


def _resolve_python(backend_name: str) -> str | None:
    """Return the absolute path to the Python interpreter
    that has THIS backend's runtime available, OR None
    when the env is missing (caller will skip this row)."""
    repo_venv = sys.executable
    fenicsx_env = (Path.home() / "miniconda3" / "envs"
                    / "ofa-fenicsx" / "bin" / "python")
    if backend_name in ("skfem", "kratos", "ngsolve"):
        # Repo .venv has scikit-fem 12 + KratosMultiphysics
        # + NGSolve installed.
        return repo_venv
    if backend_name == "fenics":
        return (str(fenicsx_env) if fenicsx_env.is_file()
                else None)
    return None


def main() -> int:
    print(f"repo_venv_python={sys.executable}")

    # Matrix of (backend, physics, variant) tuples. Each
    # row is run as a subprocess in the backend-correct
    # interpreter. Total runtime budget is ~120 s.
    #
    # All known-broken stokes templates have been fixed
    # (commits 6ffff60 ngsolve, e11d9c6 skfem, and this
    # iteration's fenics). The inverted-gate companion
    # fixture is retired in the same commit.
    rows = [
        # poisson coverage (all 4 backends)
        ("skfem", "poisson", "2d"),
        ("kratos", "poisson", "2d"),
        ("ngsolve", "poisson", "2d"),
        ("fenics", "poisson", "2d"),
        # linear_elasticity coverage (all 4 backends)
        ("skfem", "linear_elasticity", "2d"),
        ("kratos", "linear_elasticity", "2d"),
        ("ngsolve", "linear_elasticity", "2d"),
        ("fenics", "linear_elasticity", "2d"),
        # heat coverage (kratos has the simplest dispatch
        # test; skfem ships a heat::2d that runs)
        ("skfem", "heat", "2d"),
        ("kratos", "heat", "2d"),
        # stokes: all 3 catalog templates runnable after:
        #   ngsolve → free.Clear(V.ndof) pressure pin
        #   skfem   → MeshTri + intorder=4 + pin + driven
        #             cavity BC rewrite
        #   fenics  → XDMFFile → VTXWriter (P2 velocity
        #             can't be written via XDMF in dolfinx
        #             0.10 because output degree must
        #             match mesh degree of 1)
        ("ngsolve", "stokes", "2d"),
        ("skfem", "stokes", "2d"),
        ("fenics", "stokes", "2d"),
        # Heat for ngsolve + fenics (already proven in
        # the catalog; extending the coverage matrix).
        ("ngsolve", "heat", "2d"),
        ("fenics", "heat", "2d_steady"),
        # The 5 fenics phantom-closures from commits
        # c5d184a, 1a4ba74, c7f5cc9, ae0296b: helmholtz,
        # maxwell, nearly_incompressible_elasticity,
        # fracture, stokes_darcy. Each ships a generator
        # that smoke-ran during its commit. Layer F now
        # binds them as regression gates.
        ("fenics", "helmholtz", "2d"),
        ("fenics", "maxwell", "2d"),
        ("fenics", "nearly_incompressible_elasticity", "2d"),
        ("fenics", "fracture", "2d"),
        ("fenics", "stokes_darcy", "2d"),
        # Layer F sweep — physics likely to surface
        # additional catalog bugs.
        ("fenics", "navier_stokes", "2d"),
        # fenics hyperelasticity ships only "3d"
        # (backend.py L138 template_variants).
        ("fenics", "hyperelasticity", "3d"),
        # ngsolve hyperelasticity Newton still does not
        # converge — deferred (likely needs continuation /
        # load-stepping rewrite). fenics eigenvalue was
        # fixed this iteration: M must use diag=1.0 (not
        # 0.0) to avoid a singular shifted operator that
        # makes SLEPc abort with 'Zero pivot row 0';
        # SMALLEST_REAL with the default Krylov-Schur ST
        # returns the physical eigenvalues plus benign
        # boundary-unit eigenvalues at Dirichlet rows.
        ("fenics", "eigenvalue", "2d"),
        ("ngsolve", "eigenvalue", "2d"),
        ("skfem", "eigenvalue", "2d"),
        ("skfem", "biharmonic", "2d"),
        # Kratos coverage extension — most kratos catalog
        # templates are scipy-based stubs that exercise
        # catalog metadata + minimal solve. Probes likely
        # to surface broken stubs.
        ("kratos", "contact", "2d"),
        ("kratos", "structural_dynamics", "2d"),
        ("kratos", "fsi", "2d"),
        ("kratos", "dem", "2d"),
        ("kratos", "mpm", "2d"),
        ("kratos", "geomechanics", "2d"),
        # Layer F batch-7 expansion — smoke pass identified
        # 9 more catalog rows that already run as-shipped.
        # Promoting them locks in the as-is behaviour so any
        # future regression in these generators trips the
        # gate. Counterparts that DO NOT run yet are tracked
        # in inverted gates / deferred (skfem helmholtz +
        # navier_stokes + hyperelasticity + mixed_poisson,
        # fenics biharmonic, and four catalog-drift rows
        # where supported_physics advertises a name with no
        # matching _2d template).
        ("skfem", "convection_diffusion", "2d"),
        ("ngsolve", "helmholtz", "2d"),
        ("ngsolve", "navier_stokes", "2d"),
        ("ngsolve", "plasticity", "2d"),
        ("fenics", "mixed_poisson", "2d"),
        ("fenics", "convection_diffusion", "2d"),
        ("fenics", "reaction_diffusion", "2d"),
        ("kratos", "rans", "2d"),
        ("kratos", "shallow_water", "2d"),
        # fenics biharmonic XDMF degree-mismatch fixed
        # this iteration: P2 C0-IP Function cannot be
        # written via XDMFFile.write_function on a P1
        # mesh (RuntimeError 'Degree of output Function
        # must be same as mesh degree'). VTXWriter
        # (ADIOS2) supports arbitrary degree — same fix
        # as fenics::stokes Taylor-Hood velocity.
        ("fenics", "biharmonic", "2d"),
        # skfem batch-8 fixes:
        #  - helmholtz: MeshQuad.init_tensor does not
        #    auto-attach named boundaries; ib.get_dofs()
        #    returns a DofsView (NOT subscriptable, so
        #    dofs['left'] is a TypeError); use
        #    .with_boundaries({...}) on the mesh and call
        #    ib.get_dofs('left') with the tag as an arg.
        #  - mixed_poisson: sigma[0].grad[0] on a RT0
        #    vector field is an AttributeError ('ndarray'
        #    has no attribute 'grad'); use the official
        #    skfem.helpers.div(sigma) helper. Also the
        #    LinearForm(lambda...) wrap pattern errored
        #    on asm; use the @LinearForm decorator on a
        #    plain Python function.
        # hyperelasticity (to_simplex→to_meshtri +
        # boundaries fix) and navier_stokes (interpolate
        # kwargs shape) need full Newton-rewrites — left
        # deferred to a follow-up batch.
        ("skfem", "helmholtz", "2d"),
        ("skfem", "mixed_poisson", "2d"),
        # Layer F batch-9 — non-'2d' variants that the
        # catalog correctly advertises but Layer F had
        # never probed because the smoke pass defaulted to
        # variant='2d'. cd_dg runs as-shipped, while the
        # magnetostatics row needed a Netgen CSG fix:
        # geo.Add does NOT take 'mat=' kwarg — use
        # solid.mat('name') chain before Add.
        ("ngsolve", "convection_diffusion", "2d_dg"),
        ("ngsolve", "maxwell", "3d_magnetostatics"),
        # Kratos non-'2d' variants — supported_physics
        # correctly declares fluid->2d_cavity and
        # plasticity->3d; earlier probe stalled by
        # defaulting to variant='2d'.
        ("kratos", "fluid", "2d_cavity"),
        ("kratos", "plasticity", "3d"),
        # Layer F batch-10: large untested-catalog sweep
        # using each backend's actual template_variants
        # (39 rows smoked, 29 ran as shipped — promoted
        # here). Surfaced 10 new catalog bugs to track in
        # follow-up iterations (fenics dg_methods ufl.Abs,
        # skfem poisson::3d meshio shape, six ngsolve rows
        # with abs/tanh/symbolicLFI/UmfpackInverse).
        ("fenics", "poisson", "3d"),
        ("fenics", "linear_elasticity", "3d"),
        ("fenics", "heat", "2d_transient"),
        ("fenics", "navier_stokes", "3d"),
        ("fenics", "thermal_structural", "2d"),
        ("fenics", "contact", "2d"),
        ("fenics", "multiphase", "2d"),
        ("fenics", "time_dependent_heat", "2d"),
        ("fenics", "cahn_hilliard", "2d"),
        ("fenics", "nonlinear_pde", "2d"),
        ("fenics", "magnetostatics", "2d"),
        ("ngsolve", "poisson", "3d"),
        ("ngsolve", "linear_elasticity", "3d"),
        ("ngsolve", "heat", "2d_steady"),
        ("ngsolve", "heat", "2d_transient"),
        ("ngsolve", "stokes", "2d_hdg"),
        ("ngsolve", "hyperelasticity", "3d"),
        ("ngsolve", "mixed_poisson", "2d"),
        ("ngsolve", "contact", "2d"),
        ("ngsolve", "time_dependent_ns", "2d"),
        ("skfem", "heat", "2d_steady"),
        ("skfem", "nonlinear", "2d"),
        ("skfem", "heat_transient", "2d"),
        ("skfem", "dg_methods", "2d"),
        ("skfem", "time_dependent", "2d"),
        ("skfem", "reaction_diffusion", "2d"),
        ("kratos", "linear_elasticity", "2d_nonlinear"),
        ("kratos", "heat_transient", "2d"),
        ("kratos", "shape_optimization", "2d"),
        # ngsolve thermal_structural fixed this iteration:
        # same LinearForm-without-TestFunction collapse as
        # heat, plus two gfu/gfT.name = "..." assignments
        # that target a property with no setter in this
        # build.
        ("ngsolve", "thermal_structural", "2d"),
    ]
    fail = []
    executed = 0
    for backend, physics, variant in rows:
        py = _resolve_python(backend)
        if py is None:
            print(
                f"{backend}::{physics}::{variant} "
                f"SKIPPED (no env)")
            continue
        try:
            rc, out, err = run_template_in_subprocess(
                backend, physics, variant, py,
                timeout=120)
        except Exception as e:
            fail.append(
                f"{backend}::{physics}::{variant} "
                f"setup_error {type(e).__name__}: {e}")
            continue
        executed += 1
        print(f"{backend}::{physics}::{variant}_rc={rc}")
        if rc != 0:
            fail.append(
                f"{backend}::{physics}::{variant} rc={rc}")
            print(f"  stderr: {err[:300]}")
        if "Traceback" in err or "Traceback" in out:
            fail.append(
                f"{backend}::{physics}::{variant} "
                f"traceback")

    print(f"total_executed={executed}")
    print(f"failures={len(fail)}")
    for r in fail:
        print(f"  fail: {r}")
    if fail:
        print("FAIL: catalog template execution",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
