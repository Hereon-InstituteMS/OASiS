"""Regression: per-backend `Signal:`-marker coverage on pitfalls
must not slip below the 2026-06-02 baseline.

WHY this matters
================
A pitfall's `Signal:` line is the critic-gate retrieval anchor.
When a simulation fails, the post-execution critic searches the
pitfall library for `Signal:` snippets that match the error
output and surfaces the matching pitfall + post-mortem record.
A pitfall without a `Signal:` line is invisible to that
retrieval path — the LLM may have a perfectly-written failure
diagnosis sitting in the catalog and never find it.

CURRENT BASELINE (2026-06-02)
=============================
A sweep across all 1068 catalog pitfalls in 198 physics rows
across 8 backends:

  kratos    : 147 / 147  (100.0%)   # Layer A/B promotion done
  dealii    :  96 / 138  ( 69.6%)
  skfem     :  51 / 103  ( 49.5%)
  ngsolve   :  64 / 135  ( 47.4%)
  febio     :   6 /  13  ( 46.2%)
  fenics    :  41 / 129  ( 31.8%)
  fourc     :  29 / 335  (  8.7%)   # heaviest gap
  dune      :   0 /  68  (  0.0%)   # complete miss

This test pins **percentage floors** per backend so:
  - new pitfalls added without Signal: markers degrade
    coverage and trip the test
  - existing Signal-less pitfalls that get rewritten WITH a
    Signal: line raise the floor naturally on re-record

If a new commit IMPROVES coverage, raise the floor in
SIGNAL_COVERAGE_MIN below to lock in the improvement.

This test is **inverted by design**: it accepts the current gaps
as known debt (fourc + dune especially) and prevents regression,
rather than pretending coverage is universally high.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Floor percentages locked in at the 2026-06-02 audit. Each
# floor is set ~1.5 percentage points below the measured value
# so a small reordering / re-counting noise does not trip the
# test, but a real regression (a new pitfall without Signal:
# pulling the average down) WILL.
SIGNAL_COVERAGE_MIN = {
    "kratos":  99.0,   # measured 100.0
    "dealii":  99.0,   # measured 100.0 — dealii at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 84.0 after pass 3 in
                       #                  advanced.py: 20 short bullet
                       #                  pitfalls rewritten as
                       #                  Signal-tagged paragraphs
                       #                  across the remaining 10
                       #                  physics —
                       #                    mixed_laplacian (2),
                       #                    time_dependent_heat (2),
                       #                    time_dependent_wave (2),
                       #                    time_dependent_ns (2),
                       #                    multiphysics_dealii (2),
                       #                    error_estimation (2),
                       #                    phase_field (2),
                       #                    dg_advection_reaction (2),
                       #                    cg_dg_coupled (2),
                       #                    optimal_control (2).
                       #                  dealii is the THIRD backend
                       #                  (after kratos and febio) to
                       #                  reach 100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 69.6% -> 76.8% ->
                       #                  85.5% -> 100.0% in three
                       #                  passes.)
    "skfem":   99.0,   # measured 100.0 — skfem at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 74.0 after pass 3:
                       #                  25 untagged pitfalls re-
                       #                  cast across 6 physics —
                       #                    hyperelasticity (8: no
                       #                    built-in NH, neo-
                       #                    Hookean energy sign,
                       #                    PK1 vs PK2 wiring,
                       #                    F=I+grad(u), C4 with
                       #                    I4_sym_C^{-1}, geometric
                       #                    stiffness, ib.interpolate),
                       #                    helmholtz (8: cast to
                       #                    complex, 10 elem/wavelength,
                       #                    +i*k*u ABC, PML thickness,
                       #                    P2/P3 for k>20, eigsh
                       #                    non-Hermitian, real/abs
                       #                    output, pollution P1
                       #                    O(k^3 h^2)),
                       #                    dg_methods (3: FacetBasis
                       #                    vs InteriorFacetBasis,
                       #                    single-sided IFB, SUPG-
                       #                    CG vs DG),
                       #                    time_dependent (3: BE
                       #                    convergence slope,
                       #                    accuracy vs stability,
                       #                    ib.doflocs),
                       #                    reaction_diffusion (2:
                       #                    Schnakenberg steady,
                       #                    Fisher-KPP scalar),
                       #                    convection_diffusion (1:
                       #                    MeshPeriodic).
                       #                  skfem joins kratos, febio,
                       #                  dealii as the FOURTH backend
                       #                  at 100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 49.5% -> 60.2% ->
                       #                  75.7% -> 100.0%.)
    "ngsolve": 99.0,   # measured 100.0 — ngsolve at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 68.0 after pass 4:
                       #                  41 untagged pitfalls re-cast
                       #                  across 11 physics —
                       #                    navier_stokes (1: DFG
                       #                    Schafer-Turek Cd~5.57,
                       #                    Cl 0.0104-0.0110),
                       #                    thermal_structural (3:
                       #                    isotropic alpha*T*Id,
                       #                    (3*lam+2*mu) bulk factor,
                       #                    two-way iterate),
                       #                    surface_pde (3: grad
                       #                    auto-tangential, ALE for
                       #                    evolving surfaces, OCC
                       #                    .faces selection),
                       #                    dg_methods (7: dgjumps=
                       #                    True, u.Other(), skeleton
                       #                    dx vs ds, alpha=4*(p+1)^2,
                       #                    IfPos upwind, SIP penalty
                       #                    even at high Pe, GMRES
                       #                    not CG for advection),
                       #                    contact (2: active set
                       #                    vs penalty O(1/gamma)
                       #                    floor, frictional
                       #                    tangential penalty +
                       #                    Coulomb),
                       #                    time_dependent_ns (1:
                       #                    DFG transient Cd~5.57,
                       #                    lid-cavity Ghia
                       #                    streamfunction),
                       #                    mhd (7: low-Rm limit,
                       #                    HCurl for A not VectorH1,
                       #                    Hartmann Ha boundary
                       #                    layer 1/Ha, mesh
                       #                    refinement near walls,
                       #                    splitting O(dt) error,
                       #                    div(B)=0 via HDiv or
                       #                    grad-div, grad-div u),
                       #                    hdivdiv (7: NN-cty vs
                       #                    H2 conformity, clamped
                       #                    Nitsche dw/dn, simply-
                       #                    supported only w=0,
                       #                    HHJ order-optimal, Regge
                       #                    for 3D elasticity, no-
                       #                    locking mixed, w_max =
                       #                    qL^4/(64D)),
                       #                    nonlinear_elasticity (7:
                       #                    det(F)>0, load stepping,
                       #                    NH Tr(C)-d, Variation()
                       #                    auto-AD, F-bar / mixed
                       #                    for nu~0.5, Cauchy via
                       #                    PK2 not PK1, Newton
                       #                    dampfactor),
                       #                    phase_field (3: semi-
                       #                    implicit W'(c^n),
                       #                    staggered vs monolithic,
                       #                    l0 and h scale together).
                       #                  ngsolve joins kratos, febio,
                       #                  dealii, skfem, fenics as the
                       #                  SIXTH backend at 100% Signal:
                       #                  coverage. Trajectory across
                       #                  this session: 47.4% -> 55.6%
                       #                  -> 61.5% -> 69.6% -> 100.0%
                       #                  across four passes.)
    "febio":   99.0,   # measured 100.0 — FEBio at FULL Signal
                       #                  coverage (raised
                       #                  2026-06-02 from 87.0
                       #                  after pass 2d Signal-
                       #                  tagged the remaining 7
                       #                  untagged pitfalls in
                       #                  linear_elasticity (4: v
                       #                  not nu, 1-indexed nodes,
                       #                  MeshDomains v4 required,
                       #                  LoadData lc=N) and
                       #                  hyperelasticity (3:
                       #                  STATIC vs DYNAMIC, step-
                       #                  size for large strain,
                       #                  line search for
                       #                  convergence). FEBio:
                       #                  the second backend
                       #                  after kratos to reach
                       #                  100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 46.2% -> 75.9%
                       #                  -> 84.4% -> 88.5% ->
                       #                  100.0%.)
    "fenics":  99.0,   # measured 100.0 — fenics at FULL Signal
                       #                 coverage (raised 2026-06-02
                       #                 from 82.0 after pass 4:
                       #                 21 untagged pitfalls re-cast
                       #                 across 9 physics —
                       #                   heat (1: insulated BC =
                       #                   natural / do-nothing),
                       #                   thermal_structural (1:
                       #                   alpha*DeltaT*I inside
                       #                   sigma = C:(eps-alpha*DT*I)),
                       #                   mixed_poisson (2: BDM(k)+
                       #                   DG(k-1) vs RT(k), K^{-1}
                       #                   weight for heterogeneous
                       #                   permeability),
                       #                   dg_methods (3: upwind vs
                       #                   centred flux instability,
                       #                   drop diffusion entirely
                       #                   for eps=0, block-diagonal
                       #                   mass matrix),
                       #                   multiphase (4: Allen-Cahn
                       #                   W(phi) 1/(4eps) prefactor,
                       #                   mobility kappa = eps^2,
                       #                   smoothed Heaviside
                       #                   coupling, surface tension
                       #                   jump),
                       #                   time_dependent_heat (3:
                       #                   BE slope-1 sanity,
                       #                   piecewise-Function for
                       #                   layered k, dt by accuracy
                       #                   not stability),
                       #                   nonlinear_pde (2: SNES
                       #                   line-search for strongly
                       #                   nonlinear D(u),
                       #                   snes_monitor/ksp_monitor),
                       #                   magnetostatics (5: 2D
                       #                   scalar Az vs Nedelec
                       #                   over-DOF, 2D auto-gauge,
                       #                   B = curl(A) sign in 2D,
                       #                   J=0 in iron via MeshTags,
                       #                   piecewise mu_r DG0).
                       #                 fenics joins kratos, febio,
                       #                 dealii, skfem as the FIFTH
                       #                 backend at 100% Signal
                       #                 coverage. Trajectory: 31.8%
                       #                 -> 49% -> 65% -> 77% -> 83.7%
                       #                 -> 100.0% across four passes.)
    "fourc":   88.0,   # measured  89.0 — CROSSED 88% (raised
                       #                 2026-06-02 from 78.0 after
                       #                 pass 12: 6 small files
                       #                 retyped (sti, shell,
                       #                 cardiovascular0d, membrane,
                       #                 mixture, constraint,
                       #                 brownian_dynamics) — ~31
                       #                 new Signal lines, mostly
                       #                 4-5 entries each.
                       #                 fourc this session: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% -> 79.7% -> 89.0%
                       #                 across thirteen passes —
                       #                 80pp progress.)
                       #                 2026-06-02 from 60.0 after
                       #                 pass 11: 60 new Signal lines
                       #                 across 10 six-pitfall blocks
                       #                 — ssi, ale, level_set, ssti,
                       #                 fbi, pasi, lubrication,
                       #                 cardiac_monodomain,
                       #                 arterial_network,
                       #                 fluid_turbulence. Each block
                       #                 retyped ~6 pitfalls as full
                       #                 Signal-tagged paragraphs.
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% -> 79.7% across twelve
                       #                 passes — 71pp improvement.)
                       #                 2026-06-02 from 48.0 after
                       #                 pass 10: 41 new Signal lines
                       #                 across six 7-pitfall blocks
                       #                 — electrochemistry (EQUPOT
                       #                 ENC vs divi, MATID matlist,
                       #                 NUMMAT+1 phi scalars,
                       #                 INITIALFIELD COMPONENT,
                       #                 CALCFLUX_DOMAIN total, S2I
                       #                 dual setup, no_stab default),
                       #                 fpsi (ALE+CLONING required,
                       #                 dual interface sets,
                       #                 MAT_StructPoro nested,
                       #                 INITPOROSITY (0,1),
                       #                 permeability stiffness,
                       #                 fluid -> ALE only, BJS slip
                       #                 coefficient), fsi_xfem
                       #                 (no-ALE, ghost-penalty,
                       #                 water-tight cutter, Nitsche
                       #                 gamma_N tuning, dt < h/v
                       #                 CFL, cut-cell ParaView,
                       #                 NA: Euler not ALE), fs3i
                       #                 (5-field setup, dual CLONING
                       #                 mappings, fluid-scatra SUPG,
                       #                 diffusivity contrast under-
                       #                 relax, NA: ALE for ALE-vel
                       #                 in scatra, FS3I vs FSI
                       #                 DYNAMIC, matching dt), ehl
                       #                 (lubrication/structural mesh
                       #                 compat, h->0 singularity,
                       #                 piezoviscous Picard
                       #                 divergence, alpha ramp,
                       #                 correct face for Neumann,
                       #                 transient squeeze-film,
                       #                 consistent SI units),
                       #                 structural_mechanics (SOLID
                       #                 QUAD4 2D syntax, KINEM
                       #                 linear vs nonlinearTotLag,
                       #                 SOLIDSCATRA for TSI,
                       #                 MAXITER per linear/nonlinear,
                       #                 PREDICT TangDis, NUMDOF
                       #                 spatial dim, BEAM3* not
                       #                 SOLID/WALL).
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% across eleven passes.)
                       #                 2026-06-02 from 39.0 after
                       #                 pass 9: 30 new Signal lines —
                       #                 tsi (10: CLONING required,
                       #                 THEXPANS units, INITTEMP
                       #                 reference, TSI DYNAMIC vs
                       #                 per-field, ITEMAX=1 one-way,
                       #                 SOLIDSCATRA TYPE Undefined,
                       #                 COUPVARIABLE: Temperature,
                       #                 Belos for monolithic,
                       #                 DESIGN VOL THERMO DIRICH,
                       #                 SOLIDSCATRA 11 TYPE enum) +
                       #                 particles (11: mandatory SPH
                       #                 section, IO/RUNTIME VTK
                       #                 PARTICLES, regular grid,
                       #                 INTERACTION_HORIZON = m*dx,
                       #                 PERIDYNAMIC_GRID_SPACING
                       #                 match, PRE_CRACKS syntax,
                       #                 PDBODYID, phase TYPE,
                       #                 CFL dt < 0.5*dx/c_wave,
                       #                 BIN_SIZE > horizon,
                       #                 DOMAINBOUNDINGBOX) +
                       #                 beams (9: Exodus
                       #                 unsupported for beams,
                       #                 NUMDOF match, TRIADS req,
                       #                 LINE3 ordering ep1-ep2-mid,
                       #                 Hermite NUMDOF=9,
                       #                 GenAlphaLieGroup for
                       #                 finite rotations,
                       #                 MASSLIN rotations,
                       #                 consistent A/I/J,
                       #                 DNODE/DLINE TOPOLOGY).
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6%.)
                       #                 2026-06-02 from 34.0 after
                       #                 pass 8: all 20 untagged fsi
                       #                 pitfalls Signal-tagged in
                       #                 data/fourc_knowledge.py —
                       #                 NA: ALE requirement, ALE
                       #                 Dirichlet on all outer
                       #                 walls except FSI interface,
                       #                 CLONING MATERIAL MAP,
                       #                 SHAPEDERIVATIVES, separate
                       #                 SOLVER N per field, LINE vs
                       #                 SURF coupling-condition by
                       #                 dim, NUMDOF per field,
                       #                 shared-node NUMDOF in multi-
                       #                 field DIRICH, DESIGN FLUID
                       #                 LINE LIFT&DRAG (3D only),
                       #                 EVERY_ITERATION not a
                       #                 parameter, FUNCT COMPONENT
                       #                 requirement, separate-nodes
                       #                 at FSI interface, IO/RUNTIME
                       #                 VTK STRUCTURE conflict, 2D
                       #                 VTK NaN artifact, Gmsh
                       #                 fragment quad-mesh failure,
                       #                 SLAVE-cannot-carry-Dirichlet,
                       #                 IO/RUNTIME VTK ALE crash,
                       #                 valid COUPALGO enum,
                       #                 slow inflow ramp.
                       #                 fourc trajectory thru this
                       #                 session: 8.7% -> 14.0% ->
                       #                 18.2% -> 22.1% -> 25.4% ->
                       #                 28.1% -> 30.1% -> 34.6% ->
                       #                 40.6% across nine passes.)
    "dune":     0.0,   # measured   0.0 — known total gap
}


def _pitfall_text(pit) -> str:
    if isinstance(pit, str):
        return pit
    if isinstance(pit, dict):
        return pit.get("text", "") or pit.get("description", "") or ""
    return str(pit)


class TestPitfallSignalCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")

    def test_signal_coverage_meets_floor(self) -> None:
        """No backend's Signal-marker coverage falls below its
        2026-06-02 floor. If you intentionally add Signal:
        markers (good!) and coverage climbs, RAISE the
        corresponding floor in SIGNAL_COVERAGE_MIN to lock the
        improvement in."""
        failures = []
        # Stable, sorted output for diagnostics.
        rows = []
        for b in self.backends:
            total = 0
            with_sig = 0
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if not isinstance(k, dict):
                    continue
                for pit in k.get("pitfalls", []):
                    total += 1
                    if "Signal:" in _pitfall_text(pit):
                        with_sig += 1
            if total == 0:
                continue
            pct = 100.0 * with_sig / total
            rows.append((b.name(), with_sig, total, pct))
            floor = SIGNAL_COVERAGE_MIN.get(b.name())
            if floor is None:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     "no floor recorded — add one to "
                     "SIGNAL_COVERAGE_MIN at the 2026-06-02 "
                     "baseline value"))
                continue
            if pct < floor:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     f"below {floor:.1f}% floor"))
        # Always render the per-backend table so failures and
        # green-builds both surface the current numbers.
        diagnostic = "\n".join(
            f"  {n:10s}: {s:4d}/{t:4d} ({p:5.1f}%)"
            for n, s, t, p in sorted(rows))
        if failures:
            fail_lines = "\n".join(
                f"  {n}: {s}/{t} ({p:.1f}%) -- {note}"
                for n, s, t, p, note in failures)
            self.fail(
                f"{len(failures)} backend(s) regressed below the "
                f"Signal:-coverage floor.\n\n"
                f"Current per-backend coverage:\n{diagnostic}\n\n"
                f"Regressions:\n{fail_lines}\n\n"
                "Either add a Signal: line to the new pitfall(s), "
                "or accept the new floor by editing "
                "SIGNAL_COVERAGE_MIN — but only AFTER confirming "
                "the new pitfall really has no observable signal.")


if __name__ == "__main__":
    unittest.main()
