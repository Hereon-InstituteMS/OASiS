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
    "dealii":  75.0,   # measured  76.8 (raised 2026-06-02 from
                       #                  68.0 after pass 1 on
                       #                  dealii: 14 pitfalls
                       #                  tagged across 4 physics
                       #                  in deep_knowledge.py —
                       #                  advection_dg (4: flux-
                       #                  sparsity pattern, IP
                       #                  alpha, face-normal
                       #                  sign, downstream DoF
                       #                  renumbering),
                       #                  compressible_euler (4:
                       #                  CG instability,
                       #                  LF vs HLLC, CFL, slope
                       #                  limiter), contact (3:
                       #                  active-set iteration,
                       #                  penalty mis-tune,
                       #                  AffineConstraints),
                       #                  nonlinear_elasticity
                       #                  (3: three-field for
                       #                  nu>0.49, Newton load
                       #                  steps, Sacado AD).)
    "skfem":   74.0,   # measured  75.7 (raised 2026-06-02 from
                       #                  58.0 after pass 2 on
                       #                  skfem: 6 reaction_
                       #                  diffusion (block J,
                       #                  reaction-Jacobian
                       #                  mass form, IC
                       #                  perturbation, Turing
                       #                  d_v/d_u ratio,
                       #                  natural BC vs Dirichlet,
                       #                  gamma * L^2) + 7
                       #                  navier_stokes (no built-
                       #                  in NS, missing C(u),
                       #                  ElementVector split,
                       #                  pressure pin, Picard
                       #                  -> Newton, Picard
                       #                  linearisation, ib_u.N
                       #                  for block split) + 3
                       #                  convection_diffusion
                       #                  (u.grad attr,
                       #                  InteriorFacetBasis,
                       #                  high-Pe SUPG/DG).
                       #                  skfem 49.5% -> 60.2%
                       #                  -> 75.7% in two
                       #                  commits.)
    "ngsolve": 68.0,   # measured  69.6 — ties dealii (raised
                       #                  2026-06-02 from 60.0
                       #                  after pass 3: 5
                       #                  convection_diffusion +
                       #                  3 navier_stokes + 3
                       #                  mixed_poisson pitfalls
                       #                  Signal-tagged.
                       #                  ngsolve 47.4% -> 55.6%
                       #                  -> 61.5% -> 69.6% over
                       #                  three commits.)
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
    "fenics":  82.0,   # measured  83.7 (raised 2026-06-02 from
                       #                 77.0 after pass 6:
                       #                 6 partial-coverage pitfalls
                       #                 tagged in advanced.py —
                       #                 dg_methods (3: IP alpha,
                       #                 FacetNormal '+'/'-' side,
                       #                 inflow weak BC),
                       #                 time_dependent_heat (2:
                       #                 Robin BC omission,
                       #                 Neumann BC omission),
                       #                 nonlinear_pde (1:
                       #                 NonlinearProblem missing
                       #                 J argument). Remaining
                       #                 untagged in fenics are
                       #                 mostly informational tips
                       #                 (formulation conventions,
                       #                 not failure modes).)
    "fourc":   29.0,   # measured  30.1 — CROSSED 30% (raised
                       #                 2026-06-02 from 27.0 after
                       #                 a sixth pass: all 7
                       #                 xfem_fluid pitfalls
                       #                 received Signal: lines.
                       #                 fourc has gone 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1%
                       #                 -> 25.4% -> 28.1% ->
                       #                 30.1% across seven
                       #                 commits — a 3.5x
                       #                 improvement.)
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
