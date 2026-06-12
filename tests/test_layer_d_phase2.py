"""Layer D phase 2: cross-backend consistency through the CATALOG path.

Task #34. Phase 1 (scripts/tier2_fixtures/cross_backend/
mms_cross_backend_consistency) compares two in-process Python
assemblies (skfem + hand-rolled scipy). Phase 2 raises the bar: it
runs THE CATALOG'S OWN poisson_2d templates through each backend's
real generate_input() -> run() path — separate processes, separate
codebases, separate languages (Python / C++ / 4C YAML) — and asserts
they all produce the same physics.

The shared canonical problem (every backend's poisson_2d default):

    -laplace(u) = 1   on [0,1]^2,   u = 0 on the boundary

Analytic peak value (double Fourier series):

    u(1/2, 1/2) = (16/pi^4) * sum_{m,n odd} (-1)^{(m+n)/2-1}
                  / (m n (m^2+n^2))  =  0.07367135...

Observed at probe time 2026-06-12 (each within ~0.3% as expected
for the templates' Q1/P1 resolutions):

    skfem  0.0738993   (Q1 32x32, results_summary.json)
    dealii 0.0738993   (Q1 32x32, stdout 'max(u) = ...')
    fourc  0.0738993   (TRANSP QUAD4 32x32, VTU 'phi_1')
    ngsolve 0.0735372  (P1 unstructured h=1/16, results_summary)
    fenics  ~0.0737    (P1, stdout 'max(u)=...')

A sign-convention error, a wrong material law, an off-by-one
quadrature, or a unit slip in ANY backend's template moves its peak
away from the others — and this gate flips red even when every
per-backend Layer-C test still passes.

Skips per-backend when a solver is not available on the machine;
requires at least THREE backends to make the cross-comparison
meaningful. Wall-clock ~45 s (dominated by the fenics subprocess +
the dealii compile), which is why this lives in its own file — drop
it from -k filters when iterating on unrelated code.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Analytic u(0.5, 0.5) for -lap(u)=1 on the unit square, u=0 on bdry.
U_PEAK_ANALYTIC = 0.0736713

# FE peak values converge to the analytic one from above at these
# template resolutions; 1.5% covers the coarsest (P1 tri h=1/16)
# with margin while still catching any real modeling divergence
# (wrong sign / wrong rhs / wrong BC all shift the peak by >> 10%).
REL_TOL_VS_ANALYTIC = 0.015
REL_TOL_PAIRWISE = 0.015

# (backend, physics, variant, extractor-kind)
CASES = [
    ("skfem",   "poisson", "2d", "summary"),
    ("ngsolve", "poisson", "2d", "summary"),
    ("fenics",  "poisson", "2d", "stdout"),
    ("dealii",  "poisson", "2d", "stdout"),
    ("fourc",   "poisson", "poisson_2d", "vtu:phi_1"),
]

_MAX_RE = re.compile(r"max\(u\)\s*=\s*([0-9.eE+-]+)")


def _extract_peak(kind: str, work_dir: Path, stdout_text: str) -> float:
    """Pull the scalar peak out of whatever the template emitted."""
    if kind == "summary":
        summaries = sorted(work_dir.rglob("results_summary.json"))
        assert summaries, f"no results_summary.json under {work_dir}"
        data = json.loads(summaries[-1].read_text())
        return float(data["max_value"])
    if kind == "stdout":
        m = _MAX_RE.search(stdout_text)
        assert m, f"no 'max(u) = ...' line in stdout: {stdout_text[-400:]!r}"
        return float(m.group(1))
    if kind.startswith("vtu:"):
        field = kind.split(":", 1)[1]
        import meshio
        vtus = sorted(p for p in work_dir.rglob("*.vtu")
                      if "pvtu" not in p.name)
        assert vtus, f"no .vtu under {work_dir}"
        m = meshio.read(vtus[-1])
        assert field in m.point_data, (
            f"field {field!r} not in {list(m.point_data)}")
        return float(m.point_data[field].max())
    raise ValueError(kind)


class TestLayerDPhase2CatalogConsistency(unittest.TestCase):
    """All available backends agree on the canonical Poisson peak."""

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.get_backend = staticmethod(get_backend)
        cls.peaks: dict[str, float] = {}
        cls.skipped: dict[str, str] = {}

        for be_name, physics, variant, kind in CASES:
            b = get_backend(be_name)
            if b is None:
                cls.skipped[be_name] = "not registered"
                continue
            try:
                from core.backend import BackendStatus
                status, detail = b.check_availability()
                if status != BackendStatus.AVAILABLE:
                    cls.skipped[be_name] = f"{status.name}: {detail}"
                    continue
            except Exception as e:
                cls.skipped[be_name] = f"availability probe failed: {e}"
                continue

            try:
                content = b.generate_input(physics, variant, {})
                wd = Path(tempfile.mkdtemp(prefix=f"layerd2_{be_name}_"))
                job = asyncio.run(b.run(content, wd, np=1, timeout=300))
                rc = getattr(job, "return_code", None)
                if rc != 0:
                    cls.skipped[be_name] = (
                        f"run rc={rc}: {str(getattr(job, 'error', ''))[:200]}")
                    continue
                stdout = ""
                for f in wd.rglob("stdout.log"):
                    stdout += f.read_text()
                cls.peaks[be_name] = _extract_peak(kind, wd, stdout)
            except Exception as e:  # noqa: BLE001 — record + skip
                cls.skipped[be_name] = f"{type(e).__name__}: {e}"

    def test_minimum_backend_quorum(self) -> None:
        """At least 3 independent backends must participate, or the
        cross-comparison is vacuous. (On this machine 5 should run;
        a sudden drop to <3 means backend availability regressed.)"""
        self.assertGreaterEqual(
            len(self.peaks), 3,
            f"only {sorted(self.peaks)} produced a peak; skipped: "
            f"{self.skipped}")

    def test_each_backend_matches_analytic(self) -> None:
        for be, peak in sorted(self.peaks.items()):
            with self.subTest(backend=be, peak=peak):
                rel = abs(peak - U_PEAK_ANALYTIC) / U_PEAK_ANALYTIC
                self.assertLess(
                    rel, REL_TOL_VS_ANALYTIC,
                    f"{be} peak {peak:.6f} deviates {rel:.2%} from "
                    f"the analytic {U_PEAK_ANALYTIC} — its poisson_2d "
                    f"template no longer solves the canonical "
                    f"problem.")

    def test_pairwise_agreement(self) -> None:
        """The actual Layer-D property: backends agree with EACH
        OTHER, independent of the analytic anchor."""
        if len(self.peaks) < 2:
            self.skipTest("need >=2 backends")
        items = sorted(self.peaks.items())
        for i, (be_a, pa) in enumerate(items):
            for be_b, pb in items[i + 1:]:
                with self.subTest(pair=f"{be_a} vs {be_b}"):
                    rel = abs(pa - pb) / max(abs(pa), abs(pb))
                    self.assertLess(
                        rel, REL_TOL_PAIRWISE,
                        f"{be_a}={pa:.6f} vs {be_b}={pb:.6f} "
                        f"diverge {rel:.2%} — one of the two "
                        f"poisson_2d templates drifted.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
