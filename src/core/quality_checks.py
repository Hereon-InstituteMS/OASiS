"""
General-purpose simulation quality checks.

These checks provide warnings about common issues. They do NOT prescribe
specific numbers — the agent must determine appropriate resolution, time
steps, etc. based on the physics of each specific problem.
"""

import logging
from typing import Optional

logger = logging.getLogger("oasis.quality")


def check_time_step(
    dt: float,
    h: float,
    wave_speed: Optional[float] = None,
    diffusivity: Optional[float] = None,
    scheme: str = "explicit",
) -> list[str]:
    """Check time step stability (CFL, Fourier number).

    These are mathematical stability conditions, not guidelines —
    violating them WILL cause the simulation to blow up.
    """
    warnings = []

    if scheme == "explicit":
        if wave_speed is not None and wave_speed > 0:
            cfl = dt * wave_speed / h
            if cfl > 1.0:
                warnings.append(
                    f"CFL = {cfl:.2f} > 1.0 — UNSTABLE for explicit scheme. "
                    f"Reduce dt to below {h / wave_speed:.2e}."
                )

        if diffusivity is not None and diffusivity > 0:
            fourier = dt * diffusivity / (h * h)
            if fourier > 0.5:
                warnings.append(
                    f"Fourier number = {fourier:.2f} > 0.5 — UNSTABLE for explicit diffusion. "
                    f"Reduce dt to below {0.5 * h * h / diffusivity:.2e}."
                )

    return warnings


def check_material_consistency(
    E: Optional[float] = None,
    nu: Optional[float] = None,
    density: Optional[float] = None,
) -> list[str]:
    """Check material parameter sanity — catches obvious errors."""
    warnings = []

    if nu is not None:
        if nu >= 0.5:
            warnings.append(
                f"Poisson ratio nu={nu} >= 0.5 — incompressible material. "
                f"Standard displacement formulations will lock. Use mixed method."
            )
        if nu < 0:
            warnings.append(f"Negative Poisson ratio nu={nu} — verify this is intended (auxetic).")
        if nu < -1.0 or nu > 0.5:
            warnings.append(f"Poisson ratio nu={nu} is outside physical range [-1, 0.5].")

    if E is not None and E <= 0:
        warnings.append(f"Non-positive Young's modulus E={E} — this is unphysical.")

    if density is not None and density <= 0:
        warnings.append(f"Non-positive density={density} — this is unphysical.")

    return warnings


def check_output_configured(solver: str, input_content: str) -> list[str]:
    """Check that the simulation will produce viewable output files."""
    warnings = []

    if solver == "fourc":
        if "IO/RUNTIME VTK OUTPUT" not in input_content:
            warnings.append(
                "No IO/RUNTIME VTK OUTPUT section found. "
                "Without it, no ParaView-readable output will be produced."
            )

    return warnings


# ── output-side validators (physics-agnostic; consume RESULTS, not setup) ──────
# Philosophy: catch silent-wrong results with checks that need NO physics knowledge
# and NO benchmark answer — finiteness, convergence honesty, conservation balance,
# and (when available) consistency against an independent monolithic re-solve.
# These feed the critic / result payload as warnings; they never hardcode a number
# tied to one physics (no Biot, no k*dt — those are problem-specific anchors).
import numpy as _np


def check_finite(values, label: str = "result") -> list[str]:
    """Flag NaN/Inf in a result array — a universal broken-run signal."""
    w = []
    a = _np.asarray(values, float)
    if a.size and not _np.all(_np.isfinite(a)):
        n = int((~_np.isfinite(a)).sum())
        w.append(f"{label}: {n}/{a.size} non-finite (NaN/Inf) values — result is invalid.")
    return w


def check_convergence(converged: bool, residual: float, tol: float) -> list[str]:
    """A non-converged coupled/iterative solve must NOT be reported as a result.
    The single most general silent-wrong guard."""
    w = []
    if not converged:
        w.append(
            f"NOT CONVERGED (residual {residual:.3e} > tol {tol:.1e}) — the reported "
            f"quantities are NOT trustworthy and must not be treated as a solution."
        )
    return w


def check_interface_balance(export_a, export_b, label_a="A", label_b="B",
                            rtol: float = 0.05) -> list[str]:
    """Conservation across a coupling interface: the net flux leaving A should equal
    the net flux entering B (global balance). Pure arithmetic on the exchanged
    normal_fluxes — no physics. `export_*` are InterfaceData-like dicts/objects."""
    w = []
    def _flux(e):
        f = e.get("normal_fluxes") if isinstance(e, dict) else getattr(e, "normal_fluxes", None)
        return None if f is None else float(_np.sum(_np.asarray(f, float)))
    fa, fb = _flux(export_a), _flux(export_b)
    if fa is None or fb is None:
        return w
    denom = max(abs(fa), abs(fb), 1e-30)
    rel = abs(fa + fb) / denom            # A exports +flux, B imports -flux → sum≈0
    if rel > rtol:
        w.append(
            f"Interface flux NOT balanced: net({label_a})={fa:.4g}, net({label_b})={fb:.4g}, "
            f"imbalance {rel:.1%} > {rtol:.0%} — coupling may be non-conservative (silent error)."
        )
    return w


def check_monolithic_consistency(coupled_qoi: float, monolithic_qoi: float,
                                 rtol: float = 0.05, qoi: str = "QoI") -> list[str]:
    """If the same problem can be solved un-split in one code, the coupled answer must
    match it. The most decisive silent-wrong detector — needs no external benchmark,
    only a monolithic re-solve. Returns a warning if they disagree beyond rtol."""
    w = []
    if monolithic_qoi is None or coupled_qoi is None:
        return w
    denom = max(abs(monolithic_qoi), 1e-30)
    rel = abs(coupled_qoi - monolithic_qoi) / denom
    if rel > rtol:
        w.append(
            f"{qoi}: coupled={coupled_qoi:.5g} vs monolithic re-solve={monolithic_qoi:.5g} "
            f"differ by {rel:.1%} > {rtol:.0%} — the coupled result is likely WRONG."
        )
    return w
