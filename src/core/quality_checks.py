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


# Field/mesh formats meshio reads ROBUSTLY and that carry numeric solution data.
# .xdmf/.xmf are deliberately excluded: meshio's XDMF reader can raise SystemExit
# on multi-grid files (killing the process), and solvers that emit XDMF also emit
# a companion .vtu here, so nothing is lost by scanning the .vtu instead.
_FINITE_SCANNABLE = (".vtu", ".vtk", ".vtp", ".pvtu", ".msh", ".vtkhdf")


def _scan_bp_finite(path) -> tuple[list[str], bool]:
    """Best-effort finiteness scan of an ADIOS2 .bp dataset (dolfinx VTXWriter).

    Mac stress audit 2026-07-18: an all-NaN field written ONLY via VTXWriter
    (.bp) was stamped VERIFIED because meshio cannot read .bp — the exact
    'fabricated result' the gate exists to catch (dolfinx Stokes/Taylor-Hood
    templates emit .bp exclusively). Scans via adios2 when importable.
    Returns (warnings, scanned?) — scanned=False when adios2 is unavailable
    so the caller can report that finiteness was NOT asserted.
    """
    try:
        import adios2
        import numpy as _np2
    except Exception:
        return [], False
    w = []
    try:
        with adios2.FileReader(str(path)) as f:
            for name in list(f.available_variables() or {}):
                try:
                    arr = _np2.asarray(f.read(name), float)
                except (ValueError, TypeError):
                    continue  # non-numeric variable (labels, connectivity strings)
                w += check_finite(arr, label=f"{getattr(path, 'name', path)}:{name}")
        return w, True
    except BaseException:
        return [], False


def check_result_files_finite(paths, max_files: int = 25) -> list[str]:
    """Best-effort finiteness scan of a run's OUTPUT files.

    Attestation binds a claim to run evidence, but "a file exists" is not enough:
    a solve can exit 0 and write an output full of NaN/Inf — a fabricated-looking
    result. This reads each result file with meshio (plus adios2 for .bp) and
    flags non-finite values in any point/cell data, so the verification gate can
    reject it. If NONE of the output files could be scanned, that is reported
    too — a VERIFIED verdict must not silently imply a finiteness check that
    never ran. This never raises.
    """
    w = []
    scannable_format_seen = False
    considered = 0
    try:
        import meshio
    except Exception:
        return w
    from pathlib import Path as _Path
    for p in list(paths)[:max_files]:
        p = p if hasattr(p, "suffix") else _Path(str(p))
        suffix = p.suffix.lower()
        if suffix == ".bp":
            considered += 1
            bp_w, bp_scanned = _scan_bp_finite(p)
            w += bp_w
            # .bp counts as scannable only when adios2 actually read it —
            # without adios2 the format is unscannable in this environment.
            scannable_format_seen = scannable_format_seen or bp_scanned
            continue
        considered += 1
        if suffix not in _FINITE_SCANNABLE:
            continue
        scannable_format_seen = True
        try:
            m = meshio.read(str(p))
        except BaseException:
            # A best-effort scan must NEVER take down the run — some meshio
            # readers even raise SystemExit on malformed input.
            continue
        for name, arr in list(getattr(m, "point_data", {}).items()):
            w += check_finite(arr, label=f"{p.name}:{name}")
        for name, blocks in list(getattr(m, "cell_data", {}).items()):
            for i, arr in enumerate(blocks):
                w += check_finite(arr, label=f"{p.name}:{name}[{i}]")
    if considered and not scannable_format_seen:
        # Not a NaN finding — an honesty note: no output file was in a
        # format scannable in this environment (e.g. only .xplt, or .bp
        # without adios2), so finiteness is NOT asserted by the gate.
        w.append(
            "finiteness not asserted: none of the output files are in a "
            "scannable format (meshio: "
            + ", ".join(_FINITE_SCANNABLE)
            + "; .bp needs the adios2 python package) — verify field values "
            "independently.")
    return w


def _walk_nonfinite(obj, label: str) -> list[str]:
    import math
    w = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            w += _walk_nonfinite(v, f"{label}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            w += _walk_nonfinite(v, f"{label}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            w.append(f"{label}: non-finite value ({obj}) — result is invalid.")
    return w


# stdout NaN/Inf only where it clearly denotes a numeric RESULT (after = or :,
# optionally bracketed), so prose/paths can't trigger a false downgrade.
import re as _re
_STDOUT_NONFINITE = _re.compile(r"[=:]\s*[\[(]?\s*-?(?:nan|inf|infinity)\b", _re.I)


def check_summary_finite(work_dir, stdout_text: str = "") -> list[str]:
    """Scan a run's HEADLINE numbers — results_summary.json and stdout — for
    NaN/Inf. The mesh-file scan alone misses these: a summary can report
    ``"max_value": Infinity`` (the number the user actually reads) while the VTU
    field stays finite. json.loads parses bare Infinity/NaN to floats, which the
    walk then catches. Never raises.
    """
    import json as _json
    from pathlib import Path as _P
    w = []
    try:
        wd = _P(work_dir)
        for js in sorted(wd.rglob("results_summary.json")):
            try:
                w += _walk_nonfinite(_json.loads(js.read_text()), js.name)
            except Exception:
                continue
    except Exception:
        pass
    if stdout_text and _STDOUT_NONFINITE.search(stdout_text):
        w.append("stdout reports a non-finite (NaN/Inf) numeric result.")
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


def is_stub_output(content: str) -> str | None:
    """Detect a placeholder/stub generator output that advertises physics but does
    NOT produce a runnable, solving deck. Returns a reason string if stub, else None.

    Catches the silent-wrong catalog landmines the audits found across backends:
    deal.II print-and-exit placeholders, Kratos availability-probe stubs, 4C one-line
    comment templates, and `<...>`-placeholder decks. Turning these into a LOUD refusal
    (rather than fake output that passes validation) is the paper's own principle applied
    to OASiS itself.
    """
    if content is None:
        return "empty generator output"
    c = content.strip()
    if not c:
        return "empty generator output"
    low = c.lower()
    # one-line / comment-only templates (4C stubs like "# Membrane template — use ...")
    non_comment = [ln for ln in c.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    if not non_comment:
        return "template is comment-only — not a runnable deck (stub)"
    # explicit placeholder markers
    markers = [
        "see deal.ii tutorial for full implementation",  # dealii print-and-exit
        "placeholder: implement",                        # dealii NS / others
        "# placeholder", "// placeholder", "placeholder template",
        "not pip-installable", "not installed",          # kratos probe stubs
        '"note": "not installed"', "format template",     # kratos rom/iga/topology
        "use this as a starting point — not a self-contained",  # reduced_lung
    ]
    for m in markers:
        if m in low:
            return f"placeholder marker present ('{m}') — generator is a stub, not a real solve"
    # unfilled angle-bracket scalar placeholders (4C <...> YAML templates that abort).
    # Skip for XML (FEBio tags like <time_steps>) and C++ (deal.II templates <double>).
    import re
    is_xml = c.startswith("<") or "<?xml" in low or "</" in c
    is_cpp = "#include" in low or "int main" in low
    if not is_xml and not is_cpp and len(re.findall(r"<[a-z]+_[a-z_]+>", low)) >= 3:
        return "contains unfilled <...> placeholders — deck would not run (stub)"
    return None
