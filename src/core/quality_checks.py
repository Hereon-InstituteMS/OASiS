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
        # adios2 IS present but could not read the dataset: that is a CORRUPT
        # result file, not an unscannable format — report it as hard evidence
        # failure (verdict-flipping), unlike the missing-adios2 case above.
        return [
            f"{getattr(path, 'name', path)}: unreadable/corrupt result file — "
            "the gate could not read it to assert finiteness; the output "
            "cannot serve as verified run evidence."
        ], False


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
        try:
            m = meshio.read(str(p))
        except BaseException:
            # A best-effort scan must NEVER take down the run — some meshio
            # readers even raise SystemExit on malformed input. But a file with
            # a SCANNABLE suffix that fails to parse is a CORRUPT result file,
            # not an unscannable format (stress audit F1: a garbage-only .vtu
            # was stamped VERIFIED with the honesty note relegated to
            # 'validation'). Report it as a hard, verdict-flipping finding —
            # a result the gate cannot read is not verified run evidence.
            w.append(
                f"{p.name}: unreadable/corrupt result file — the gate could "
                "not read it to assert finiteness; the output cannot serve as "
                "verified run evidence.")
            continue
        # Only mark as scanned AFTER a successful read — otherwise an unreadable
        # .vtu would suppress the honesty note without any check having run.
        scannable_format_seen = True
        for name, arr in list(getattr(m, "point_data", {}).items()):
            w += check_finite(arr, label=f"{p.name}:{name}")
        for name, blocks in list(getattr(m, "cell_data", {}).items()):
            for i, arr in enumerate(blocks):
                w += check_finite(arr, label=f"{p.name}:{name}[{i}]")
    if considered and not scannable_format_seen and not w:
        # `not w`: when a hard corrupt-file finding was already emitted, the
        # note below would be misleading (the format IS scannable here — the
        # file is corrupt) and redundant (the hard finding flips the verdict).
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
# Also matches arrow ("max(u) -> nan") and copula ("residual is nan") headline
# forms (stress audit F2) — still anchored to a result-introducing token so
# prose ("infinite domain", "information") cannot false-trigger.
_STDOUT_NONFINITE = _re.compile(
    r"(?:[=:]|->|→|\bis\b)\s*[\[(]?\s*[+-]?(?:nan|inf|infinity)\b", _re.I)


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
    normal_fluxes — no physics. `export_*` are InterfaceData-like dicts/objects.

    Returns findings only. Whether the check COULD run at all is a separate
    question, answered by `interface_balance_coverage` — a coupling that
    exchanges no fluxes gets an empty finding list here, and an empty finding
    list must never be read as "conservation was checked and is fine".
    """
    w = []
    def _flux(e):
        f = e.get("normal_fluxes") if isinstance(e, dict) else getattr(e, "normal_fluxes", None)
        return None if f is None else float(_np.sum(_np.asarray(f, float)))
    fa, fb = _flux(export_a), _flux(export_b)
    if fa is None or fb is None:
        return w
    # A non-finite net flux makes every comparison below False (nan > rtol is
    # False), so without this the check would report nothing at all on the most
    # broken data it can be handed.
    if not (_np.isfinite(fa) and _np.isfinite(fb)):
        return [f"Interface flux balance could NOT be evaluated: net({label_a})={fa}, "
                f"net({label_b})={fb} — a non-finite exchanged flux. Conservation "
                "is unchecked and the exchanged data is invalid."]
    denom = max(abs(fa), abs(fb), 1e-30)
    rel = abs(fa + fb) / denom            # A exports +flux, B imports -flux → sum≈0
    if rel > rtol:
        # Name the convention. The most common cause of this warning is not a
        # non-conservative coupling but both sides exporting their flux with
        # the SAME sign — which a correct coupling does if nobody said which
        # normal to use. Saying only "not balanced" sends the agent looking for
        # a physics bug that is not there.
        same_sign = fa * fb > 0 and abs(abs(fa) - abs(fb)) / denom <= rtol
        hint = (" The two magnitudes match but the signs agree, which is the "
                "signature of a SIGN-CONVENTION error rather than a "
                "conservation error: each participant must export the flux "
                "through the interface with respect to ITS OWN outward normal, "
                "and those normals are anti-parallel, so the two sums should "
                "cancel. Note this is the opposite of the BC value you APPLY, "
                "which is the same number on both sides."
                if same_sign else "")
        if not hint:
            hint = _unit_ratio_hint(fa, fb)
        w.append(
            f"Interface flux NOT balanced: net({label_a})={fa:.4g}, net({label_b})={fb:.4g}, "
            f"imbalance {rel:.1%} > {rtol:.0%} — coupling may be non-conservative (silent error)."
            + hint
        )
    return w


# Ratios that show up when two participants agree on the physics and disagree on
# the units. Naming the suspect turns "your coupling is non-conservative" (which
# sends the agent hunting a physics bug that is not there) into one thing to
# check. Deliberately conservative: only near-exact ratios, and always phrased as
# a candidate, never as a diagnosis.
_UNIT_RATIOS = [
    (1e3, "a 1000x factor — the classic W/mW, m/mm, kg/g, kPa/Pa mix-up"),
    (1e-3, "a 1000x factor — the classic W/mW, m/mm, kg/g, kPa/Pa mix-up"),
    (1e6, "a 1e6 factor — e.g. Pa/MPa, m^2/mm^2, W/uW"),
    (1e-6, "a 1e6 factor — e.g. Pa/MPa, m^2/mm^2, W/uW"),
    (1e9, "a 1e9 factor — e.g. Pa/GPa, m^3/mm^3"),
    (1e-9, "a 1e9 factor — e.g. Pa/GPa, m^3/mm^3"),
    (1e4, "a 1e4 factor"), (1e-4, "a 1e4 factor"),
    (60.0, "a factor of 60 — a per-second / per-minute rate mismatch"),
    (1 / 60.0, "a factor of 60 — a per-second / per-minute rate mismatch"),
    (3600.0, "a factor of 3600 — a per-second / per-hour rate mismatch"),
    (1 / 3600.0, "a factor of 3600 — a per-second / per-hour rate mismatch"),
]


def _unit_ratio_hint(fa: float, fb: float, rtol: float = 0.02) -> str:
    """Name a UNIT MISMATCH when the two net fluxes differ by a suspicious factor.

    A sign error makes the magnitudes match; a unit error makes them differ by a
    clean power of ten (or 60 / 3600). Both look identical to a plain imbalance
    number, and the second one converges to a confidently wrong answer.
    """
    if fa == 0.0 or fb == 0.0:
        return ""
    ratio = abs(fb) / abs(fa)
    for target, what in _UNIT_RATIOS:
        if abs(ratio / target - 1.0) <= rtol:
            return (f" The magnitudes differ by {what}, not by a small amount: "
                    "that is the signature of a UNIT MISMATCH between the two "
                    "participants rather than a conservation error. Check that "
                    "both sides express the exchanged quantity in the same units "
                    "before looking for a physics bug.")
    return ""


def check_monolithic_consistency(coupled_qoi: float, monolithic_qoi: float,
                                 rtol: float = 0.05, qoi: str = "QoI") -> list[str]:
    """If the same problem can be solved un-split in one code, the coupled answer must
    match it. The most decisive silent-wrong detector — needs no external benchmark,
    only a monolithic re-solve. Returns a warning if they disagree beyond rtol.

    Unit mismatches, a wrongly applied interface sign, a participant that never
    reads its imports and a lossy mesh mapping all end at the same place: a
    coupled number that is clean, converged and wrong. This is the only check in
    the file that compares that number against an independent answer to the same
    question, which is why `couple` reports loudly when it was not run.
    """
    w = []
    if monolithic_qoi is None or coupled_qoi is None:
        return w
    if not (_np.isfinite(coupled_qoi) and _np.isfinite(monolithic_qoi)):
        return [f"{qoi}: coupled={coupled_qoi} vs monolithic re-solve="
                f"{monolithic_qoi} — a non-finite value, so the two could not be "
                "compared and the coupled result is not corroborated."]
    denom = max(abs(monolithic_qoi), 1e-30)
    rel = abs(coupled_qoi - monolithic_qoi) / denom
    if rel > rtol:
        w.append(
            f"{qoi}: coupled={coupled_qoi:.5g} vs monolithic re-solve={monolithic_qoi:.5g} "
            f"differ by {rel:.1%} > {rtol:.0%} — the coupled result is likely WRONG."
        )
    return w


# ── coupling-machinery checks (consume the driver's recorded evidence) ────────
# Each one answers a question a partitioned coupling can otherwise get wrong
# while reporting a clean convergence. They return (findings, not_checked):
# findings flip the verdict, not_checked is reported so that "this check could
# not look at anything" is never silently indistinguishable from "this check
# looked and was happy".

def check_coupling_directionality(graph: dict, max_iter: int = 0
                                  ) -> tuple[list[str], list[str]]:
    """Is the coupling wired the way the caller thinks it is?

    A partitioned coupling is a directed graph, and the two ways it goes wrong
    are silent: a participant that declares no partner at all, and an edge to a
    partner whose name is misspelled. Both make the iteration a one-way transfer
    that converges quickly and looks excellent — the residual really is zero,
    because nothing is feeding back.

    A deliberate one-way transfer is declared by asking for a single pass
    (max_iter=1). Iterating a one-way graph is the confusion this catches.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    names = list(graph.get("participants") or [])
    edges = dict(graph.get("declared_edges") or {})
    if not names or not edges:
        return findings, ["coupling directionality: the participant graph was "
                          "not recorded, so one-way/two-way could not be checked"]
    unknown = {n: [s for s in srcs if s not in names] for n, srcs in edges.items()}
    unknown = {n: u for n, u in unknown.items() if u}
    if unknown:
        findings.append(
            f"Coupling graph names unknown participants: {unknown} — those edges "
            f"carry no data. Known participants: {names}.")
    isolated = [n for n in names if not edges.get(n)]
    if isolated and max_iter != 1:
        findings.append(
            f"ONE-WAY coupling: {isolated} import from nobody, so no information "
            "flows back to them and iterating to 'convergence' is meaningless — "
            "the residual falls to zero because nothing changes, not because the "
            "coupled problem was solved. If one-way IS intended, ask for a single "
            "pass (max_iter=1), which declares it; otherwise set imports_from on "
            "both sides.")
    elif isolated:
        not_checked.append(
            f"two-way convergence: {isolated} import from nobody and this was run "
            "as a declared single pass, so nothing was iterated and no coupled "
            "fixed point was established")
    return findings, not_checked


def check_participant_responsiveness(responsiveness: dict) -> tuple[list[str], list[str]]:
    """Did every participant's answer actually depend on what it was given?

    This is the check for the participant that exits 0 having done nothing: it
    re-emits its initial condition (or a cached first answer) every iteration, so
    the export-vector change is exactly zero at iteration 2 and the coupling
    reports converged with a residual of 0.0 and no other complaint. A real solve
    handed different boundary data does not return byte-identical output.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    if not responsiveness:
        return findings, ["participant responsiveness: the driver recorded no "
                          "per-iteration trace, so a do-nothing participant "
                          "could not be ruled out"]
    dead = [n for n, s in responsiveness.items() if s == "unresponsive"]
    frozen = [n for n, s in responsiveness.items() if s == "imports never changed"]
    if dead:
        findings.append(
            f"Participant(s) {dead} produced byte-identical output while the data "
            "handed to them CHANGED — their answer does not depend on their "
            "imports. Either the script never reads imports.json, or it re-serves "
            "a cached/initial result. Any convergence reported here is the "
            "coupling standing still, not a solution.")
    if frozen:
        not_checked.append(
            f"participant responsiveness for {frozen}: the data handed to them "
            "never changed during the run, so whether they read it could not be "
            "established")
    return findings, not_checked


def check_interface_meshes(export_a, export_b, label_a="A", label_b="B",
                           rtol: float = 1e-6) -> tuple[list[str], list[str]]:
    """Compare the two sides' interface discretisations.

    Non-matching interface meshes are legitimate and routine, so this is NOT an
    error — but it changes what the other numbers mean. Every exchange then goes
    through an interpolation that is lossy and does not conserve the integrated
    quantity unless the mapping was built to, and a converged residual is
    completely silent about that. Nothing here can inspect the caller's mapping,
    so the honest report is: say the interfaces do not match, and say that
    conservation across them is established by the flux balance or not at all.
    That belongs in the coverage list, not in the findings — reporting the
    geometry as a failure would be as wrong as reporting nothing.
    """
    findings: list[str] = []
    not_checked: list[str] = []

    def _co(e):
        c = e.get("coordinates") if isinstance(e, dict) else getattr(e, "coordinates", None)
        return None if c is None else _np.atleast_2d(_np.asarray(c, float))

    def _has_flux(e):
        f = e.get("normal_fluxes") if isinstance(e, dict) else getattr(e, "normal_fluxes", None)
        return f is not None and len(_np.asarray(f, float).ravel()) > 0

    ca, cb = _co(export_a), _co(export_b)
    if ca is None or cb is None or ca.size == 0 or cb.size == 0:
        return findings, ["interface mesh conformity: one or both participants "
                          "exported no interface coordinates, so matching / "
                          "non-matching discretisation could not be checked"]
    na, nb = len(ca), len(cb)
    if na == nb and ca.shape == cb.shape:
        span = float(_np.max(_np.abs(ca))) or 1.0
        if float(_np.max(_np.abs(ca - cb))) <= rtol * span:
            return findings, not_checked          # matching, node-for-node
    both_flux = _has_flux(export_a) and _has_flux(export_b)
    note = (f"conservation across a NON-MATCHING interface ({label_a} exports {na} "
            f"point(s), {label_b} exports {nb}): every exchange passes through an "
            "interpolation, which is lossy and does not conserve the integrated "
            "quantity unless the mapping was built to — a nearest-neighbour or "
            "plain linear map is not. ")
    note += ("The interface flux balance is the only evidence here that it did "
             "conserve; the residual is silent about it."
             if both_flux else
             "Neither side exported `normal_fluxes`, so NOTHING here checked "
             "whether the interpolation conserved. Export the normal flux from "
             "both sides to make that checkable.")
    return findings, [note]


def check_residual_blocks(block_residuals: dict, tol: float,
                          slack: float = 10.0) -> tuple[list[str], list[str]]:
    """Is the reported global residual actually representative?

    The driver converges on ONE relative norm over every participant's stacked
    export vector. When the exchanged quantities live on different scales — the
    standard case in FSI (forces ~1e3, displacements ~1e-5) and TSI (temperature
    ~1e3, displacement ~1e-5) — the large block sets the denominator and the small
    block can still be moving by a large fraction of itself while the global
    number sits below tolerance. That is a converged-looking, wrong answer with no
    other symptom.
    """
    findings: list[str] = []
    not_checked: list[str] = []
    if not block_residuals:
        return findings, ["per-block convergence: the driver recorded no "
                          "per-block residuals, so scale masking in the global "
                          "residual could not be ruled out"]
    finite = {k: v for k, v in block_residuals.items() if v == v and abs(v) != float("inf")}
    if not finite:
        return findings, ["per-block convergence: every per-block residual was "
                          "non-finite or unavailable"]
    limit = tol * slack
    bad = {k: v for k, v in finite.items() if v > limit}
    if bad:
        worst = max(bad.items(), key=lambda kv: kv[1])
        findings.append(
            "Global residual is NOT representative: block(s) "
            + ", ".join(f"{k}={v:.2e}" for k, v in sorted(bad.items()))
            + f" are still changing by more than {limit:.1e} relative, while the "
            "global norm — which is dominated by the largest-magnitude block — "
            f"reports convergence. {worst[0]} is the one to look at. Converge each "
            "exchanged quantity in its own units, or scale the blocks before "
            "taking the norm.")
    return findings, not_checked


def check_returncodes(returncodes: dict) -> tuple[list[str], list[str]]:
    """Every participant's LAST run must have exited 0.

    A solver that diverges commonly writes its last iterate and then aborts; the
    file handshake sees a perfectly well-formed exports.json and couples on it.
    """
    findings: list[str] = []
    if not returncodes:
        return findings, ["participant exit codes: none were recorded"]
    bad = {n: rc for n, rc in returncodes.items() if rc != 0}
    if bad:
        findings.append(
            f"Participant(s) exited non-zero: {bad} — the exchanged data on that "
            "iteration is the output of a FAILED solve, whatever the residual says.")
    return findings, []



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
