"""General, physics-agnostic partitioned coupling driver.

Replaces the overfit `problem=`-enum coupled_solve. The driver owns ONLY the
iteration math (data exchange + relaxation + convergence). It knows nothing about
heat/elasticity/flux, no geometry, no benchmark answer. Physics lives entirely in
the participant scripts (which the agent writes) and the data currency is
InterfaceData JSON (file handshake).

Contract for a PARTICIPANT (any solver, any code, any physics):
  It is a runnable command. Each iteration the driver:
    1. writes <work_dir>/imports.json  = the InterfaceData this participant must
       consume this iteration (boundary values from its coupling partners), or
       an empty file on iteration 0.
    2. runs the participant command in <work_dir>.
    3. reads <work_dir>/exports.json   = the InterfaceData the participant produced
       on the shared interface (whatever quantities it exports — opaque to driver).
  The participant decides HOW to apply imports (Dirichlet, Neumann, Robin, traction,
  flux, concentration, ...) and WHAT to export. The driver treats both as opaque
  numbers on coordinates -> works for ANY coupling.

Convergence is on the stacked export-vector change between iterations. If it does
not converge within max_iter, the driver returns success=False LOUDLY (the most
general silent-wrong guard: never frame a non-converged run as a result).

STOCHASTIC PARTICIPANTS HAVE A RESIDUAL FLOOR, and the driver knows about it.
A Monte-Carlo participant (DSMC, any sampled estimator) returns a slightly
different export every time it is asked the SAME question. The residual is the
change in the export vector, so it cannot fall below the size of that sampling
scatter no matter how well the physics has settled — and a `tol` underneath the
floor therefore ends every run as "did not converge", on a coupling that is
right. That verdict is honest but useless, and it is the reason a stochastic
coupling could not be graded on convergence at all.

`noise_replicates` / `noise_floor` fix it without ever softening the guard:

  * the floor is MEASURED, not assumed. With `noise_replicates=N` the driver
    runs each participant N times on the SAME imports and evaluates its OWN
    residual expression across independent replicates. That number IS the
    residual a perfectly converged run would still report. Use N >= 4: the
    floor is itself an estimate and three samples is a bad one;
  * convergence is then declared against `max(tol, floor)` — never below the
    floor, never above the tolerance the caller asked for;
  * the stopping statistic becomes a BLOCK MEAN of the last `noise_block`
    residuals once a floor is in play, so a single lucky dip into the noise
    does not end the run;
  * `noise_floor` on the result is what a grader must use: a tolerance tighter
    than the floor is measuring the sampler, not the coupling;
  * a floor measured as exactly zero is REPORTED, because for a Monte-Carlo
    participant that means a fixed seed, and a residual that falls under a
    fixed seed proves only that the same draw was repeated.

The feature is inert for deterministic participants: their replicates are
bit-identical, the floor is 0, and `max(tol, 0) == tol`.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.field_transfer import InterfaceData


@dataclass
class Participant:
    """One coupled solver. `command` reads imports.json / writes exports.json in work_dir."""
    name: str
    command: list[str]        # e.g. ["python", "subdomain_A.py"] or ["/path/4C", "deckB.yaml", "out"]
    work_dir: Path
    # which partner-export this participant imports (edge): partner_name -> None (take its export)
    imports_from: list[str] = field(default_factory=list)
    timeout: int = 3600
    # Files the solver needs in work_dir (species/surface/mesh/config data).
    # Staged (copied in) once before the iteration loop. A missing file is a
    # LOUD setup error — the alternative is the solver dying mid-iteration
    # with an opaque 'Cannot open ...' (the T15 SPARTA failure mode).
    data_files: list[str] = field(default_factory=list)


@dataclass
class CouplingResult:
    converged: bool
    iterations: int
    residual: float
    exports: dict[str, dict]          # name -> InterfaceData.to_dict()
    history: list[float]
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    # Stochastic branch. `noise_floor` is None when no floor was measured or
    # declared; 0.0 means it WAS established and came out zero, which is a
    # different statement and the reason these are not collapsed.
    noise_floor: Optional[float] = None
    tol_effective: Optional[float] = None
    stopped_at_noise_floor: bool = False
    # PROVENANCE, not warnings. How the floor was measured, and the fixed-seed
    # caveat, belong next to the number rather than in `validation` — the tool
    # copies `warnings` into `validation`, and an agent is told that an empty
    # validation block is what a correct coupling looks like. Putting a routine
    # measurement note there would make every stochastic-aware run look flagged
    # and every deterministic one that merely asked for a floor look flagged
    # too. Only a floor that actually CHANGED the verdict is a warning.
    notes: list[str] = field(default_factory=list)


def _stack(ifd: InterfaceData) -> np.ndarray:
    v = np.asarray(ifd.values, float).ravel()
    if ifd.normal_fluxes is not None:
        v = np.concatenate([v, np.asarray(ifd.normal_fluxes, float).ravel()])
    return v


def _relax(prev: np.ndarray, new: np.ndarray, theta: float) -> np.ndarray:
    return (1 - theta) * prev + theta * new


def _aitken(prev_relaxed, new_raw, prev_raw, theta_prev):
    """Aitken dynamic relaxation on the residual r = new_raw - prev_relaxed.
    Lifted generically (no physics). Falls back to theta_prev if denominator ~0."""
    r_new = new_raw - prev_relaxed
    if prev_raw is None:
        return min(max(theta_prev, 0.1), 1.0), r_new
    r_old = prev_raw
    dr = r_new - r_old
    denom = float(np.dot(dr, dr))
    if denom < 1e-30:
        return min(max(theta_prev, 0.1), 1.0), r_new
    theta = -theta_prev * float(np.dot(r_old, dr)) / denom
    theta = min(max(theta, 0.05), 1.0)
    return theta, r_new


def _invoke(p: Participant, imp: dict) -> tuple[Optional[InterfaceData], Optional[str]]:
    """Write imports, run the participant once, read its export back.

    Returns (InterfaceData, None) or (None, error). Factored out of the
    iteration loop so the noise-floor measurement drives participants through
    exactly the same path — a floor measured by a different mechanism than the
    residual it is compared against would not be a floor for that residual.
    """
    (p.work_dir / "imports.json").write_text(json.dumps(imp, indent=2))
    ep = p.work_dir / "exports.json"
    if ep.exists():
        ep.unlink()
    try:
        r = subprocess.run(p.command, cwd=str(p.work_dir), capture_output=True,
                           text=True, timeout=p.timeout)
    except subprocess.TimeoutExpired:
        return None, f"participant {p.name} timed out"
    if not ep.exists():
        return None, (f"participant {p.name} wrote no exports.json "
                      f"(rc={r.returncode}). stderr tail: {r.stderr[-300:]}")
    try:
        return InterfaceData.from_json(ep), None
    except Exception as e:
        return None, f"participant {p.name} bad exports.json: {e}"


def _residual_of(new: dict[str, np.ndarray],
                 prev: dict[str, np.ndarray]) -> float:
    """The driver's own residual expression, on two sets of export vectors.

    Kept as one function so the convergence test and the noise-floor
    measurement cannot drift apart.
    """
    num = 0.0
    ref = 0.0
    for n, v in new.items():
        num += float(np.sum((v - prev[n]) ** 2))
        ref += float(np.sum(v ** 2)) + 1e-30
    return float(np.sqrt(num / ref))


def _measure_noise_floor(participants: list[Participant], replicates: int,
                         imports_for: dict[str, dict], where: str,
                         against: Optional[dict] = None
                         ) -> tuple[Optional[float], Optional[str], list[str]]:
    """Run every participant `replicates` times on the SAME imports and report
    the residual the driver would still see between independent answers.

    Returns (floor, error, notes). `floor` is None only when the measurement
    could not be made at all.

    WHY REPLICATES AND NOT A STANDARD DEVIATION. The quantity that has to be
    beaten is the residual, and the residual is a normalised difference of two
    export vectors. So the floor is that same expression, evaluated on
    independent answers to one question — no distributional assumption, no
    conversion factor, and directly comparable to the number the loop reports.

    `against` decides WHICH comparison, and the two are genuinely different:

      * `against=<the loop's relaxed_prev>` is the faithful one. Each replicate
        is scored against the very vector the loop compares to, so the value IS
        the residual the loop reports, measured several times.
      * `against=None` (before the loop, where no relaxed_prev exists) scores
        replicates against EACH OTHER. That is a LOWER BOUND, not the same
        number. It was first written down here as conservative on the argument
        that the relaxed blend averages noise down — and measuring it showed the
        opposite: the relaxed vector is a lagged average carrying its own
        accumulated noise, and noise that has propagated through a partner over
        earlier iterations is missing from the pairwise figure entirely. On one
        two-participant case the pairwise estimate came out around a third of
        the faithful one. It is used to avoid a pointless long run, never as the
        final word.
    """
    notes: list[str] = []
    draws: list[dict[str, np.ndarray]] = []
    for k in range(replicates):
        one: dict[str, np.ndarray] = {}
        for p in participants:
            ifd, err = _invoke(p, imports_for.get(p.name, {}))
            if err:
                return None, f"noise-floor measurement ({where}): {err}", notes
            one[p.name] = _stack(ifd)
        if draws and any(one[n].size != draws[0][n].size for n in one):
            return None, (f"noise-floor measurement ({where}): a participant "
                          f"changed its export size between replicate runs, so "
                          f"no floor can be defined for it"), notes
        draws.append(one)
    if against is not None:
        # THE FAITHFUL MEASUREMENT, available only once the loop has a previous
        # relaxed vector to compare against: repeat the iteration N times from
        # the same state and take the residual it ACTUALLY REPORTS each time.
        # Nothing is modelled — this is the same expression on the same two
        # operands the loop uses.
        vals = [_residual_of(d, against) for d in draws]
        n_ind = len(vals)
    else:
        # PRE-LOOP there is no previous relaxed vector, so the only thing
        # available is the scatter BETWEEN independent answers. That is a LOWER
        # BOUND on the loop's floor and not the same number: in the loop the
        # comparison is against a lagged relaxed average which carries its own
        # accumulated noise, and noise that has propagated through a partner
        # over earlier iterations is not in this at all. It is used to avoid a
        # pointless long run; the verdict is re-judged against the faithful
        # measurement if the loop ends un-converged.
        pairs = [(i, j) for i in range(replicates)
                 for j in range(i + 1, replicates)]
        vals = [_residual_of(draws[i], draws[j]) for i, j in pairs]
        n_ind = len(pairs)
    floor = float(sum(vals) / len(vals))
    notes.append(f"noise floor {floor:.3e} from {replicates} replicate runs "
                 f"per participant ({n_ind} samples, {where})")
    # THE FLOOR IS ITSELF AN ESTIMATE, and at three replicates it is a bad one.
    # Measured here: the same coupling gave 1.2e-03 from three replicates and
    # 9.6e-03 from five — a factor of eight, from estimator scatter alone, on a
    # quantity the convergence verdict is compared against. Three replicates
    # give only three pairs and they are not independent. Say so rather than let
    # a lucky low draw make the criterion look tighter than it is.
    if n_ind < 6:
        notes.append(
            f"this floor rests on only {n_ind} replicate samples and is "
            f"itself a noisy estimate — raise noise_replicates to 4 or more "
            f"(6+ pairs) before relying on the number, especially before using "
            f"it as a grading tolerance")
    return floor, None, notes


def run_coupling(participants: list[Participant], max_iter: int = 50,
                 tol: float = 1e-6, accelerator: str = "aitken",
                 theta0: float = 0.5, noise_floor: Optional[float] = None,
                 noise_replicates: int = 0,
                 noise_block: int = 3) -> CouplingResult:
    """Run a general fixed-point partitioned coupling. Physics-agnostic.

    Each iteration: every participant consumes its partners' latest exports (relaxed),
    runs, and produces new exports. Converges when the relaxed export-vector stops
    changing. Returns success=False if not converged within max_iter.

    STOCHASTIC PARTICIPANTS (see the module docstring):
      * `noise_replicates >= 2` measures the residual noise floor by running
        every participant that many times on the same imports, before the loop.
      * `noise_floor` declares one instead (or raises a measured one), for a
        caller who established it independently.
      * whichever is larger of `tol` and the floor becomes the convergence
        criterion, and once a non-zero floor is in play the stopping statistic
        is the mean of the last `noise_block` residuals rather than a single
        one, so one lucky dip into the noise cannot end the run.
      * if the loop still ends un-converged, the floor is RE-MEASURED at the
        final state — the pre-loop estimate is taken with the participants in
        their iteration-1 fallback, which need not carry the same relative
        scatter as the settled state — and the verdict is re-judged against it.
        That second measurement is paid only on failure.
    """
    # ── stage participant data files BEFORE the loop (loud on missing) ──
    for p in participants:
        p.work_dir.mkdir(parents=True, exist_ok=True)
        for df in p.data_files:
            src = Path(df).expanduser()
            if not src.is_file():
                return CouplingResult(
                    False, 0, float("nan"), {}, [],
                    error=f"participant {p.name}: data file not found: {df}")
            dest = p.work_dir / src.name
            if dest.resolve() != src.resolve():
                shutil.copy(src, dest)

    exports: dict[str, InterfaceData] = {}      # latest relaxed exports per participant
    raw_prev: dict[str, np.ndarray] = {}
    relaxed_prev: dict[str, np.ndarray] = {}
    theta: dict[str, float] = {p.name: theta0 for p in participants}
    history: list[float] = []
    warnings: list[str] = []
    notes: list[str] = []

    # ── the noise floor, before anything iterates ──────────────────────────
    floor: Optional[float] = None if noise_floor is None else float(noise_floor)
    if noise_replicates and noise_replicates >= 2:
        measured, err, got = _measure_noise_floor(
            participants, int(noise_replicates),
            {p.name: {} for p in participants}, "iteration-1 state")
        if err:
            return CouplingResult(False, 0, float("nan"), {}, history,
                                  error=err, warnings=warnings, notes=notes)
        notes.extend(got)
        floor = measured if floor is None else max(floor, measured)
        if measured == 0.0:
            notes.append(
                f"noise floor measured as EXACTLY 0 from {noise_replicates} "
                f"replicate runs: every participant returned a bit-identical "
                f"export. For a deterministic solver that is what should "
                f"happen. For a Monte-Carlo participant it means the SEED IS "
                f"FIXED — the residual then measures the repeatability of one "
                f"random draw, not whether the physics settled, and a run that "
                f"meets tol under a fixed seed is not evidence it would meet it "
                f"under another. Vary the seed between replicates to measure a "
                f"real floor.")
    tol_eff = tol if not floor else max(tol, float(floor))
    block = max(1, int(noise_block)) if floor else 1
    if floor and tol_eff > tol:
        warnings.append(
            f"CONVERGENCE IS AT THE NOISE FLOOR, NOT AT tol: the requested "
            f"tol={tol:.1e} is below the residual noise floor {floor:.3e}, "
            f"which no amount of iterating can cross. The run is judged against "
            f"{tol_eff:.3e} instead, over a block mean of the last {block} "
            f"residuals. ANY TOLERANCE APPLIED TO THIS RESULT — including a "
            f"grading tolerance — MUST BE AT LEAST {floor:.3e} RELATIVE.")

    for it in range(1, max_iter + 1):
        new_exports: dict[str, InterfaceData] = {}
        for p in participants:
            # assemble imports = latest exports of the partners this participant reads
            imp = {src: exports[src].to_dict() for src in p.imports_from if src in exports}
            ifd, err = _invoke(p, imp)
            if err:
                return CouplingResult(False, it, float("nan"), {}, history,
                                      error=err, warnings=warnings,
                                      notes=notes)
            new_exports[p.name] = ifd
            v = _stack(ifd)
            if not np.all(np.isfinite(v)):
                warnings.append(f"{p.name}: non-finite export values at iter {it}")

        # relaxation + residual on the concatenated export vector
        if it == 1:
            for n, ifd in new_exports.items():
                exports[n] = ifd; relaxed_prev[n] = _stack(ifd); raw_prev[n] = _stack(ifd)
            history.append(float("nan")); continue

        # A participant that changes its export LENGTH between iterations used
        # to reach _relax and raise a bare numpy broadcast ValueError straight
        # out of run_coupling — or, when the new length was 1, broadcast
        # silently and be misreported as a non-convergence with no clue why.
        # Both are the same setup error, so name it here.
        for p in participants:
            m, k = _stack(new_exports[p.name]).size, relaxed_prev[p.name].size
            if m != k:
                return CouplingResult(
                    False, it, float("nan"), {}, history,
                    error=(f"participant {p.name} changed its export size from "
                           f"{k} to {m} at iteration {it}. Every participant must "
                           f"export the SAME number of points, in the same order, "
                           f"every iteration (and keep normal_fluxes present or "
                           f"absent consistently) — the driver relaxes export "
                           f"vectors element by element."),
                    warnings=warnings, notes=notes)

        total_res = 0.0; total_ref = 0.0
        for p in participants:
            n = p.name; raw_new = _stack(new_exports[n]); prev = relaxed_prev[n]
            if accelerator == "aitken":
                th, _ = _aitken(prev, raw_new, raw_prev.get(n), theta[n]); theta[n] = th
            else:
                th = theta0
            relaxed = _relax(prev, raw_new, th)
            total_res += float(np.sum((raw_new - prev) ** 2))
            total_ref += float(np.sum(raw_new ** 2)) + 1e-30
            # write relaxed values back into the InterfaceData carrier
            ifd = new_exports[n]
            ncomp = ifd.values.size
            ifd.values = relaxed[:ncomp].reshape(ifd.values.shape)
            if ifd.normal_fluxes is not None:
                ifd.normal_fluxes = relaxed[ncomp:].reshape(ifd.normal_fluxes.shape)
            exports[n] = ifd
            raw_prev[n] = raw_new; relaxed_prev[n] = relaxed

        res = float(np.sqrt(total_res / total_ref)); history.append(res)
        if _stat(history, block) < tol_eff:
            return CouplingResult(
                True, it, res, {n: e.to_dict() for n, e in exports.items()},
                history, warnings=warnings, notes=notes, noise_floor=floor,
                tol_effective=tol_eff, stopped_at_noise_floor=bool(tol_eff > tol))

    # ── did not converge. If a floor was in play, the pre-loop estimate was
    # taken with the participants in their iteration-1 fallback, which need not
    # carry the same relative scatter as the settled state. Re-measure THERE
    # before calling a stochastic coupling a failure. Paid only on failure.
    last = history[-1] if history else float("nan")
    if noise_replicates and noise_replicates >= 2:
        imports_now = {p.name: {s: exports[s].to_dict()
                                for s in p.imports_from if s in exports}
                       for p in participants}
        measured, err, got = _measure_noise_floor(
            participants, int(noise_replicates), imports_now,
            "final state, against the residual the loop reports",
            against=relaxed_prev)
        if not err:
            notes.extend(got)
            if measured is not None and measured > (floor or 0.0):
                floor = measured
                tol_eff = max(tol, float(floor))
            if _stat(history, block) < tol_eff:
                warnings.append(
                    f"CONVERGED AT THE RE-MEASURED NOISE FLOOR. The iteration did "
                    f"not reach tol={tol:.1e}, but the residual noise floor "
                    f"measured with the participants in their FINAL state is "
                    f"{floor:.3e}, and the block mean of the last {block} "
                    f"residuals is below it. The residual has stopped measuring "
                    f"the coupling and started measuring the sampler. Any "
                    f"tolerance applied to this result must be at least "
                    f"{floor:.3e} relative.")
                return CouplingResult(
                    True, max_iter, last,
                    {n: e.to_dict() for n, e in exports.items()}, history,
                    warnings=warnings, notes=notes, noise_floor=floor,
                    tol_effective=tol_eff, stopped_at_noise_floor=True)

    err_msg = (f"did not converge to tol={tol_eff:g} in {max_iter} iters "
               f"(last residual {last:.2e}) — result is NOT trustworthy")
    if floor is None and _stalled(history):
        # The residual stopped falling rather than never having fallen. That is
        # what a sampling floor looks like from outside, and it is also what a
        # theta above the stability limit looks like; the driver cannot tell
        # them apart without a measurement, so it names the measurement.
        err_msg += ("; the residual STOPPED FALLING rather than falling too "
                    "slowly — if any participant is a Monte-Carlo / sampled "
                    "estimator, re-run with noise_replicates>=2 so the driver "
                    "can measure its residual floor and judge against it "
                    "instead of against an unreachable tol")
    return CouplingResult(False, max_iter, last,
                          {n: e.to_dict() for n, e in exports.items()}, history,
                          error=err_msg, warnings=warnings, notes=notes,
                          noise_floor=floor,
                          tol_effective=tol_eff, stopped_at_noise_floor=False)


def _stat(history: list[float], block: int) -> float:
    """The stopping statistic: the last residual, or the mean of the last
    `block` of them once a noise floor is in play.

    Block-averaging is the whole discipline for a stochastic coupling. A single
    residual that dips under the floor says nothing — the noise puts it there
    roughly half the time — so with a floor active the run only stops when a
    WHOLE BLOCK of consecutive residuals averages below it.
    """
    vals = [v for v in history[-block:] if np.isfinite(v)]
    if not vals:
        return float("inf")
    if len(vals) < block:
        # Not enough post-NaN history yet to fill the block; refuse to stop.
        return float("inf")
    return float(sum(vals) / len(vals))


def _stalled(history: list[float], window: int = 6) -> bool:
    """Has the residual stopped falling? Compares the mean of the last half of
    a window against the first half. Used only to make a failure message name
    the right next step, never to declare convergence."""
    vals = [v for v in history if np.isfinite(v)]
    if len(vals) < window:
        return False
    half = window // 2
    early = sum(vals[-window:-half]) / half
    late = sum(vals[-half:]) / half
    return late > 0.5 * early
