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

WHAT A PARTITIONED COUPLING GETS WRONG QUIETLY, and what this driver records so
the validators can catch it (each of these was demonstrated against the previous
version of this file, which reported several of them as a clean convergence):

  * a participant that exits NON-ZERO but leaves an exports.json behind — the
    iteration used to continue on the output of a crashed solve;
  * a participant that never reads imports.json (or reads a stale copy of its own
    last answer) — its export never moves, the residual is zero at iteration 2 and
    the coupling "converges" instantly to the initial condition;
  * a partner name in `imports_from` that matches no participant (a typo) — the
    edge used to be dropped silently, turning a two-way coupling into a one-way
    one with no trace in the output;
  * an export whose length changes between iterations — the relaxation reshape
    either broadcast silently or raised out of the driver;
  * a residual computed as ONE relative norm over every participant and every
    block at once, so a large, quickly-settled block (forces, fluxes) masks a
    small one (displacements, temperatures) that is still moving.

None of these is decided here: the driver records evidence
(`returncodes`, `responsiveness`, `block_residuals`, `graph`, `theta`) and
core.quality_checks turns it into verdict-bearing findings. Enforcement is always
by verdict, never by exception.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.field_transfer import InterfaceData

# How many per-iteration non-finite warnings to keep before collapsing the rest
# into a count. An unbounded list turned a single NaN into ~80 near-identical
# lines, which buries every other finding in the validation block.
_MAX_NONFINITE_WARNINGS = 4


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
    # ── evidence the validators consume (never interpreted here) ──────────
    # participant name -> exit code of its LAST run
    returncodes: dict[str, int] = field(default_factory=dict)
    # "<participant>.values" / "<participant>.normal_fluxes" -> relative change
    # of that block alone on the final iteration. A global norm hides a small
    # block behind a large one; these do not.
    block_residuals: dict[str, float] = field(default_factory=dict)
    # participant -> "responsive" | "unresponsive" | "imports never changed"
    #              | "no imports declared"
    responsiveness: dict[str, str] = field(default_factory=dict)
    # the coupling graph as the driver resolved it (declared vs. actually wired)
    graph: dict = field(default_factory=dict)
    # relaxation actually applied: {"mode": "aitken"|"constant", "theta0": float,
    #                               "final": {participant: theta}}
    theta: dict = field(default_factory=dict)


def _stack(ifd: InterfaceData) -> np.ndarray:
    v = np.asarray(ifd.values, float).ravel()
    if ifd.normal_fluxes is not None:
        v = np.concatenate([v, np.asarray(ifd.normal_fluxes, float).ravel()])
    return v


def _relax(prev: np.ndarray, new: np.ndarray, theta: float) -> np.ndarray:
    return (1 - theta) * prev + theta * new


def _aitken(prev_relaxed, new_raw, res_prev, theta_prev, lo=0.05, hi=1.0):
    """Aitken dynamic relaxation on the GLOBAL interface residual r_k = G(x_k) - x_k.

    theta_k = -theta_{k-1} * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2

    Two things this function used to get wrong, both of which cost convergence on
    correct setups (a false 'NOT CONVERGED' on a good coupling is as corrosive as
    a missed silent-wrong):

      * `res_prev` MUST be the previous RESIDUAL r_{k-1}. It used to be handed the
        previous RAW EXPORT G(x_{k-1}), which makes theta an arbitrary number in
        [lo, hi] with no relation to the iteration.
      * there must be ONE theta for the whole interface state. Aitken's derivation
        is for a single scalar sequence extrapolated from the composite fixed-point
        map; giving each participant its own theta relaxes the two halves of one
        coupled system by different amounts, which on a Dirichlet-Neumann split
        drove the two thetas apart (to the clamp at either end) and made the
        iteration diverge where constant relaxation converged.

    Returns (theta, r_k) — the caller must store r_k and hand it back next time.
    """
    r_new = new_raw - prev_relaxed
    if res_prev is None or res_prev.shape != r_new.shape:
        return min(max(theta_prev, lo), hi), r_new
    dr = r_new - res_prev
    denom = float(np.dot(dr, dr))
    if not np.isfinite(denom) or denom < 1e-30:
        return min(max(theta_prev, lo), hi), r_new
    theta = -theta_prev * float(np.dot(res_prev, dr)) / denom
    if not np.isfinite(theta):
        return min(max(theta_prev, lo), hi), r_new
    return min(max(theta, lo), hi), r_new


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _rel_change(new: np.ndarray, prev: np.ndarray) -> float:
    """Relative L2 change of one block; NaN if it cannot be formed."""
    if new.shape != prev.shape or new.size == 0:
        return float("nan")
    ref = float(np.sqrt(np.sum(prev ** 2))) + float(np.sqrt(np.sum(new ** 2)))
    if ref <= 0:
        return 0.0
    return float(np.sqrt(np.sum((new - prev) ** 2)) * 2.0 / ref)


def _blocks(ifd: InterfaceData) -> dict[str, np.ndarray]:
    """Split an export into the pieces that must each converge on their own.

    Components matter, not just arrays: a TSI or FSI interface carries
    temperature and displacement (or force and displacement) inside ONE `values`
    array of shape (N, n_comp), on scales that differ by many orders of
    magnitude. Lumping them into a single norm is exactly how the small one stops
    being visible.
    """
    b: dict[str, np.ndarray] = {}
    for label, arr in (("values", ifd.values),
                       ("normal_fluxes", ifd.normal_fluxes)):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.ndim >= 2 and a.shape[-1] > 1:
            for c in range(a.shape[-1]):
                b[f"{label}[{c}]"] = a[..., c].ravel()
        else:
            b[label] = a.ravel()
    return b


def run_coupling(participants: list[Participant], max_iter: int = 50,
                 tol: float = 1e-6, accelerator: str = "aitken",
                 theta0: float = 0.5) -> CouplingResult:
    """Run a general fixed-point partitioned coupling. Physics-agnostic.

    Each iteration: every participant consumes its partners' latest exports (relaxed),
    runs, and produces new exports. Converges when the relaxed export-vector stops
    changing. Returns success=False if not converged within max_iter.

    accelerator: "aitken" (dynamic theta, recomputed per participant per iteration,
        starting from theta0) or "constant" (theta fixed at theta0 for the whole
        run). theta0 is the ONLY relaxation knob; there is no separate per-field or
        per-participant theta.
    """
    names = [p.name for p in participants]
    # ── coupling graph: an unknown partner name is a SETUP error, not a no-op ──
    # `imports_from` used to be filtered with `if src in exports`, so a typo
    # ("Bee" for "B") silently deleted that edge and the run became a one-way
    # coupling that converged in a few iterations and reported nothing unusual.
    for p in participants:
        unknown = [s for s in p.imports_from if s not in names]
        if unknown:
            return CouplingResult(
                False, 0, float("nan"), {}, [],
                error=(f"participant {p.name}: imports_from names no such "
                       f"participant: {unknown} (participants are {names}). "
                       "Refusing to run: dropping the edge would silently turn "
                       "this into a one-way coupling."),
                graph={"participants": names,
                       "declared_edges": {q.name: list(q.imports_from)
                                          for q in participants}})
        if p.name in p.imports_from:
            return CouplingResult(
                False, 0, float("nan"), {}, [],
                error=f"participant {p.name}: imports_from includes itself.",
                graph={"participants": names,
                       "declared_edges": {q.name: list(q.imports_from)
                                          for q in participants}})
    graph = {"participants": names,
             "declared_edges": {p.name: list(p.imports_from) for p in participants}}

    # ── stage participant data files BEFORE the loop (loud on missing) ──
    for p in participants:
        p.work_dir.mkdir(parents=True, exist_ok=True)
        for df in p.data_files:
            src = Path(df).expanduser()
            if not src.is_file():
                return CouplingResult(
                    False, 0, float("nan"), {}, [],
                    error=f"participant {p.name}: data file not found: {df}",
                    graph=graph)
            dest = p.work_dir / src.name
            if dest.resolve() != src.resolve():
                shutil.copy(src, dest)

    exports: dict[str, InterfaceData] = {}      # latest relaxed exports per participant
    res_prev: dict[str, np.ndarray] = {}        # previous Aitken residual r_{k-1}
    relaxed_prev: dict[str, np.ndarray] = {}
    prev_blocks: dict[str, dict[str, np.ndarray]] = {}
    theta_global: float = theta0
    history: list[float] = []
    warnings: list[str] = []
    returncodes: dict[str, int] = {}
    block_residuals: dict[str, float] = {}
    # participant -> list of (imports digest, exports digest) per iteration
    trace: dict[str, list[tuple[str, str]]] = {p.name: [] for p in participants}
    nonfinite_hits = 0

    def _finish(**kw) -> CouplingResult:
        kw.setdefault("warnings", warnings)
        kw.setdefault("returncodes", returncodes)
        kw.setdefault("block_residuals", block_residuals)
        kw.setdefault("responsiveness", _responsiveness(trace, participants))
        kw.setdefault("graph", graph)
        kw.setdefault("theta", {"mode": accelerator, "theta0": theta0,
                                "applied": (theta_global if accelerator == "aitken"
                                            else theta0)})
        return CouplingResult(**kw)

    for it in range(1, max_iter + 1):
        new_exports: dict[str, InterfaceData] = {}
        for p in participants:
            # assemble imports = latest exports of the partners this participant reads
            imp = {src: exports[src].to_dict() for src in p.imports_from if src in exports}
            imp_text = json.dumps(imp, indent=2, sort_keys=True)
            (p.work_dir / "imports.json").write_text(imp_text)
            ep = p.work_dir / "exports.json"
            if ep.exists():
                ep.unlink()
            try:
                r = subprocess.run(p.command, cwd=str(p.work_dir), capture_output=True,
                                   text=True, timeout=p.timeout)
            except subprocess.TimeoutExpired:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} timed out after "
                                      f"{p.timeout}s at iteration {it} — the "
                                      "coupling was killed, no result"))
            except (OSError, ValueError) as e:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} could not be launched: {e}")
            returncodes[p.name] = int(r.returncode)
            if not ep.exists():
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} wrote no exports.json "
                                     f"(rc={r.returncode}). stderr tail: {r.stderr[-300:]}")
            # A NON-ZERO exit code is a failed solve even when exports.json is
            # present: a solver that diverges often writes its last iterate and
            # then aborts. Continuing on that output produced a converged-looking
            # coupling built on a crashed participant.
            if r.returncode != 0:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={n: e.to_dict() for n, e in exports.items()},
                               history=history,
                               error=(f"participant {p.name} exited with code "
                                      f"{r.returncode} at iteration {it}; its "
                                      "exports.json is the output of a FAILED run "
                                      "and must not be coupled on. stderr tail: "
                                      f"{r.stderr[-300:]}"))
            try:
                new_exports[p.name] = InterfaceData.from_json(ep)
            except Exception as e:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} bad exports.json: {e}")
            trace[p.name].append((_digest(imp_text), _digest(ep.read_text(errors="replace"))))
            ifd = new_exports[p.name]
            v = _stack(ifd)
            # An EMPTY export is not a converged one. With nothing in the stacked
            # vector the residual is 0/1e-30 = 0 at iteration 2, so a participant
            # that writes a well-formed but empty interface converges instantly
            # and every value-based check has nothing to look at and says nothing.
            if v.size == 0:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} exported an EMPTY "
                                      "interface (no values) at iteration "
                                      f"{it} — there is nothing to couple, and an "
                                      "empty exchange would otherwise report a "
                                      "residual of zero"))
            coords = np.asarray(ifd.coordinates, float)
            bad = (not np.all(np.isfinite(v))) or (
                coords.size and not np.all(np.isfinite(coords)))
            if bad:
                nonfinite_hits += 1
                if nonfinite_hits <= _MAX_NONFINITE_WARNINGS:
                    where = "values/fluxes" if not np.all(np.isfinite(v)) else "coordinates"
                    warnings.append(
                        f"{p.name}: non-finite export {where} at iter {it}")
            # An export whose length changes between iterations breaks relaxation:
            # numpy either broadcasts a length-1 block up to the new length or
            # raises out of the driver. Neither is a result.
            if p.name in relaxed_prev and v.shape != relaxed_prev[p.name].shape:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} changed its export "
                                      f"length from {relaxed_prev[p.name].shape[0]} to "
                                      f"{v.shape[0]} at iteration {it} — the exported "
                                      "interface must have the same layout every "
                                      "iteration or relaxation is meaningless"))

        # relaxation + residual on the concatenated export vector
        if it == 1:
            for n, ifd in new_exports.items():
                exports[n] = ifd
                relaxed_prev[n] = _stack(ifd)
                prev_blocks[n] = _blocks(ifd)
            history.append(float("nan"))
            continue

        total_res = 0.0
        total_ref = 0.0
        # ONE theta for the whole interface state (see _aitken): Aitken is applied
        # to the composite fixed-point map, not to each participant separately.
        if accelerator == "aitken":
            raw_all = np.concatenate([_stack(new_exports[p.name]) for p in participants])
            prev_all = np.concatenate([relaxed_prev[p.name] for p in participants])
            th, r_k = _aitken(prev_all, raw_all, res_prev.get("*"), theta_global)
            theta_global = th
            res_prev["*"] = r_k
        else:
            th = theta0
        for p in participants:
            n = p.name
            raw_new = _stack(new_exports[n])
            prev = relaxed_prev[n]
            relaxed = _relax(prev, raw_new, th)
            total_res += float(np.sum((raw_new - prev) ** 2))
            total_ref += float(np.sum(raw_new ** 2)) + 1e-30
            # per-block relative change, so a large settled block cannot mask a
            # small moving one in the single global norm below
            nb = _blocks(new_exports[n])
            for bname, arr in nb.items():
                pb = prev_blocks.get(n, {}).get(bname)
                block_residuals[f"{n}.{bname}"] = (
                    _rel_change(arr, pb) if pb is not None else float("nan"))
            prev_blocks[n] = nb
            # write relaxed values back into the InterfaceData carrier
            ifd = new_exports[n]
            ncomp = ifd.values.size
            ifd.values = relaxed[:ncomp].reshape(ifd.values.shape)
            if ifd.normal_fluxes is not None:
                ifd.normal_fluxes = relaxed[ncomp:].reshape(ifd.normal_fluxes.shape)
            exports[n] = ifd
            relaxed_prev[n] = relaxed

        res = float(np.sqrt(total_res / total_ref))
        history.append(res)
        if res < tol:
            if nonfinite_hits > _MAX_NONFINITE_WARNINGS:
                warnings.append(f"... {nonfinite_hits - _MAX_NONFINITE_WARNINGS} "
                                "further non-finite export warnings suppressed")
            return _finish(converged=True, iterations=it, residual=res,
                           exports={n: e.to_dict() for n, e in exports.items()},
                           history=history)

    if nonfinite_hits > _MAX_NONFINITE_WARNINGS:
        warnings.append(f"... {nonfinite_hits - _MAX_NONFINITE_WARNINGS} further "
                        "non-finite export warnings suppressed")
    return _finish(converged=False, iterations=max_iter,
                   residual=history[-1] if history else float("nan"),
                   exports={n: e.to_dict() for n, e in exports.items()},
                   history=history,
                   error=f"did not converge to tol={tol} in {max_iter} iters "
                         f"(last residual {history[-1]:.2e}) — result is NOT trustworthy")


def _responsiveness(trace: dict[str, list[tuple[str, str]]],
                    participants: list[Participant]) -> dict[str, str]:
    """Did each participant's export ever MOVE when its imports moved?

    The discriminator between a real converged solve and a participant that
    exits 0 having done nothing (or that re-serves a cached answer, or that
    never opened imports.json): compare the byte digest of what it was handed
    with the byte digest of what it produced. A solver whose input changed and
    whose output is byte-identical is not a function of its input.

    A genuinely converged participant does not trip this: its export becomes
    byte-identical only once the imports it is given have also stopped changing,
    and the check only looks at iteration pairs where the imports DID change.
    """
    declared = {p.name: list(p.imports_from) for p in participants}
    out: dict[str, str] = {}
    for name, hist in trace.items():
        if not declared.get(name):
            out[name] = "no imports declared"
            continue
        moved_in = moved_out = 0
        for (i0, e0), (i1, e1) in zip(hist, hist[1:]):
            if i0 != i1:
                moved_in += 1
                if e0 != e1:
                    moved_out += 1
        if moved_in == 0:
            out[name] = "imports never changed"
        elif moved_out == 0:
            out[name] = "unresponsive"
        else:
            out[name] = "responsive"
    return out
