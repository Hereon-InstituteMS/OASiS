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
"""
from __future__ import annotations
import json
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


@dataclass
class CouplingResult:
    converged: bool
    iterations: int
    residual: float
    exports: dict[str, dict]          # name -> InterfaceData.to_dict()
    history: list[float]
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


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


def run_coupling(participants: list[Participant], max_iter: int = 50,
                 tol: float = 1e-6, accelerator: str = "aitken",
                 theta0: float = 0.5) -> CouplingResult:
    """Run a general fixed-point partitioned coupling. Physics-agnostic.

    Each iteration: every participant consumes its partners' latest exports (relaxed),
    runs, and produces new exports. Converges when the relaxed export-vector stops
    changing. Returns success=False if not converged within max_iter.
    """
    exports: dict[str, InterfaceData] = {}      # latest relaxed exports per participant
    raw_prev: dict[str, np.ndarray] = {}
    relaxed_prev: dict[str, np.ndarray] = {}
    theta: dict[str, float] = {p.name: theta0 for p in participants}
    history: list[float] = []
    warnings: list[str] = []

    for it in range(1, max_iter + 1):
        new_exports: dict[str, InterfaceData] = {}
        for p in participants:
            # assemble imports = latest exports of the partners this participant reads
            imp = {src: exports[src].to_dict() for src in p.imports_from if src in exports}
            (p.work_dir / "imports.json").write_text(json.dumps(imp, indent=2))
            ep = p.work_dir / "exports.json"
            if ep.exists():
                ep.unlink()
            try:
                r = subprocess.run(p.command, cwd=str(p.work_dir), capture_output=True,
                                   text=True, timeout=p.timeout)
            except subprocess.TimeoutExpired:
                return CouplingResult(False, it, float("nan"), {}, history,
                                      error=f"participant {p.name} timed out", warnings=warnings)
            if not ep.exists():
                return CouplingResult(False, it, float("nan"), {}, history,
                                      error=f"participant {p.name} wrote no exports.json "
                                            f"(rc={r.returncode}). stderr tail: {r.stderr[-300:]}",
                                      warnings=warnings)
            try:
                new_exports[p.name] = InterfaceData.from_json(ep)
            except Exception as e:
                return CouplingResult(False, it, float("nan"), {}, history,
                                      error=f"participant {p.name} bad exports.json: {e}",
                                      warnings=warnings)
            v = _stack(new_exports[p.name])
            if not np.all(np.isfinite(v)):
                warnings.append(f"{p.name}: non-finite export values at iter {it}")

        # relaxation + residual on the concatenated export vector
        if it == 1:
            for n, ifd in new_exports.items():
                exports[n] = ifd; relaxed_prev[n] = _stack(ifd); raw_prev[n] = _stack(ifd)
            history.append(float("nan")); continue

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
        if res < tol:
            return CouplingResult(True, it, res, {n: e.to_dict() for n, e in exports.items()},
                                  history, warnings=warnings)

    return CouplingResult(False, max_iter, history[-1] if history else float("nan"),
                          {n: e.to_dict() for n, e in exports.items()}, history,
                          error=f"did not converge to tol={tol} in {max_iter} iters "
                                f"(last residual {history[-1]:.2e}) — result is NOT trustworthy",
                          warnings=warnings)
