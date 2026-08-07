"""Shared machinery for the TWO-WAY thermo-structural (TSI) coupling fixtures.

WHAT MAKES A TSI FIXTURE DIFFERENT FROM THE HEAT ONES
-----------------------------------------------------
The existing coupling fixtures are DOMAIN DECOMPOSITION: two subdomains, one
scalar field, a shared interface, Dirichlet on one side and Neumann on the
other. TSI is a FIELD coupling — the two participants occupy the SAME body and
exchange volume fields, temperature one way and volumetric strain the other.
Nothing about the interface machinery applies: there is no flux to balance
across, no normal to get the sign of, and `couple`'s conservation checks report
themselves as not-run rather than passing.

So the verification has to come from somewhere else, and the whole point of
these fixtures is that it does:

  1. CONVERGENCE, with a relaxation that is actually needed. At the exaggerated
     coupling strength the un-relaxed Jacobi iteration DIVERGES (the composite
     map's eigenvalues are +-i*sqrt(delta) and delta > 1), so a converged run is
     not free.
  2. AGREEMENT WITH A MONOLITHIC RE-SOLVE of the same problem — one mesh, one
     assembly, no exchange, no iteration (see tsi_monolithic.py).
  3. BOTH DIRECTIONS ACTIVE, and this is the one that matters. A "two-way"
     coupling whose reverse direction changes nothing is one-way with extra
     steps. Two independent demonstrations, and the second is the real one:
       (a) suppressing mechanical -> thermal MOVES the answer, by a margin
           reported next to the tolerance the agreement check uses;
       (b) the suppressed run then matches the ONE-WAY monolithic reference,
           which was computed independently. So the reverse direction does not
           merely perturb the answer, it moves it BY THE RIGHT AMOUNT — a
           coupling that exchanged the wrong quantity would also move when
           switched off, and would land somewhere else.
  4. A MUTATION CONTROL, in each fixture.json, that perturbs the physics and
     makes the checks fail.

HOW THE TWO DIRECTIONS ARE SWITCHED OFF, and why it is done through the tool.
`imports_from` is enough for both, so no participant script has to be edited to
run the controls:
  * thermal with `imports_from=[]` falls back to EVOL_INIT, which is set to the
    volumetric strain of the initial state, so the (e - e_old) source term is
    identically zero. That is EXACTLY the one-way problem.
  * mech with `imports_from=[]` falls back to THETA_INIT, a uniform temperature,
    so the structure no longer sees the thermal field.
The thermal participant ALSO carries a `COUPLING` switch, and the fixtures check
that the two routes agree — a cross-check on both.

NO MEASURED NUMBER FROM THESE RUNS IS WRITTEN INTO ANY SERVED TEXT, and no
fixture pins one: every threshold here is a tolerance the physics must beat, and
the errors are PRINTED so each run reports its own numbers.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import couplinglib as L                                          # noqa: E402
import tsi_monolithic as M                                       # noqa: E402

Absent = L.Absent
TsiProblem = M.TsiProblem
STEEL, STRONG = M.STEEL, M.STRONG

# Which mesh each side uses. Deliberately different, in both directions, so
# every exchange goes through a real non-matching mesh-to-mesh map.
MESH_T = (40, 10)
MESH_M = (32, 8)
MESH_MONO = (80, 20)


def shipped_tsi(kind: str, backend: str) -> Path:
    p = L.PARTICIPANT_DIR / f"participant_tsi_{kind}_{backend}.py"
    if not p.is_file():
        raise Absent(f"no shipped TSI {kind} participant for {backend!r}")
    return p


def thermal_edits(p: TsiProblem, mesh: tuple[int, int], partner: str,
                  coupling: float = 1.0) -> dict:
    return {"PARTNER": json.dumps(partner),
            "X0, X1": f"0.0, {p.lx!r}", "Y0, Y1": f"0.0, {p.ly!r}",
            "NX, NY": f"{mesh[0]}, {mesh[1]}",
            "K_COND": repr(p.k_cond), "RHO_C": repr(p.rho_c), "DT": repr(p.dt),
            "T_REF": repr(p.t_ref), "T_OLD": repr(p.t_old),
            "T_HOT": repr(p.t_hot), "T_HOT_DY": repr(p.t_hot_dy),
            "T_COLD": repr(p.t_cold), "BETA": repr(p.beta),
            "COUPLING": repr(float(coupling)),
            "EVOL_OLD": repr(p.evol_old), "EVOL_INIT": repr(p.evol_old)}


def mech_edits(p: TsiProblem, mesh: tuple[int, int], partner: str,
               alpha_mech: float | None = None) -> dict:
    beta = p.beta if alpha_mech is None else (3 * p.lam + 2 * p.mu) * alpha_mech
    return {"PARTNER": json.dumps(partner),
            "X0, X1": f"0.0, {p.lx!r}", "Y0, Y1": f"0.0, {p.ly!r}",
            "NX, NY": f"{mesh[0]}, {mesh[1]}",
            "E_MOD": repr(p.e_mod), "NU": repr(p.nu), "BETA": repr(beta),
            "THETA_INIT": repr(p.t_old - p.t_ref)}


def stage_tsi(root: Path, name: str, backend: str, kind: str,
              edits: dict) -> dict:
    """Write one configured TSI participant into its own work_dir and return the
    spec `couple` takes. Mirrors couplinglib.stage, for the TSI script names."""
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    full = dict(edits)
    if backend in L.BINARY_EDITS:
        full.update(L.BINARY_EDITS[backend]())
    script = wd / f"participant_{name}.py"
    script.write_text(L.edit(shipped_tsi(kind, backend).read_text(), full))
    interp = L.interpreter(backend)
    print(f"participant[{name}]={backend}/{kind} interpreter={interp}")
    return {"name": name, "command": [interp, script.name],
            "work_dir": str(wd), "timeout": 1800}


def tsi_identity(p: TsiProblem):
    """The reference solver's own self-check (see tsi_monolithic)."""
    return M.check_effective_capacity_identity(p, *MESH_MONO)


# ── one two-way TSI run, checked against the monolithic re-solve ───────────

def theta_opt(p: TsiProblem) -> float:
    """The relaxation the served theta rule gives for this coupling.

    The composite fixed-point map here is block-antidiagonal with product
    -delta, so its Jacobian's eigenvalues are +-i*sqrt(delta) and the relaxed
    iteration's amplification is sqrt((1-th)^2 + delta*th^2) — the SAME
    expression the coupling knowledge writes for a Dirichlet-Neumann split with
    delta in place of the conductance ratio rho. Its minimiser is 1/(1+delta).
    At delta > 1 an un-relaxed iteration (th=1) has amplification sqrt(delta) > 1
    and diverges, which is why these runs need relaxation at all.
    """
    return 1.0 / (1.0 + p.delta)


def amplification(p: TsiProblem, theta: float) -> float:
    return ((1.0 - theta) ** 2 + p.delta * theta ** 2) ** 0.5


def run_tsi(tag: str, backend_thermal: str, backend_mech: str,
            p: TsiProblem | None = None, mesh_t=MESH_T, mesh_m=MESH_M,
            theta: float | None = None, accelerator: str = "constant",
            max_iter: int = 300, tol: float = 1e-10,
            coupling: float = 1.0, thermal_reads: bool = True,
            mech_reads: bool = True, alpha_mech: float | None = None,
            with_tool_monolithic: bool = True, quiet: bool = False) -> dict:
    """Stage, couple, and return {result, theta_field, evol_field, coords}.

    Does NOT assert: `assert_tsi` does. Split so the direction controls can run
    the same coupling and only compare.
    """
    p = p or STRONG
    if theta is None:
        theta = theta_opt(p)
    root = L.workroot(tag)
    specs = [stage_tsi(root, "thermal", backend_thermal, "thermal",
                       thermal_edits(p, mesh_t, "mech", coupling)),
             stage_tsi(root, "mech", backend_mech, "mech",
                       mech_edits(p, mesh_m, "thermal", alpha_mech))]
    specs[0]["imports_from"] = ["mech"] if thermal_reads else []
    specs[1]["imports_from"] = ["thermal"] if mech_reads else []
    args = {"participants": json.dumps(specs), "max_iter": max_iter, "tol": tol,
            "accelerator": accelerator, "theta": theta, "critic_approved": True}
    if with_tool_monolithic:
        mono_wd = root / "monolithic"
        cmd = M.write_reference(mono_wd, p, *MESH_MONO,
                                coupling=(coupling if thermal_reads else 0.0))
        args["monolithic"] = json.dumps({"command": cmd,
                                         "work_dir": str(mono_wd),
                                         "timeout": 1800})
    raw = L.call_tool("couple", args)
    try:
        res = json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"couple returned non-JSON: {raw[:400]}")
    out = {"result": res, "theta_used": theta, "problem": p}
    ex = res.get("exports") or {}
    if "thermal" in ex:
        out["theta_field"] = np.asarray(ex["thermal"]["values"], float)
        out["theta_coords"] = np.asarray(ex["thermal"]["coordinates"], float)
    if "mech" in ex:
        out["evol_field"] = np.asarray(ex["mech"]["values"], float)
        out["evol_coords"] = np.asarray(ex["mech"]["coordinates"], float)
    if not quiet:
        print(f"{tag}_converged={bool(res.get('converged'))}")
        print(f"{tag}_iterations={res.get('iterations')}")
        print(f"{tag}_residual={float(res.get('residual', float('nan'))):.3e}")
    return out


def _interp(coords_src, vals_src, coords_dst):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    src = np.asarray(coords_src, float)[:, :2]
    out = LinearNDInterpolator(src, np.asarray(vals_src, float))(
        np.asarray(coords_dst, float)[:, :2])
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = NearestNDInterpolator(src, np.asarray(vals_src, float))(
            np.asarray(coords_dst, float)[:, :2][bad])
    return out


_MONO_CACHE: dict = {}


def monolithic(p: TsiProblem, coupling: float = 1.0, nx=None, ny=None) -> dict:
    key = (p, coupling, nx, ny)
    if key not in _MONO_CACHE:
        _MONO_CACHE[key] = M.solve_monolithic(
            p, nx or MESH_MONO[0], ny or MESH_MONO[1], coupling=coupling)
    return _MONO_CACHE[key]


def compare_to_monolithic(tag: str, run: dict, coupling: float = 1.0,
                          rtol: float = 5e-3) -> dict:
    """Compare BOTH exchanged fields against the un-split solve.

    `couple`'s own `monolithic=` check covers only the field the reference
    declares (temperature_change); the structural participant exports a
    different quantity and the tool says so rather than comparing. This covers
    both, using `core.quality_checks.check_monolithic_consistency` — the same
    detector an agent gets — so the fixture proves the detector too.
    """
    from core.quality_checks import check_monolithic_consistency
    p = run["problem"]
    ref = monolithic(p, coupling)
    errs = {}
    w: list[str] = []
    for name, key in (("theta", "theta"), ("evol", "evol")):
        got = run.get(f"{name}_field")
        if got is None:
            L.check(False, f"{tag}_{name}_missing", "participant exported nothing")
            continue
        at = _interp(ref["coordinates"], ref[key], run[f"{name}_coords"])
        denom = float(np.linalg.norm(at)) or 1e-30
        rel = float(np.linalg.norm(got - at)) / denom
        errs[name] = rel
        print(f"{tag}_{name}_vs_monolithic_relL2={rel:.3e}")
        L.check(rel <= rtol, f"{tag}_{name}_disagrees_with_monolithic",
                f"relative L2 {rel:.3e} > {rtol:.1e}")
        w += check_monolithic_consistency(float(np.mean(got)), float(np.mean(at)),
                                          rtol=rtol, qoi=f"{tag}_{name}_mean")
    L.check(not w, f"{tag}_monolithic_detector_complained", "; ".join(w)[:300])
    print(f"{tag}_matches_monolithic={bool(not w and all(v <= rtol for v in errs.values()))}")
    return errs


def assert_run_clean(tag: str, run: dict, expect_one_way: bool = False) -> bool:
    """Convergence, an empty validation block, and — the discriminator for a
    coupling that only looks alive — that both participants responded to their
    imports and that their answers actually DEPEND on them."""
    res = run["result"]
    if not L.check(bool(res.get("converged")), f"{tag}_did_not_converge",
                   str(res.get("error"))[:300]):
        return False
    ok = L.check(not res.get("validation"), f"{tag}_validation_not_empty",
                 "; ".join(res.get("validation") or [])[:400])
    print(f"{tag}_validation_empty={bool(ok)}")
    resp = res.get("responsiveness") or {}
    sens = res.get("interface_sensitivity") or {}
    want = ["mech"] if expect_one_way else ["thermal", "mech"]
    for n in want:
        ok &= L.check(resp.get(n) == "responsive", f"{tag}_{n}_unresponsive",
                      f"responsiveness={resp.get(n)!r}")
        s = (sens.get(n) or {}).get("S")
        print(f"{tag}_{n}_sensitivity_S={s}")
        ok &= L.check(s is not None and s > 1e-6, f"{tag}_{n}_insensitive",
                      f"finite-difference interface sensitivity S={s!r}: this "
                      f"participant's answer does not depend on what it is handed")
    return bool(ok)


# ── the check the whole thing exists for ──────────────────────────────────

def reverse_direction_is_active(tag: str, two_way: dict, one_way: dict,
                                agreement: float, margin: float = 50.0) -> bool:
    """Does switching the mechanical -> thermal direction off change the answer,
    and by ENOUGH to be a signal rather than the agreement tolerance?

    `agreement` is how far the two-way coupled run sat from its monolithic
    reference. The effect has to be much larger than that, or "the answer moved"
    is indistinguishable from noise in the comparison. `margin` is how much
    larger; it is reported either way.
    """
    a, b = two_way["theta_field"], one_way["theta_field"]
    if a.shape != b.shape:
        return L.check(False, f"{tag}_shape_mismatch",
                       f"{a.shape} vs {b.shape}")
    scale = float(np.max(np.abs(a))) or 1.0
    eff = float(np.max(np.abs(a - b))) / scale
    print(f"{tag}_reverse_direction_effect_rel={eff:.3e}")
    print(f"{tag}_agreement_with_monolithic_rel={agreement:.3e}")
    print(f"{tag}_reverse_over_agreement={eff / max(agreement, 1e-30):.1f}")
    ok = L.check(eff > margin * agreement, f"{tag}_reverse_direction_is_inert",
                 f"suppressing mechanical->thermal moved the temperature field "
                 f"by only {eff:.3e} relative, against an agreement tolerance of "
                 f"{agreement:.3e}. This coupling is EFFECTIVELY ONE-WAY and must "
                 f"be reported as such.")
    print(f"{tag}_both_directions_active={bool(ok)}")
    return bool(ok)
