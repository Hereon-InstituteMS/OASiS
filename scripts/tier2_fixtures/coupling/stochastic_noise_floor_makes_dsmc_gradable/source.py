"""A Monte-Carlo participant has a residual FLOOR, and until the driver could
measure it, a physically correct DSMC coupling was reported as a FAILURE.

THE CLAIM UNDER TEST. The served SPARTA payload says it plainly: "its sampling
noise does NOT shrink as the coupling iterates, so the driver's relative
residual has a FLOOR at the noise level and `tol` below that floor can never be
met — the run ends as 'did not converge', which is honest, not a bug." The sides
table says the same in its one starred capability row: SPARTA's Dirichlet role
"ran end to end and the physics agreed, but `couple` reported FAILURE". That is
the flagship's hardest instance and it was ungradable: a coupling whose verdict
is always FAILURE cannot be scored on convergence at all.

WHAT THIS FIXTURE PROVES. `couple` now takes `noise_replicates`, and this runs
the SAME real DSMC-to-solid coupling twice to show what it changes:

  ARM PLAIN        tol far below the sampling noise, no noise handling. The
                   physics settles and then the residual sits on its floor for
                   the whole iteration budget. Verdict: FAILURE.
  ARM NOISE-AWARE  identical, plus `noise_replicates`. The driver runs each
                   participant several times on the SAME imports, evaluates its
                   OWN residual expression between independent replicates, and
                   judges convergence against max(tol, that floor) over a block
                   mean. Verdict: CONVERGED, with the floor reported so a grader
                   cannot apply a tolerance tighter than the sampler allows.

Both arms have the same physics, and the fixture checks that the physics is
RIGHT in the noise-aware arm rather than only that the verdict flipped — against
a fixed point computed INDEPENDENTLY here, by running SPARTA standalone at
several uniform wall temperatures and solving the scalar balance the shell
imposes. No part of that reference is served to any agent.

  ARM DETERMINISTIC  the ordinary two-code conduction coupling, with the same
                   `noise_replicates` switched on. Its floor measures EXACTLY
                   zero, nothing about the run changes, and the closed form is
                   still met. The branch has to be inert where there is no
                   noise, or it is a way of passing failures.

WHAT IT DOES NOT PROVE. Both sides here discretise the interface identically
(they read the same surface file), so this fixture says nothing about
non-matching interface meshes — the pair fixtures cover that. The shell passes
the imported flux straight through to its outer boundary, so interface
conservation holds by construction and is not evidence of anything here. And the
shell is ISOTHERMAL in-plane, which is what makes the independent scalar
reference exact; a shell reacting element by element would sit at a fixed point
no uniform-wall probe can predict, and the fixture would then be grading against
a number it could not defend.

A REAL LIMIT OF THE METHOD, stated because the fixture cannot rule it out: a
physics drift whose per-iteration change is SMALLER than the sampling noise, but
which accumulates over many iterations, is invisible to a residual floor. The
floor answers "can this sampler still see the iteration moving", not "has the
iteration finished". Block-averaging widens the window; it does not close that
gap.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

# Which cells of the served sides table this fixture establishes. Read by
# sides_table_backed_by_runs. SPARTA is the Dirichlet side — it imports a wall
# temperature and exports the flux the gas deposits — and it CONVERGES here,
# which is the part the table could not claim before.
SIDES_COVERED = [("sparta", "dirichlet"), ("skfem", "dirichlet"),
                 ("skfem", "neumann")]

# ── the knobs the mutations move ────────────────────────────────────────────
NOISE_REPLICATES = 5        # >=2 turns the driver's stochastic branch on.
                            # FIVE and not the bare minimum: the floor is
                            # itself an estimate, and three replicates give
                            # only three samples. The driver says so in its
                            # notes below six.
SEED_MODE = "vary"          # "vary" = a fresh DSMC seed per invocation
TOL = 1e-8                  # deliberately far below any Monte-Carlo floor
MAX_ITER = 18

# The thermal shell the gas sees, and the reference it defines.
T_OUTER = 300.0             # temperature on the shell's outer face [K]
C_SHELL = 8.0               # lumped through-thickness conductance [W/(m^2 K)]
T_START = 800.0             # iteration-1 wall temperature [K]
Q_START = 4.0e3             # the shell's iteration-1 flux fallback
T_REFERENCE_RTOL = 0.03     # how far the coupled wall T may sit from the
                            # independently computed fixed point. Set by the
                            # measured flux scatter propagated through 1/C, with
                            # room to spare; every pathology this could hide is
                            # far larger.

NRUN, NAVE = 2000, 1000     # DSMC steps / sampling window
PROBE_T = (400.0, 700.0, 1000.0)     # wall temperatures for the reference curve
PROBE_REPEATS = 3           # the reference is itself a Monte-Carlo estimate, so
                            # each probe is averaged; one draw per temperature
                            # would put the sampler's noise into the slope and
                            # then into the number the coupling is graded on.

SURF, SPECIES, VSS = "circle.surf", "ar.species", "ar.vss"


# ── the solid side: a lumped thermal shell around the same surface ──────────
#
# Written by the fixture rather than taken from data/coupling_participants,
# because no shipped participant solves an annular shell. It is a genuine
# Neumann-side participant: it imports a FLUX and exports the resulting VALUE.
SHELL = '''\
import json, sys
from pathlib import Path

PARTNER = "gas"
SURF    = "circle.surf"
T_OUTER = {T_OUTER}
C_SHELL = {C_SHELL}
Q_INIT  = {Q_START}


def centroids():
    """The same interface elements the gas side uses, from the same file."""
    txt = Path(SURF).read_text().splitlines()
    pts, lines, sec = {{}}, [], None
    for ln in txt:
        s = ln.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low == "points":
            sec = "p"; continue
        if low == "lines":
            sec = "l"; continue
        if low.endswith("points") or low.endswith("lines"):
            continue
        f = s.split()
        if sec == "p" and len(f) >= 3:
            pts[int(f[0])] = (float(f[1]), float(f[2]))
        elif sec == "l" and len(f) >= 3:
            lines.append((int(f[0]), int(f[-2]), int(f[-1])))
    lines.sort()
    return [[0.5 * (pts[a][0] + pts[b][0]), 0.5 * (pts[a][1] + pts[b][1])]
            for (_, a, b) in lines]


co = centroids()
n = len(co)
q = [Q_INIT] * n
p = Path("imports.json")
if p.is_file():
    try:
        d = (json.loads(p.read_text() or "{{}}") or {{}}).get(PARTNER)
    except json.JSONDecodeError:
        d = None
    if d and d.get("normal_fluxes") and len(d["normal_fluxes"]) == n:
        # Apply the partner's number UNCHANGED: this is the Neumann side.
        q = [float(v) for v in d["normal_fluxes"]]

# Steady lumped shell, ISOTHERMAL in-plane: a thin shell whose tangential
# conductivity is high compared with its through-thickness one holds ONE wall
# temperature, set by the TOTAL heat it must pass to its outer face. That is
# what makes the fixture's independent reference exact rather than approximate:
# the reference solves the same scalar balance from standalone runs at a
# UNIFORM wall temperature, and here the wall really is uniform. A shell that
# reacted element by element would sit at a different fixed point than any
# uniform-wall probe, by an amount nothing in the fixture could bound.
qbar = sum(q) / len(q)
t = [T_OUTER + qbar / C_SHELL] * n

Path("exports.json").write_text(json.dumps({{
    "field_name": "wall_temperature",
    "n_points": n,
    "coordinates": co,
    "values": t,
    # Exported with respect to THIS side's outward normal, which points the
    # other way, so the two sides carry opposite signs.
    "normal_fluxes": [-v for v in q],
}}, indent=2))
print(f"shell n={{n}} q=[{{min(q):.4e}},{{max(q):.4e}}] "
      f"T=[{{min(t):.2f}},{{max(t):.2f}}]")
'''


def sparta_binary() -> str:
    from backends.sparta.backend import _find_sparta_binary
    b = _find_sparta_binary()
    if not b:
        raise L.Absent("no SPARTA binary resolved from the registry")
    return b


def sparta_data(binary: str) -> dict:
    """Locate the surface / species / vss files inside the SPARTA distribution
    the resolved binary belongs to. Never a hard-coded host path."""
    from backends.sparta.backend import _sparta_data_dirs
    found: dict = {}
    for root in _sparta_data_dirs(binary):
        for name in (SURF, SPECIES, VSS):
            if name in found:
                continue
            hits = sorted(Path(root).rglob(name))
            if hits:
                found[name] = hits[0]
    missing = [n for n in (SURF, SPECIES, VSS) if n not in found]
    if missing:
        raise L.Absent(f"SPARTA data files not found in the distribution: "
                       f"{missing}")
    return found


def stage_gas(root: Path, name: str, binary: str, data: dict,
              t_init: float, partner: str) -> dict:
    """One configured SPARTA participant in its own work_dir."""
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    for src in data.values():
        shutil.copy(src, wd / src.name)
    text = L.edit(L.shipped("sparta").read_text(), {
        "PARTNER": json.dumps(partner),
        "SPARTA": json.dumps(binary),
        "SURF_FILE": json.dumps(SURF),
        "SPECIES": json.dumps(SPECIES),
        "VSS": json.dumps(VSS),
        "T_INIT": f"{t_init}",
        "NRUN": f"{NRUN}",
        "NAVE": f"{NAVE}",
        "SEED_MODE": json.dumps(SEED_MODE),
    })
    (wd / f"participant_{name}.py").write_text(text)
    return {"name": name, "command": [sys.executable, f"participant_{name}.py"],
            "work_dir": str(wd), "timeout": 600}


def stage_shell(root: Path, name: str, data: dict) -> dict:
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(data[SURF], wd / SURF)
    (wd / f"participant_{name}.py").write_text(SHELL.format(
        T_OUTER=T_OUTER, C_SHELL=C_SHELL, Q_START=Q_START))
    return {"name": name, "command": [sys.executable, f"participant_{name}.py"],
            "work_dir": str(wd), "timeout": 300}


def mean_wall_T(res: dict) -> float:
    v = res["exports"]["shell"]["values"]
    return sum(float(x) for x in v) / len(v)


# ── the independent reference: SPARTA alone, then the scalar balance ────────

def independent_fixed_point(binary: str, data: dict) -> tuple[float, float]:
    """Run SPARTA STANDALONE at several uniform wall temperatures, fit the mean
    flux it returns, and solve the shell's own balance for the wall temperature.

    This never touches `couple`, the driver or any relaxation, so agreement
    between it and the coupled answer is evidence about the coupling rather than
    a restatement of it. Returns (T_fixed_point, |dq/dT|)."""
    import subprocess
    root = L.workroot("dsmc_ref")
    qs = []
    for t in PROBE_T:
        spec = stage_gas(root, f"probe_{int(t)}", binary, data, t, "nobody")
        wd = Path(spec["work_dir"])
        draws = []
        for _ in range(PROBE_REPEATS):
            (wd / "imports.json").unlink(missing_ok=True)
            r = subprocess.run(spec["command"], cwd=str(wd),
                               capture_output=True, text=True,
                               timeout=spec["timeout"])
            ep = wd / "exports.json"
            if not ep.is_file():
                raise L.Absent(f"standalone SPARTA probe at T={t} produced no "
                               f"exports.json (rc={r.returncode}): "
                               f"{r.stderr[-300:]}")
            d = json.loads(ep.read_text())
            draws.append(sum(float(x) for x in d["normal_fluxes"])
                         / len(d["normal_fluxes"]))
        q = sum(draws) / len(draws)
        qs.append(q)
        print(f"reference_probe_T={t:.0f}_qbar={q:.6e}")
    # Least-squares line q(T) = a + b T over the probes.
    n = len(PROBE_T)
    mt = sum(PROBE_T) / n
    mq = sum(qs) / n
    sxx = sum((t - mt) ** 2 for t in PROBE_T)
    b = sum((t - mt) * (q - mq) for t, q in zip(PROBE_T, qs)) / sxx
    a = mq - b * mt
    # Solve T = T_OUTER + (a + bT)/C  ->  T (1 - b/C) = T_OUTER + a/C
    t_fp = (T_OUTER + a / C_SHELL) / (1.0 - b / C_SHELL)
    return t_fp, abs(b)


def body() -> None:
    L.require_available("sparta", "skfem")
    binary = sparta_binary()
    data = sparta_data(binary)
    print(f"sparta_binary_resolved={bool(binary)}")
    print(f"sparta_data_staged={len(data)}")
    print(f"seed_mode={SEED_MODE}")

    # ── the reference, before any coupling ────────────────────────────────
    t_ref, slope = independent_fixed_point(binary, data)
    print(f"independent_fixed_point_T={t_ref:.4f}")
    L.check(math.isfinite(t_ref) and 200.0 < t_ref < 3000.0,
            "reference_fixed_point_is_not_physical",
            f"the standalone probes gave T={t_ref}, so the reference this "
            f"fixture grades against is meaningless")
    L.check(slope > 0.0, "reference_flux_does_not_respond_to_wall_temperature",
            f"|dq/dT| = {slope:.3e}: if the gas flux does not depend on the "
            f"wall temperature there is no coupling to measure")

    def run(tag: str, replicates: int) -> dict:
        root = L.workroot(tag)
        gas = stage_gas(root, "gas", binary, data, T_START, "shell")
        shell = stage_shell(root, "shell", data)
        gas["imports_from"] = ["shell"]
        shell["imports_from"] = ["gas"]
        res = L.couple([gas, shell], max_iter=MAX_ITER, tol=TOL,
                       accelerator="constant", theta=0.5,
                       noise_replicates=replicates)
        print(f"{tag}_converged={bool(res.get('converged'))}")
        print(f"{tag}_iterations={res.get('iterations')}")
        print(f"{tag}_residual={float(res.get('residual', float('nan'))):.3e}")
        return res

    # ── ARM PLAIN: the failure the whole feature exists to remove ─────────
    plain = run("arm_plain", 0)
    L.check(not plain.get("converged"), "plain_arm_converged_anyway",
            "a DSMC coupling with tol below the sampling noise must NOT reach "
            "tol — if it did, either the noise vanished or the seed is fixed, "
            "and this fixture is not demonstrating anything")
    print(f"arm_plain_reports_failure={not bool(plain.get('converged'))}")
    print(f"arm_plain_residual_stalled_above_tol="
          f"{bool(float(plain.get('residual', 0.0)) > TOL)}")

    # ── ARM NOISE-AWARE: same run, floor measured ─────────────────────────
    aware = run("arm_noise_aware", NOISE_REPLICATES)
    print(f"arm_noise_aware_converged_at_floor="
          f"{bool(aware.get('converged')) and bool(aware.get('stopped_at_noise_floor'))}")
    L.check(bool(aware.get("converged")), "noise_aware_arm_did_not_converge",
            str(aware.get("error"))[:300])
    L.check(bool(aware.get("stopped_at_noise_floor")),
            "noise_aware_arm_did_not_use_the_floor",
            "it converged, but not because of the floor — then the feature is "
            "not what produced the verdict")
    floor = float(aware.get("noise_floor") or 0.0)
    print(f"measured_noise_floor={floor:.4e}")
    L.check(floor > TOL, "measured_floor_is_below_tol",
            f"the floor came out {floor:.3e}, at or under tol={TOL:.1e}, so "
            f"nothing was actually blocked by sampling noise")
    print(f"floor_exceeds_requested_tol={bool(floor > TOL)}")
    # WHERE the floor is announced matters as much as THAT it is. `validation`
    # is the findings channel and `couple` treats any entry in it as making the
    # coupling untrustworthy — so a correct stochastic run whose notice sat
    # there came back NOT VERIFIED, which is the verdict this feature exists to
    # stop being unavoidable. The notice belongs in the coverage channel, which
    # is always printed inside `verification` and never flips it. Both halves
    # are checked: it must be SAID, and `validation` must be EMPTY.
    said = " | ".join(str(w) for w in aware.get("checks_not_run", []))
    in_verdict = "NOISE FLOOR" in str(aware.get("verification", "")).upper()
    print(f"result_names_the_floor_to_the_agent="
          f"{bool('NOISE FLOOR' in said.upper() and in_verdict)}")
    L.check("NOISE FLOOR" in said.upper(), "floor_not_reported_to_the_agent",
            "a verdict reached at the noise floor that does not SAY so is a "
            "softened failure, which is worse than the honest one it replaced")
    L.check(in_verdict, "floor_not_carried_into_the_verdict",
            "the coverage channel is appended to `verification`; if the floor "
            "is not there, an agent reading only the verdict never sees it")
    print(f"noise_aware_validation_stayed_empty={not bool(aware.get('validation'))}")
    L.check(not aware.get("validation"), "noise_aware_validation_not_empty",
            "a correct coupling judged at its measured floor must leave the "
            "findings block EMPTY — a non-empty one stamps NOT VERIFIED: "
            + " | ".join(str(w) for w in aware.get("validation", []))[:300])

    # ── and the physics is right, against the independent fixed point ─────
    t_got = mean_wall_T(aware)
    rel = abs(t_got - t_ref) / abs(t_ref)
    print(f"coupled_wall_T={t_got:.4f}")
    print(f"coupled_vs_independent_fixed_point_rel={rel:.4e}")
    L.check(rel <= T_REFERENCE_RTOL, "coupled_answer_misses_the_reference",
            f"the coupled wall temperature {t_got:.3f} K is {rel:.2%} from the "
            f"independently computed fixed point {t_ref:.3f} K")
    print(f"coupled_matches_independent_reference={bool(rel <= T_REFERENCE_RTOL)}")

    # ── ARM DETERMINISTIC: the branch must be inert without noise ─────────
    p = L.DEFAULT
    root = L.workroot("det")
    specs = [
        L.stage(root, "left", "skfem",
                L.heat_edits(p, "left", "dirichlet", "right", (16, 16))),
        L.stage(root, "right", "skfem",
                L.heat_edits(p, "right", "neumann", "left", (14, 12))),
    ]
    specs[0]["imports_from"] = ["right"]
    specs[1]["imports_from"] = ["left"]
    det = L.couple(specs, max_iter=200, tol=1e-8, accelerator="constant",
                   theta=p.theta_opt("left"),
                   noise_replicates=NOISE_REPLICATES)
    det_floor = det.get("noise_floor")
    print(f"deterministic_floor_is_exactly_zero={det_floor == 0.0}")
    L.check(det_floor == 0.0, "deterministic_participants_measured_noise",
            f"two deterministic FEM participants must return bit-identical "
            f"exports across replicate runs; got a floor of {det_floor!r}")
    L.check(bool(det.get("converged")), "deterministic_arm_did_not_converge",
            str(det.get("error"))[:300])
    L.check(not det.get("stopped_at_noise_floor"),
            "deterministic_arm_stopped_at_a_floor",
            "a zero floor must not change the criterion — max(tol, 0) is tol")
    # The measurement's provenance must NOT land in `validation`: an agent is
    # told an empty validation block is what a correct coupling looks like, so a
    # routine note there would train it to ignore the block.
    dnotes = " | ".join(str(x) for x in det.get("noise_notes", []))
    print(f"deterministic_validation_stayed_empty={not bool(det['validation'])}")
    print(f"fixed_seed_caveat_is_reported_out_of_band="
          f"{bool('SEED IS FIXED' in dnotes)}")
    L.check("SEED IS FIXED" in dnotes, "zero_floor_caveat_missing",
            "a floor of exactly zero on a Monte-Carlo participant means a fixed "
            "seed, and the result must say so somewhere the agent reads")
    print(f"deterministic_arm_unaffected="
          f"{bool(det.get('converged')) and not det.get('stopped_at_noise_floor')}")
    L._check_interface_physics("deterministic", det, p.t_iface, p.q, 1e-3, 5e-3)
    L.assert_against_monolithic(
        "deterministic",
        0.5 * sum(L.span(det["exports"]["left"]["values"])),
        0.5 * sum(L.span(det["exports"]["left"]["normal_fluxes"])), p)

    print("arms_compared=3")


L.main(body)
