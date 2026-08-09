"""4C's NATIVE Thermo_Structure_Interaction, as an independent reference.

FIXTURE-SIDE ONLY, like tsi_monolithic.py: it produces the answer the coupled
runs are graded against and is never reachable through any tool.

WHY A NATIVE REFERENCE AS WELL AS A MONOLITHIC ONE. The monolithic reference in
tsi_monolithic.py shares its author, its element technology and its
understanding of the physics with the participants. It cannot catch a mistake
that is in the MODEL rather than in the partitioning. 4C's TSI can: it is a
production code, written by other people, with its own element formulation
(trilinear hexahedra rather than triangles), its own time integrators and its
own monolithic Newton solve of the coupled system.

WHAT MATCHES EXACTLY AND WHAT DOES NOT, stated up front because a reference
whose modelling difference is discovered afterwards is not a reference:

  * FORWARD (thermal -> mechanical). 4C's `MAT_Struct_ThermoStVenantK` uses the
    stress-temperature modulus -(3 lambda + 2 mu) * THEXPANS with the reference
    temperature INITTEMP (4C source: Mat::ThermoStVenantKirchhoff::st_modulus).
    That is exactly beta and exactly T_ref here, so this direction is the same
    model in both codes and must agree to discretisation error.

  * REVERSE (mechanical -> thermal). 4C's thermal element assembles
    `- N^T . ctemp : (B_L . d') . N . T` (4C source: TemperImpl::linear_dissipation
    _coupled_term), i.e. + beta * T * d/dt tr(eps) with the CURRENT temperature
    T. The classical linear theory this project implements uses T_ref there
    instead. The two differ by the factor T/T_ref.

    That difference is NOT worked around, it is made negligible and then
    bounded: the 4C comparison runs the same problem with a temperature
    excursion of a fraction of a kelvin, so T/T_ref - 1 is ~1e-3 and the
    reverse term — which is itself ~10% of the answer — differs between the two
    models by ~1e-4 of that. Nothing else about the problem changes: every
    equation here is linear in the excursion, so the RELATIVE size of the
    reverse direction, the coupling parameter delta and the relaxation the
    iteration needs are all identical to the full-size problem. The bound is
    computed and printed by `linearisation_bound` rather than asserted.

EVERY NEWTON TOLERANCE HERE IS ABOVE ITS OWN ROUNDOFF FLOOR, and that is not
slack. 4C's default NORMCOMBI_RESFINC ("Coupl_And_Single") requires the coupled
residual, the coupled increment AND each field's own residual and increment to
be met at once, and several of those are ABSOLUTE norms of quantities that carry
an offset: the temperature increment norm cannot go below ~1e-12 when T is ~293,
and the coupled relative residual floors near 1e-12 on this mesh. Asking for
1e-14 and 1e-13 made a Newton that was fully converged at iteration 3 grind to
ITEMAX and abort with "Newton unconverged in 50 iterations" — an error that
looks like a physics failure and is a tolerance below the floor. The problem is
LINEAR, so the state at iteration 3 is exact to roundoff and every tolerance
below is orders of magnitude tighter than the discretisation.

QUASI-STATIC MECHANICS OUT OF A DYNAMIC INTEGRATOR. 4C's structural velocity is
what enters the coupling term, so the structure cannot be run as `Statics` — it
would have no velocity and the reverse direction would silently vanish (which is
exactly what 4C's own `tsi_oneway` + COUPVARIABLE=Displacement decks do on
purpose). It is run as OneStepTheta with THETA=1, which makes v_{n+1} exactly
(d_{n+1} - d_n)/dt — the same difference quotient the participants use — and
with a density of 1e-9 so the inertia it drags in is ~20 orders below the
elastic force.

THE MESH MUST BE COARSER THAN 1e-3 METRES PER ELEMENT, and nothing says so.
4C builds the structure <-> thermo node correspondence with a geometric octree
whose default tolerance is an ABSOLUTE 1e-3 (Coupling::Adapter::Coupling::
match_nodes). Below that spacing distinct nodes collapse into one match and the
run aborts with "Did not get 1:1 correspondence. masternodes.size()=324
(structure), coupling.size()=320 (thermo)" — a message about node COUNTS whose
cause is geometric SCALE, and which does not change when the mesh, the
conditions or the physics are corrected. The whole TSI problem is therefore
posed at metre scale; see TsiProblem.

EVERY KEY IN THE DECK IS FROM `4C -p`. A mis-cased key and an invented key
produce the same 'Failed to match specification' message, so the grammar dump is
the only way to tell them apart; scripts/audit_named_input_keys.py checks the
ones this file writes.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tsi_monolithic import TsiProblem

# A tiny temperature excursion: see the header. Everything is linear in it, so
# nothing about the coupling changes, and the one model difference between 4C
# and the linear theory shrinks with it.
#
# TWO PROBLEMS, AND THE REASON IS NOT COSMETIC.
#
#   NATIVE starts at t_old == t_ref. That is what makes 4C's own uniaxial
#   effective-capacity identity hold (see TsiProblem.unstrained_start), so it is
#   the problem on which 4C's reverse term is identified black-box and on which
#   4C is compared against the in-house monolithic solve. Neither of those runs
#   a partitioned coupling.
#
#   COUPLED offsets every temperature above t_ref. It has to: with t_old ==
#   t_ref the exchanged fields are ZERO over most of the body after one step
#   (the heat has not arrived), and `couple`'s per-block convergence check —
#   which is the worst ENTRY-WISE relative change, not a norm — then reports
#   blocks "still changing" at 1e-06 on a run whose global residual is 8e-13.
#   Measured, on exactly this problem. Those entries carry no information: a
#   relative change of 1e-06 on an entry that is 1e-08 of the largest is an
#   absolute change of 1e-14 of the field. The check is not wrong to be
#   conservative, and the answer is not to loosen it but to exchange fields
#   whose DYNAMIC RANGE is narrow, which an offset gives.
#
#   HOW BIG THE OFFSET HAS TO BE WAS MEASURED, and the first guess was too
#   small. At t_old = t_ref + 0.3 the reverse direction — which on an
#   unstrained start releases the pre-strain and therefore COOLS the body —
#   pulls the far field down to theta = 0.16 against a boundary value of 0.9,
#   a dynamic range of 5.6, and the worst block landed at 1.09e-11 against a
#   limit of 1.0e-11. Nine percent of margin is a coin flip, and tightening
#   `tol` does not help: the limit is 10*tol, so the ratio of block to limit is
#   unchanged by it. Only narrowing the range does.
FOURC_NATIVE = TsiProblem(alpha=1.2e-4, t_ref=293.0, t_old=293.0,
                          t_hot=293.3, t_hot_dy=0.0, t_cold=293.0)
#   The offset also cannot be made arbitrarily large: it raises T/T_ref - 1,
#   which is exactly the modelling difference against 4C's reverse term. At
#   theta ~ 2 the bound is 8e-03 and the difference it allows in the answer
#   (~25% of which is the reverse direction) reaches the 2e-03 the comparison
#   is held to. theta ~ 1 keeps the bound at 5e-03 and the range at ~2.6.
FOURC_COUPLED = TsiProblem(alpha=1.2e-4, t_ref=293.0, t_old=294.0,
                           t_hot=294.4, t_hot_dy=0.0, t_cold=294.0,
                           unstrained_start=True)


def linearisation_bound(p: TsiProblem) -> float:
    """How far 4C's reverse term (beta*T*d/dt tr eps) can sit from the linear
    theory's (beta*T_ref*d/dt tr eps), as a fraction of the reverse term."""
    hi = max(p.t_hot, p.t_cold, p.t_old) + max(p.t_hot_dy, 0.0)
    return abs(hi - p.t_ref) / p.t_ref


# ── every key in the deck, against 4C's own accepted grammar ───────────────
#
# A MIS-CASED KEY AND AN INVENTED KEY PRODUCE THE SAME MESSAGE. 4C answers both
# with "Failed to match specification in section '<SECTION>'", so the message
# cannot tell them apart and a deck that fails to parse gives no clue whether
# the key is wrong or the value is. `4C -p` dumps the grammar the binary
# actually accepts, and that is the only thing that can. This checks the deck
# this file generates against it, key by key, so an invented key is caught by
# name here instead of as an opaque parse failure later.
#
# Sections that carry MESH DATA rather than parameters are excluded by name:
# their content is node and element lines, not key/value pairs.
_MESH_SECTIONS = {"NODE COORDS", "STRUCTURE ELEMENTS", "DSURF-NODE TOPOLOGY",
                  "DVOL-NODE TOPOLOGY", "TITLE"}


def _grammar() -> dict:
    """`4C -p`, parsed. Cached on disk under TMPDIR — it is ~70k lines and the
    binary takes a moment, and every fixture that audits keys wants the same
    dump."""
    import json as _json
    import yaml
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "oasis_4c_grammar.json"
    if cache.is_file():
        return _json.loads(cache.read_text())
    env = dict(os.environ)
    for cand in ("/opt/4C-dependencies/lib",):
        if Path(cand).is_dir():
            env["LD_LIBRARY_PATH"] = cand + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run(["stdbuf", "-oL", _find_binary(), "-p"],
                       capture_output=True, text=True, timeout=900, env=env)
    text = r.stdout
    # An X11 warning on a headless box prepends bytes to the first line and
    # turns `metadata:` into a different key; drop anything before it.
    i = text.find("metadata:")
    if i > 0:
        text = text[i:]
    doc = yaml.safe_load(text)
    # `sections` is {"type": "all_of", "specs": [ {name, type: group, specs}, ...]}
    # and the big repeated sub-grammars (every condition's E/NUMDOF/ONOFF/VAL/
    # FUNCT block, the element spec inside a generated domain) are FACTORED OUT
    # into `$references` and pulled in with `{"$ref": "<id>"}`. Not following
    # those makes every condition section look as if it accepts no keys at all,
    # which is how a first version of this auditor reported thirty-three
    # perfectly valid keys as fabrications.
    refs = doc.get("$references") or {}
    entries = (doc.get("sections") or {}).get("specs") or []
    out = {"sections": {}}
    for sec in entries:
        if isinstance(sec, dict) and isinstance(sec.get("name"), str):
            # `type: group` sections carry `specs`; `type: list` sections
            # (MATERIALS, FUNCT<n>, every DESIGN ... CONDITIONS) carry a single
            # `spec` describing one entry. Reading only `specs` made every
            # list-valued section look as if it accepted no keys at all.
            body = sec.get("specs") if "specs" in sec else sec.get("spec")
            out["sections"][sec["name"]] = sorted(_names(body, refs=refs))
    if not out["sections"]:
        raise RuntimeError("`4C -p` produced no parseable section list")
    cache.write_text(_json.dumps(out))
    return out


def _names(node, acc=None, refs=None, seen=None) -> set:
    acc = set() if acc is None else acc
    refs = refs or {}
    seen = set() if seen is None else seen
    if isinstance(node, dict):
        r = node.get("$ref")
        if isinstance(r, (str, int)) and str(r) not in seen:
            seen.add(str(r))
            _names(refs.get(str(r)), acc, refs, seen)
        if isinstance(node.get("name"), str):
            acc.add(node["name"])
        for k, v in node.items():
            if k != "$ref":
                _names(v, acc, refs, seen)
    elif isinstance(node, list):
        for v in node:
            _names(v, acc, refs, seen)
    return acc


def audit_deck_keys(p: TsiProblem, nx: int = 4) -> list[str]:
    """Return the sections and keys this generator writes that 4C's grammar
    does not know. An empty list is the only acceptable result."""
    import yaml
    g = _grammar()["sections"]
    bad: list[str] = []
    d = yaml.safe_load(deck(p, nx))
    for sec, body in d.items():
        if sec in _MESH_SECTIONS:
            continue
        # 4C writes indexed sections as a pattern: FUNCT1/FUNCT2/... are all
        # `FUNCT<n>` in the dump, and SOLVER 1 is `SOLVER <n>`.
        key = re.sub(r"\d+$", "<n>", sec)
        key = key if key in g else sec
        if key not in g:
            bad.append(f"section {sec!r} is not in 4C's grammar")
            continue
        allowed = set(g[key])
        for key in _deck_keys(body):
            if key not in allowed:
                bad.append(f"{sec} -> {key!r} is not in 4C's grammar")
    return bad


def _deck_keys(body) -> set:
    """The parameter names a deck section sets, at any nesting depth."""
    out: set = set()
    if isinstance(body, dict):
        for k, v in body.items():
            out.add(str(k))
            out |= _deck_keys(v)
    elif isinstance(body, list):
        for v in body:
            out |= _deck_keys(v)
    return out


def _find_binary() -> str:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
    from backends.fourc.backend import _find_fourc_binary
    b = _find_fourc_binary()
    if not b:
        raise RuntimeError("no 4C binary resolved from the registry")
    return str(b)


def deck(p: TsiProblem, nx: int = 160, two_way: bool = True,
         capa_scale: float = 1.0, lz: float = 0.1) -> str:
    """The 4C input deck for one coupled step of the same problem.

    One element through the thickness in y and z with u_y = u_z = 0 imposed on
    every node is the finite-element realisation of uniaxial strain, which is
    what the participants' roller boundary conditions produce when the boundary
    temperature has no y-variation.
    """
    import yaml

    ly = p.ly
    nid = lambda i, j, k: 1 + i + (nx + 1) * (j + 2 * k)      # noqa: E731
    nodes, coords = [], []
    for k in range(2):
        for j in range(2):
            for i in range(nx + 1):
                x, y, z = p.lx * i / nx, ly * j, lz * k
                coords.append((nid(i, j, k), x, y, z))
    coords.sort()
    for n, x, y, z in coords:
        nodes.append(f"NODE {n} COORD {x:.16e} {y:.16e} {z:.16e}")
    elems = []
    for i in range(nx):
        c = [nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 1, 0), nid(i, 1, 0),
             nid(i, 0, 1), nid(i + 1, 0, 1), nid(i + 1, 1, 1), nid(i, 1, 1)]
        elems.append(f"{i + 1} SOLIDSCATRA HEX8 " + " ".join(map(str, c))
                     + " MAT 1 KINEM linear TYPE Undefined")

    left = [nid(0, j, k) for k in range(2) for j in range(2)]
    right = [nid(nx, j, k) for k in range(2) for j in range(2)]
    topo = ([f"NODE {n} DSURFACE 1" for n in left]        # u_x = 0
            + [f"NODE {n} DSURFACE 2" for n in left]      # T = t_hot
            + [f"NODE {n} DSURFACE 3" for n in right])    # T = t_cold
    vol = [f"NODE {n} DVOL 1" for n, _, _, _ in coords]

    d = {
        "TITLE": ["OASiS TSI cross-code reference: one implicit step of coupled "
                  "linear thermoelasticity, uniaxial strain."],
        "PROBLEM TYPE": {"PROBLEMTYPE": "Thermo_Structure_Interaction"},
        "IO": {"STRUCT_STRESS": "No", "STRUCT_STRAIN": "No"},
        "IO/RUNTIME VTK OUTPUT": {"INTERVAL_STEPS": 1},
        "IO/RUNTIME VTK OUTPUT/STRUCTURE": {"OUTPUT_STRUCTURE": True,
                                            "DISPLACEMENT": True},
        "THERMAL DYNAMIC/RUNTIME VTK OUTPUT": {"OUTPUT_THERMO": True,
                                               "TEMPERATURE": True},
        "STRUCTURAL DYNAMIC": {
            "DYNAMICTYPE": "OneStepTheta", "TIMESTEP": p.dt, "MAXTIME": p.dt,
            "NUMSTEP": 1, "TOLDISP": 1e-12, "TOLRES": 1e-6,
            "LINEAR_SOLVER": 1},
        "STRUCTURAL DYNAMIC/ONESTEPTHETA": {"THETA": 1.0},
        "THERMAL DYNAMIC": {
            "DYNAMICTYPE": "OneStepTheta", "INITIALFIELD": "field_by_function",
            "INITFUNCNO": 1, "TIMESTEP": p.dt, "MAXTIME": p.dt, "NUMSTEP": 1,
            "TOLTEMP": 1e-9, "TOLRES": 1e-6, "LINEAR_SOLVER": 1},
        "THERMAL DYNAMIC/ONESTEPTHETA": {"THETA": 1.0},
        "TSI DYNAMIC": {
            "COUPALGO": "tsi_monolithic" if two_way else "tsi_oneway",
            "NUMSTEP": 1, "MAXTIME": p.dt, "TIMESTEP": p.dt, "ITEMAX": 50},
        "TSI DYNAMIC/MONOLITHIC": {
            "CONVTOL": 1e-9, "TOLINC": 1e-9, "NORM_RESF": "Rel",
            "MERGE_TSI_BLOCK_MATRIX": True, "LINEAR_SOLVER": 1},
        "TSI DYNAMIC/PARTITIONED": {"COUPVARIABLE": "Temperature"},
        "SOLVER 1": {"SOLVER": "UMFPACK"},
        "MATERIALS": [
            {"MAT": 1,
             "MAT_Struct_ThermoStVenantK": {
                 "YOUNGNUM": 1, "YOUNG": [p.e_mod], "NUE": p.nu,
                 "DENS": 1e-9, "THEXPANS": p.alpha, "INITTEMP": p.t_ref,
                 "THERMOMAT": 2}},
            {"MAT": 2,
             "MAT_Fourier": {"CAPA": p.rho_c * capa_scale,
                             "CONDUCT": {"constant": [p.k_cond]}}},
        ],
        "CLONING MATERIAL MAP": [{"SRC_FIELD": "structure", "SRC_MAT": 1,
                                  "TAR_FIELD": "thermo", "TAR_MAT": 2}],
        "FUNCT1": [{"COMPONENT": 0,
                    "SYMBOLIC_FUNCTION_OF_SPACE_TIME": f"{p.t_old}"}],
        "DESIGN SURF DIRICH CONDITIONS": [
            {"E": 1, "NUMDOF": 3, "ONOFF": [1, 0, 0], "VAL": [0, 0, 0],
             "FUNCT": [0, 0, 0]}],
        "DESIGN VOL DIRICH CONDITIONS": [
            {"E": 1, "NUMDOF": 3, "ONOFF": [0, 1, 1], "VAL": [0, 0, 0],
             "FUNCT": [0, 0, 0]}],
        "DESIGN SURF THERMO DIRICH CONDITIONS": [
            {"E": 2, "NUMDOF": 1, "ONOFF": [1], "VAL": [p.t_hot], "FUNCT": [0]},
            {"E": 3, "NUMDOF": 1, "ONOFF": [1], "VAL": [p.t_cold], "FUNCT": [0]}],
        "DSURF-NODE TOPOLOGY": topo,
        "DVOL-NODE TOPOLOGY": vol,
        "NODE COORDS": nodes,
        "STRUCTURE ELEMENTS": elems,
    }
    return yaml.safe_dump(d, sort_keys=False, default_flow_style=None,
                          width=10 ** 6)


def run(work_dir: Path, p: TsiProblem, nx: int = 160, two_way: bool = True,
        capa_scale: float = 1.0, timeout: int = 1800) -> dict:
    """Write the deck, run 4C, read the temperature and x-displacement back.

    Returns {"x", "T", "ux"} sorted along the bar. `stdbuf -oL` is mandatory:
    without it an MPI_Abort swallows the diagnostic that says why.
    """
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    (work_dir / "tsi.4C.yaml").write_text(deck(p, nx, two_way, capa_scale))
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    for cand in ("/opt/4C-dependencies/lib",):
        if Path(cand).is_dir() and cand not in ld:
            env["LD_LIBRARY_PATH"] = (cand + os.pathsep + ld) if ld else cand
    cmd = ["stdbuf", "-oL", _find_binary(), "tsi.4C.yaml", "out"]
    r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True,
                       timeout=timeout, env=env)
    (work_dir / "4c_stdout.txt").write_text(r.stdout)
    (work_dir / "4c_stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        tail = (r.stdout + "\n" + r.stderr)[-1500:]
        raise RuntimeError(f"4C exited {r.returncode}. tail:\n{tail}")
    return _read_fields(work_dir, p)


def _last_vtu(work_dir: Path, which: str) -> Path:
    cands = sorted(work_dir.rglob(f"*{which}*.vtu"))
    if not cands:
        cands = sorted(work_dir.rglob("*.vtu"))
    if not cands:
        raise RuntimeError(f"4C wrote no {which} .vtu under {work_dir}")

    def step(pth: Path) -> int:
        m = re.findall(r"(\d+)", pth.stem)
        return int(m[-1]) if m else -1

    return max(cands, key=step)


def _read_fields(work_dir: Path, p: TsiProblem) -> dict:
    import meshio
    th = meshio.read(_last_vtu(work_dir, "thermo"))
    st = meshio.read(_last_vtu(work_dir, "structure"))
    tkey = next((k for k in th.point_data
                 if "temper" in k.lower()), None)
    dkey = next((k for k in st.point_data
                 if "displac" in k.lower()), None)
    if tkey is None or dkey is None:
        raise RuntimeError(f"4C VTU has no temperature/displacement: "
                           f"thermo={list(th.point_data)} "
                           f"structure={list(st.point_data)}")
    xt = th.points[:, 0]
    T = np.asarray(th.point_data[tkey], float).ravel()
    xs = st.points[:, 0]
    ux = np.asarray(st.point_data[dkey], float)[:, 0]
    # collapse the (y, z) duplicates: the solution is uniaxial, so every node at
    # the same x carries the same value; averaging them is also a check that it
    # really is one-dimensional.
    def collapse(x, v):
        xs_ = np.unique(np.round(x, 12))
        m = np.array([v[np.isclose(x, xx, atol=1e-12)].mean() for xx in xs_])
        s = np.array([v[np.isclose(x, xx, atol=1e-12)].std() for xx in xs_])
        return xs_, m, float(np.max(s))

    x1, Tm, Tspread = collapse(xt, T)
    x2, um, uspread = collapse(xs, ux)
    return {"x": x1, "T": Tm, "ux": np.interp(x1, x2, um),
            "transverse_spread_T": Tspread, "transverse_spread_ux": uspread}
