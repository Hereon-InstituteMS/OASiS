"""Structural guards for the TWO-WAY thermo-structural coupling.

Nothing here runs a coupling — the tier-2 fixtures under
scripts/tier2_fixtures/coupling/tsi_* do that, against a monolithic re-solve and
against 4C's native TSI. These are the properties that must hold whether or not
a backend is installed, and the ones a fixture cannot check because it would
have to be wrong in the same way to miss them:

  * the shipped participants exist, parse, and expose every knob the fixture
    library configures — a `.replace()` that silently matched nothing would
    leave a fixture verifying a script it never configured;
  * they contain NO reference values. The whole verification rests on the
    coupled answer being compared against something computed elsewhere; a
    participant that carried the answer would make the comparison circular;
  * the answer-bearing code (the monolithic reference, the 4C reference) is
    NOT importable from src/, so no tool can serve it;
  * `coupled_solve(problem='tsi_dd')` no longer reports a fabricated
    convergence;
  * the served TSI knowledge names the reverse-direction term and the control
    that proves it is alive.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PARTICIPANTS = REPO / "data" / "coupling_participants"
BACKENDS = ("skfem", "fenics", "ngsolve")

# The knobs tsilib.thermal_edits / mech_edits set. `couplinglib.edit` raises on a
# key that is not a top-level assignment, so a rename here is caught at fixture
# time too — but only on a host that can run the fixture, and only for the
# backend it happens to use. This checks every shipped script on every host.
THERMAL_KEYS = ["PARTNER", "X0, X1", "Y0, Y1", "NX, NY", "K_COND", "RHO_C",
                "DT", "T_REF", "T_OLD", "T_HOT", "T_HOT_DY", "T_COLD", "BETA",
                "COUPLING", "EVOL_OLD", "EVOL_INIT"]
MECH_KEYS = ["PARTNER", "X0, X1", "Y0, Y1", "NX, NY", "E_MOD", "NU", "BETA",
             "THETA_INIT"]


def _src(kind: str, backend: str) -> str:
    p = PARTICIPANTS / f"participant_tsi_{kind}_{backend}.py"
    assert p.is_file(), f"missing shipped TSI participant {p.name}"
    return p.read_text()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind,keys", [("thermal", THERMAL_KEYS),
                                       ("mech", MECH_KEYS)])
def test_participant_exposes_every_configured_knob(backend, kind, keys):
    text = _src(kind, backend)
    for k in keys:
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        assert pat.search(text), (
            f"participant_tsi_{kind}_{backend}.py has no top-level `{k} = ...` "
            f"assignment, so the fixture library's edit of it would be a silent "
            f"no-op and the fixture would verify a script it did not configure")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["thermal", "mech"])
def test_participant_parses(backend, kind):
    ast.parse(_src(kind, backend))


@pytest.mark.parametrize("backend", BACKENDS)
def test_thermal_participant_carries_the_reverse_direction(backend):
    """The mechanical -> thermal term IS the two-way capability. A thermal
    participant without it is a heat solver."""
    text = _src("thermal", backend)
    assert "COUPLING" in text and "EVOL_OLD" in text
    # the term must multiply the imported strain by T_ref*BETA/DT and be
    # switchable — the fixtures' direction control depends on that switch
    assert re.search(r"COUPLING\s*\*\s*.*T_REF\s*\*\s*BETA\s*/\s*DT", text) or \
        re.search(r"COUPLING\s*\*\s*T_REF\s*\*\s*BETA\s*/\s*DT", text), (
        f"participant_tsi_thermal_{backend}.py does not apply "
        f"COUPLING * T_REF * BETA / DT to the imported volumetric strain, so "
        f"either the reverse direction or its control switch is missing")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["thermal", "mech"])
def test_participant_exchanges_the_right_quantity(backend, kind):
    text = _src(kind, backend)
    want = "temperature_change" if kind == "thermal" else "volumetric_strain"
    assert f'"field_name": "{want}"' in text, (
        f"participant_tsi_{kind}_{backend}.py must export field_name "
        f"{want!r}: `couple`'s monolithic check compares like with like by "
        f"field name and silently skips a participant whose name differs")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["thermal", "mech"])
def test_participant_carries_no_reference_values(backend, kind):
    """A participant that shipped the answer would make every comparison
    circular. Nothing here may look like a stored result."""
    text = _src(kind, backend)
    # Prose is fine and wanted — the headers explain that only a monolithic or
    # native comparison can catch a wrong sign. What must not be there is a
    # stored ANSWER or a check against one.
    body = "\n".join(l for l in text.splitlines()
                      if not l.lstrip().startswith("#"))
    body = re.sub(r'"""[\s\S]*?"""', "", body, count=1)     # drop the header
    for bad in ("EXPECTED", "REFERENCE", "assert ", "relL2",
                "tsi_monolithic", "tsi_fourc", "monolithic.json"):
        assert bad not in body, (
            f"participant_tsi_{kind}_{backend}.py contains {bad!r} outside its "
            f"header — a shipped participant must carry no reference values "
            f"and no checks against them")


def test_the_reference_solvers_are_not_reachable_from_src():
    """The monolithic and 4C references compute the graded answer. They live
    under scripts/tier2_fixtures/ and must stay there: anything importable from
    src/ can be reached by a tool and therefore quoted back as an answer."""
    for name in ("tsi_monolithic.py", "tsi_fourc.py"):
        hits = list((REPO / "src").rglob(name))
        assert not hits, f"{name} is importable from src/: {hits}"
    lib = REPO / "scripts" / "tier2_fixtures" / "coupling" / "_lib"
    assert (lib / "tsi_monolithic.py").is_file()
    assert (lib / "tsi_fourc.py").is_file()


def test_no_served_payload_mentions_the_reference_modules():
    """Not even by name: a payload that told an agent where the answer lives
    would defeat the separation above."""
    # The MODULE, not the token: `tsi_monolithic` is also 4C's own COUPALGO
    # enum value and appears legitimately in a deck generator, so a bare
    # substring test accuses that file of leaking a reference it has never
    # heard of.
    pats = [re.compile(r"\b(?:import|from)\s+tsi_(?:monolithic|fourc)\b"),
            re.compile(r"tsi_(?:monolithic|fourc)\.py"),
            re.compile(r"monolithic_full\.json"),
            re.compile(r"\bsolve_monolithic\b"),
            re.compile(r"\bcheck_effective_capacity_identity\b")]
    for path in (REPO / "src").rglob("*.py"):
        text = path.read_text(errors="replace")
        for pat in pats:
            m = pat.search(text)
            assert not m, f"{path} names the fixture-side reference: {m.group(0)!r}"


def test_legacy_tsi_dd_no_longer_reports_a_fabricated_convergence():
    """`coupled_solve(problem='tsi_dd')` used to write `converged: True,
    iterations: 1, residual: 0.0` after one thermal solve and one one-way
    structural solve, with nothing exchanged back."""
    src = (REPO / "src" / "tools" / "coupling.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_twoway_tsi_coupling")
    # The docstring QUOTES the fabricated literals on purpose — that is the
    # record of what was removed — so this looks at the executable body, which
    # must be nothing but a return.
    stmts = [n for n in fn.body if not (isinstance(n, ast.Expr)
                                        and isinstance(n.value, ast.Constant)
                                        and isinstance(n.value.value, str))]
    assert len(stmts) == 1 and isinstance(stmts[0], ast.Return), (
        "the removed path must not run anything; its body is a single return")
    text = ast.unparse(stmts[0])
    assert "REMOVED" in text
    assert "couple" in text, "the refusal must name the tool that replaces it"
    # The returned string DESCRIBES the fabricated literals; what matters is
    # that they are prose in one string constant and not a dict this function
    # builds. Nothing here may construct a result object at all.
    assert not [n for n in ast.walk(stmts[0]) if isinstance(n, ast.Dict)], (
        "the removed path must not build a result dict")
    assert not [n for n in ast.walk(fn) if isinstance(n, (ast.Await, ast.Call))
                and not (isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute))], (
        "the removed path must not run or await anything")


def test_transfer_field_documents_the_volume_mode():
    """A field coupling has no interface plane to slice, so the tool has to be
    able to say so."""
    from core.field_transfer import extract_interface_from_vtu
    doc = extract_interface_from_vtu.__doc__ or ""
    assert "interface_axis = -1" in doc or "interface_axis=-1" in doc
    con = (REPO / "src" / "tools" / "consolidated.py").read_text()
    i = con.index("async def transfer_field(")
    assert "interface_axis=-1" in con[i:i + 4000]


def test_transfer_field_volume_mode_takes_every_point(tmp_path):
    import numpy as np
    meshio = pytest.importorskip("meshio")
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0]], float)
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    f = tmp_path / "vol.vtu"
    meshio.write_points_cells(str(f), pts, cells,
                              point_data={"temperature": vals})
    from core.field_transfer import extract_interface_from_vtu
    plane = extract_interface_from_vtu(f, "temperature", 0.0, 0)
    assert len(plane.coordinates) == 2, "the plane slice must still slice"
    whole = extract_interface_from_vtu(f, "temperature", 0.0, -1)
    assert len(whole.coordinates) == 4
    assert sorted(np.asarray(whole.values, float).ravel()) == [1.0, 2.0, 3.0, 4.0]
    # the order must be reproducible: the driver relaxes export vectors entry
    # by entry, so a participant whose point order moved between iterations
    # would be relaxing unrelated numbers against each other
    again = extract_interface_from_vtu(f, "temperature", 0.0, -1)
    assert np.array_equal(np.asarray(whole.coordinates),
                          np.asarray(again.coordinates))


def _tsi_knowledge() -> str:
    from mcp.server.fastmcp import FastMCP
    from tools.knowledge import register_knowledge_tools
    m = FastMCP("t")
    register_knowledge_tools(m)
    fn = None
    for attr in ("_tool_manager",):
        mgr = getattr(m, attr, None)
        if mgr is None:
            continue
        for t in mgr.list_tools() if hasattr(mgr, "list_tools") else []:
            pass
    import inspect
    import tools.knowledge as K
    src = inspect.getsource(K)
    i = src.index("def get_tsi_knowledge")
    j = src.index("def get_precice_knowledge")
    return src[i:j]


@pytest.mark.parametrize("token", [
    "TWO-WAY TSI ACROSS TWO CODES",
    "T_ref*beta*(tr eps(u) - tr eps(u_old))/dt",
    "delta = T_ref * beta^2 / (rho_c * (lam + 2 mu))",
    "theta = 1/(1+delta)",
    "imports_from: []",
    "temperature CHANGE",
    "participant_tsi_thermal_",
    "Did not get 1:1 correspondence",
    "Newton unconverged in 50 iterations",
    "Statics",
    "COUPVARIABLE",
])
def test_tsi_knowledge_documents_the_two_way_path(token):
    """Every one of these is either the capability, the control that proves it,
    or a failure whose message points somewhere other than its cause."""
    assert token in _tsi_knowledge(), f"{token!r} missing from knowledge(topic='tsi')"


def test_tsi_knowledge_no_longer_calls_a_one_way_recipe_a_coupling():
    """It used to say: FEniCS solves heat, 4C receives the SAME thermal BCs,
    compare displacements. Nothing is exchanged back in that; it is two
    independent one-way runs."""
    k = _tsi_knowledge()
    assert "4C TSI receives same thermal BCs" not in k
    assert "one-way" in k.lower(), (
        "the knowledge must distinguish one-way from two-way explicitly — most "
        "published 'TSI couplings' are one-way and do not say so")
