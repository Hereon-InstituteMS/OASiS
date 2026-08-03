"""The critic requirement, exercised through the real MCP tools.

The unit tests in test_critic_gate.py prove the registry refuses forged, stale,
reused and mismatched tokens. They say nothing about whether OASiS ASKS it. An
audit found exactly that gap: `critic_gate` existed, was tested, and was wired
into nothing, so `run_simulation(..., critic_approved=True)` still stamped a
result VERIFIED with no critic anywhere in the process.

These tests therefore drive the tools themselves, with a real solver run, and
assert on the verdict the agent actually receives.
"""
from __future__ import annotations

import json

import pytest

from tools import consolidated


# A genuine, tiny scikit-fem Poisson solve. Real run, real output files, so
# `evidence_ok` is True and the ONLY thing standing between it and a VERIFIED
# stamp is the critic. That isolation is the point: a test that fails the
# numerical checks would pass for the wrong reason.
DECK = """
import numpy as np, skfem
from skfem import Basis, ElementTriP1, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad

m = skfem.MeshTri().refined(3)
basis = Basis(m, ElementTriP1())

@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def L(v, w):
    return 1.0 * v

A, b = asm(a, basis), asm(L, basis)
x = solve(*condense(A, b, D=basis.get_dofs()))
m.save('solution.vtu', {'u': x})
print('max u =', float(x.max()))
"""

FINDINGS = (
    "Checked the weak form against -laplace(u)=1 on the unit square, confirmed "
    "homogeneous Dirichlet data is applied on the whole boundary via "
    "basis.get_dofs(), verified P1 elements are adequate for this smooth "
    "problem, and confirmed the units are consistent (dimensionless)."
)


@pytest.fixture
def tools(monkeypatch):
    """Fresh server + registry per test, so one test's review cannot verify
    another's run."""
    from core.registry import load_all_backends
    load_all_backends()
    monkeypatch.setattr(consolidated, "_CRITIC_REGISTRY",
                        consolidated.CriticRegistry())
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    consolidated.register_consolidated_tools(_Recorder())
    return captured


def _skfem_available() -> bool:
    try:
        import skfem  # noqa: F401
        return True
    except Exception:
        return False


needs_skfem = pytest.mark.skipif(not _skfem_available(),
                                 reason="scikit-fem not installed")


# ── the hole that was open ────────────────────────────────────────────────
@needs_skfem
@pytest.mark.asyncio
async def test_asserting_critic_approved_does_not_verify_a_run(tools):
    """THE regression. `critic_approved=True` with no review on record used to
    stamp a real run VERIFIED. It must not."""
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_approved=True))
    assert out["trustworthy_result"] is False
    # It must fail on the CRITIC, not on missing evidence — otherwise this test
    # would still pass with the critic requirement torn out, which is exactly
    # the kind of green that means nothing.
    assert "MANDATORY independent critic" in out["verification"], out["verification"]
    # and the verdict must NAME the false claim rather than quietly ignoring it
    assert "critic_approved=True" in out["critic_review"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_submitted_review_verifies_that_setup(tools):
    sub = json.loads(await tools["submit_critic_review"](
        solver="skfem", setup=DECK, findings=FINDINGS))
    assert sub["accepted"] is True and sub["critic_token"]

    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_approved=True))
    assert out["trustworthy_result"] is True, out["verification"]
    assert "VERIFIED" in out["verification"]


@needs_skfem
@pytest.mark.asyncio
async def test_editing_the_deck_after_review_invalidates_it(tools):
    """Review a clean deck, run a different one — the obvious way to defeat a
    critic requirement."""
    await tools["submit_critic_review"](solver="skfem", setup=DECK,
                                        findings=FINDINGS)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK + "\n# one extra character of change",
        critic_approved=True))
    assert out["trustworthy_result"] is False
    assert "changed after it was reviewed" in out["critic_review"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_review_for_another_solver_does_not_transfer(tools):
    await tools["submit_critic_review"](solver="fenics", setup=DECK,
                                        findings=FINDINGS)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_approved=True))
    assert out["trustworthy_result"] is False


# ── the registry's own refusals, reached through the tool ─────────────────
@pytest.mark.asyncio
async def test_an_empty_approval_is_refused(tools):
    """"Looks fine" is indistinguishable from no review at all."""
    sub = json.loads(await tools["submit_critic_review"](
        solver="skfem", setup=DECK, findings="LGTM"))
    assert sub["accepted"] is False and "characters" in sub["error"]


@pytest.mark.asyncio
async def test_a_review_must_name_exactly_one_setup(tools):
    both = json.loads(await tools["submit_critic_review"](
        solver="skfem", setup=DECK, coupling_args="{}", findings=FINDINGS))
    neither = json.loads(await tools["submit_critic_review"](
        solver="skfem", findings=FINDINGS))
    assert both["accepted"] is False and neither["accepted"] is False


@needs_skfem
@pytest.mark.asyncio
async def test_a_token_is_single_use(tools):
    """The token path additionally bounds how many runs one review covers."""
    tok = json.loads(await tools["submit_critic_review"](
        solver="skfem", setup=DECK, findings=FINDINGS))["critic_token"]

    first = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_token=tok))
    assert first["trustworthy_result"] is True, first["verification"]

    second = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_token=tok))
    assert second["trustworthy_result"] is False
    assert "already used" in second["critic_review"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_self_issued_token_is_refused(tools):
    """An agent cannot mint its own: the store is server-side."""
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK,
        critic_token="obviously-not-a-real-token"))
    assert out["trustworthy_result"] is False
    assert "not known to this server" in out["critic_review"]


@needs_skfem
@pytest.mark.asyncio
async def test_an_expired_review_does_not_verify(tools):
    await tools["submit_critic_review"](solver="skfem", setup=DECK,
                                        findings=FINDINGS, ttl_s=-1.0)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=DECK, critic_approved=True))
    assert out["trustworthy_result"] is False


# ── numbers come from the data, not from what the run said ────────────────
@needs_skfem
@pytest.mark.asyncio
async def test_oasis_computes_the_numbers_from_the_runs_own_data(tools):
    """The gate bound its verdict to the RUN but never to a NUMBER.

    A deck that solves honestly and then prints a flattering value used to be
    indistinguishable from one that printed the true one — nothing recomputed
    it. Here the script prints a false L2 norm and writes real data; the value
    OASiS reports must come from the data.
    """
    lying = DECK + "\nprint('L2 error = 1.0000e-12')\nprint('max u = 999.0')\n"
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=lying, job_name="attest_vs_narration"))

    computed = out["oasis_computed"]
    assert computed["max_abs"]["available"] is True, computed
    # -laplace(u)=1 on the unit square with u=0 on the boundary peaks at about
    # 0.0737; P1 on this mesh lands just under it. Nowhere near the 999 claimed.
    assert 0.06 < computed["max_abs"]["value"] < 0.08, computed
    assert computed["l2_norm"]["value"] > 1e-6, computed
    # and it must say which artefact it came from, so the claim is checkable
    assert computed["max_abs"]["from_file"].endswith(".vtu")
    assert computed["max_abs"]["sha256"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_run_with_no_data_output_attests_nothing(tools):
    """A script that only prints cannot have a number attested — the honest
    outcome is "unattestable", never a value borrowed from stdout."""
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content="print('L2 error = 1e-12')\n",
        job_name="attest_no_data"))
    # No artefacts at all, so the run is not verified and nothing is computed.
    assert out["trustworthy_result"] is False
    assert "oasis_computed" not in out or not any(
        v.get("available") for v in out["oasis_computed"].values()
        if isinstance(v, dict))


# ── does the output actually solve the problem? ───────────────────────────
MMS_SOLVE = """
import numpy as np, skfem
from skfem import Basis, ElementTriP1, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad

m = skfem.MeshTri().refined(5)
basis = Basis(m, ElementTriP1())

@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def L(v, w):
    return 2*np.pi**2*np.sin(np.pi*w.x[0])*np.sin(np.pi*w.x[1]) * v

A, b = asm(a, basis), asm(L, basis)
x = solve(*condense(A, b, D=basis.get_dofs()))
m.save('solution.vtu', {'u': x})
"""

# The eight-line fabricator. It never assembles anything: it writes the analytic
# field straight to disk. The result is MORE accurate than the genuine solve and
# passes a mesh-independence study, so accuracy cannot be the discriminator.
MMS_FORGERY = """
import numpy as np, skfem
m = skfem.MeshTri().refined(5)
p = m.p.T
u = np.sin(np.pi*p[:, 0])*np.sin(np.pi*p[:, 1])
m.save('solution.vtu', {'u': u})
print('L2 error = 3.1e-06')
"""

PDE = json.dumps({"operator": "diffusion",
                  "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                  "dim": 2, "domain_measure": 1.0})


@needs_skfem
@pytest.mark.asyncio
async def test_a_genuine_solve_satisfies_its_declared_equations(tools):
    await tools["submit_critic_review"](solver="skfem", setup=MMS_SOLVE,
                                        findings=FINDINGS)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=MMS_SOLVE, verify_pde=PDE,
        job_name="residual_genuine"))
    assert out["residual_check"]["verdict"] == "SOLVES", out["residual_check"]
    assert out["residual_check"]["relative_residual"] < 1e-8
    assert out["trustworthy_result"] is True, out["verification"]
    assert "SATISFIES the equations" in out["verification"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_verified_run_says_when_nothing_checked_the_physics(tools):
    """The residual check is opt-in, so a run that skips it still verifies. If
    the verdict were silent about that, "checked and solves the problem" and
    "nothing looked" would read identically in the one place an agent reads."""
    await tools["submit_critic_review"](solver="skfem", setup=MMS_SOLVE,
                                        findings=FINDINGS)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=MMS_SOLVE, job_name="residual_absent"))
    assert out["trustworthy_result"] is True
    assert "nothing here checked whether this output satisfies any equations" \
        in out["verification"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_written_out_analytic_field_is_refused(tools):
    """Reviewed, finite, structurally sane, more accurate than the real solve —
    and not a solve. Nothing that inspects only the data can tell; the residual
    can."""
    await tools["submit_critic_review"](solver="skfem", setup=MMS_FORGERY,
                                        findings=FINDINGS)
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=MMS_FORGERY, verify_pde=PDE,
        job_name="residual_forgery"))
    assert out["residual_check"]["verdict"] == "DOES_NOT_SOLVE", out["residual_check"]
    assert out["trustworthy_result"] is False
    assert "does NOT satisfy the equations" in out["verification"]


@needs_skfem
@pytest.mark.asyncio
async def test_an_unsupported_problem_is_not_checked_rather_than_passed(tools):
    """'Not checked' must never be reachable as 'checked and fine'."""
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=MMS_SOLVE, job_name="residual_unsupported",
        verify_pde=json.dumps({"operator": "navier_stokes", "source": "0",
                               "dim": 2})))
    assert out["residual_check"]["verdict"] == "REFUSED"
    assert "not one OASiS can assemble" in out["residual_check"]["detail"]


@needs_skfem
@pytest.mark.asyncio
async def test_a_source_term_is_an_expression_not_a_program(tools):
    out = json.loads(await tools["run_simulation"](
        solver="skfem", input_content=MMS_SOLVE, job_name="residual_injection",
        verify_pde=json.dumps({"operator": "diffusion", "dim": 2,
                               "source": "__import__('os').system('true')"})))
    assert out["residual_check"]["verdict"] == "REFUSED"


# ── the bypass that was removed ───────────────────────────────────────────
def test_no_environment_variable_can_lift_the_critic_requirement():
    """OFA_DISABLE_CRITIC used to stamp unreviewed runs VERIFIED. A mandatory
    gate with an environment-variable off-switch is not mandatory: any stray
    export in a harness silently converts every verdict.

    Checked on the AST, not the text: the comment recording WHY the switch was
    removed necessarily names it, and a substring search would either fail on
    that comment or force someone to delete the explanation to go green.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(consolidated))
    reads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value == "OFA_DISABLE_CRITIC"
    ]
    assert not reads, (
        "the critic ablation switch is back in executable code at line(s) "
        + ", ".join(str(n.lineno) for n in reads))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "_ABLATE_CRITIC" not in names
