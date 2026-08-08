"""Proof that the blind harness works — including that its gate fires on REAL leaks.

A leak gate that has only ever been run on clean inputs is not known to work.
So this suite drives it from both ends:

  * the two leaks the pre-existing campaign's own DESIGN.md admits to (``B1``,
    ``D3``), plus one it does not know about (``B2``) — the genuine task texts
    and keys as they shipped, frozen under ``fixtures/blind_leaks/`` so the
    evidence outlives the fix, not reconstructions written to pass;
  * the four leaks in the older HOE-v2 prompt generator, verbatim;
  * negative controls that would fire a naive gate — above all a source term
    packed with ``sin(pi x)``, which is the false alarm that ruined an earlier
    attempt and would make the harness unusable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

from blind_eval import derive, keyvault, leakgate, selfconv
from blind_eval.spec import BlindSpec, Probe

# The three real leaks are frozen as fixtures rather than read from the live
# campaign.  Reading the campaign was self-defeating: rebuilding B1, B2 and D3
# removed the very leaks these tests exist to fire on, so the proof that the
# gate works evaporated at exactly the moment the defect was fixed.  A gate's
# evidence has to outlive the bug it found.
LEAKS = Path(__file__).resolve().parent / "fixtures" / "blind_leaks"
needs_fixtures = pytest.mark.skipif(
    not LEAKS.is_dir(), reason="frozen leak fixtures not present")

X, Y = derive.X, derive.Y


def _load(pid):
    task = (LEAKS / f"{pid}_task.txt").read_text()
    key = json.loads((LEAKS / f"{pid}_key.json").read_text())
    return task, key


# ══════════════════════════════════════════════════════════════════════
# 1. The gate fires on REAL leaks
# ══════════════════════════════════════════════════════════════════════
@needs_fixtures
@pytest.mark.parametrize("pid,expect_rule", [
    ("B1", "L4_PRINTED_SUBSUM"),   # DESIGN.md: "leaks by naked inspection"
    ("D3", "L4_PRINTED_SUBSUM"),   # DESIGN.md: the other admitted one
])
def test_gate_fires_on_admitted_real_leaks(pid, expect_rule):
    task, key = _load(pid)
    rep = leakgate.scan(task, key, pid)
    assert not rep.clean, f"{pid} is a known real leak but the gate passed it"
    assert any(f.rule == expect_rule for f in rep.findings), \
        f"{pid}: expected {expect_rule}, got {[f.rule for f in rep.findings]}"


@needs_fixtures
def test_gate_finds_a_leak_the_design_did_not_know_about():
    """B2 embeds 4*pi**2 * u_exact; DESIGN.md lists only B1 and D3."""
    task, key = _load("B2")
    rep = leakgate.scan(task, key, "B2")
    rules = [f.rule for f in rep.findings]
    assert any(r.startswith("L4_SOURCE_EMBEDS") or r == "L4_PRINTED_SUBSUM"
               for r in rules), f"B2 embedding not detected; got {rules}"


@needs_fixtures
def test_b1_leak_is_real_not_a_gate_artefact():
    """One PRINTED term of B1's source term is exactly 12*pi**2 * u_exact."""
    _, key = _load("B1")
    S = leakgate.SYMS
    u = sp.sympify(key["exact_solution"], locals=S)
    f = sp.sympify(key["source_term"], locals=S)
    comb, c = leakgate.printed_subsum(f, u)
    assert comb is not None and len(comb) == 1
    assert sp.simplify(comb[0] - c * u) == 0
    assert sp.simplify(c - 12 * sp.pi ** 2) == 0


# ══════════════════════════════════════════════════════════════════════
# 2. The trap: trigonometry in the source term is NOT a leak
# ══════════════════════════════════════════════════════════════════════
def test_source_term_full_of_sines_is_not_flagged():
    """The false alarm that ruined the earlier attempt.

    Hidden solution is a polynomial-times-exponential bubble; the source term
    is deliberately stuffed with sin(pi x) via a trigonometric coefficient. A
    gate that pattern-matches "sin(pi" fires here and is useless.
    """
    u = 8 * X * (1 - X) * Y * (1 - Y) * sp.exp(X + Y)
    kappa = 2 + sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    f = sp.simplify(-derive.divergence(
        [kappa * g for g in derive.grad(u, (X, Y))], (X, Y)))
    assert "sin(pi*x)" in sp.sstr(f).replace(" ", "")
    task = (f"Solve -div(k grad u) = f on the unit square.\n"
            f"COEFFICIENT: k(x,y) = {sp.sstr(kappa)}\n"
            f"SOURCE TERM: f(x, y) = {sp.sstr(f)}\n"
            f"BOUNDARY CONDITIONS: u = 0 on the entire boundary.\n")
    key = {"exact_solution": sp.sstr(u), "source_term": sp.sstr(f)}
    rep = leakgate.scan(task, key, "trap")
    assert rep.clean, f"false alarm on a legitimate source term: {[f_.detail for f_ in rep.findings]}"


def test_boundary_data_and_probe_coordinates_are_not_flagged():
    """Numbers that must appear (BC values, probe coordinates) stay clean."""
    u = 8 * X * (1 - X) * Y * (1 - Y) * sp.exp(X + Y)
    f = sp.simplify(-derive.laplacian(u, (X, Y)))
    task = (f"SOURCE TERM: f(x, y) = {sp.sstr(f)}\n"
            "BOUNDARY CONDITIONS: u = 0 on the entire boundary.\n"
            "PROBE POINTS: x = (i+0.5)/32, y = (j+0.5)/32.\n"
            "MESH SEQUENCE: N = 8, 16, 32, 64.\n")
    key = {"exact_solution": sp.sstr(u), "source_term": sp.sstr(f)}
    assert leakgate.scan(task, key, "bc").clean


# ══════════════════════════════════════════════════════════════════════
# 3. Leaks a substring test cannot catch
# ══════════════════════════════════════════════════════════════════════
def test_algebraically_equivalent_disclosure_is_caught():
    """The solution rewritten in a different but equal form.

    build_problems.leak_check does ``normalise(u) in normalise(task)``, so any
    reordering or refactoring of the same function slips straight through.
    """
    u = X * Y * (1 - X) * (1 - Y) * sp.cos(2 * sp.pi * X)
    disguised = "x*(1 - y)*y*(1 - x)*cos(2*pi*x)"        # same function, different string
    assert sp.simplify(sp.sympify(disguised, locals=leakgate.SYMS) - u) == 0
    assert sp.sstr(u).replace(" ", "") not in disguised.replace(" ", "")   # substring test misses it

    task = f"Solve the problem. For reference, the field satisfies w = {disguised}.\n"
    key = {"exact_solution": sp.sstr(u), "source_term": "0"}
    rep = leakgate.scan(task, key, "equiv")
    assert not rep.clean
    assert any(f.rule == "L2_SYMBOLIC" for f in rep.findings)


def test_scaled_disclosure_is_caught():
    u = X * Y * (1 - X) * (1 - Y)
    task = "A useful auxiliary field is g = 7*x*y*(1 - x)*(1 - y).\n"
    key = {"exact_solution": sp.sstr(u), "source_term": "0"}
    rep = leakgate.scan(task, key, "scaled")
    assert any(f.rule == "L2_SYMBOLIC" for f in rep.findings)


def test_numeric_probe_value_disclosure_is_caught():
    u = X * Y * (1 - X) * (1 - Y)
    true_vals = [float(u.subs({X: 0.5, Y: 0.5}))]          # 0.0625
    task = "Sanity check: at the centre the field takes the value 0.0625.\n"
    key = {"exact_solution": sp.sstr(u), "source_term": "0"}
    rep = leakgate.scan(task, key, "numeric", true_probe_values=true_vals)
    assert any(f.rule == "L3_NUMERIC" for f in rep.findings)


# ══════════════════════════════════════════════════════════════════════
# 4. The older HOE-v2 prompts, verbatim
# ══════════════════════════════════════════════════════════════════════
HOE_V2_LEAKS = {
    "E2": ("Derive the body force from the manufactured displacement field\n"
           "  u1 = sin(pi x) sin(pi y) sin(pi z)\n"
           "compute the L2 displacement error against u_exact for each N"),
    "E3": ("Solve -Laplacian(u) = 2 pi^2 sin(pi x) sin(pi y) on the unit square "
           "with u = 0 on the whole boundary (exact solution u = sin(pi x) sin(pi y))\n"
           "RESULT err_A = <relative L2 error of subdomain-A solution vs exact>"),
    "E7": ("The exact solution is the 1D boundary-layer profile "
           "u(x, y) = (exp(x/eps) - 1) / (exp(1/eps) - 1).\n"
           "Report the L2 error against the exact solution."),
    "E8": ("The exact eigenvalues are squares of Bessel-function zeros: "
           "lambda_1 = 5.78319, and the double eigenvalue lambda_2 = 14.68197."),
}


@pytest.mark.parametrize("tid", sorted(HOE_V2_LEAKS))
def test_old_hoe_v2_prompts_are_caught(tid):
    key = {"exact_solution": "sin(pi*x)*sin(pi*y)", "source_term": "0"}
    rep = leakgate.scan(HOE_V2_LEAKS[tid], key, tid)
    assert not rep.clean, f"{tid} leaks but the gate passed it"


def test_e3_source_term_is_an_eigenfunction_pair():
    """-Lap u = 2 pi^2 sin sin: the source term IS the solution, scaled."""
    u = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    src = derive.derive_source("poisson", u, {"kappa": 1}, 2)
    checks = derive.design_checks(u, src, 2, "poisson")
    assert any(c.startswith("PROPORTIONAL") for c in checks["failures"])


# ══════════════════════════════════════════════════════════════════════
# 5. Derivation correctness and design checks
# ══════════════════════════════════════════════════════════════════════
def test_derivation_is_proved_by_substitution():
    u = 8 * X * (1 - X) * Y * (1 - Y) * sp.exp(X + Y)
    src = derive.derive_source("poisson", u, {"kappa": 1}, 2)
    assert derive.verify_residual("poisson", u, src, {"kappa": 1}, 2) == {}


def test_a_wrong_source_term_fails_the_proof():
    u = 8 * X * (1 - X) * Y * (1 - Y) * sp.exp(X + Y)
    src = derive.derive_source("poisson", u, {"kappa": 1}, 2)
    src["f"] = src["f"] + 1                      # inject a sign/offset error
    assert derive.verify_residual("poisson", u, src, {"kappa": 1}, 2) != {}


def test_recommended_field_passes_all_design_checks():
    u = 8 * X * (1 - X) * Y * (1 - Y) * sp.exp(X + Y)
    src = derive.derive_source("poisson", u, {"kappa": 1}, 2)
    checks = derive.design_checks(u, src, 2, "poisson")
    assert checks["failures"] == [] and checks["warnings"] == []
    assert derive.boundary_trace_is_zero(u, 2, ((0, 1), (0, 1)))


def test_elasticity_derivation_and_proof():
    phi = X * (1 - X) * Y * (1 - Y)
    u = [phi * sp.exp(X), phi * (1 + Y) / 2]
    co = {"E": 1000, "nu": sp.Rational(3, 10)}
    src = derive.derive_source("elasticity", u, co, 2)
    assert derive.verify_residual("elasticity", u, src, co, 2) == {}
    assert derive.boundary_trace_is_zero(u, 2, ((0, 1), (0, 1)))


# ══════════════════════════════════════════════════════════════════════
# 6. Order by mesh halving, with no exact solution anywhere
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_self_convergence_recovers_the_order(p):
    """u_h = u + C h^p, sampled on four halvings, at ten probe points."""
    import random
    rng = random.Random(7)
    truth = [rng.uniform(-1, 1) for _ in range(10)]
    coef = [rng.uniform(0.5, 2.0) for _ in range(10)]
    levels = [[t + c * (0.5 ** (k * p)) for t, c in zip(truth, coef)]
              for k in range(4)]
    r = selfconv.self_convergence(levels, theoretical_order=p)
    assert r.order == pytest.approx(p, abs=1e-6)
    assert r.heuristics["monotone_differences"]
    assert r.heuristics["order_near_theory"]
    assert r.ok


def test_self_convergence_needs_no_key():
    """The estimator's signature admits no exact solution, by construction."""
    import inspect
    params = set(inspect.signature(selfconv.self_convergence).parameters)
    for banned in ("exact", "u_exact", "key", "reference", "truth"):
        assert banned not in params


def test_self_convergence_rejects_a_stalled_sequence():
    levels = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    r = selfconv.self_convergence(levels, theoretical_order=2.0)
    assert r.order is None


def test_self_convergence_flags_nonmonotone_and_unbounded():
    bad = selfconv.self_convergence([[1.0], [float("inf")], [1.0]])
    assert bad.heuristics["bounded"] is False
    noisy = selfconv.self_convergence([[1.0], [1.5], [1.0], [1.4]])
    assert noisy.heuristics["monotone_differences"] is False


def test_richardson_extrapolation_beats_the_finest_level():
    p, truth = 2.0, 3.0
    levels = [[truth + 1.7 * (0.5 ** (k * p))] for k in range(4)]
    r = selfconv.self_convergence(levels, theoretical_order=p)
    assert abs(r.richardson_estimate[0] - truth) < abs(levels[-1][0] - truth)
    assert r.gci is not None and r.gci > 0


def test_cross_check_detects_disagreeing_orders():
    assert selfconv.cross_check(2.0, 1.98)["agree"] is True
    assert selfconv.cross_check(2.0, 0.9)["agree"] is False


# ══════════════════════════════════════════════════════════════════════
# 7. Structural blindness of the spec container
# ══════════════════════════════════════════════════════════════════════
def _payload(**over):
    base = dict(task_id="T", title="t", domain="unit square", pde="-lap u = f",
                coefficients={}, source_term={"f": "1"}, boundary_conditions=["u=0"],
                mesh_sequence=[8, 16, 32], probes=[], result_keys=[])
    base.update(over)
    return base


def test_blindspec_refuses_an_exact_solution_field():
    for bad in ("u_exact", "exact_solution", "solution", "reference_values"):
        with pytest.raises(ValueError):
            BlindSpec.from_payload(_payload(**{bad: "sin(pi*x)"}))


def test_blindspec_round_trips_and_has_no_answer_shaped_field():
    s = BlindSpec.from_payload(_payload(probes=[
        {"name": "p1", "kind": "point", "quantity": "u", "location": [0.5, 0.5]}]))
    assert isinstance(s.probes[0], Probe)
    assert BlindSpec.from_payload(s.to_payload()).task_id == "T"
    for f in BlindSpec.field_names():
        assert not any(w in f for w in ("exact", "solution", "answer", "truth"))


# ══════════════════════════════════════════════════════════════════════
# 8. Key custody
# ══════════════════════════════════════════════════════════════════════
def test_encrypt_decrypt_round_trip_and_wrong_passphrase(tmp_path):
    blob = keyvault.encrypt_bytes(b'{"exact_solution": "x*y"}', "080294")
    assert b"exact_solution" not in blob and b"x*y" not in blob
    assert keyvault.decrypt_bytes(blob, "080294") == b'{"exact_solution": "x*y"}'
    with pytest.raises(Exception):
        keyvault.decrypt_bytes(blob, "wrong")


def test_encrypt_tree_removes_plaintext(tmp_path):
    d = tmp_path / "keys" / "B1"
    d.mkdir(parents=True)
    (d / "key.json").write_text('{"exact_solution": "x*(1-x)*y*(1-y)"}')
    keyvault.encrypt_tree(tmp_path / "keys", "080294")
    assert not (d / "key.json").exists()
    enc = d / "key.json.enc"
    assert enc.exists() and b"exact_solution" not in enc.read_bytes()
    assert json.loads(keyvault.decrypt_bytes(enc.read_bytes(), "080294"))


def test_absence_is_not_a_seal(tmp_path):
    """The pre-existing check called a missing or empty keys dir 'sealed'."""
    assert keyvault.is_sealed(tmp_path / "nope") is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert keyvault.is_sealed(empty) is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores DAC permissions")
def test_sealing_makes_keys_unreadable_to_a_separate_process(tmp_path):
    d = tmp_path / "keys" / "B1"
    d.mkdir(parents=True)
    (d / "key.json").write_text('{"exact_solution": "SEALED-CANARY"}')
    try:
        keyvault.seal(tmp_path / "keys")
        assert keyvault.is_sealed(tmp_path / "keys")
        v = keyvault.verify_unreadable(tmp_path / "keys")
        assert v["sealed"] is True, v
        assert v["files_opened"] == 0
        assert "SEALED-CANARY" not in v["shell_cat"]
    finally:
        keyvault.unseal(tmp_path / "keys")
    assert (d / "key.json").read_text()


def test_hash_manifest_detects_a_changed_key(tmp_path):
    d = tmp_path / "keys" / "B1"
    d.mkdir(parents=True)
    k = d / "key.json"
    k.write_text('{"exact_solution": "x*(1-x)"}')
    man = keyvault.build_manifest(tmp_path / "keys", campaign="test")
    assert keyvault.verify_manifest(man, tmp_path / "keys")["verdict"] == "PASS"
    k.write_text('{"exact_solution": "SOMETHING ELSE"}')      # post-hoc edit
    bad = keyvault.verify_manifest(man, tmp_path / "keys")
    assert bad["verdict"] == "FAIL" and bad["mismatched"] == ["B1/key.json"]


def test_manifest_detects_tampering_with_itself(tmp_path):
    d = tmp_path / "keys"
    d.mkdir()
    (d / "k.json").write_text("{}")
    man = keyvault.build_manifest(d, campaign="test")
    man["entries"][0]["sha256"] = "0" * 64
    assert keyvault.verify_manifest(man, d)["manifest_self_consistent"] is False
