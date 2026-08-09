"""One theta, two components: which rho does it have to be chosen from?

THE QUESTION THIS SETTLES. The driver applies ONE relaxation factor to the
whole interface state. For a SCALAR interface the served rule is
theta = 1/(1+rho), with rho the ratio of the two subdomains' interface
conductances, and the coupling knowledge states it as such. A VECTOR interface
has a rho PER COMPONENT — the two subdomains' stiffness ratio is not the same
number for the normal and the tangential direction unless they share a Poisson
ratio, because M/mu = 2(1-nu)/(1-2nu) depends on nu alone. So the single theta
has to be chosen from one of them, and the choice is not cosmetic.

The driver's iteration is Jacobi, so its amplification for component c is

    sqrt( (1-theta)^2 + rho_c theta^2 )

which is below one only while theta < 2/(1+rho_c). The LARGEST rho is therefore
the binding one, and 1/(1+rho_max) is not a cautious choice but the only one
that converges: whenever rho_max > 1 + 2 rho_min, picking theta from rho_min
DIVERGES on the other component while the first one settles — a half-converging
coupling whose only symptom in the driver's global residual is "did not
converge", with no indication that one component was fine.

This fixture runs both choices on the same problem and asserts the outcomes:
the wrong theta must NOT converge, the right one must converge AND land on the
closed form. Both are asserted, because a fixture that only showed the good
case would pass with the rule inverted.

T2_MUTATE is not used here — the mutation is the rule itself, declared in
fixture.json as a one-word change from max to min in VectorProblem.theta_opt.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


def _amp(rho: float, theta: float) -> float:
    return math.sqrt((1.0 - theta) ** 2 + rho * theta * theta)


def body() -> None:
    L.require_available("skfem")
    # Two DIFFERENT Poisson ratios, which is what splits the two components'
    # conductance ratios apart. With one nu they collapse to a single number
    # and the question this fixture asks does not arise.
    p = L.VectorProblem(nul=0.48, nur=0.15)
    rx, ry = p.rho("left")
    r_max, r_min = max(rx, ry), min(rx, ry)
    print(f"rho_x={rx:.6f}")
    print(f"rho_y={ry:.6f}")
    print(f"rho_components_differ_by={r_max / r_min:.4g}x")
    L.check(r_max / r_min > 3.0, "rhos_too_close",
            "the two components' conductance ratios are nearly equal here, so "
            "this fixture does not exercise the choice it exists for")
    th_bad, th_good = 1.0 / (1.0 + r_min), p.theta_opt("left")
    print(f"theta_from_smaller_rho={th_bad:.6f}")
    print(f"theta_from_larger_rho={th_good:.6f}")
    # The prediction, before anything runs. A fixture that only reports what
    # happened cannot be wrong; one that predicts first can be.
    print(f"predicted_amp_bad={max(_amp(rx, th_bad), _amp(ry, th_bad)):.4f}")
    print(f"predicted_amp_good={max(_amp(rx, th_good), _amp(ry, th_good)):.4f}")
    L.check(max(_amp(rx, th_bad), _amp(ry, th_bad)) > 1.0,
            "bad_theta_predicted_to_converge",
            "the smaller-rho theta is predicted stable here, so the divergence "
            "this fixture asserts would not be the effect it claims")

    # ── the wrong choice: must NOT converge ────────────────────────────────
    root = L.workroot("theta_bad")
    specs = [L.stage(root, "left", "skfem",
                     L.vector_edits(p, "left", "dirichlet", "right", (16, 16)),
                     kind="vector"),
             L.stage(root, "right", "skfem",
                     L.vector_edits(p, "right", "neumann", "left", (13, 11)),
                     kind="vector")]
    bad = L.pair(specs, max_iter=60, tol=1e-8, accelerator="constant",
                 theta=th_bad)
    print(f"smaller_rho_converged={bool(bad.get('converged'))}")
    print(f"smaller_rho_last_residual={float(bad.get('residual', float('nan'))):.3e}")
    L.check(not bad.get("converged"), "smaller_rho_theta_converged",
            "theta from the SMALLER rho was expected to diverge on the other "
            "component and did not, so the rule this fixture asserts is not "
            "what is being measured")
    # …and it is the OTHER component that runs away, not both. That is the part
    # a global residual hides.
    br = bad.get("block_residuals") or {}
    small = "values[1]" if rx > ry else "values[0]"
    large = "values[0]" if rx > ry else "values[1]"
    got_bad = float(br.get(f"left.{large}", float("nan")))
    got_ok = float(br.get(f"left.{small}", float("nan")))
    print(f"smaller_rho_block_stiff={got_bad:.3e} block_compliant={got_ok:.3e}")
    print(f"smaller_rho_only_one_component_ran_away="
          f"{bool(L.check(got_bad > got_ok, 'both_components_behaved_alike', f'{got_bad:.2e} vs {got_ok:.2e}'))}")

    # ── the right choice: must converge, and be right ──────────────────────
    res = L.vector_arrangement("theta_from_larger_rho", "skfem", "skfem",
                               "left", problem=p, max_iter=400, tol=1e-8,
                               u_atol=1e-9, t_atol=1e-3)
    print(f"larger_rho_iterations={res.get('iterations')}")
    print("relaxation_rule_established=True")


L.main(body)
