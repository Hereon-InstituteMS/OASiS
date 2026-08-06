"""FEniCSx <-> scikit-fem: does the pair the sides table claims actually solve
the problem, in both role/position arrangements?

THE CLAIM UNDER TEST (src/tools/coupling_knowledge.py, the sides table plus the
FEniCSx and scikit-fem payloads): FEniCSx can take either side, scikit-fem can
take either side "in either subdomain", the two were coupled to each other on
this install with non-matching interface meshes, and the run CONVERGED.

Why convergence is not the assertion. A partitioned fixed-point iteration
converges to a FIXED POINT. That is the solution only if the two participants
exchange the right quantity, with the right sign, in the right units. Get any of
those wrong and the loop still converges — smoothly, and with a perfect flux
balance. So this fixture checks the physics against the closed form of the split
conduction problem: the interface temperature, the interface flux density, and
that the two sides' fluxes cancel.

Nothing here is pinned. The tolerances are what the physics has to beat; the
errors are printed so the run reports its own numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def one(tag: str, dirichlet: str, mesh_l, mesh_r,
        backend_left: str, backend_right: str, theta: float) -> None:
    """Couple the two backends once, in the given role arrangement."""
    p = L.DEFAULT
    root = L.workroot(tag)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    specs = [
        L.stage(root, "left", backend_left,
                L.heat_edits(p, "left", roles["left"], "right", mesh_l)),
        L.stage(root, "right", backend_right,
                L.heat_edits(p, "right", roles["right"], "left", mesh_r)),
    ]
    res = L.pair(specs, max_iter=200, tol=1e-8,
                 accelerator="constant", theta=theta)

    print(f"--- {tag}: left={backend_left}/{roles['left']} "
          f"right={backend_right}/{roles['right']} theta={theta}")
    if not L.check(bool(res.get("converged")), f"{tag}_did_not_converge",
                   str(res.get("error"))[:300]):
        return
    print(f"{tag}_iterations={res['iterations']}")
    print(f"{tag}_residual={res['residual']:.3e}")

    ex = res["exports"]
    # Non-matching interface discretisation is the normal case for a
    # partitioned coupling, and the claim says so explicitly.
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    L.check(nl != nr, f"{tag}_matching_meshes",
            f"both sides used {nl} interface points, so the claim about "
            f"NON-matching interface meshes was not exercised")

    # (1) the interface temperature, both sides, against the closed form
    for side in ("left", "right"):
        lo, hi = L.span(ex[side]["values"])
        print(f"{tag}_{side}_T_span=[{lo:.9f},{hi:.9f}]")
        L.close(0.5 * (lo + hi), p.t_iface, 1e-3, f"{tag}_{side}_T_err")
        L.check(hi - lo < 1e-3, f"{tag}_{side}_T_not_uniform",
                f"the exact interface temperature has no y-variation, "
                f"got a spread of {hi - lo:.3e}")

    # (2) the interface flux density. The two sides export with respect to
    # their OWN outward normals, which are anti-parallel, so the signs differ.
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = L.span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.9f},{hi:.9f}]")
        L.close(0.5 * (lo + hi), sign * p.q, 5e-3, f"{tag}_{side}_q_err")

    # (3) conservation: what leaves one subdomain enters the other.
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    scale = max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={abs(net_l + net_r) / scale:.3e}")
    L.check(abs(net_l + net_r) / scale < 1e-4, f"{tag}_flux_not_balanced",
            f"net(left)={net_l:.6e} net(right)={net_r:.6e}")
    bal = L.check_balance(res)
    L.check(not bal, f"{tag}_balance_check_complained", "; ".join(bal)[:300])

    # (4) and the driver's own validation block must be empty for a coupling
    # that is right — an empty block is what an agent is told to expect.
    L.check(not res["validation"], f"{tag}_validation_not_empty",
            "; ".join(res["validation"])[:300])


def body() -> None:
    L.require_available("fenics", "skfem")
    # theta = 1/(1+rho) for each arrangement, which is what the knowledge says
    # to compute before running anything.
    p = L.DEFAULT
    one("fenics_D_skfem_N", "left", (16, 16), (14, 12), "fenics", "skfem",
        p.theta_opt("left"))
    one("skfem_D_fenics_N", "left", (16, 16), (14, 12), "skfem", "fenics",
        p.theta_opt("left"))
    # …and with the roles swapped, so FEniCSx is proven on BOTH sides against
    # this partner rather than only in one arrangement.
    one("fenics_N_skfem_D", "right", (16, 16), (14, 12), "fenics", "skfem",
        p.theta_opt("right"))
    print(f"rho_dirichlet_left={p.rho('left'):.6f}")
    print(f"rho_dirichlet_right={p.rho('right'):.6f}")
    print("pairs_run=3")


L.main(body)
