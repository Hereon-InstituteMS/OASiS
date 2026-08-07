"""TWO-WAY TSI, same code on both halves — the control the cross-code pairs are
read against.

THE CLAIM UNDER TEST: OASiS can run a genuinely two-way thermo-structural
coupling through the generic `couple` path — thermal expansion drives the
deformation AND the deformation feeds back into the energy equation — and the
converged answer is the answer, not just A fixed point.

Both halves are scikit-fem here on purpose. A cross-code pair that disagrees
with the monolithic reference leaves two suspects: the coupling, and the second
code. This one removes the second, so it is the fixture that says whether the
COUPLING is right. The cross-code pairs then say whether it survives crossing a
code boundary.

WHAT IT RUNS, in order:
  * the exaggerated-coupling problem (delta ~ 1.25), where the un-relaxed
    iteration diverges and the reverse direction moves the answer by ~10%;
  * the same problem with the reverse direction switched off THROUGH THE
    COUPLING GRAPH (`imports_from=[]` on the thermal side), checked against the
    one-way monolithic reference;
  * the same suppression through the participant's own `COUPLING` switch, to
    check the two routes agree;
  * a real metal's coupling strength (delta ~ 0.012), where the reverse
    direction is a fraction of a percent — small, but still four orders of
    magnitude above the agreement tolerance, and the fixture reports the
    margin rather than asserting the effect is "large".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                          # noqa: E402
import tsilib as T                                               # noqa: E402
import numpy as np                                               # noqa: E402


def body() -> None:
    L.require_available("skfem")

    # ── the reference solver is itself checked, before anything is graded
    # against it: with no y-variation the body is in uniaxial strain and the
    # two-way solve at rho_c must equal the one-way solve at rho_c*(1+delta).
    for name, p in (("steel", T.STEEL), ("strong", T.STRONG)):
        dev, size = T.tsi_identity(p)
        print(f"reference_{name}_delta={p.delta:.6f}")
        print(f"reference_{name}_effective_capacity_identity_dev={dev:.3e}")
        print(f"reference_{name}_reverse_direction_size={size:.3e}")
        L.check(dev < 1e-4 and size > 1e3 * dev,
                f"reference_{name}_identity_failed",
                f"deviation {dev:.3e}, effect under test {size:.3e}")
        print(f"reference_{name}_identity_holds="
              f"{bool(dev < 1e-4 and size > 1e3 * dev)}")

    for tag, p, margin in (("strong", T.STRONG, 500.0), ("steel", T.STEEL, 50.0)):
        print(f"--- {tag}: delta={p.delta:.6f} theta_opt={T.theta_opt(p):.6f} "
              f"amplification(theta=1)={T.amplification(p, 1.0):.4f}")
        two = T.run_tsi(f"{tag}_2way", "skfem", "skfem", p=p)
        if not T.assert_run_clean(f"{tag}_2way", two):
            continue
        errs = T.compare_to_monolithic(f"{tag}_2way", two, coupling=1.0)

        # the reverse direction switched off THROUGH THE COUPLING GRAPH
        one = T.run_tsi(f"{tag}_1way", "skfem", "skfem", p=p,
                        thermal_reads=False)
        T.assert_run_clean(f"{tag}_1way", one, expect_one_way=True)
        T.compare_to_monolithic(f"{tag}_1way", one, coupling=0.0)
        T.reverse_direction_is_active(tag, two, one,
                                      agreement=max(errs.values()),
                                      margin=margin)

        # ... and through the participant's own physics switch. Same answer, or
        # one of the two routes is not doing what it says.
        off = T.run_tsi(f"{tag}_off", "skfem", "skfem", p=p, coupling=0.0,
                        with_tool_monolithic=False, quiet=True)
        d = float(np.max(np.abs(off["theta_field"] - one["theta_field"]))) / \
            max(float(np.max(np.abs(one["theta_field"]))), 1e-30)
        print(f"{tag}_graph_vs_switch_suppression_rel={d:.3e}")
        L.check(d < 1e-9, f"{tag}_suppression_routes_disagree",
                f"switching the reverse direction off through `imports_from` and "
                f"through COUPLING=0.0 gave answers {d:.3e} apart")

    print("pairs_run=1")


L.main(body)
