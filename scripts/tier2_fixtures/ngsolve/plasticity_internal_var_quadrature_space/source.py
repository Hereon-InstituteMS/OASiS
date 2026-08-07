"""Tier-2: L2(mesh, order=2*disp_order-1) does NOT have one dof per quadrature
point -- IntegrationRuleSpace does, and it lives only in ngsolve.comp.

Claim: ngsolve plasticity#3 -- internal variables eps_p and alpha must be stored
on a space sized at the quadrature points of the displacement space, "NOT on the
H1 nodes.  Use L2(mesh, order=2*disp_order-1) or the IntegrationRule machinery".
The stated signal is: "the L2 ndof matches num_elements * num_quad_points; if it
does not, the storage layout will not align."

Wrong variant: take the recommended L2 space at disp_order = 2, i.e.
L2(mesh, order=3), and compare its ndof against mesh.ne * (number of quadrature
points of the collocated rule) as the claim instructs.

Observed on NGSolve 6.2.2604 (unit_square maxh 0.5, 6 triangles):
  * L2(order=3) has 10 dofs per element  (dim P_3 on a triangle),
    IntegrationRule(TRIG, 3) has 6 points and the rule that the actual
    quadrature-collocated space uses has 12 -- so the claimed identity
    ndof == ne * nip is FALSE for every order tested (1, 2, 3, 4, 5).
  * ngsolve.comp.IntegrationRuleSpace(mesh, order=k) DOES satisfy it exactly:
    ndof == ne * len(irs.GetIntegrationRules()[ET.TRIG]) for k = 0..5.
  * IntegrationRuleSpace is NOT reachable as ngsolve.IntegrationRuleSpace and
    NOT exported by `from ngsolve import *`; it must be imported from
    ngsolve.comp.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- the internal-variable space under test is built as
IntegrationRuleSpace(mesh, order=IVAR_ORDER) instead of L2(mesh, order=IVAR_ORDER).
It then has 12 dofs per element and ndof == ne*nip holds, so the fixture goes red
on l2_dofs_per_element=10 and l2_ndof_equals_ne_times_quadpoints=False.
Re-run: T2_MUTATE=1 python source.py
"""

from __future__ import annotations

import os
import sys

import ngsolve
import ngsolve.fem as ngfem
from ngsolve import TRIG, H1, L2, Mesh, VectorH1, unit_square
from ngsolve.comp import IntegrationRuleSpace

MUTATE = os.environ.get("T2_MUTATE") == "1"

DISP_ORDER = 2
# The claim's recommended internal-variable space order.
IVAR_ORDER = 2 * DISP_ORDER - 1


def main() -> int:
    ok = True

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    ne = mesh.ne
    print(f"n_elements={ne}")

    disp = VectorH1(mesh, order=DISP_ORDER)
    print(f"disp_space_type={type(disp).__name__} h1_type={type(H1(mesh, order=1)).__name__}")

    # --- API inventory: where does the quadrature space actually live? ------
    ns: dict = {}
    exec("from ngsolve import *", ns)  # noqa: S102 - deliberate inventory probe
    print(f"top_level_IntegrationRuleSpace={hasattr(ngsolve, 'IntegrationRuleSpace')}")
    print(f"star_import_IntegrationRuleSpace={'IntegrationRuleSpace' in ns}")
    print(f"comp_IntegrationRuleSpace_type={IntegrationRuleSpace.__name__}")

    # --- WRONG variant: the L2 sizing the claim recommends ------------------
    # T2_MUTATE=1 swaps in the fix (IntegrationRuleSpace) at this site.
    l2 = (IntegrationRuleSpace(mesh, order=IVAR_ORDER) if MUTATE
          else L2(mesh, order=IVAR_ORDER))
    ir_claim = ngfem.IntegrationRule(TRIG, IVAR_ORDER)
    irs = IntegrationRuleSpace(mesh, order=IVAR_ORDER)
    ir_real = irs.GetIntegrationRules()[TRIG]
    print(f"l2_order={IVAR_ORDER} l2_ndof={l2.ndof} l2_dofs_per_element={l2.ndof // ne}")
    print(f"integrationrule_trig_order{IVAR_ORDER}_npoints={len(ir_claim)}")
    print(f"collocated_rule_npoints={len(ir_real)}")
    l2_matches_claim = (l2.ndof == ne * len(ir_claim))
    l2_matches_real = (l2.ndof == ne * len(ir_real))
    print(f"l2_ndof_equals_ne_times_quadpoints={l2_matches_claim or l2_matches_real}")
    if l2_matches_claim or l2_matches_real:
        print("FAIL: L2 ndof now does equal ne*nip -- the falsification "
              "recorded in this fixture no longer holds", file=sys.stderr)
        ok = False

    # the mismatch is systematic, not an artefact of one order
    mismatched = []
    for k in range(1, 6):
        n_l2 = L2(mesh, order=k).ndof
        n_ir = len(ngfem.IntegrationRule(TRIG, k))
        if n_l2 != ne * n_ir:
            mismatched.append(k)
    print(f"orders_where_l2_ndof_mismatches_quadpoints={mismatched}")
    if mismatched != [1, 2, 3, 4, 5]:
        print(f"FAIL: expected the L2/quadrature mismatch at every order 1..5, "
              f"got {mismatched}", file=sys.stderr)
        ok = False

    # --- RIGHT variant: IntegrationRuleSpace is dof-per-quadrature-point ----
    aligned = []
    for k in range(0, 6):
        space = IntegrationRuleSpace(mesh, order=k)
        rule = space.GetIntegrationRules()[TRIG]
        if space.ndof == ne * len(rule):
            aligned.append(k)
    print(f"irs_ndof_equals_ne_times_own_rule_for_orders={aligned}")
    print(f"irs_aligned_all_orders_0_to_5={aligned == [0, 1, 2, 3, 4, 5]}")
    if aligned != [0, 1, 2, 3, 4, 5]:
        print(f"FAIL: IntegrationRuleSpace no longer allocates one dof per "
              f"quadrature point (aligned={aligned})", file=sys.stderr)
        ok = False

    # and it is genuinely element-local storage, like L2
    print(f"irs_type={type(irs).__name__}")
    print(f"irs_dofs_per_element={irs.ndof // ne}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
