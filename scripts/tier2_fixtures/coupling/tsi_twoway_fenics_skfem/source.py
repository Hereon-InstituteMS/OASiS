"""TWO-WAY TSI ACROSS TWO CODES: FEniCSx and scikit-fem, both arrangements.

THE CLAIM UNDER TEST: the two-way thermo-structural coupling is not a property
of one code talking to itself. Either code can take either half — the energy
equation with the mechanical feedback term, or the elasticity with the thermal
stress — and the coupled answer still lands on the un-split solution.

BOTH ARRANGEMENTS, because they are different couplings and not a relabelling.
The two halves are not symmetric: the thermal side carries the reverse
direction's source term and the Dirichlet data, the structural side carries the
thermal stress and the strain projection, and the two codes discretise those
differently (dolfinx builds the strain by an L2 projection out of a vector-P2
space, scikit-fem out of its own). A pair that works one way round and not the
other has a bug in the half that moved.

WHAT IS BEING EXCHANGED, and why it is not an interface: both participants own
the WHOLE body. The thermal one exports the temperature change at its nodes and
the structural one exports the volumetric strain at its nodes, on meshes that do
not match. There is no interface, no normal, and nothing to balance — so
`couple`'s flux-conservation checks report themselves as not-run, and the
verification rests on the monolithic comparison and on the direction controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                          # noqa: E402
import tsilib as T                                               # noqa: E402
import numpy as np                                               # noqa: E402


def body() -> None:
    L.require_available("fenics", "skfem")

    a = T.full_pair_check("fenics_T_skfem_M", "fenics", "skfem")
    b = T.full_pair_check("skfem_T_fenics_M", "skfem", "fenics")

    # The two arrangements solve the SAME problem with the codes swapped, so
    # their answers must agree with each other as well as with the reference.
    # This catches a bias that both arrangements share with the reference but
    # not with each other — nothing else here would.
    if "theta_field" in a and "theta_field" in b:
        at = T._interp(a["theta_coords"], a["theta_field"], b["theta_coords"])
        d = float(np.linalg.norm(b["theta_field"] - at)) / \
            max(float(np.linalg.norm(at)), 1e-30)
        print(f"arrangements_agree_relL2={d:.3e}")
        L.check(d < 5e-3, "arrangements_disagree",
                f"swapping which code takes which half changed the converged "
                f"temperature field by {d:.3e} relative")
        print(f"arrangements_agree={bool(d < 5e-3)}")

    print("pairs_run=2")


L.main(body)
