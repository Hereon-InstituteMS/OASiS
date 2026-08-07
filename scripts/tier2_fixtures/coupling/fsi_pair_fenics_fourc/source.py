"""FSI with the structure in 4C: FEniCSx ALE Navier-Stokes + 4C plane-strain wall.

The second cross-code FSI pair, and the one that matters most for the claim,
because 4C is an established multiphysics code with its own input grammar, its
own element library and its own binary. Everything crosses a process boundary
here: the fluid runs under the conda FEniCSx interpreter, the structure is a
plain-Python WRAPPER that writes a 4C YAML deck with an inline QUAD4 mesh, runs
the 4C binary, and reads the VTU back.

WHAT THIS ADDS over fsi_partitioned_two_way, which already runs FEniCSx against
scikit-fem: a THIRD structure discretisation, in a code that shares no library
with either of the other two. If the converged interface displacement is a
property of the problem rather than of one implementation, all three must land
on it. They are compared here against each other, not against a stored number.

WHAT IT DOES NOT ADD: the closed-form anchors, the suppression controls and the
sign-flip limit are all in fsi_partitioned_two_way and are not repeated. This
fixture is about whether 4C can take the structure side at all, and whether the
answer moves when it does.

THE 4C STRUCTURE IS MESH-SENSITIVE AND THAT IS WHY NXS IS LARGE HERE. 4C's WALL
QUAD4 is a bilinear element and shear-locks in bending; on a plate four elements
thick it under-predicts the deflection by several percent, which is an element
property and not a coupling error. The plate mesh is refined until that bias is
below the tolerance this fixture asserts, and the tolerance is stated in terms
of the DIFFERENCE BETWEEN CODES rather than against any stored value.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                        # noqa: E402
import fsilib as F                                             # noqa: E402


def body() -> None:
    L.require_available("fenics", "fourc", "skfem")

    # 4C's QUAD4 plate needs the refinement; scikit-fem's P2 plate does not.
    # Both are the SAME physical plate.
    case_4c = F.FsiCase(nxs=160, nys=16)
    case_sk = F.FsiCase()

    a = F.run_pair("fourc", "fourc", case_4c, with_reference=True,
                   max_iter=60, tol=1e-9)
    ok = F.report_verdict(a, "fourc")
    ok &= F.interface_equilibrium(a, "fourc")
    ok &= F.kinematic_continuity(a, "fourc")
    mono = a.get("monolithic_check") or {}
    rel = float((mono.get("solid") or {}).get("relative_l2", float("nan")))
    print(f"fourc_reference_status={mono.get('status')}")
    print(f"fourc_reference_rel_l2={rel:.3e}")
    L.check(mono.get("status") == "checked", "fourc_reference_ran", str(mono)[:250])
    L.check(rel == rel and rel < 1e-5, "fourc_matches_reference",
            f"relative L2 against the Newton-Krylov root = {rel:.3e}")

    d4 = F.max_dy(a)
    n4 = len(a["exports"]["solid"]["coordinates"])
    nf = len(a["exports"]["fluid"]["coordinates"])
    print(f"fourc_max_dy={d4:.9e}")
    print(f"fourc_n_points_solid={n4} fourc_n_points_fluid={nf}")
    L.check(n4 != nf, "fourc_meshes_nonmatching",
            "the two interface discretisations are identical, so 4C's side of "
            "the mapping was never exercised")
    print("fourc_meshes_nonmatching=True")

    # The interface sensitivity, which is what separates a real two-way run
    # from a fluid that never moved its mesh. Asserted on the number.
    for who in ("fluid", "solid"):
        sv = ((a.get("interface_sensitivity") or {}).get(who) or {})
        print(f"fourc_sensitivity_{who}_S={sv.get('S')} "
              f"noise={sv.get('noise')} signal={sv.get('signal')}")
    sf = ((a.get("interface_sensitivity") or {}).get("fluid") or {}).get("S")
    ss = ((a.get("interface_sensitivity") or {}).get("solid") or {}).get("S")
    L.check(sf is not None and sf > 1e-4, "fourc_fluid_responds",
            f"the fluid's interface sensitivity is {sf}")
    L.check(ss is not None and ss > 1e-4, "fourc_solid_responds",
            f"4C's interface sensitivity is {ss} — its answer does not depend "
            f"on the traction it is handed")

    # ── the comparison, run here rather than stored ────────────────────────
    b = F.run_pair("skfem", "skfem", case_sk, max_iter=60, tol=1e-9)
    ok &= F.report_verdict(b, "skfem")
    ds = F.max_dy(b)
    print(f"skfem_max_dy={ds:.9e}")
    gap = abs(d4 - ds) / ds
    print(f"fourc_vs_skfem_rel={gap:.6f}")
    L.check(gap < 0.01, "fourc_agrees_with_skfem",
            f"the 4C structure and the scikit-fem structure disagree by "
            f"{gap:.3%} on the converged interface deflection under the same "
            f"fluid; at that size one of them is not solving this problem")

    print("pairs_run=2")


L.main(body)
