"""Tier-2: EmpiricalSpringElement3D2N returns a force that is right
only when the spring points along global x.

The claim (kratos.cable_net #2) is that ``GlobalizeVector`` does an
in-place ``noalias(v) = prod(trans(T), v)`` — it reads the vector it is
overwriting — so the globalised nodal force is not ``F * ex`` but a
cascade built from ``ex_x``. The consequence a user meets is not an
exception: it is a spring that carries NO force at all when its axis is
global-y or global-z, and a wrong force in both magnitude and direction
when the axis is skew.

What makes this the dangerous half of the family: the LEFT hand side is
built with ``Matrix = prod(...)``, which makes a temporary, so the
tangent is correct at every orientation. A correct tangent with a wrong
internal force is the "wrong stress" failure mode — Newton still
converges quadratically, in FEW iterations, onto the wrong equilibrium.
Watching iteration counts cannot find this.

Every probe here is on the element itself: build one two-node spring
with a linear force law F(dl) = k*dl, stretch it purely along its own
axis by dl, and read CalculateRightHandSide / CalculateLeftHandSide.
The exact axial force is then k*dl regardless of orientation, and the
axial stiffness is k, so both sides have a closed-form reference with
no discretisation error to argue about.

The direction is checked with a SIGNED alignment (the cosine between
the returned force and the element's own unit axis), not with a norm:
a magnitude test alone cannot tell a rotated force from a correct one,
and the skew cases are exactly where the magnitude is least wrong.

Mutation control: T2_MUTATE=1 replaces the element's own CalculateRightHandSide force with a CORRECT reference force, F = K_SPRING * STRETCH directed along the spring's unit axis, leaving the model, the stretch and the stiffness probe untouched. That removes the aliasing defect the fixture exists to detect: the y- and z-aligned springs then carry a real force instead of exactly zero, and the skew springs point along their own axis. It also shows the assertions respond to what Kratos actually returns rather than to the fixture's own arithmetic.
"""
from __future__ import annotations

import math
import os
import sys

sys.excepthook = sys.__excepthook__
os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA   # noqa: F401
import KratosMultiphysics.CableNetApplication as CN

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=internal_force_taken_from_a_correct_reference_instead_of_the_element_rhs")

K_SPRING = 1000.0        # F(dl) = K_SPRING * dl  (poly1d, highest first)
STRETCH = 0.01           # pure axial elongation applied to node 2
EXACT_F = K_SPRING * STRETCH


def probe(axis):
    """Stretch a spring along `axis` by STRETCH and read both sides."""
    model = KM.Model()
    mp = model.CreateModelPart("spring")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for var in (KM.DISPLACEMENT, KM.REACTION, KM.VELOCITY, KM.ACCELERATION):
        mp.AddNodalSolutionStepVariable(var)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    n2 = mp.CreateNewNode(2, float(axis[0]), float(axis[1]), float(axis[2]))
    for node in mp.Nodes:
        node.AddDof(KM.DISPLACEMENT_X)
        node.AddDof(KM.DISPLACEMENT_Y)
        node.AddDof(KM.DISPLACEMENT_Z)
    props = mp.CreateNewProperties(1)
    props.SetValue(CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL,
                   KM.Vector([K_SPRING, 0.0]))
    el = mp.CreateNewElement("EmpiricalSpringElement3D2N", 1, [1, 2], props)
    el.Initialize(mp.ProcessInfo)

    length = math.sqrt(sum(a * a for a in axis))
    unit = [a / length for a in axis]
    disp = [STRETCH * e for e in unit]
    n2.SetSolutionStepValue(KM.DISPLACEMENT, list(disp))
    n2.X = axis[0] + disp[0]
    n2.Y = axis[1] + disp[1]
    n2.Z = axis[2] + disp[2]

    rhs = KM.Vector(6)
    el.CalculateRightHandSide(rhs, mp.ProcessInfo)
    lhs = KM.Matrix(6, 6)
    el.CalculateLeftHandSide(lhs, mp.ProcessInfo)

    if MUTATE:
        # Defect removed: a correct internal force, k*dl along the axis.
        force = [EXACT_F * e for e in unit]
    else:
        force = [rhs[i] for i in range(3)]
    magnitude = math.sqrt(sum(f * f for f in force))
    dotted = sum(f * e for f, e in zip(force, unit))
    alignment = dotted / magnitude if magnitude > 0.0 else float("nan")
    axial_k = sum(unit[i] * sum(lhs[i, j] * unit[j] for j in range(3))
                  for i in range(3))
    return magnitude, alignment, axial_k


def main() -> int:
    fail: list[str] = []
    cases = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "xy": (1.0, 1.0, 0.0),
        "xyz": (1.0, 1.0, 1.0),
    }
    measured = {}
    for name, axis in cases.items():
        magnitude, alignment, axial_k = probe(axis)
        measured[name] = (magnitude, alignment, axial_k)
        print(f"axis[{name}]_force_magnitude={magnitude:.6e}_exact="
              f"{EXACT_F:.6e}")
        print(f"axis[{name}]_force_alignment_with_axis={alignment:.6f}")
        print(f"axis[{name}]_axial_stiffness={axial_k:.6e}_exact="
              f"{K_SPRING:.6e}")

    # 1. Only the x-aligned spring returns the exact force, and it
    #    returns it along its own axis.
    mag_x, align_x, _ = measured["x"]
    ok_x = (abs(mag_x - EXACT_F) < 1e-9 * EXACT_F and abs(align_x - 1.0) < 1e-9)
    print(f"x_aligned_spring_is_exact={ok_x}")
    if not ok_x:
        fail.append(f"the x-aligned spring did not return F = k*dl along "
                    f"its own axis ({mag_x:.6e} vs {EXACT_F:.6e}, "
                    f"alignment {align_x:.6f}); the claim is that this one "
                    f"orientation IS correct, so the aliasing story does "
                    f"not describe this build")

    # 2. y- and z-aligned springs carry EXACTLY no force. Not small —
    #    zero, because every predicted component carries a factor ex_x.
    for name in ("y", "z"):
        mag, _, _ = measured[name]
        silent = mag == 0.0
        print(f"{name}_aligned_spring_force_is_exactly_zero={silent}")
        if not silent:
            fail.append(f"the {name}-aligned spring returned a force of "
                        f"{mag:.6e}; the claim is that it returns exactly "
                        f"zero while carrying a real elongation")

    # 3. The skew springs return a force that is neither the right size
    #    nor the right direction. Alignment is the load-bearing check:
    #    a norm alone cannot see a rotated force.
    for name in ("xy", "xyz"):
        mag, align, _ = measured[name]
        wrong_dir = align < 0.999
        wrong_mag = abs(mag - EXACT_F) > 1e-6 * EXACT_F
        print(f"{name}_skew_force_is_misdirected={wrong_dir}")
        print(f"{name}_skew_force_magnitude_is_wrong={wrong_mag}")
        if not (wrong_dir and wrong_mag):
            fail.append(f"the {name}-skew spring returned a force with "
                        f"alignment {align:.6f} and magnitude {mag:.6e} "
                        f"against the exact {EXACT_F:.6e}; the claim is "
                        f"that a skewed spring is wrong in BOTH")

    # 4. The tangent is correct at every orientation — that is why
    #    nothing crashes and why iteration counts cannot find this.
    tangent_ok = all(abs(k - K_SPRING) < 1e-9 * K_SPRING
                     for _, _, k in measured.values())
    print(f"tangent_is_correct_at_every_orientation={tangent_ok}")
    if not tangent_ok:
        fail.append("the axial stiffness was not k at every orientation; "
                    "the claim is that only the RHS is aliased and the "
                    "LHS is untouched, which is what makes Newton "
                    "converge onto a wrong answer")

    # 5. The pair of 4 and 2/3 together IS the pathology: right tangent,
    #    wrong force. State it as one line so a mutation that fixes
    #    either half flips it.
    silent_wrong = (
        tangent_ok
        and measured["y"][0] == 0.0
        and measured["z"][0] == 0.0
        and measured["xyz"][1] < 0.999
        and abs(measured["x"][0] - EXACT_F) < 1e-9 * EXACT_F
    )
    print(f"right_tangent_with_wrong_internal_force={silent_wrong}")
    if not silent_wrong:
        fail.append("the combination the claim describes — correct "
                    "tangent, force correct only along global x — was "
                    "not observed")

    if not fail:
        print("cablenet_empirical_spring_rhs_aliasing_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
