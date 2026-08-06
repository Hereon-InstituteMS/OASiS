"""Tier-2: HARDENING_CURVE=0 never becomes perfectly plastic, however
large FRACTURE_ENERGY is made; HARDENING_CURVE=3 is perfectly plastic
at every FRACTURE_ENERGY.

The claim (kratos.plasticity #4) advises curve 3 with a large fracture
energy and warns that curve 0 "still softens unless FRACTURE_ENERGY is
very large". Measured, the second half is stronger than it reads: the
droop does not stop at some large value, it decays as 1/FRACTURE_ENERGY
and stays strictly negative. There is no number that switches it off.

The law is driven directly — no mesh, no solver — through a monotonic
strain path, and the observable is the von Mises equivalent stress, not
sigma_xx. That distinction is the whole reason a naive version of this
test reports "no softening at all": under a prescribed lateral strain
the axial stress keeps climbing at the elastic BULK slope after yield
because plastic flow is incompressible and the imposed lateral strain
is not, so sigma_xx rises with pressure while the deviator is capped.
Measuring sigma_xx therefore hides the softening completely.
"""
from __future__ import annotations

import math
import os
import sys

sys.excepthook = sys.__excepthook__
os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA    # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

LAW = "SmallStrainIsotropicPlasticity3DVonMisesVonMises"
E, NU, SY = 2.0e11, 0.3, 2.5e8
EPS_MAX, NSTEP = 0.02, 200
# Decades of fracture energy, all above the value the law itself
# rejects as "too low" on this material.
GF_DECADES = (1.0e7, 1.0e8, 1.0e9, 1.0e10)


def sweep(hardening_curve, fracture_energy):
    """Return the (strain, von Mises stress) curve for one setting."""
    model = KM.Model()
    mp = model.CreateModelPart("cell")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                   (0, 0, 1)], 1):
        mp.CreateNewNode(i, float(x), float(y), float(z))
    props = mp.CreateNewProperties(1)
    props.SetValue(KM.YOUNG_MODULUS, E)
    props.SetValue(KM.POISSON_RATIO, NU)
    props.SetValue(CLA.YIELD_STRESS_TENSION, SY)
    props.SetValue(CLA.YIELD_STRESS_COMPRESSION, SY)
    props.SetValue(KM.FRACTURE_ENERGY, fracture_energy)
    props.SetValue(CLA.HARDENING_CURVE, hardening_curve)
    law = getattr(CLA, LAW)()
    props.SetValue(KM.CONSTITUTIVE_LAW, law)
    el = mp.CreateNewElement("SmallDisplacementElement3D4N", 1,
                             [1, 2, 3, 4], props)

    geom = el.GetGeometry()
    shape = KM.Vector([0.25] * 4)
    law.InitializeMaterial(props, geom, shape)
    par = KM.ConstitutiveLawParameters()
    par.SetMaterialProperties(props)
    par.SetElementGeometry(geom)
    par.SetProcessInfo(mp.ProcessInfo)
    flags = KM.Flags()
    flags.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)
    flags.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, True)
    flags.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, True)
    par.SetOptions(flags)
    par.SetShapeFunctionsValues(shape)
    par.SetDeterminantF(1.0)
    par.SetDeformationGradientF(KM.Matrix(3, 3, 0.0))
    strain, stress, tangent = KM.Vector(6), KM.Vector(6), KM.Matrix(6, 6)
    par.SetStrainVector(strain)
    par.SetStressVector(stress)
    par.SetConstitutiveMatrix(tangent)

    curve = []
    for i in range(NSTEP + 1):
        eps = EPS_MAX * i / NSTEP
        for j in range(6):
            strain[j] = 0.0
        strain[0] = eps
        strain[1] = -NU * eps
        strain[2] = -NU * eps
        par.SetStrainVector(strain)
        law.CalculateMaterialResponseCauchy(par)
        law.FinalizeMaterialResponseCauchy(par)
        s = par.GetStressVector()
        mean = (s[0] + s[1] + s[2]) / 3.0
        dev = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]]
        vm = math.sqrt(1.5 * (dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2)
                       + 3.0 * (dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2))
        curve.append((eps, vm))
    return curve


def post_yield_slope(curve):
    """d(vm)/d(eps) over the last half of the path — entirely plastic."""
    half = len(curve) // 2
    return (curve[-1][1] - curve[half][1]) / (curve[-1][0] - curve[half][0])


def main() -> int:
    fail: list[str] = []

    # The claim's own wording ("despite HARDENING_MODULUS=0") names a
    # variable that does not exist here — record it, do not assume it.
    has_hm = (hasattr(CLA, "HARDENING_MODULUS")
              or hasattr(KM, "HARDENING_MODULUS"))
    print(f"hardening_modulus_variable_exists={has_hm}")

    slopes = {}
    peaks = {}
    for curve_id in (0, 3):
        for gf in GF_DECADES:
            data = sweep(curve_id, gf)
            slope = post_yield_slope(data)
            peak = max(v for _, v in data)
            slopes[(curve_id, gf)] = slope
            peaks[(curve_id, gf)] = peak
            print(f"curve{curve_id}_gf{gf:.0e}_peak_vm={peak:.6e} "
                  f"post_yield_slope={slope:.6e}")

    # 1. Curve 3 is perfectly plastic: the deviator caps at the yield
    #    stress and stays there, at EVERY fracture energy.
    flat = all(abs(slopes[(3, gf)]) < 1.0e-6 * E for gf in GF_DECADES)
    capped = all(abs(peaks[(3, gf)] - SY) < 1.0e-6 * SY for gf in GF_DECADES)
    print(f"curve3_is_flat_at_every_fracture_energy={flat}")
    print(f"curve3_plateau_equals_yield_stress={capped}")
    if not (flat and capped):
        fail.append("HARDENING_CURVE=3 was not perfectly plastic at every "
                    "fracture energy tested; the claim recommends it "
                    "precisely because it is")

    # 2. Curve 0 droops at every fracture energy, including the largest.
    droops = {gf: slopes[(0, gf)] < 0.0 for gf in GF_DECADES}
    all_droop = all(droops.values())
    for gf in GF_DECADES:
        print(f"curve0_gf{gf:.0e}_slope_is_negative={droops[gf]}")
    print(f"curve0_droops_at_every_fracture_energy={all_droop}")
    if not all_droop:
        fail.append(f"HARDENING_CURVE=0 did not soften at every fracture "
                    f"energy: {slopes}. The claim is that it softens "
                    f"unless the fracture energy is very large; measured, "
                    f"it softens at 1e10 too.")

    # 3. The MECHANISM, not a threshold: the droop decays like
    #    1/FRACTURE_ENERGY, one decade of slope per decade of Gf, so no
    #    finite value switches it off.
    ratios = []
    for a, b in zip(GF_DECADES, GF_DECADES[1:]):
        ratios.append(abs(slopes[(0, a)]) / abs(slopes[(0, b)]))
    per_decade = all(5.0 < r < 20.0 for r in ratios)
    print(f"curve0_slope_ratio_per_decade="
          f"{','.join(f'{r:.3f}' for r in ratios)}")
    print(f"curve0_softening_decays_as_one_over_fracture_energy="
          f"{per_decade}")
    if not per_decade:
        fail.append(f"the curve-0 softening slope did not fall by about a "
                    f"decade per decade of fracture energy (ratios "
                    f"{ratios}); without that the claim reduces to a "
                    f"threshold, and a threshold would have a value")

    # 4. The single statement the advice rests on.
    only_three = flat and all_droop
    print(f"only_curve_three_gives_perfect_plasticity={only_three}")
    if not only_three:
        fail.append("curve 0 and curve 3 were not distinguishable on this "
                    "material, so the advice to use curve 3 has no "
                    "operational content")

    if not fail:
        print("plasticity_perfect_plasticity_gate_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
