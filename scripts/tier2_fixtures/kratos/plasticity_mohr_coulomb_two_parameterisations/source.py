"""Tier-2: Kratos ships two Mohr-Coulomb laws whose property sets do
not overlap, and only one of them puts the yield surface where the
YIELD_STRESS_* values say.

The claim (kratos.plasticity #3) says Modified MC is parameterised by
YIELD_STRESS_TENSION / YIELD_STRESS_COMPRESSION while Classical MC uses
cohesion and friction angle, and that mixing them puts the yield
surface on the wrong axes. Both halves are checked here, and the second
one is checked against a closed-form reference rather than against a
remembered number: for a Coulomb surface with cohesion c and friction
angle phi the uniaxial intercepts are

    sigma_c = 2 c cos(phi) / (1 - sin(phi))
    sigma_t = 2 c cos(phi) / (1 + sin(phi))

so the two laws can be told apart by asking each one where it actually
yields under an identical strain path, and comparing that with the
number its own parameterisation predicts.

Mutation control: T2_MUTATE=1 hands the MODIFIED-style property set to both Mohr-Coulomb laws, removing the crossed parameterisation that is the pathology, so the classical law is no longer asked for a property set it does not accept and stops naming COHESION.
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

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=both_laws_probed_with_the_modified_style_property_set")

MODIFIED = "SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb"
CLASSICAL = "SmallStrainIsotropicPlasticity3DMohrCoulombMohrCoulomb"

E, NU = 2.0e10, 0.25
PHI_DEG = 30.0
COHESION = 1.0e6
YIELD_T, YIELD_C = 3.0e6, 3.0e7          # Modified MC's parameterisation
YIELD_SCALAR = 3.0e6                     # what Classical MC is handed

# The parameterisation each law is documented to take.
MODIFIED_STYLE = {
    KM.FRACTURE_ENERGY: 1.0e10,
    CLA.HARDENING_CURVE: 3,
    CLA.YIELD_STRESS_TENSION: YIELD_T,
    CLA.YIELD_STRESS_COMPRESSION: YIELD_C,
    CLA.FRICTION_ANGLE: PHI_DEG,
    CLA.DILATANCY_ANGLE: PHI_DEG,
}
CLASSICAL_STYLE = {
    KM.FRACTURE_ENERGY: 1.0e10,
    CLA.HARDENING_CURVE: 3,
    CLA.COHESION: COHESION,
    CLA.FRICTION_ANGLE: PHI_DEG,
    KM.YIELD_STRESS: YIELD_SCALAR,
}


def coulomb_intercepts(cohesion, phi_deg):
    phi = math.radians(phi_deg)
    common = 2.0 * cohesion * math.cos(phi)
    return common / (1.0 - math.sin(phi)), common / (1.0 + math.sin(phi))


def cell(law_name, props):
    model = KM.Model()
    mp = model.CreateModelPart("cell")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                   (0, 0, 1)], 1):
        mp.CreateNewNode(i, float(x), float(y), float(z))
    p = mp.CreateNewProperties(1)
    p.SetValue(KM.YOUNG_MODULUS, E)
    p.SetValue(KM.POISSON_RATIO, NU)
    for key, value in props.items():
        p.SetValue(key, value)
    law = getattr(CLA, law_name)()
    p.SetValue(KM.CONSTITUTIVE_LAW, law)
    el = mp.CreateNewElement("SmallDisplacementElement3D4N", 1,
                             [1, 2, 3, 4], p)
    return mp, p, law, el


def check(law_name, props):
    mp, p, law, el = cell(law_name, props)
    try:
        law.Check(p, el.GetGeometry(), mp.ProcessInfo)
    except Exception as exc:                                  # noqa: BLE001
        return str(exc).strip().splitlines()[0].replace("Error: ", "")
    return None


def yield_intercept(law_name, props, sign, eps_max=0.01, nstep=2000):
    """The axial stress where the response leaves the elastic line."""
    mp, p, law, el = cell(law_name, props)
    geom = el.GetGeometry()
    shape = KM.Vector([0.25] * 4)
    law.InitializeMaterial(p, geom, shape)
    par = KM.ConstitutiveLawParameters()
    par.SetMaterialProperties(p)
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
    for i in range(nstep + 1):
        eps = sign * eps_max * i / nstep
        for j in range(6):
            strain[j] = 0.0
        strain[0] = eps
        strain[1] = -NU * eps
        strain[2] = -NU * eps
        par.SetStrainVector(strain)
        law.CalculateMaterialResponseCauchy(par)
        law.FinalizeMaterialResponseCauchy(par)
        s = par.GetStressVector()
        if i > 2 and abs(s[0] - E * eps) > 1.0e-3 * abs(E * eps):
            return abs(s[0])
    return float("nan")


def close(a, b, tol=0.10):
    return abs(a - b) <= tol * abs(b)


def main() -> int:
    fail: list[str] = []

    # 1. Each law accepts exactly one of the two property sets, and
    #    names the property the other one is missing.
    pairs = {
        ("modified_style", "modified"): (MODIFIED, MODIFIED_STYLE),
        ("modified_style", "classical"): (CLASSICAL, MODIFIED_STYLE),
        ("classical_style", "modified"): (
            MODIFIED, MODIFIED_STYLE if MUTATE else CLASSICAL_STYLE),
        ("classical_style", "classical"): (
            CLASSICAL, MODIFIED_STYLE if MUTATE else CLASSICAL_STYLE),
    }
    reported = {}
    for (style, law_label), (law_name, props) in pairs.items():
        message = check(law_name, props)
        reported[(style, law_label)] = message
        print(f"check[{style}+{law_label}]={message or 'PASS'}")

    matched = (reported[("modified_style", "modified")] is None
               and reported[("classical_style", "classical")] is None)
    crossed_cohesion = "COHESION" in (
        reported[("modified_style", "classical")] or "")
    crossed_dilatancy = "DILATANCY_ANGLE" in (
        reported[("classical_style", "modified")] or "")
    print(f"each_law_accepts_its_own_parameterisation={matched}")
    print(f"classical_rejects_modified_style_on_cohesion="
          f"{crossed_cohesion}")
    print(f"modified_rejects_classical_style_on_dilatancy="
          f"{crossed_dilatancy}")
    if not (matched and crossed_cohesion and crossed_dilatancy):
        fail.append(f"the two Mohr-Coulomb laws did not reject each "
                    f"other's property set by name: {reported}. The "
                    f"claim is that the parameterisations are different, "
                    f"and Check() is where that first shows.")

    # 2. Modified MC yields at the YIELD_STRESS_* values it was given.
    mod_t = yield_intercept(MODIFIED, MODIFIED_STYLE, +1.0)
    mod_c = yield_intercept(MODIFIED, MODIFIED_STYLE, -1.0)
    print(f"modified_tensile_intercept={mod_t:.6e}_given={YIELD_T:.6e}")
    print(f"modified_compressive_intercept={mod_c:.6e}_given="
          f"{YIELD_C:.6e}")
    mod_ok = close(mod_t, YIELD_T) and close(mod_c, YIELD_C)
    print(f"modified_intercepts_match_yield_stress_properties={mod_ok}")
    if not mod_ok:
        fail.append(f"Modified MC did not yield at the YIELD_STRESS_* "
                    f"values it was handed ({mod_t:.3e} vs {YIELD_T:.3e}, "
                    f"{mod_c:.3e} vs {YIELD_C:.3e})")

    # 3. Classical MC ignores YIELD_STRESS and lands on the Coulomb
    #    cohesion/friction intercepts instead.
    sig_c, sig_t = coulomb_intercepts(COHESION, PHI_DEG)
    cls_t = yield_intercept(CLASSICAL, CLASSICAL_STYLE, +1.0)
    cls_c = yield_intercept(CLASSICAL, CLASSICAL_STYLE, -1.0)
    print(f"coulomb_hand_calc_tensile={sig_t:.6e}")
    print(f"coulomb_hand_calc_compressive={sig_c:.6e}")
    print(f"classical_tensile_intercept={cls_t:.6e}")
    print(f"classical_compressive_intercept={cls_c:.6e}")
    follows_c_phi = close(cls_t, sig_t) and close(cls_c, sig_c)
    ignores_ys = not close(cls_t, YIELD_SCALAR)
    print(f"classical_intercepts_follow_cohesion_and_friction="
          f"{follows_c_phi}")
    print(f"classical_ignores_the_yield_stress_property={ignores_ys}")
    if not follows_c_phi:
        fail.append(f"Classical MC's intercepts ({cls_t:.3e}, "
                    f"{cls_c:.3e}) did not follow 2 c cos(phi) / "
                    f"(1 -+ sin(phi)) = ({sig_t:.3e}, {sig_c:.3e}); "
                    f"without that the two laws cannot be told apart by "
                    f"a hand-calc, which is the claim's Signal")
    if not ignores_ys:
        fail.append(f"Classical MC yielded at the YIELD_STRESS it was "
                    f"handed ({cls_t:.3e} vs {YIELD_SCALAR:.3e}); the "
                    f"claim is that this parameterisation is not the one "
                    f"it reads")

    # 4. The ratio is a pure function of phi — the tan(phi)-related
    #    factor the claim's Signal names, stated as a mechanism.
    ratio = cls_c / cls_t
    phi = math.radians(PHI_DEG)
    expected = (1.0 + math.sin(phi)) / (1.0 - math.sin(phi))
    print(f"classical_compression_over_tension={ratio:.6f}_expected="
          f"{expected:.6f}")
    ratio_ok = close(ratio, expected, 0.05)
    print(f"classical_ratio_is_a_function_of_friction_angle={ratio_ok}")
    if not ratio_ok:
        fail.append(f"the classical law's compression/tension ratio "
                    f"{ratio:.4f} is not (1+sin phi)/(1-sin phi) = "
                    f"{expected:.4f}")

    distinct = (matched and crossed_cohesion and crossed_dilatancy
                and mod_ok and follows_c_phi and ignores_ys)
    print(f"two_parameterisations_are_not_interchangeable={distinct}")
    if not distinct:
        fail.append("the two laws were not distinguishable both by the "
                    "properties they demand and by where they put the "
                    "yield surface")

    if not fail:
        print("plasticity_mohr_coulomb_parameterisations_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
