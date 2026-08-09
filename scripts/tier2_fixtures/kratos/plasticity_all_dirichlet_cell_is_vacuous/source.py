"""Tier-2: a single-element plasticity test with every DOF prescribed
converges in one Newton iteration per step and proves nothing.

The claim (kratos.plasticity #8) is that a fully displacement-controlled
cell "can pass spuriously" because Newton never has a free DOF to
correct, so the algorithmic tangent is never exercised — and that at
least one Neumann-loaded face is needed before the iteration count
means anything.

The same hex8 cell, the same material and the same axial path are run
twice: once with all 24 DOFs fixed, once with the lateral DOFs of the
loaded face free and nodal loads applied to it. The number reported is
ProcessInfo[NL_ITERATION_NUMBER] after each step.

To keep the first case from being dismissed as "it converged in one
step because it was still elastic", the fixture also reads the
element's plastic state back through CalculateOnIntegrationPoints at
the end of the path. The cell is deep in the plastic range and still
reports one iteration — that is what makes the pass vacuous rather
than merely easy.

Mutation control: T2_MUTATE=1 runs the supposedly all-Dirichlet cell with the Neumann-loaded face and free lateral DOFs, i.e. it removes the fully displacement-controlled setup that makes the one-iteration convergence vacuous. Real iterations are then needed and the one-iteration-every-step claim collapses.
"""
from __future__ import annotations

import os
import sys

sys.excepthook = sys.__excepthook__
os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication as CLA
from KratosMultiphysics import python_linear_solver_factory as LSF

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=all_dirichlet_cell_given_a_neumann_loaded_face")

E, NU, SY = 2.0e11, 0.3, 2.5e8
UNIT_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
BOTTOM, TOP = (1, 2, 3, 4), (5, 6, 7, 8)
NSTEPS, AXIAL, LATERAL = 6, -0.004, -2.0e6


def build(with_neumann):
    model = KM.Model()
    mp = model.CreateModelPart("Structure", 2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for var in (KM.DISPLACEMENT, KM.REACTION, SMA.POINT_LOAD):
        mp.AddNodalSolutionStepVariable(var)
    for i, (x, y, z) in enumerate(UNIT_CUBE, 1):
        mp.CreateNewNode(i, float(x), float(y), float(z))
    props = mp.CreateNewProperties(1)
    props.SetValue(KM.YOUNG_MODULUS, E)
    props.SetValue(KM.POISSON_RATIO, NU)
    props.SetValue(CLA.YIELD_STRESS_TENSION, SY)
    props.SetValue(CLA.YIELD_STRESS_COMPRESSION, SY)
    props.SetValue(KM.FRACTURE_ENERGY, 1.0e10)
    props.SetValue(CLA.HARDENING_CURVE, 3)
    props.SetValue(KM.CONSTITUTIVE_LAW,
                   CLA.SmallStrainIsotropicPlasticity3DVonMisesVonMises())
    mp.CreateNewElement("SmallDisplacementElement3D8N", 1,
                        list(range(1, 9)), props)
    for node in mp.Nodes:
        node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
        node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
        node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)

    if with_neumann:
        for i in BOTTOM:
            node = mp.GetNode(i)
            node.Fix(KM.DISPLACEMENT_X)
            node.Fix(KM.DISPLACEMENT_Y)
            node.Fix(KM.DISPLACEMENT_Z)
        for i in TOP:                       # axial control only
            mp.GetNode(i).Fix(KM.DISPLACEMENT_Z)
        load_props = mp.CreateNewProperties(2)
        for k, i in enumerate(TOP, 1):
            mp.CreateNewCondition("PointLoadCondition3D1N", k, [i],
                                  load_props)
    else:
        for node in mp.Nodes:               # every DOF prescribed
            node.Fix(KM.DISPLACEMENT_X)
            node.Fix(KM.DISPLACEMENT_Y)
            node.Fix(KM.DISPLACEMENT_Z)
    return model, mp


def newton(mp):
    solver = LSF.ConstructSolver(
        KM.Parameters('{"solver_type": "skyline_lu_factorization"}'))
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    criterion = KM.ResidualCriteria(1e-8, 1e-10)
    criterion.SetEchoLevel(0)
    builder = KM.ResidualBasedBlockBuilderAndSolver(solver)
    strategy = KM.ResidualBasedNewtonRaphsonStrategy(
        mp, scheme, criterion, builder, 30, False, False, False)
    strategy.SetEchoLevel(0)
    strategy.Initialize()
    return strategy


def run(with_neumann):
    _model, mp = build(with_neumann)
    strategy = newton(mp)
    counts = []
    for step in range(1, NSTEPS + 1):
        mp.CloneTimeStep(float(step))
        mp.ProcessInfo[KM.STEP] = step
        frac = step / float(NSTEPS)
        for i in TOP:
            mp.GetNode(i).SetSolutionStepValue(KM.DISPLACEMENT_Z,
                                               AXIAL * frac)
            if with_neumann:
                mp.GetNode(i).SetSolutionStepValue(
                    SMA.POINT_LOAD,
                    [LATERAL * frac * 0.25, LATERAL * frac * 0.25, 0.0])
        strategy.Solve()
        counts.append(mp.ProcessInfo[KM.NL_ITERATION_NUMBER])
    el = mp.GetElement(1)
    plastic = el.CalculateOnIntegrationPoints(CLA.EQUIVALENT_PLASTIC_STRAIN,
                                              mp.ProcessInfo)
    return counts, max(plastic)


def main() -> int:
    fail: list[str] = []

    fixed_counts, fixed_plastic = run(bool(MUTATE))
    free_counts, free_plastic = run(True)
    print(f"all_dirichlet_iteration_counts="
          f"{','.join(str(c) for c in fixed_counts)}")
    print(f"all_dirichlet_max_equivalent_plastic_strain="
          f"{fixed_plastic:.6e}")
    print(f"with_neumann_iteration_counts="
          f"{','.join(str(c) for c in free_counts)}")
    print(f"with_neumann_max_equivalent_plastic_strain="
          f"{free_plastic:.6e}")

    # 1. One iteration per step, every step.
    always_one = all(c == 1 for c in fixed_counts)
    print(f"all_dirichlet_is_one_iteration_every_step={always_one}")
    if not always_one:
        fail.append(f"the fully displacement-controlled cell took "
                    f"{fixed_counts} iterations; the claim is that it "
                    f"reports 1 per step because there is no free DOF to "
                    f"correct")

    # 2. And it is genuinely plastic while doing so — otherwise "one
    #    iteration" would just mean "still linear".
    yielded = fixed_plastic > 0.0
    print(f"all_dirichlet_cell_actually_yielded={yielded}")
    if not yielded:
        fail.append(f"the fully displacement-controlled cell never left "
                    f"the elastic range (max equivalent plastic strain "
                    f"{fixed_plastic:.3e}), so its one-iteration "
                    f"convergence would be honest rather than vacuous "
                    f"and this fixture would not be testing the claim")

    # 3. The same material on the same cell needs real iterations once
    #    one face is Neumann-loaded and the lateral DOFs are free.
    exercised = max(free_counts) > 1
    print(f"with_neumann_takes_more_than_one_iteration={exercised}")
    print(f"with_neumann_max_iterations={max(free_counts)}")
    if not exercised:
        fail.append(f"adding a Neumann-loaded face did not raise the "
                    f"iteration count above 1 ({free_counts}); without "
                    f"that contrast the first measurement says nothing "
                    f"about the tangent being exercised")

    # 4. The claim as one line: same material, same path, and the
    #    iteration count only carries information in the second setup.
    vacuous = always_one and yielded and exercised
    print(f"displacement_only_cell_passes_without_testing_the_tangent="
          f"{vacuous}")
    if not vacuous:
        fail.append("the contrast the claim rests on was not observed")

    if not fail:
        print("plasticity_all_dirichlet_cell_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
