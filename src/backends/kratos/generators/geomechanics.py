"""Kratos GeoMechanics generators and knowledge.

Covers soil mechanics, consolidation, groundwater flow, slope stability.
Application: GeoMechanicsApplication.
"""


def _geomechanics_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Geomechanics consolidation/slope analysis."""
    return '''\
"""GeoMechanics analysis — Kratos GeoMechanicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.GeoMechanicsApplication as GMA
    from KratosMultiphysics.GeoMechanicsApplication.geo_mechanics_analysis import GeoMechanicsAnalysis
    print("GeoMechanicsApplication available")
    summary = {"note": "GeoMechanicsApplication available — use ProjectParameters.json + .mdpa workflow",
               "capabilities": ["consolidation", "groundwater_flow", "slope_stability", "excavation"]}
except ImportError:
    print("GeoMechanicsApplication not installed")
    print("Install: pip install KratosGeoMechanicsApplication")
    summary = {"note": "KratosGeoMechanicsApplication not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "geomechanics": {
        "description": "Geomechanics: soil mechanics, consolidation, groundwater flow, slope stability",
        "application": "GeoMechanicsApplication (pip install KratosGeoMechanicsApplication)",
        "elements": {
            "2D": ["UPwSmallStrainElement2D3N", "UPwSmallStrainElement2D4N",
                   "UPwSmallStrainElement2D6N", "UPwSmallStrainElement2D8N",
                   "UPwSmallStrainElement2D9N", "UPwSmallStrainElement2D10N",
                   "UPwSmallStrainElement2D15N"],
            "3D": ["UPwSmallStrainElement3D4N", "UPwSmallStrainElement3D8N",
                   "UPwSmallStrainElement3D10N", "UPwSmallStrainElement3D20N",
                   "UPwSmallStrainElement3D27N"],
            "interface": ["UPwSmallStrainInterfaceElement2D4N", "UPwSmallStrainInterfaceElement3D6N",
                          "UPwSmallStrainInterfaceElement3D8N"],
        },
        # Real registered names from KratosGeoMechanicsApplication
        # 10.4.2 binary scan (libKratosGeoMechanicsCore.so).
        # CAVEAT: ModifiedCamClay and DruckerPrager were in the
        # prior catalog but DO NOT exist as registered laws in
        # GeoMechanicsApplication at all — see pitfall #0.
        "constitutive_laws": [
            "GeoLinearElasticPlaneStrain2DLaw",
            "GeoIncrementalLinearElastic3DLaw",
            "GeoIncrementalLinearElasticInterfaceLaw",
            "LinearElastic2DInterfaceLaw",
            "LinearElastic3DInterfaceLaw",
            "GeoMohrCoulombWithTensionCutOff2D",
            "GeoMohrCoulombWithTensionCutOff3D",
            "SmallStrainUDSM2DPlaneStrainLaw",
            "SmallStrainUDSM3DLaw",
            "SmallStrainUDSM2DInterfaceLaw",
            "SmallStrainUDSM3DInterfaceLaw",
            "TrussBackboneConstitutiveLaw",
        ],
        "solver_types": ["U-Pw (displacement-water pressure coupled)",
                         "Pw (groundwater flow only)", "U (structural only)"],
        "analysis_types": ["consolidation", "groundwater_flow", "slope_stability",
                           "excavation_staged", "dam_safety"],
        "pitfalls": [
                        '[API] Kratos GeoMechanicsApplication 10.4.2 '
                        'has the following CL families (verified via '
                        'binary scan of libKratosGeoMechanicsCore.so): '
                        'Geo-prefixed LinearElastic + Mohr-Coulomb-with-'
                        'tension-cutoff variants, UDSM (user-defined soil '
                        'model) variants, plus 2D/3D Interface laws and '
                        'TrussBackboneConstitutiveLaw. NOTABLY ABSENT: '
                        'no ModifiedCamClay anywhere; no DruckerPrager '
                        'anywhere. The prior catalog listed both as '
                        'available — they were never registered in this '
                        'Application. Real Mohr-Coulomb is '
                        '"GeoMohrCoulombWithTensionCutOff2D" (or 3D), '
                        'NOT plain "MohrCoulomb". Linear elastic is '
                        '"GeoLinearElasticPlaneStrain2DLaw" / '
                        '"GeoIncrementalLinearElastic3DLaw", NOT '
                        '"LinearElastic2DPlaneStrain" / '
                        '"LinearElastic3DLaw". Signal: in '
                        'ProjectParameters.json or MaterialsFile, '
                        'constitutive_law.name = "ModifiedCamClay" '
                        'raises RuntimeError "Trying to add a non '
                        'registered ConstitutiveLaw" at AnalysisStage.'
                        'Initialize; same for DruckerPrager, MohrCoulomb '
                        '(without "WithTensionCutOff2D"), LinearElastic2'
                        'DPlaneStrain, LinearElastic3DLaw. (Verified '
                        'empirically 2026-06-01 — Tier-2 fixture '
                        'geomechanics_cl_naming in scripts/tier2_fixtures'
                        '/kratos/.)',
                        '[Physics] U-Pw elements require both DISPLACEMENT and WATER_PRESSURE DOFs '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                        '[Numerical] Gravity loading via body_force_per_unit_mass: [0, -9.81, 0] '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Initial stress state often needed via K0 procedure '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Time stepping critical for consolidation (geometric progression recommended) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Integration] Material parameters: use effective stress parameters, not total stress '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                    ],
    },
}

GENERATORS = {
    "geomechanics_2d": _geomechanics_2d,
}
