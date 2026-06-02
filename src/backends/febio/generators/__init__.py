"""FEBio generator registry — maps physics_variant -> generator function.

Mirrors the layout used by skfem, fenics, ngsolve etc.: one file per
FEBio physics module exposes a per-module ``GENERATORS`` + ``KNOWLEDGE``
dict, and this aggregator merges them into the top-level ``GENERATORS``
and ``KNOWLEDGE`` dicts that ``backend.py`` consumes.

When adding a new FEBio physics module (multiphasic, fluid, fluid-FSI,
etc.), drop a new file alongside this one and add it to the imports and
the merge loops below.
"""
from .linear_elasticity import GENERATORS as _le_gen, KNOWLEDGE as _le_kn
from .hyperelasticity import GENERATORS as _he_gen, KNOWLEDGE as _he_kn
from .biphasic import GENERATORS as _bi_gen, KNOWLEDGE as _bi_kn
from .heat import GENERATORS as _ht_gen, KNOWLEDGE as _ht_kn
from .multiphasic import GENERATORS as _mp_gen, KNOWLEDGE as _mp_kn
from .fluid import GENERATORS as _fl_gen, KNOWLEDGE as _fl_kn
from .fluid_fsi import GENERATORS as _fsi_gen, KNOWLEDGE as _fsi_kn
from .rigid_body import GENERATORS as _rb_gen, KNOWLEDGE as _rb_kn
from .viscoelasticity import GENERATORS as _ve_gen, KNOWLEDGE as _ve_kn
from .plasticity import GENERATORS as _pl_gen, KNOWLEDGE as _pl_kn
from .fiber_reinforced import GENERATORS as _fr_gen, KNOWLEDGE as _fr_kn
from .active_contraction import GENERATORS as _ac_gen, KNOWLEDGE as _ac_kn
from .biphasic_fsi import GENERATORS as _bfs_gen, KNOWLEDGE as _bfs_kn
from .polar_fluid import GENERATORS as _pf_gen, KNOWLEDGE as _pf_kn
from .damage import GENERATORS as _dm_gen, KNOWLEDGE as _dm_kn
from .growth_remodeling import GENERATORS as _gr_gen, KNOWLEDGE as _gr_kn


GENERATORS: dict[str, callable] = {}
for _g in (_le_gen, _he_gen, _bi_gen, _ht_gen,
           _mp_gen, _fl_gen, _fsi_gen, _rb_gen,
           _ve_gen, _pl_gen, _fr_gen, _ac_gen,
           _bfs_gen, _pf_gen, _dm_gen, _gr_gen):
    GENERATORS.update(_g)


KNOWLEDGE: dict[str, dict] = {}
for _k in (_le_kn, _he_kn, _bi_kn, _ht_kn,
           _mp_kn, _fl_kn, _fsi_kn, _rb_kn,
           _ve_kn, _pl_kn, _fr_kn, _ac_kn,
           _bfs_kn, _pf_kn, _dm_kn, _gr_kn):
    KNOWLEDGE.update(_k)


KNOWLEDGE["_general"] = {
    "description": "FEBio general capabilities and C++ embedding surface",
    "adaptive_mesh_refinement": {
        "description": (
            "FEAMR module (FEBio::FEAMR library) registers .feb XML tag "
            "names users type in the <Adaptive> / <Step> blocks. "
            "Source: FEAMR/FEAMR.cpp -> FEAMR::InitModule()."
        ),
        "mesh_adaptors": {
            "erosion":       "FEErosionAdaptor — remove elements failing a criterion",
            "hex_refine":    "FEHexRefine — uniform 3D hex split",
            "hex_refine2d":  "FEHexRefine2D — 2D analogue",
            "tet_refine":    "FETetRefine — uniform tet split",
            "mmg_remesh":    "FEMMGRemesh — anisotropic remesh via MMG library (requires MMG build)",
            "test_refine":   "FETestRefine — EXPERIMENTAL adaptor flagged FECORE_EXPERIMENTAL in source; do not use in production",
        },
        "adaptor_criteria": {
            "max_variable":      "FEVariableCriterion — threshold on a field variable",
            "element_selection": "FEElementSelectionCriterion",
            "math":              "FEScaleAdaptorCriterion — math-expression scaling",
            "min-max filter":    "FEMinMaxFilterAdaptorCriterion — SPACED-AND-HYPHEN tag name",
            "relative error":    "FEDomainErrorCriterion — SPACED tag name",
            "element data":      "FEElementDataCriterion — SPACED tag name",
            "tet-quality":       "FETetQualityCriterion",
            "mean-ratio":        "FEMeanRatioQualityCriterion",
            "scaled Jacobian":   "FEScaledJacobianQualityCriterion — SPACED-AND-CAMELCASE tag name",
        },
        "plot_variables": {
            "tet-quality":     "FEPlotTetQuality",
            "mean-ratio":      "FEPlotMeanRatio",
            "scaled-Jacobian": "FEPlotScaledJacobian — HYPHEN-AND-CAMELCASE",
        },
        "Signal": (
            "[Input] Some FEAMR .feb tag names contain SPACES and "
            "hyphens that look non-XML-like — they ARE the canonical "
            "strings the FECoreKernel registers and parses. Examples: "
            "'min-max filter', 'relative error', 'element data', "
            "'scaled Jacobian', 'scaled-Jacobian' (plot). Replacing "
            "spaces with underscores ('min_max_filter') silently fails "
            "to match in the FECoreKernel registry — FEBio reports the "
            "unknown tag at parse time. Also note test_refine has "
            "FECORE_EXPERIMENTAL flag in the source — production use "
            "is unsupported. (File walk FEAMR/FEAMR.cpp 2026-06-02.)"
        ),
    },
    "cmake_embedding": {
        "find_package": "find_package(FEBio) — defined by FEBioConfig.cmake "
                        "at the FEBio source tree root; detects in-tree-build "
                        "vs install layout automatically via "
                        "_FEBIO_IS_SOURCE_TREE.",
        "imported_targets": [
            "FEBio::FECore     — core FE framework",
            "FEBio::FEBioMech  — solid-mechanics module",
            "FEBio::FEBioMix   — biphasic / multiphasic mixture mechanics",
            "FEBio::FEBioFluid — fluid + biphasic-fluid",
            "FEBio::FEBioRVE   — RVE / multiscale support",
            "FEBio::FEBioPlot  — output plot file writer (.xplt)",
            "FEBio::FEBioXML   — .feb XML input parser",
            "FEBio::FEBioLib   — top-level orchestration / module registry",
            "FEBio::FEAMR      — adaptive mesh refinement",
            "FEBio::FEBioOpt   — parameter optimization",
            "FEBio::FEImgLib   — image-based meshing / DICOM support",
        ],
        "build_configurations": "_febio_find_library probes lib/Release "
                                "+ lib/Debug + lib/MinSizeRel + "
                                "lib/RelWithDebInfo, with a fallback that "
                                "matches anything under lib/. Multi-config "
                                "users (Visual Studio) get correct per-config "
                                "binaries.",
        "Signal": (
            "[Output] FEBioConfig.cmake sets FEBio_FOUND=FALSE if ANY of "
            "the 11 imported targets fails to locate its library — the "
            "config aborts with `return()` after the first missing lib. "
            "A partial build (e.g. only Release + FECore + FEBioMech, no "
            "FEImgLib because OpenCV wasn't installed) makes "
            "find_package(FEBio) report not-found even though the "
            "subset would suffice for mechanics-only embedding. "
            "Workaround: build all 11 libs even if the user only needs a "
            "subset, OR maintain a local fork of FEBioConfig.cmake with "
            "the unwanted libs removed from _FEBIO_LIBS. "
            "(File-walk audit of FEBioConfig.cmake 2026-06-02.)"
        ),
    },
}
