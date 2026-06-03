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
        "mesh_adaptor_details": {
            "erosion (FEErosionAdaptor)": {
                "description": (
                    "Iteratively deactivates elements that match a "
                    "criterion. Operates IN-PLACE on the mesh — no "
                    "remeshing, just .setInactive() on selected "
                    "elements + optional surface/topology cleanup. "
                    "Source: FEAMR/FEErosionAdaptor.cpp"),
                "parameters": {
                    "max_iters": (
                        "int, default -1 = UNLIMITED. Apply() only "
                        "self-terminates via 'Max iterations reached' "
                        "when max_iters >= 0 and iteration crosses "
                        "it. Default behaviour: erosion never stops "
                        "voluntarily — relies on the outer time-"
                        "step / load controller to bound the loop."),
                    "max_elems": (
                        "int, default 0 = NO CAP. Caps how many "
                        "elements get deactivated per Apply() call. "
                        "When 0, the entire criterion-selection is "
                        "processed in one call."),
                    "sort": (
                        "int enum {0=none (default), 1=largest first "
                        "(DECREASING), 2=smallest first (INCREASING)}. "
                        "Only consulted when max_elems > 0 — with "
                        "max_elems == 0 the sort value is IGNORED, "
                        "no log line. Sorts by the criterion's "
                        "per-element scalar. Header-comment drift: "
                        "FEErosionAdaptor.h:58 says '1 = smallest to "
                        "largest, 2 = largest to smallest' — the "
                        "OPPOSITE of what the .cpp dispatch "
                        "(FEErosionAdaptor.cpp:89-100) implements "
                        "(1 -> SORT_DECREASING, 2 -> SORT_INCREASING). "
                        "Trust the .cpp; the header doc is wrong."),
                    "remove_islands": (
                        "bool, default false. When true, calls "
                        "RemoveIslands(FEMeshTopo) after deactivating "
                        "the selection — flood-fills active elements "
                        "and removes any disconnected component whose "
                        "ALL nodes have BC == DOF_OPEN on DOFs 0/1/2."),
                    "erode_surfaces": (
                        "enum, default 'yes' (ERODE). Values: "
                        "{'no' (DONT_ERODE), 'yes' (ERODE), 'grow' "
                        "(GROW), 'reconstruct' (RECONSTRUCT)}. "
                        "Registered via setEnums(\"no\\0yes\\0grow\\0"
                        "reconstruct\\0\")."),
                    "criterion": (
                        "REQUIRED <criterion> child property "
                        "(FEMeshAdaptorCriterion). Without it, "
                        "Apply() returns false silently — NO log "
                        "line, no error."),
                },
                "erode_surfaces_semantics": {
                    "no": (
                        "DONT_ERODE — leave surfaces untouched; faces "
                        "attached to deactivated elements remain "
                        "active. Useful when surfaces represent "
                        "external loading patches that should persist."),
                    "yes": (
                        "ERODE (default) — ErodeSurfaces(): for each "
                        "surface, deactivate any face whose owning "
                        "element became inactive. Calls surf.Init() "
                        "only when faces were actually eroded."),
                    "grow": (
                        "GROW — GrowErodedSurfaces(): same as 'yes' "
                        "PLUS rebuild boundary as the eroded surface "
                        "grows inward. New faces are INVERTED "
                        "(node[l] = node[nf-1-l]) to keep normals "
                        "outward-pointing. Materializes new FE_TRI3G3 "
                        "/ FE_QUAD4G4 surface elements."),
                    "reconstruct": (
                        "RECONSTRUCT — discard existing surface faces, "
                        "rebuild from FEDomainBoundary(domainList) "
                        "where the surface name matches a registered "
                        "FEDomainList name. Surfaces NOT generated "
                        "from part lists with matching names are "
                        "silently skipped — no log, no error."),
                },
            },
        },
        "adaptor_criteria": {
            "max_variable":      "FEVariableCriterion — threshold on a field variable",
            "element_selection": (
                "FEElementSelectionCriterion — UNDERSCORE tag "
                "name. Required children: <element_list> (vector "
                "of element IDs, 1-based, GetID() not "
                "GetLocalID()) and <value> (double, default 1.0 "
                "= refinement scale assigned to each matched "
                "element). Iterates over active elements only "
                "(el.isActive() gate; inactive/erased elements "
                "silently skipped). LOOKUP IS O(N·M) per call: "
                "for N total elements and M IDs in the list, a "
                "linear scan runs N×M times — source comment "
                "(FEAMR/FEElementSelectionCriterion.cpp:52) "
                "explicitly says 'TODO: This is really slow. "
                "Need to speed this up!'. Acceptable for "
                "M ≤ ~100 on small meshes; becomes the dominant "
                "cost for large meshes with long lists. Source: "
                "FEAMR/FEElementSelectionCriterion.cpp"),
            "math":              "FEScaleAdaptorCriterion — math-expression scaling",
            "min-max filter": (
                "FEMinMaxFilterAdaptorCriterion — SPACED-AND-HYPHEN "
                "tag name. Parameters: <min> (double, default -1e37), "
                "<max> (double, default +1e37), <clamp> (bool, default "
                "true). Required child: <data> (FEMeshAdaptorCriterion "
                "property). Behavior depends on <clamp>: when true "
                "(default), values are CLAMPED to [min,max] and ALL "
                "elements remain in the selection; when false, "
                "elements with values outside [min,max] are REJECTED "
                "(GetElementValue returns false). Default range "
                "[-1e37,+1e37] makes the criterion a silent pass-"
                "through regardless of clamp mode — users must set "
                "min/max explicitly. Missing <data> property is "
                "silently no-op (return false; same pattern as "
                "FEElementDataCriterion and FEErosionAdaptor's "
                "criterion). Source: FEAMR/FEFilterAdaptorCriterion.cpp"),
            "relative error": (
                "FEDomainErrorCriterion — SPACED tag name. Required "
                "child params: <error> (double, target relative-error "
                "tolerance) and <data> (FEMeshAdaptorCriterion property "
                "providing per-Gauss-point scalar via "
                "GetMaterialPointValue). Error formula per element: "
                "max_j |s_j - s_hat_j| / (smax - smin), where s_hat is "
                "the recovered nodal projection evaluated at Gauss "
                "point j. Returns size-scale s = error/max_err when "
                "max_err>error (shrink), else s=1 (keep). Source: "
                "FEAMR/FEDomainErrorCriterion.cpp"),
            "element data": (
                "FEElementDataCriterion — SPACED tag name on the "
                "outer <criterion type=\"element data\"> wrapper. "
                "Required child uses UNDERSCORE form: "
                "<element_data>...</element_data>. The string value "
                "is dispatched via fecore_new<FELogElemData>(name, "
                "fem) to look up a registered log-element-data "
                "variant; unknown name silently returns nullptr → "
                "Init() returns false → criterion is silently "
                "disabled with no log line. Vocabulary matches the "
                "<plot> tag set (use the same registered "
                "FELogElemData names you see in .feb plot output). "
                "Source: FEAMR/FEElementDataCriterion.cpp"),
            "tet-quality":       "FETetQualityCriterion",
            "mean-ratio": (
                "FEMeanRatioQualityCriterion — HYPHEN tag name. "
                "Required child: <min_quality> (double, default 1.0 "
                "= refines everything; mean-ratio metric ∈ [0,1] "
                "with 1=perfect equilateral). Per-node formula: "
                "MR = 3·det(J)^(2/3) / (|e0|² + |e1|² + |e2|²), "
                "returns min over nodes. Element-shape gate "
                "(GetElementValue): ONLY ET_TET4 / ET_HEX8 / "
                "ET_PENTA6 — linear quads, higher-order elements "
                "(ET_TET10, ET_HEX20, ...), and all 2D shells/beams "
                "return false silently. Source: "
                "FEAMR/FEElementQualityCriterion.cpp:35-127"),
            "scaled Jacobian": (
                "FEScaledJacobianQualityCriterion — SPACED-AND-"
                "CAMELCASE tag name (outer); ADD_PARAMETER uses "
                "UNDERSCORE \"min_quality\" inner form (default "
                "1.0). Per-node formula: SJ = det(J) / "
                "(|e0|·|e1|·|e2|), returns min over nodes (range "
                "[-1, 1] with 1=perfect, near 0=bad, negative "
                "=inverted element). Same element-shape gate as "
                "mean-ratio: ONLY TET4/HEX8/PENTA6. Unsupported "
                "shapes return false silently. Source: "
                "FEAMR/FEElementQualityCriterion.cpp:131-194"),
        },
        "plot_variables": {
            "tet-quality":     ("FEPlotTetQuality — silently writes 0.0 for "
                                "non-ET_TET4 elements on mixed meshes "
                                "(FEAMR/FEAMRPlot.cpp:44 explicit else branch); "
                                "tag name warns but no log message"),
            "mean-ratio":      "FEPlotMeanRatio — applies to ALL element shapes",
            "scaled-Jacobian": "FEPlotScaledJacobian — HYPHEN-AND-CAMELCASE; applies to ALL element shapes",
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
            "is unsupported. Plus a quiet behavior in 'relative error' "
            "(FEDomainErrorCriterion::GetElementSelection): if the "
            "data-field range across the element set is below 1e-12 "
            "(fabs(smin - smax) < 1e-12 — e.g. an undisturbed step-0 "
            "state with zero stress everywhere), the function returns "
            "an EMPTY FEMeshAdaptorSelection and refinement is silently "
            "a no-op. Users probing the criterion before the first "
            "solve see no refinement and no warning. Two more quirks "
            "in the 'mean-ratio' and 'scaled Jacobian' criteria "
            "(FEElementQualityCriterion.cpp): (a) BOTH default "
            "<min_quality> to 1.0 — since mean-ratio ∈ [0,1] and "
            "scaled-Jacobian ∈ [-1,1] both peak at 1, users who omit "
            "<min_quality> trigger refinement on every non-perfect "
            "element (essentially the entire mesh on the first call). "
            "(b) BOTH criterion classes silently SKIP elements whose "
            "Shape() is not ET_TET4 / ET_HEX8 / ET_PENTA6 — "
            "GetElementValue returns false. Higher-order solids "
            "(ET_TET10, ET_HEX20), 2D shells, beams, etc. are "
            "invisible to these criteria. Mixed meshes get partial "
            "coverage with no warning. Five FEErosionAdaptor "
            "edges users routinely hit on damage / erosion runs: "
            "(c) <erode_surfaces> is a 4-VALUE enum {'no', 'yes', "
            "'grow', 'reconstruct'} registered via setEnums("
            "\"no\\0yes\\0grow\\0reconstruct\\0\"). Any other "
            "spelling (e.g. 'true', 'erode', 'remove') silently "
            "FAILS the enum gate at XML parse time — the parameter "
            "stays at its default ERODE. (d) Default <max_iters> "
            "is -1 (UNLIMITED) — Apply() only emits 'Max "
            "iterations reached' when max_iters >= 0; with the "
            "default the loop NEVER self-terminates and relies on "
            "the outer load step. (e) <sort> is silently IGNORED "
            "when <max_elems> == 0 (default). Users setting "
            "sort=1 expecting biggest-failures-first behaviour "
            "must ALSO set max_elems > 0. (f) <criterion> is "
            "REQUIRED — omitting it makes Apply() return false "
            "silently (no feLog, no error). The mesh adaptor is "
            "effectively a no-op with no diagnostic. (g) "
            "RemoveIslands has a HARDCODED 'TODO: mechanics "
            "only!' check — it tests nj.get_bc(0/1/2) != DOF_OPEN. "
            "For biphasic / multiphasic / thermal / scalar-only "
            "problems where mechanics DOFs are open by default, "
            "EVERY component looks isolated → remove_islands=true "
            "deletes the entire mesh. Source comment line "
            "FEAMR/FEErosionAdaptor.cpp:206. Three "
            "FEMinMaxFilterAdaptorCriterion edges users routinely "
            "hit on the 'min-max filter' criterion: "
            "(h) Default <min> = -1e37 and <max> = +1e37 — with "
            "the defaults the filter is a SILENT PASS-THROUGH "
            "regardless of <clamp>. Users must set both bounds "
            "explicitly; forgetting them yields a refinement "
            "selection equal to the input (no filtering happens). "
            "(i) <clamp> default is TRUE — meaning out-of-range "
            "values are clamped to the bounds and the element "
            "stays in the selection. Users who want to FILTER "
            "OUT out-of-range elements (the natural "
            "interpretation of 'filter') must explicitly set "
            "<clamp>false</clamp>; otherwise the criterion is a "
            "soft-clamp that keeps every element. "
            "(j) Missing <data> child property silently returns "
            "false (no log, no error) — same pattern as "
            "FEElementDataCriterion's missing-data and "
            "FEErosionAdaptor's missing-criterion (edge f above). "
            "Three FECORE criterion / adaptor classes share this "
            "silent-no-op fault pattern, and they all consult the "
            "same nullable property field. "
            "(File walks "
            "FEAMR/FEAMR.cpp 2026-06-02, "
            "FEAMR/FEDomainErrorCriterion.cpp 2026-06-03, "
            "FEAMR/FEElementQualityCriterion.cpp 2026-06-03, "
            "FEAMR/FEErosionAdaptor.cpp 2026-06-03, "
            "FEAMR/FEFilterAdaptorCriterion.cpp 2026-06-03.)"
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
