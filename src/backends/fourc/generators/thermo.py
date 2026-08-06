"""Pure thermal analysis generator for 4C.

Covers standalone heat conduction without structural coupling (use TSI for coupled).
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ThermoGenerator(BaseGenerator):
    """Generator for pure thermal (heat conduction) problems in 4C."""

    module_key = "thermo"
    display_name = "Pure Thermal Analysis"
    problem_type = "Thermo"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Standalone thermal analysis — heat conduction, convection BCs, "
                "radiation.  For thermal-structure coupling, use TSI instead."
            ),
            "yaml_section": "THERMAL DYNAMIC",
            "elements": {
                "2D": ["THERMO QUAD4", "THERMO QUAD9", "THERMO TRI3"],
                "3D": ["THERMO HEX8", "THERMO HEX27", "THERMO TET4", "THERMO TET10"],
            },
            "time_integration": {
                "Statics": "Steady-state thermal analysis",
                "OneStepTheta": "Transient with theta method (theta=1 for backward Euler)",
                "GenAlpha": "Generalized-alpha for transient thermal",
            },
            "materials": {
                "MAT_Fourier": {
                    "parameters": {
                        "CAPA": "Heat capacity [J/(m^3 K)]",
                        "CONDUCT": "Thermal conductivity [W/(m K)]",
                    },
                },
            },
            "boundary_conditions": {
                "DESIGN SURF THERMO DIRICH CONDITIONS": "Prescribed temperature",
                "DESIGN SURF THERMO NEUMANN CONDITIONS": "Prescribed heat flux",
                "DESIGN SURF THERMO CONVECTION CONDITIONS": "Convective heat transfer (h, T_inf)",
            },
            "pitfalls": [
                "[Syntax] MAT_Fourier.CONDUCT is a tensor-typed "
                "input — even for isotropic conductivity the value "
                "must be wrapped as 'constant: [k]' (a list under a "
                "'constant:' sub-key). A bare scalar "
                "'CONDUCT: 1.0' fails to match the MAT_Fourier "
                "input spec at "
                "core/io/src/4C_io_input_spec_builders.cpp:633 and "
                "4C echoes the whole MAT_Fourier block as 'remains "
                "unused'. Signal: stderr contains 'Failed to match "
                "specification in section \\'MATERIALS\\'' + "
                "'Could not match this input'. (Verified empirically "
                "2026-06-01 — 'CONDUCT: 1.0' rejected; "
                "'CONDUCT: {constant: [1.0]}' progresses to "
                "fill_complete on discretization 'thermo'. Real-input "
                "anisotropic example uses 'constant: [k11..k33]' "
                "9-vector.)",
                "[Syntax] PROBLEMTYPE must be 'Thermo' (NOT "
                "'Thermal'). The valid enum value list is enumerated "
                "in core/io as a deprecated_selection — a wrong "
                "value is rejected with 'Candidate deprecated_"
                "selection PROBLEMTYPE has wrong value, possible "
                "values: ...|Thermo|...' from the InputSpec match "
                "tree. Signal: stderr contains 'PROBLEMTYPE' + "
                "'has wrong value' + 'possible values:' enumerated "
                "by the MatchTree.assert_match() call in InputSpec "
                "(emitted from 4C_io_input_spec_builders). The "
                "printed list is the useful part: it contains "
                "'Thermo' and no 'Thermal'. Note this is a VALUE "
                "rejection, not a section-name one — 'is not a valid "
                "section name' does NOT appear, which is what "
                "separates it from the THERMAL DYNAMIC pitfall. "
                "(Verified by execution 2026-08-06.)",
                "[Syntax] Standalone thermal uses section "
                "'THERMAL DYNAMIC' (NOT 'THERMO DYNAMIC' or "
                "'THERMO'). 'THERMO DYNAMIC' is rejected at YAML "
                "parse with 'Section \\'THERMO DYNAMIC\\' is not a "
                "valid section name.' from core/io/src/"
                "4C_io_input_file.cpp:546. Signal: stderr contains "
                "the offending section name + 'not a valid section "
                "name'. (Same code path as fourc fluid section-name "
                "pitfall; verified by execution 2026-08-06 — "
                "'THERMO DYNAMIC' rejected, 'THERMAL DYNAMIC' "
                "accepted. The rejected deck never reaches a "
                "discretisation, so nothing else in it is checked.)",
                "[Physics] MAT_Fourier.CAPA is *volumetric* heat "
                "capacity rho*c_p [J/(m^3 K)], NOT specific heat "
                "capacity c_p [J/(kg K)]. Mixing the two silently "
                "produces a transient simulation with the wrong "
                "thermal time constant tau = rho*c_p*L^2 / k — "
                "off by a factor of rho. Signal: MAT_Fourier.CAPA "
                "in MATERIALS that produces a transient whose time "
                "constant is off by the factor rho, with NO warning "
                "of any kind — the decay constant is strictly "
                "proportional to CAPA, so the same field is reached "
                "after a time shorter by exactly rho. Compare "
                "against MAT_Fourier.CONDUCT (same material card) "
                "to flag a units mismatch. (Verified by execution "
                "2026-08-06 on a mesh with a single interior DOF, "
                "where the discrete problem is a scalar ODE and "
                "nothing else can absorb the error.)",
                "[Integration] For coupled thermal-structural problems "
                "use PROBLEMTYPE 'Thermo_Structure_Interaction' "
                "(TSI), NOT 'Thermo' with manual STRUCTURE/THERMO "
                "elements. TSI uses its own monolithic/staggered "
                "machinery from src/tsi/ that wires the discretiza"
                "tions together; constructing the coupling by hand "
                "in a Thermo problem will run thermo alone with "
                "the structure block silently ignored. Signal: only "
                "'fill_complete() on discretization thermo' appears "
                "in stderr — no structure discretization printed, "
                "no warning that the block was dropped, and exit 0. "
                "A real TSI run prints BOTH banners, so the absence "
                "is a usable check. (Verified by execution "
                "2026-08-06: a Thermo deck carrying a STRUCTURE "
                "ELEMENTS block and a structural material ran heat "
                "conduction alone and passed its thermal result "
                "test.)",
                "[Syntax] THERMO element block format is "
                "'<id> THERMO <celltype> <node_ids...> MAT <id>' "
                "(eletype string 'THERMO' from src/thermo/src/"
                "element/4C_thermo_element.cpp:45, celltype keys "
                "hex8/hex20/hex27/tet4/tet10/quad4/quad9/tri3 from "
                "setup_element_definition l.107). The section does "
                "NOT gate the element category: the element "
                "vocabulary is global and the section name only "
                "chooses which discretisation the elements land in. "
                "Writing a TRANSP element in THERMO ELEMENTS is "
                "therefore NOT caught as a category error — with "
                "its required TYPE key supplied it is accepted into "
                "the thermo discretisation, survives fill_complete, "
                "and the run then dies inside the thermal time "
                "integrator with an UNCAUGHT C++ exception "
                "(Teuchos::Exceptions::InvalidParameterType naming "
                "FourC::ScaTra::Action), i.e. SIGABRT and no 4C "
                "error block at all. Signal: without TYPE, "
                "\"Required value 'TYPE' not found in input line\"; "
                "with it, an abort carrying no 4C diagnostic. A "
                "wrong CELLTYPE is caught, but by "
                "core/fem/.../4C_fem_general_cell_type_traits.hpp "
                "('Unknown celltype HEX9'), not by the input-spec "
                "builder. In no case does 4C say the element does "
                "not belong in the section. (Corrected by execution "
                "2026-08-06; the earlier entry predicted an "
                "element-spec mismatch from "
                "4C_io_input_spec_builders.cpp, which does not "
                "happen.)",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "thermo_2d", "description": "2D steady-state heat conduction"},
                {"name": "thermo_3d", "description": "3D transient heat conduction"}]

    def get_template(self, variant: str = "thermo_2d") -> str:
        from ..inline_mesh import (
            matched_thermo_2d_input, matched_thermo_3d_input)
        if variant in ("thermo_2d", "default"):
            return matched_thermo_2d_input()
        if variant == "thermo_3d":
            return matched_thermo_3d_input()
        raise ValueError(f"Unknown variant {variant!r}")

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
