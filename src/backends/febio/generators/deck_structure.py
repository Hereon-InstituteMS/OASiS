"""Executed-verified .feb deck-authoring surface for FEBio 4.12.

Everything in this module was obtained by RUNNING the installed
binary, not by reading documentation. Two ground-truth sources
were used and cross-checked:

  1. The binary's own factory registry. ``febio4`` has an
     interactive console; the ``list`` command dumps every
     registered factory as ``<module>.<type string> [SUPER_CLASS]``
     and ``list -m`` dumps the module names. Piping
     ``printf 'list\\nquit\\n' | febio4 -nosplash`` therefore
     yields the authoritative type-string vocabulary FOR THE
     INSTALLED BUILD — including which optional libraries were
     compiled in. This is strictly better than grepping
     REGISTER_FECORE_CLASS out of the sources, because a source
     tree lists features that a given cmake configuration never
     compiles (see ``linear_solvers`` below: ``pardiso`` is in
     the sources but absent from a USE_MKL=OFF build).

  2. Minimal .feb decks — one hex8 element, or a 2x2x2 patch —
     run through ``febio4 -i in.feb``, one deviation from a
     known-good baseline at a time, recording exit code, the
     "Reading file ... SUCCESS!/FAILED!" line, the verbatim
     ERROR box, the termination banner, and the numbers written
     to the logfile CSVs.

PROVENANCE: all entries below carry the marker
"live 2026-08-03 / FEBio 4.12.0.86045466d" and were produced on
a source build (github.com/febiosoftware/FEBio @ 8604546,
USE_MKL=OFF) symlinked at ~/FEBio/bin/febio4.

VERSION DRIFT WARNING: the installed build reports itself as
``version 4.12.0.86045466d``. Ten modules are registered. There
is NO ``heat`` module and NO ``biphasic-FSI`` module in this
build, although both strings appear in older OASiS catalog
entries and in FEBio documentation for earlier lines. Feeding
either string to ``<Module type=...>`` does not raise an error —
it segfaults the process (see ``module_types`` below).
"""

# ---------------------------------------------------------------
# Registered module names — `list -m` on the installed binary.
# `<Module type="X"/>` must be one of these, byte for byte.
# ---------------------------------------------------------------
FEBIO_MODULES: tuple[str, ...] = (
    "solid",
    "biphasic",
    "solute",
    "multiphasic",
    "fluid",
    "fluid-FSI",
    "multiphasic-FSI",
    "fluid-solutes",
    "thermo-fluid",
    "polar fluid",
)

# `<analysis>` is an enum whose vocabulary is installed BY THE
# ACTIVE MODULE (FE*Analysis::FindParameterFromData(&m_nanalysis)
# ->setEnums(...)), so the legal words differ per module. Verified
# live: a word from the wrong module's vocabulary is a hard parse
# error, not a silent fallback.
FEBIO_ANALYSIS_VALUES: dict[str, tuple[str, ...]] = {
    "solid": ("STATIC", "DYNAMIC"),
    "biphasic": ("STEADY-STATE", "TRANSIENT"),
    "solute": ("STEADY-STATE", "TRANSIENT"),
    "multiphasic": ("STEADY-STATE", "TRANSIENT"),
    "fluid": ("STEADY-STATE", "DYNAMIC"),
    "fluid-FSI": ("STEADY-STATE", "DYNAMIC"),
    "multiphasic-FSI": ("STEADY-STATE", "DYNAMIC"),
    "fluid-solutes": ("STEADY-STATE", "DYNAMIC"),
    "thermo-fluid": ("STEADY-STATE", "DYNAMIC"),
    "polar fluid": ("STEADY-STATE", "DYNAMIC"),
}


DECK_KNOWLEDGE = {
    "description": (
        "Executed-verified .feb authoring surface for the installed "
        "FEBio build. Every claim here was reproduced by running "
        "febio4 on a minimal deck; the counterexample decks are "
        "kept as Tier-2 fixtures under "
        "scripts/tier2_fixtures/febio/."),
    "verified_against": (
        "FEBio 4.12.0.86045466d (source build of "
        "github.com/febiosoftware/FEBio @ 8604546, USE_MKL=OFF, "
        "skyline default linear solver), live 2026-08-03."),

    "how_to_enumerate_features_yourself": (
        "febio4 has an interactive console. `printf 'list -m\\nquit\\n' "
        "| febio4 -nosplash` prints the registered MODULE names; "
        "`printf 'list\\nquit\\n' | febio4 -nosplash` prints every "
        "registered factory as `<module>.<type string> "
        "[SUPER_CLASS_ID]`; `list -c <SUPER_CLASS_ID>` filters and "
        "`list -s <substring>` greps. This is the only reliable way "
        "to learn which optional features a particular BUILD has — "
        "the source tree registers many more than any one cmake "
        "configuration compiles. On the verified build the dump has "
        "1429 factories: 159 FEMATERIAL_ID, 288 FEPLOTDATA_ID, "
        "286 FELOGELEMDATA_ID, 75 FELOAD_ID, 49 FEBC_ID, "
        "20 FELINEARSOLVER_ID, 10 FEANALYSIS_ID."),

    "required_skeleton": (
        "A deck that actually solves needs, at minimum: "
        "<febio_spec version=\"4.0\"> (the version attribute is "
        "mandatory) / <Module type=...> as the FIRST child / "
        "<Control> containing a <solver type=...> child / "
        "<Material> with at least one named <material> / <Mesh> "
        "with <Nodes> + <Elements> / <MeshDomains> binding the "
        "element block to the material BY NAME / <Boundary>. "
        "<Output> is optional: FEBio writes <jobname>.xplt with "
        "`displacement` and `stress` even with no <Output> section "
        "at all (verified: identical 1629-byte .xplt with and "
        "without the section). TEXT output is NOT free — CSV files "
        "only appear if you ask for them with <Output><logfile>."),

    "section_order": (
        "The .feb reader is SINGLE-PASS: a section can only "
        "reference names that an EARLIER section defined. Verified "
        "live, the hard constraints are exactly four: "
        "(1) <Module> must be the first tag — any other first tag, "
        "or no <Module> at all, aborts with `First tag must be the "
        "Module section.`; "
        "(2) <Material> must precede <MeshDomains>, else "
        "`tag \"SolidDomain\" (line N) : invalid value for "
        "attribute \"mat\"`; "
        "(3) <Mesh> must precede <MeshDomains>; "
        "(4) <MeshDomains> must precede <Boundary> — node sets are "
        "only resolvable once the domains are built, so putting "
        "<Boundary> between <Mesh> and <MeshDomains> still fails. "
        "Constraints (3) and (4) share one message, `tag \"bc\" "
        "(line N) : invalid value for attribute \"node_set\"`, "
        "which names the BC rather than the section that is out of "
        "place — check the order of Mesh / MeshDomains / Boundary "
        "before suspecting the node set itself. "
        "Everything else floats: <Control>, <LoadData> and "
        "<Output> were each verified to run correctly in first, "
        "middle and last position. The one ordering mistake that "
        "does NOT produce a diagnostic is <MeshDomains> before "
        "<Mesh> — that segfaults (exit 139)."),

    "matching_error_text": (
        "FEBio wraps its ERROR box at 71 columns, so a long "
        "diagnostic is SPLIT ACROSS LINES inside the star frame. "
        "`The selected linear solver does not support the "
        "requested matrix format.` is emitted as `... the "
        "requested matrix` / `format.` on two lines and a "
        "fixed-string grep for the whole sentence returns nothing. "
        "Match on a short distinctive fragment that cannot wrap "
        "(`does not support the requested`), or strip the frame "
        "and re-join before matching. Verified live 2026-08-03: "
        "the shorter messages quoted throughout this catalog "
        "(`Invalid load curve ID`, `First tag must be the Module "
        "section.`, `Invalid element type`, every `tag \"x\" (line "
        "N) : ...` form) all fit on one line and grep cleanly."),

    "module_types": {
        "registered": list(FEBIO_MODULES),
        "note": (
            "Byte-exact and case-SENSITIVE. `polar fluid` really "
            "does contain a space. `fluid-FSI` and "
            "`multiphasic-FSI` really do have an uppercase FSI. "
            "There is no `heat` module and no `biphasic-FSI` "
            "module in 4.12."),
        "failure_mode": (
            "FEBioModuleSection.cpp calls "
            "FEModelBuilder::SetActiveModule(szt) with NO existence "
            "check, so an unrecognised module string dereferences a "
            "null module and the process dies with SIGSEGV. "
            "Verified live for `heat`, `biphasic-FSI`, `Solid` and "
            "`SOLID` — all exit 139 with stdout truncated mid-line "
            "at `Reading file in.feb ...`, no FAILED!, no ERROR "
            "box, no log file."),
    },

    "analysis_values": {
        "per_module": {k: list(v) for k, v in
                       FEBIO_ANALYSIS_VALUES.items()},
        "note": (
            "Enum VALUES are matched case-insensitively (`static`, "
            "`Static` and `STATIC` all work) and the raw ordinal is "
            "also accepted (`<analysis>1</analysis>` == DYNAMIC in "
            "the solid module — verified by identical nodal "
            "displacements). Contrast with factory TYPE STRINGS in "
            "`type=` attributes, which are case-SENSITIVE. A word "
            "from another module's vocabulary is a clean parse "
            "error: `tag \"analysis\" (line N) : invalid value: "
            "STEADY-STATE` in a solid deck. An out-of-range ordinal "
            "passes the reader and fails at init with `Invalid "
            "value for parameter: .analysis`."),
    },

    "linear_solvers": {
        "registered_on_this_build": [
            "diagonal", "LU", "skyline", "pardiso-project", "fgmres",
            "boomeramg", "cg", "schur", "hypre_gmres", "hypre_pcg_amg",
            "block", "bipn", "bicgstab", "strategy", "test",
            "accelerate", "superlu_mt", "ilu0", "ilut", "ichol",
        ],
        "note": (
            "`pardiso` is NOT a valid type string on a USE_MKL=OFF "
            "build — `<linear_solver type=\"pardiso\">` fails at "
            "parse with `tag \"linear_solver\" (line N) : invalid "
            "value for attribute \"type\"`. The MKL-free spelling "
            "that IS registered is `pardiso-project`, and even that "
            "one rejects unsymmetric matrices in this build. The "
            "startup banner prints the fallback it picked: "
            "`Default linear solver: skyline`."),
        "unsymmetric": (
            "Anything that sets <symmetric_stiffness>non-symmetric"
            "</symmetric_stiffness> — which the biphasic, "
            "multiphasic, fluid and FSI solvers do by default — "
            "needs a solver that accepts an unsymmetric sparse "
            "format. Verified live on the USE_MKL=OFF build by "
            "sweeping all 20 registered solvers on a "
            "confined-compression biphasic deck: skyline, fgmres, "
            "cg, schur, bipn, superlu_mt, accelerate, diagonal, "
            "hypre_gmres, hypre_pcg_amg, boomeramg, ichol, "
            "pardiso-project and strategy abort with `The selected "
            "linear solver does not support the requested matrix "
            "format.` (the composite solvers schur / strategy / "
            "block may instead stop earlier at parse, demanding "
            "their sub-solver properties — `Component "
            "\"linear_solver\" needs to have property \"A_solver\" "
            "defined` etc. — depending on how they are "
            "configured); `LU` aborts with `Linear solver failed "
            "to find solution`; `ilu0` and `ilut` with `Fatal "
            "error in factorization of stiffness matrix`; `block` "
            "with `An error occurred during preprocessing of "
            "linear solver`. Two solvers RUN: `bicgstab`, which is "
            "the only real one, and `test`, which is not a solver "
            "at all (see `null_test_solver` below)."),
        "bicgstab_caveat": (
            "`bicgstab` is the recommended unsymmetric solver on a "
            "USE_MKL=OFF build, but it is registered with an "
            "OPTIONAL `pc_left` preconditioner property that "
            "defaults to none. Unpreconditioned BiCGStab converged "
            "on a single-element biphasic deck and FAILED on a "
            "2x2x2 one with `Linear solver failed to find "
            "solution. Aborting run.` — so it is a workaround for "
            "small problems, not a substitute for a direct "
            "unsymmetric solver. If real biphasic / fluid / FSI "
            "work is needed, rebuild FEBio with MKL "
            "(-DUSE_MKL=ON) so the `pardiso` factorisation is "
            "available. (Live-verified 2026-08-03.)"),
        "null_test_solver": (
            "DO NOT USE <linear_solver type=\"test\"/>. It is a "
            "registered FELINEARSOLVER_ID factory whose "
            "TestSolver::BackSolve (NumCore/TestSolver.cpp) sets "
            "every entry of the solution vector to 0 and returns "
            "true. FEBio therefore takes zero Newton increments, "
            "declares convergence, completes every time step and "
            "ends with `N O R M A L   T E R M I N A T I O N` and "
            "exit 0. It is the one verified case on this build "
            "where the termination banner AND a non-zero "
            "completed-step count are both satisfied by a run that "
            "solved nothing: on a one-element biphasic deck it "
            "reported sz = -2.00200400802 against the correct "
            "-2.00691787274 — close enough to look plausible, "
            "because the prescribed BCs alone still deform the "
            "single element. Anything with interior degrees of "
            "freedom returns the zero field. (Live-verified "
            "2026-08-03.)"),
    },

    "control_section": (
        "<Control> REQUIRES a <solver type=\"<module name>\"> child "
        "property in the 4.0 schema — omitting it aborts at parse "
        "with `Component \"\" needs to have property \"solver\" "
        "defined (line N)`. Quasi-Newton settings live INSIDE that "
        "solver, one level deeper than in 3.x: `max_ups` directly "
        "under <solver> is rejected (`tag \"max_ups\" (line N) : "
        "unrecognized tag`); the 4.0 spelling is "
        "<solver><qn_method type=\"BFGS\"><max_ups>10</max_ups>"
        "</qn_method></solver>. Registered qn_method types: BFGS, "
        "Broyden, JFNK, `modified Newton`, `full Newton`."),

    "final_time_formula": (
        "FINAL TIME = time_steps x step_size. FEAnalysis::Solve() "
        "loops `while (endtime - currentTime > eps)` with "
        "`endtime = m_dt0 * m_ntime`, so <time_steps> is NOT a step "
        "count when an auto <time_stepper> is active — it only sets "
        "the end of the interval together with <step_size>. "
        "Verified: time_steps=10 + step_size=1.0 runs to t = 10; "
        "time_steps=10 + step_size=0.1 runs to t = 1."),

    "output_naming": {
        "plot_variables": (
            "<Output><plotfile type=\"febio\"><var type=\"NAME\"/> "
            "names come from the FEPLOTDATA_ID factories and are "
            "MODULE-SCOPED and case-SENSITIVE. Solid-module basics: "
            "`displacement`, `velocity`, `acceleration`, `stress`, "
            "`relative volume`, `PK1 stress`, `PK2 stress`, "
            "`nodal stress`, `Lagrange strain`, `contact pressure`, "
            "`reaction forces`, `fiber vector`, `damage`. There is "
            "no `von Mises stress` variable — request `stress` "
            "(the full tensor) and reduce it downstream."),
        "log_variables": (
            "<Output><logfile><node_data data=\"...\"> and "
            "<element_data data=\"...\"> take a SEMICOLON-separated "
            "list of FELOG*DATA_ID names, also module-scoped and "
            "case-sensitive. Solid nodes: x y z ux uy uz vx vy vz "
            "ax ay az Rx Ry Rz. Solid elements: J, sx sy sz sxy syz "
            "sxz (note the Voigt order is xy, yz, xz), Ex..Exz "
            "(Green-Lagrange, FEElasticMaterialPoint::Strain), "
            "ex..exz (INFINITESIMAL strain, "
            "FEElasticMaterialPoint::SmallStrain — not an "
            "Almansi measure), Ux..Uxz "
            "(right stretch), `effective stress`. Omitting the "
            "file= attribute is legal and dumps the table into the "
            "run's .log instead of a separate CSV."),
        "small_strain_shear_bug": (
            "UPSTREAM BUG in FEBio 4.12: the `eyz` and `exz` log "
            "variables are SWAPPED. "
            "FEElasticMaterialPoint::SmallStrain() builds its "
            "return value as mat3ds(F00-1, F11-1, F22-1, "
            "0.5(F01+F10), 0.5(F02+F20), 0.5(F12+F21)) while the "
            "mat3ds constructor signature is (xx, yy, zz, xy, yz, "
            "xz) — so the xz shear is stored in the yz slot and "
            "vice versa. Verified live 2026-08-03 on a homogeneous "
            "F = [[1.02,0.01,0],[0,0.99,0.005],[0,0,1.01]]: FEBio "
            "reports eyz = 7.5e-18 and exz = 0.0025 where the "
            "correct values are eyz = 0.0025 and exz = 0. The "
            "diagonals and exy are right, and the Green-Lagrange "
            "Ex..Exz family is right (Eyz = 0.002475, Exz = "
            "7.7e-18 as expected). Use Ex..Exz, or swap eyz/exz in "
            "post-processing."),
        "csv_format": (
            "Each recorded step is a block of three header lines "
            "`*Step`, `*Time`, `*Data` followed by one row per "
            "item, `id,value,value,...`, at ~12 significant "
            "digits. Step 0 (the undeformed state) is always "
            "written, so a file that contains ONLY a step-0 block "
            "means nothing was solved."),
    },

    "silent_failure_checklist": (
        "febio4's exit code alone is not enough and neither is the "
        "presence of output files. Check, in this order: "
        "(1) exit status 0; "
        "(2) the literal banner `N O R M A L   T E R M I N A T I O N` "
        "in stdout/.log — `E R R O R   T E R M I N A T I O N` can "
        "appear with a fully-populated .xplt and CSVs already on "
        "disk; "
        "(3) `Number of time steps completed` > 0; "
        "(4) the last `*Time` in the logfile CSV equals "
        "time_steps x step_size — with an auto time-stepper FEBio "
        "writes every converged step, so a run that dies at t=0.335 "
        "of an intended t=1 leaves four perfectly valid-looking "
        "states behind; "
        "(5) any WARNING box. `1 isolated vertex removed.` means "
        "the connectivity is degenerate; `No force acting on the "
        "system.` means nothing is loading the model; "
        "`Model has N unreferenced load controllers.` means a "
        "load curve is not wired to anything. "
        "Two verified cases defeat checks (1)-(4) entirely and "
        "have to be excluded at deck-generation time instead: "
        "<linear_solver type=\"test\"/>, which is a null solver "
        "that reports success (see linear_solvers.null_test_solver), "
        "and a degenerate element whose connectivity repeats a "
        "node id, which only produces a WARNING."),

    "verified_working_baseline": (
        "This exact skeleton runs to normal termination on 4.12 and "
        "is the safest thing to mutate from:\n"
        "<febio_spec version=\"4.0\">\n"
        "  <Module type=\"solid\"/>\n"
        "  <Control><analysis>STATIC</analysis>"
        "<time_steps>1</time_steps><step_size>1.0</step_size>\n"
        "    <solver type=\"solid\">"
        "<symmetric_stiffness>symmetric</symmetric_stiffness>"
        "</solver></Control>\n"
        "  <Material><material id=\"1\" name=\"M1\" "
        "type=\"isotropic elastic\">"
        "<density>1.0</density><E>1000.0</E><v>0.3</v>"
        "</material></Material>\n"
        "  <Mesh><Nodes name=\"N\">...</Nodes>"
        "<Elements type=\"hex8\" name=\"P1\">...</Elements>"
        "<NodeSet name=\"bot\">1,2,3,4</NodeSet></Mesh>\n"
        "  <MeshDomains><SolidDomain name=\"P1\" mat=\"M1\"/>"
        "</MeshDomains>\n"
        "  <Boundary>...</Boundary>\n"
        "  <LoadData><load_controller id=\"1\" name=\"LC1\" "
        "type=\"loadcurve\">...</load_controller></LoadData>\n"
        "</febio_spec>"),

    "isotropic_elastic_constitutive_law": (
        "FEBio's `isotropic elastic` is NOT small-strain linear "
        "elasticity — it is St. Venant-Kirchhoff: "
        "S = lambda*tr(E)*I + 2*mu*E with E the Green-Lagrange "
        "strain, reported as Cauchy stress sigma = F S F^T / J "
        "(FEIsotropicElastic::Stress evaluates the algebraically "
        "identical b*(lambda*trE - mu)/J + b^2*mu/J). Verified "
        "numerically to 1e-12 relative by a hex8 patch test with "
        "F = [[1.02,0.01,0],[0,0.99,0.005],[0,0,1.01]] — see "
        "scripts/tier2_fixtures/febio/hex8_patch_test_exact_stress. "
        "Consequence: results agree with linear elasticity only to "
        "O(strain); a verification study must either keep strains "
        "tiny or compare against the SVK closed form."),

    "shipped_template_execution_status": (
        "All 17 templates in this backend were generated with "
        "default parameters and run through FEBio 4.12 on "
        "2026-08-03. Five reach `N O R M A L   T E R M I N A T I O N`: "
        "linear_elasticity_3d_cube, elasticity_mms_3d_cube_hex8, "
        "hyperelasticity_3d_cube, fiber_reinforced_3d_hgo, "
        "active_contraction_3d_fiber. The other twelve do not, and "
        "the reasons are catalogued in the per-physics pitfalls: "
        "seven (damage, fluid, growth_remodeling, multiphasic, "
        "polar_fluid, rigid_body, viscoelasticity) are rejected "
        "for a <material> without a name= "
        "attribute; biphasic_3d_confined and fluid_fsi_3d_block "
        "die on the unsymmetric-matrix/skyline mismatch; "
        "plasticity_3d_uniaxial uses the 3.x placement of "
        "max_ups; heat_3d_bar and biphasic_fsi_3d_block SEGFAULT "
        "because `heat` and `biphasic-FSI` are not modules in "
        "4.12. Treat those twelve as unverified stubs, not as "
        "worked examples."),
}
