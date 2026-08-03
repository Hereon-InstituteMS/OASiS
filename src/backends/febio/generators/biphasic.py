"""FEBio biphasic generators and knowledge.

FEBio Module type: 'biphasic'. Solid skeleton + interstitial fluid with
explicit permeability. The hallmark FEBio module for soft-tissue mechanics
(cartilage, intervertebral disc, intervertebral disc, etc.).
"""


def _biphasic_3d_confined(params: dict) -> str:
    """Confined-compression biphasic test — solid skeleton with
    interstitial fluid, isotropic permeability.

    Top face drained (zero pore pressure); bottom face fixed.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="biphasic"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>10</time_steps>
    <step_size>1.0</step_size>
    <solver type="biphasic">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="biphasic">
      <phi0>0.2</phi0>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </solid>
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">1,2,3,4</NodeSet>
    <NodeSet name="load_top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="load_top"/>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
</febio_spec>
'''


KNOWLEDGE = {
    "biphasic": {
        "description": "Biphasic poroelasticity — solid skeleton + interstitial fluid (FEBio Module type='biphasic')",
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (biphasic solver)",
        "materials": {
            "biphasic": {
                "phi0": "Solid volume fraction at reference",
                "solid": "Nested solid material (e.g. neo-Hookean)",
                "permeability": "Nested permeability model "
                                "(perm-const-iso, perm-Holmes-Mow, perm-exp-iso, etc.)",
            },
        },
        "pitfalls": [
            "[Syntax] Module type MUST be 'biphasic' (NOT 'solid'). "
            "Wrong Module type causes the biphasic material to be "
            "rejected at input-parse time. "
            "Signal: stderr contains 'unknown material type' or "
            "'invalid module' from FEBio.",
            "[Syntax] biphasic material requires NESTED <solid> and "
            "<permeability> elements with their own type attribute. "
            "Flat parameter lists (E, v directly inside biphasic) "
            "are not valid. "
            "Signal: input parse fails with 'invalid material parameter'.",
            "[Numerical] Pore-pressure boundary conditions use "
            "'zero fluid pressure' or 'prescribed fluid pressure' "
            "BC types — separate from displacement BCs. "
            "Signal: silent stagnation of pressure field if drainage "
            "BC is missing from the loaded surface.",
            "[Solver] Biphasic (and multiphasic / fluid / FSI) systems are "
            "NON-SYMMETRIC. The linear solver must support a non-symmetric "
            "matrix format. FEBio's default 'skyline' solver — the fallback "
            "when FEBio is built without Intel MKL — is symmetric-only. "
            "Signal: the deck READS cleanly (`Reading file ...SUCCESS!`) and "
            "then aborts with `The selected linear solver does not support "
            "the requested matrix format.`, `Number of time steps completed "
            "... : 0` and `E R R O R   T E R M I N A T I O N`, exit status 1 "
            "— note a 1353-byte mesh-only .xplt is still written. "
            "LIVE-VERIFIED 2026-08-03 on a USE_MKL=OFF source build of FEBio "
            "4.12.0, sweeping all 20 registered FELINEARSOLVER_ID factories: "
            "ONLY `bicgstab` accepts the unsymmetric format. skyline, fgmres, "
            "cg, schur, bipn, superlu_mt, accelerate, diagonal, hypre_gmres, "
            "pardiso-project and strategy all emit the message above; `LU` "
            "fails with `Linear solver failed to find solution`; `ilu0` with "
            "`Fatal error in factorization of stiffness matrix`; `block` "
            "with `An error occurred during preprocessing of linear solver`. "
            "CORRECTIONS to the previous entry: `pardiso` is not even a valid "
            "type string without MKL (`tag \"linear_solver\" (line N) : "
            "invalid value for attribute \"type\"`) — the registered "
            "MKL-free spelling is `pardiso-project`; and `accelerate`, "
            "previously recommended here, does NOT accept the unsymmetric "
            "format on this build. Use <solver type=\"biphasic\">"
            "<linear_solver type=\"bicgstab\"/></solver>.",
            "[Solver] The <analysis> vocabulary is installed by the ACTIVE "
            "MODULE, so it differs from the solid module's. In `biphasic` "
            "(and `solute` / `multiphasic`) the legal words are "
            "`STEADY-STATE` and `TRANSIENT`; the solid module's `STATIC` / "
            "`DYNAMIC` are rejected, and the fluid-family modules use a "
            "third pairing (`STEADY-STATE` / `DYNAMIC`). Signal: `tag "
            "\"analysis\" (line N) : invalid value: STATIC` with `Reading "
            "file ...FAILED!` — a clean parse error, not a silent fallback "
            "to ordinal 0. Enum words are matched case-insensitively and the "
            "raw ordinal is also accepted, so `<analysis>1</analysis>` "
            "quietly selects TRANSIENT here and DYNAMIC in a solid deck; "
            "prefer the spelled-out word. (Live-verified 2026-08-03, FEBio "
            "4.12.0; enum tables from FEBiphasicAnalysis.cpp / "
            "FESolidAnalysis.cpp setEnums().)",
            "[Solver] <symmetric_stiffness>symmetric</symmetric_stiffness> "
            "is a legitimate workaround when only skyline is available: it "
            "changes the NEWTON TANGENT only, not the residual, so the "
            "converged answer is unchanged to solver tolerance. Verified on "
            "a confined-compression deck with strain-dependent "
            "perm-Holmes-Mow permeability (where the biphasic tangent really "
            "is unsymmetric): symmetric+skyline and non-symmetric+bicgstab "
            "agreed on sz to 6 significant digits over 4 steps "
            "(-126.160499 vs -126.160529). Signal: the cost is convergence "
            "robustness, not accuracy — expect more `------- failed to "
            "converge at time :` lines and more BFGS reformations, not a "
            "wrong stress. Do not treat a symmetric-stiffness run as "
            "automatically invalid. (Live-verified 2026-08-03, FEBio "
            "4.12.0.)",
            "[Syntax] FEBio 4.x <material> requires a 'name' attribute (in "
            "addition to id), and <MeshDomains> reference the material by that "
            "name (mat=\"<name>\"). Signal: omitting the name attribute fails at "
            "parse with 'tag \"material\" ... missing attribute \"name\"' "
            "(required in FEBioXML/FEBioMaterialSection.cpp). NodeSets declared "
            "in <Mesh> take a comma-separated node-id list "
            "(<NodeSet name=\"x\">1,2,3,4</NodeSet>), NOT <n id=.../> child "
            "elements, which FEBio 4.x rejects with 'invalid value'.",
        ],
    },
}


GENERATORS = {
    "biphasic_3d_confined": _biphasic_3d_confined,
}
