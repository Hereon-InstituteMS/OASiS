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
            "when FEBio is built without Intel MKL (e.g. on Apple Silicon, "
            "where MKL is unavailable) — is symmetric-only and aborts "
            "immediately with 0 linear-solver calls: 'The selected linear "
            "solver does not support the requested matrix format'. Use a "
            "non-symmetric solver: 'pardiso' (MKL builds) or 'accelerate' "
            "(Apple Accelerate sparse solver on macOS; registered in "
            "NumCore/NumCore.cpp as \"accelerate\"). Set it install-wide via "
            "<default_linear_solver type=\"accelerate\"/> in febio.xml, or per "
            "run via <linear_solver> in the Control/solver block.",
            "[Syntax] FEBio 4.x <material> requires a 'name' attribute (in "
            "addition to id), and <MeshDomains> reference the material by that "
            "name (mat=\"<name>\"). Omitting name fails at parse with "
            "'tag \"material\" ... missing attribute \"name\"' "
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
