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
    <material id="1" type="biphasic">
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
    <NodeSet name="fix_bottom">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="load_top">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
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
        ],
    },
}


GENERATORS = {
    "biphasic_3d_confined": _biphasic_3d_confined,
}
