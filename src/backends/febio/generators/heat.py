"""FEBio heat-transfer generators and knowledge.

FEBio Module type: 'heat'. Steady-state and transient heat conduction
with isotropic Fourier conductivity.
"""


def _heat_3d_bar(params: dict) -> str:
    """Steady-state heat conduction in a 1x1x1 bar with
    Dirichlet temperatures on opposite faces."""
    k = params.get("conductivity", 1.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="heat"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
  </Control>
  <Material>
    <material id="1" type="isotropic Fourier">
      <density>1.0</density>
      <capacity>1.0</capacity>
      <conductivity>{k}</conductivity>
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
    <NodeSet name="cold_face">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="hot_face">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="cold" type="prescribed temperature" node_set="cold_face">
      <value lc="1">0.0</value>
    </bc>
    <bc name="hot" type="prescribed temperature" node_set="hot_face">
      <value lc="1">100.0</value>
    </bc>
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
    "heat": {
        "description": "Steady-state heat conduction (FEBio Module type='heat')",
        "input_format": "FEBio XML v4.0",
        "solver": "Linear or non-linear thermal solver",
        "materials": {
            "isotropic Fourier": {
                "density": "Mass density",
                "capacity": "Specific heat capacity",
                "conductivity": "Thermal conductivity",
            },
        },
        "pitfalls": [
            "[Syntax] Module type MUST be 'heat' (NOT 'solid'). "
            "Signal: stderr contains 'invalid module type' from "
            "FEBio when the wrong Module type is used.",
            "[Syntax] Temperature BCs use 'prescribed temperature' "
            "type (NOT 'prescribed displacement'). "
            "Signal: FEBio reports 'invalid BC type' at parse time.",
            "[Syntax] The thermal conductivity material in "
            "FEBio 4.0 is the <isotropic Fourier> type — NOT "
            "<thermal>, <conductive>, or <Fourier>. The exact "
            "attribute spelling 'isotropic Fourier' (lowercase "
            "noun, capitalized proper noun, space-separated) is "
            "required. "
            "Signal: emitting <material type='Fourier'> or "
            "<material type='isotropic_Fourier'> (underscore) "
            "raises an FEBio 'unknown material' parse error "
            "naming the supplied type string verbatim.",
        ],
    },
}


GENERATORS = {
    "heat_3d_bar": _heat_3d_bar,
}
