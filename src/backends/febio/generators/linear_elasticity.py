"""FEBio linear-elasticity generators and knowledge.

FEBio Module type: 'solid'. Material: 'isotropic elastic' (small strain).
"""


def _elasticity_3d_cube(params: dict) -> str:
    """Unit cube with prescribed-displacement uniaxial compression.

    Linear isotropic elastic material; STATIC analysis; FEBio v4.0 XML.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>0</T>
      <R>0</R>
      <Fc>0</Fc>
    </Constants>
  </Globals>
  <Material>
    <material id="1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
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
      <x_dof>1</x_dof>
      <y_dof>1</y_dof>
      <z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt>
        <pt>1,1</pt>
      </points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity with FEBio — isotropic elastic material",
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": "Newton-Raphson with direct linear solver",
        "materials": {
            "isotropic elastic": {"E": "Young's modulus (Pa)", "v": "Poisson's ratio"},
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "material constant 1", "c2": "material constant 2"},
        },
        "pitfalls": [
            (
                "[Input] FEBio uses lowercase 'v' for Poisson's "
                "ratio (NOT 'nu'). The 'nu' name is convention in "
                "FEniCSx / deal.II / NGSolve but FEBio's XML schema "
                "rejects it. Signal: input parser raises "
                "`unknown material parameter nu in isotropic "
                "elastic` or `material 1: parameter nu not "
                "found`; replacing <nu> with <v> resolves the "
                "error. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Element connectivity is 1-indexed. Node "
                "id=0 does not exist in FEBio's XML schema; many "
                "mesh-conversion scripts (PyVista, meshio) "
                "default to 0-indexing and need an explicit +1 "
                "offset. Signal: input parser aborts with "
                "`element references node 0 — node IDs must "
                "start at 1` at mesh-loading time. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] MeshDomains section is REQUIRED in v4.0 "
                "(links each domain to a material id). Older "
                "v3.x .feb files put the mat attribute directly "
                "on the <Elements> tag; v4.0 requires the "
                "explicit MeshDomains/SolidDomain mapping. "
                "Signal: FEBio aborts with `domain Part1 has no "
                "MeshDomains entry — required for v4.0` or "
                "silently associates the domain with material "
                "id=1 (giving the wrong material if there are "
                "multiple). (Audit 2026-06-02.)"
            ),
            (
                "[Input] LoadData section with <load_controller> "
                "is needed for any prescribed BC that uses "
                "`lc=N` attribute. Forgetting the LoadData block "
                "while keeping `lc=1` on a BC value silently "
                "leaves the BC at its initial value (no ramping). "
                "Signal: prescribed-displacement BC stays at 0 "
                "throughout the simulation; result shows no "
                "applied loading even though the input file looks "
                "correct. Look for the `lc=N` attribute on every "
                "<value> and confirm each N has a matching "
                "<load_controller id=N>. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "linear_elasticity_3d_cube": _elasticity_3d_cube,
}
