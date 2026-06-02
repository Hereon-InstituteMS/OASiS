"""FEBio hyperelasticity generators and knowledge.

FEBio Module type: 'solid'. Materials: 'neo-Hookean', 'Mooney-Rivlin',
'Ogden', 'Holmes-Mow', 'Yeoh', 'Veronda-Westmann'.
"""


def _hyperelasticity_3d_cube(params: dict) -> str:
    """Unit cube with neo-Hookean material under prescribed
    displacement (30% nominal strain), 10 load steps."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Material>
    <material id="1" type="neo-Hookean">
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
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.3</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/><var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "hyperelasticity": {
        "description": "Nonlinear hyperelasticity with FEBio — Neo-Hookean, Mooney-Rivlin",
        "materials": {
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "1st Mooney-Rivlin constant", "c2": "2nd constant",
                              "k": "bulk modulus"},
        },
        "pitfalls": [
            (
                "[Input] Use 'STATIC' analysis for quasi-static "
                "loading. 'DYNAMIC' adds an inertia term that "
                "couples acceleration to the stress balance and "
                "produces oscillation at the load-onset step on "
                "a problem where you only wanted equilibrium. "
                "Signal: visualizing displacement at step 1 shows "
                "a velocity-overshoot pattern (peak displacement "
                "exceeds the target by ~50%) instead of a clean "
                "load ramp; switching <analysis> to STATIC "
                "removes the overshoot. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Large deformations require proper "
                "step-size control. A single load step from zero "
                "to full deformation rarely converges past ~10% "
                "nominal strain for neo-Hookean / Mooney-Rivlin "
                "materials. Use ~10 time_steps with "
                "step_size = 0.1 to incrementally apply the "
                "prescribed displacement. Signal: NOX reports "
                "`DIVERGED_LINE_SEARCH` or `MaxIters` at step 1 "
                "for stretch ratios > 1.1; subdividing the load "
                "into 10 substeps converges quadratically per "
                "substep. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Convergence issues: reduce step size "
                "or enable line search. FEBio's solver has a "
                "<lstol> tag inside <solver> to enable line "
                "search with the default LS damping; aggressive "
                "load steps benefit from <max_ups>10</max_ups> "
                "to allow BFGS quasi-Newton updates between "
                "stiffness reassemblies. Signal: NOX residual "
                "oscillates between two values without "
                "converging — line search would catch this and "
                "back off the step. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "hyperelasticity_3d_cube": _hyperelasticity_3d_cube,
}
