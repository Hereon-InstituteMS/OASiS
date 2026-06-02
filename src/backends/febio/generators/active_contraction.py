"""FEBio active-contraction generators and knowledge.

FEBio Module type: 'solid' with active-contraction materials. The
'solid mixture' container combines a passive elastic skeleton with a
contractile fiber bundle whose internal stress varies with an
activation parameter (typically a calcium-controlled scalar). Two
common active models:
  - 'active fiber stress' — direct Ca-driven contractile stress
  - 'prescribed uniaxial active contraction' — table-driven activation

Canonical for cardiac chamber modeling, skeletal-muscle gait studies,
gastric peristalsis, and any biological tissue with controllable
internal contraction.
"""


def _active_contraction_3d_fiber(params: dict) -> str:
    """A passive neo-Hookean matrix + active contractile fiber bundle
    along the z-axis. The activation curve ramps from 0 to T_max,
    pulling the block in via fiber contraction.

    Demonstrates the FEBio idiom for cardiac active stress: solid
    mixture wraps a passive 'neo-Hookean' base and an active
    'prescribed uniaxial active contraction' fiber model with
    load-controller-driven activation.
    """
    E_passive = params.get("E", 50.0)
    nu = params.get("nu", 0.45)
    T_max = params.get("activation_max", 100.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="solid mixture">
      <density>1.0</density>
      <mat_axis type="vector">
        <a>0,0,1</a>
        <d>1,0,0</d>
      </mat_axis>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E_passive}</E>
        <v>{nu}</v>
      </solid>
      <solid type="prescribed uniaxial active contraction">
        <T0 lc="1">{T_max}</T0>
      </solid>
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
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>0.5,1</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="fiber vector"/>
      <var type="active fiber stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "active_contraction": {
        "description": (
            "Active contraction of fiber-reinforced tissue via "
            "FEBio's 'solid mixture' wrapping a passive elastic "
            "base + an active contractile fiber. The active "
            "component contributes additional fiber-axis stress "
            "controlled by a load-curve (or, in coupled "
            "cardiac-EM models, by a Ca transient). Canonical for "
            "cardiac chamber dynamics, skeletal-muscle gait "
            "studies, peristalsis."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, DYNAMIC analysis",
        "materials": {
            "solid mixture": {
                "mat_axis": "Material orientation frame (required to "
                            "define the active-fiber direction)",
                "solid (1)": "Passive elastic base (neo-Hookean, HGO, "
                             "Mooney-Rivlin, ...)",
                "solid (2)": "Active contractile fiber model. "
                             "Common options: "
                             "'prescribed uniaxial active contraction' "
                             "(T0 lc=N), "
                             "'active fiber stress' (sigma_max, "
                             "Ca50, n), "
                             "'Guccione cardiac contraction'.",
            },
        },
        "pitfalls": [
            (
                "[Input] Active-contraction materials live INSIDE a "
                "'solid mixture' container — they cannot stand alone "
                "as the top-level material. Signal: emitting "
                "<material type='prescribed uniaxial active "
                "contraction'> at top level raises FEBio "
                "`material must have a passive base in a solid "
                "mixture` at parse time. (Audit 2026-06-02.)"
            ),
            (
                "[Input] mat_axis defines the fiber direction; "
                "omitting it leaves the active stress pointing "
                "along global x. For cardiac chamber models with "
                "spatially-varying fiber direction, use the "
                "'vector field' or 'shell' mat_axis form to vary "
                "per-element. Signal: simulated contraction "
                "produces uniform z-strain when fibers should be "
                "helical; visualize 'fiber vector' to confirm "
                "direction matches anatomy. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] T0 (active stress amplitude) "
                "interacts with the passive stiffness — T0 << "
                "E_passive yields negligible contraction; T0 >> "
                "E_passive can cause non-physical fiber buckling "
                "or self-intersection. Realistic cardiac range: "
                "T0 ~ 50-150 kPa for a passive stiffness ~10-50 "
                "kPa. Signal: visualize the contraction "
                "trajectory — fiber-direction strain should be in "
                "the [-30%, +5%] range for biological tissues. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Time integration of active "
                "contraction is stiff — start with small dt and "
                "use SMOOTH ramping on the activation curve. "
                "Step changes in T0 produce a velocity impulse "
                "that requires many small Newton steps to damp. "
                "Signal: kinetic energy oscillates with > 50% "
                "amplitude for the first ~20 steps if "
                "interpolate=LINEAR + small dt_0 is used "
                "instead of SMOOTH. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "active_contraction_3d_fiber": _active_contraction_3d_fiber,
}
