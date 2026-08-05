"""FEBio polar-fluid generators and knowledge.

FEBio Module type: 'polar fluid'. A micropolar (Cosserat) fluid model
that augments the standard Navier-Stokes velocity field with an
independent micro-rotation field. Captures size-dependent rheology in
suspensions of rigid particles (red blood cells, polymer fluids,
granular slurries) and turbulent-boundary-layer near-wall behaviour
where classical NS over-predicts shear.
"""


def _polar_fluid_3d_channel(params: dict) -> str:
    """Pressure-driven channel flow with micropolar effects. Same
    geometry as the basic fluid template but with an additional
    micro-rotation field that's zero at the no-slip walls (matches
    the convention for a viscous polar fluid in a smooth channel).
    """
    rho = params.get("density", 1.0)
    mu = params.get("viscosity", 0.01)
    eta = params.get("micropolar_viscosity", 0.001)
    p_in = params.get("p_inlet", 0.01)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="polar fluid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="polar fluid">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="polar fluid">
      <density>{rho}</density>
      <k>1e3</k>
      <viscous type="Newtonian fluid">
        <mu>{mu}</mu>
      </viscous>
      <polar type="polar linear">
        <tau>{eta}</tau>
        <alpha>0.0</alpha>
        <beta>{eta}</beta>
        <gamma>{eta}</gamma>
      </polar>
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
    <Elements type="hex8" mat="Material1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="inlet">1,4,5,8</NodeSet>
    <NodeSet name="outlet">2,3,6,7</NodeSet>
    <NodeSet name="walls">1,2,3,4,5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
  </MeshDomains>
  <Boundary>
    <bc name="noslip" type="zero fluid velocity" node_set="walls">
      <wy_dof>1</wy_dof>
      <wz_dof>1</wz_dof>
    </bc>
    <bc name="no_microrot" type="zero fluid angular velocity" node_set="walls">
      <gx_dof>1</gx_dof><gy_dof>1</gy_dof><gz_dof>1</gz_dof>
    </bc>
    <bc name="p_in" type="prescribed fluid dilatation" node_set="inlet">
      <value lc="1">{p_in}</value>
    </bc>
    <bc name="p_out" type="prescribed fluid dilatation" node_set="outlet">
      <value lc="1">0.0</value>
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
      <var type="fluid velocity"/>
      <var type="fluid pressure"/>
      <var type="polar fluid angular velocity"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "polar_fluid": {
        "description": (
            "Micropolar (Cosserat) fluid via FEBio's 'polar fluid' "
            "module. Adds an independent micro-rotation field on "
            "top of standard fluid velocity. Used for suspensions "
            "of rigid microparticles (red blood cells in plasma, "
            "polymer fluids, granular slurries), and for near-wall "
            "turbulence corrections where classical Navier-Stokes "
            "over-predicts wall shear."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric solver with extended (velocity + rotation) DOFs",
        "materials": {
            "polar fluid": {
                "density": "Mass density rho",
                "k": "Bulk modulus",
                "viscous": "Nested viscous law (Newtonian / Carreau)",
                "micro_viscosity": "Couple-stress micropolar viscosity "
                                   "eta (lengthscale-controlling); "
                                   "set 0 to recover classical NS.",
            },
        },
        "pitfalls": [
            (
                "[Input] Module type 'polar fluid' adds a micro-"
                "rotation DOF triplet (gx_dof / gy_dof / gz_dof) "
                "that needs its own BC at walls — typically "
                "'zero micro-rotation' for a smooth wall. Forgetting "
                "the rotation BC leaves the micro-rotation field "
                "free at the boundary, producing unphysical "
                "spinning at the walls. Signal: opening the .xplt "
                "plotfile and visualizing the micro_rotation output "
                "channel near the no-slip wall shows non-zero "
                "values where gx_dof / gy_dof / gz_dof should be "
                "~0 — the BC was missed. (Audit 2026-06-02.)"
            ),
            (
                "[Input] There is no <micro_viscosity> parameter, so the question of setting it to zero does not arise. On FEBio 4.12 the polar viscosity is a PROPERTY: <polar type=\"polar linear\"> with <tau>, <alpha>, <beta> and <gamma>. "
                "WRONG: <micro_viscosity>0.001</micro_viscosity> as a parameter of the `polar fluid` material. "
                "RIGHT: <material id=\"1\" name=\"Material1\" type=\"polar fluid\"><density>1.0</density><k>1e3</k><viscous type=\"Newtonian fluid\"><mu>0.01</mu></viscous><polar type=\"polar linear\"><tau>0.001</tau><alpha>0.0</alpha><beta>0.001</beta><gamma>0.001</gamma></polar></material>. "
                "Signal: `tag \"micro_viscosity\" (line N) : unrecognized tag` and `Reading file ...FAILED!`. The material also carries its own <k> bulk modulus; the only registered type for the <polar> slot on this build is `polar linear`. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The micropolar correction scales with "
                "the length ratio (micropolar lengthscale) / "
                "(channel half-width). For typical fluids "
                "(blood, polymer melts) the correction is at the "
                "1-10% level — too small to detect on a coarse "
                "grid. Refine the cross-flow direction by at "
                "least 16 elements before claiming a polar effect. "
                "Signal: comparing the velocity profile between "
                "polar and classical fluid shows < 0.5% difference "
                "on a 4-cell mesh; the same comparison on 32 "
                "cells shows the expected 1-10%. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Syntax] The micro-rotation DOF names gx_dof / gy_dof / gz_dof are correct, but the BC TYPE that carries them is `zero fluid angular velocity` — there is no `zero micro-rotation`. The matching plot variable is `polar fluid angular velocity`, not `micro rotation`. "
                "WRONG: <bc type=\"zero micro-rotation\" node_set=\"walls\">, or <var type=\"micro rotation\"/>. "
                "RIGHT: <bc name=\"no_microrot\" type=\"zero fluid angular velocity\" node_set=\"walls\"><gx_dof>1</gx_dof><gy_dof>1</gy_dof><gz_dof>1</gz_dof></bc>, and <var type=\"polar fluid angular velocity\"/>. "
                "Signal: for the BC, `tag \"bc\" (line N) : invalid value for attribute \"type\"` at parse time. For the plot variable the failure is LATER and looks different: the deck reads `...SUCCESS!` and then aborts with `FATAL ERROR: Output variable \"micro rotation\" is not defined`. The registered polar plot variables all begin with \"polar fluid\" or \"nodal polar fluid\". (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "polar_fluid_3d_channel": _polar_fluid_3d_channel,
}
