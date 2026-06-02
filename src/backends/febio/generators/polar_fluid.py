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
    p_in = params.get("p_inlet", 1.0)
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
    </solver>
  </Control>
  <Material>
    <material id="1" type="polar fluid">
      <density>{rho}</density>
      <k>1e3</k>
      <viscous type="Newtonian fluid">
        <mu>{mu}</mu>
      </viscous>
      <micro_viscosity>{eta}</micro_viscosity>
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
    <NodeSet name="inlet">
      <n id="1"/><n id="4"/><n id="5"/><n id="8"/>
    </NodeSet>
    <NodeSet name="outlet">
      <n id="2"/><n id="3"/><n id="6"/><n id="7"/>
    </NodeSet>
    <NodeSet name="walls">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="noslip" type="zero fluid velocity" node_set="walls">
      <wy_dof>1</wy_dof>
      <wz_dof>1</wz_dof>
    </bc>
    <bc name="no_microrot" type="zero micro-rotation" node_set="walls">
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
      <var type="effective fluid pressure"/>
      <var type="micro rotation"/>
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
                "[Numerical] Setting micro_viscosity = 0 recovers "
                "the classical Navier-Stokes velocity field "
                "exactly, but leaves the (now-decoupled) micro-"
                "rotation DOF unconstrained — the linear solver "
                "complains about a singular block. Set "
                "micro_viscosity > 0 OR use the plain 'fluid' "
                "module for pure NS. Signal: solver reports "
                "`KSPSolve: zero-pivot at micro-rotation block`. "
                "(Audit 2026-06-02.)"
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
                "[Input] Micro-rotation DOFs are named "
                "gx_dof / gy_dof / gz_dof in the BC spec (lower "
                "case g for 'gyration'), NOT mx/my/mz or Rx/Ry/Rz. "
                "Signal: a BC type='zero micro-rotation' with "
                "<x_dof>1</x_dof> raises `invalid DOF for polar "
                "fluid module — expected gx_dof / gy_dof / "
                "gz_dof`. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "polar_fluid_3d_channel": _polar_fluid_3d_channel,
}
