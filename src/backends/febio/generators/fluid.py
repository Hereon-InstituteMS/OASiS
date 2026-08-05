"""FEBio fluid generators and knowledge.

FEBio Module type: 'fluid'. Incompressible Newtonian fluid solved via
FEBio's pressure-velocity fluid solver. The hallmark module for
cardiovascular CFD, blood-flow benchmarks, and the fluid half of FSI
problems before coupling with 'fluid-FSI'.
"""


def _fluid_3d_channel(params: dict) -> str:
    """Pressure-driven channel flow in a 1x1x1 cube.

    Inlet face (x=0) has prescribed effective pressure; outlet face
    (x=1) has zero effective pressure; lateral walls are no-slip.
    """
    rho = params.get("density", 1.0)
    mu = params.get("viscosity", 0.01)
    p_in = params.get("p_inlet", 0.01)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="fluid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="fluid">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="fluid">
      <density>{rho}</density>
      <k>1e3</k>
      <viscous type="Newtonian fluid">
        <mu>{mu}</mu>
      </viscous>
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
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fluid": {
        "description": (
            "Incompressible / quasi-incompressible Newtonian fluid via "
            "FEBio's fluid solver (Module type='fluid'). Velocity + "
            "dilatation primary variables; pressure is derived. Suited "
            "to cardiovascular CFD benchmarks and the fluid half of "
            "FSI workflows."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (fluid solver)",
        "materials": {
            "fluid": {
                "density": "Mass density rho",
                "k": "Bulk modulus (penalises near-incompressibility)",
                "viscous": "Nested viscous law: 'Newtonian fluid' (mu), "
                           "'Carreau' (mu_0, mu_inf, lambda, n), "
                           "'Carreau-Yasuda', or 'power-law'.",
            },
        },
        "pitfalls": [
            (
                "[Syntax] A `fluid` material needs <Module type=\"fluid\"/>, and the first thing that fails if you leave the module as solid is the SOLVER, not the material — so the message points somewhere you did not edit. "
                "WRONG: <Module type=\"solid\"/> with <solver type=\"fluid\"> and a `fluid` material. "
                "RIGHT: <Module type=\"fluid\"/> with <solver type=\"fluid\"><symmetric_stiffness>non-symmetric</symmetric_stiffness><linear_solver type=\"bicgstab\"/></solver>. "
                "Signal: `tag \"solver\" (line N) : invalid value for attribute \"type\"` and `Reading file ...FAILED!`. The solver factory is module-scoped, so an out-of-module solver name is rejected before the <Material> section is ever read. Do not go looking for a material problem. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] Fluid velocity DOFs are wx/wy/wz, and a solid-style name is a hard parse error indistinguishable from a made-up one. "
                "WRONG: <bc type=\"zero fluid velocity\" node_set=\"walls\"><y_dof>1</y_dof></bc>. "
                "RIGHT: <bc name=\"noslip\" type=\"zero fluid velocity\" node_set=\"walls\"><wy_dof>1</wy_dof><wz_dof>1</wz_dof></bc>. "
                "Signal: `tag \"y_dof\" (line N) : unrecognized tag` and `Reading file ...FAILED!`. Executed against a deliberately invented name too: the message is byte-identical, so it tells you the tag is unknown and nothing more — it will not hint that you wanted the w prefix. (Executed 2026-08-05, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] Bulk modulus k controls how strictly "
                "near-incompressibility is enforced. Too low: large "
                "spurious dilatation; too high: ill-conditioning and "
                "Newton stalls. Rule of thumb: k > 100 * mu * "
                "(u_max / L) to keep dilatation < 1%. Signal: "
                "fluid velocity field has visible volume change "
                "(div(v) != 0 by >1%), or solver reports cond > 1e14 "
                "with the k that was too aggressive. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] CFL-like restriction on dt for "
                "convection-dominated flow: dt < h / max|v|. The "
                "fluid solver is implicit but accuracy still depends "
                "on dt; very large steps smooth out transient "
                "features. Signal: a vortex-shedding benchmark "
                "(cylinder in cross-flow) shows no shedding when dt "
                "is larger than ~0.1 * shedding_period. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "fluid_3d_channel": _fluid_3d_channel,
}
