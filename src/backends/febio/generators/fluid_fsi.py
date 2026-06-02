"""FEBio fluid-FSI generators and knowledge.

FEBio Module type: 'fluid-FSI'. Strongly-coupled monolithic fluid-
structure interaction: a deformable solid and an incompressible fluid
share a moving interface. The fluid mesh deforms with the solid
(ALE). Hallmark FEBio module for arterial-wall hemodynamics, cardiac
chamber modeling, and any compliant-vessel benchmark.
"""


def _fluid_fsi_3d_block(params: dict) -> str:
    """Compliant block facing a fluid column: solid block (z<0.5) +
    fluid block (z>=0.5) sharing the z=0.5 interface. Fluid pressure
    pushes against the solid; solid deflection drives ALE mesh
    motion in the fluid domain.

    This is a deliberately compact demonstration topology — real FSI
    studies use much larger fluid/solid domains and meshed with
    multiple element blocks per region.
    """
    E = params.get("E", 1e4)
    nu = params.get("nu", 0.3)
    rho_f = params.get("density_fluid", 1.0)
    mu = params.get("viscosity", 0.01)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="fluid-FSI"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="fluid-FSI">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="neo-Hookean">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
    <material id="2" type="fluid-FSI">
      <fluid type="fluid">
        <density>{rho_f}</density>
        <k>1e3</k>
        <viscous type="Newtonian fluid">
          <mu>{mu}</mu>
        </viscous>
      </fluid>
      <solid type="neo-Hookean">
        <density>{rho_f}</density>
        <E>1.0</E>
        <v>0.0</v>
      </solid>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,0.5</node>
      <node id="6">1,0,0.5</node>
      <node id="7">1,1,0.5</node>
      <node id="8">0,1,0.5</node>
      <node id="9">0,0,1</node>
      <node id="10">1,0,1</node>
      <node id="11">1,1,1</node>
      <node id="12">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Solid">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <Elements type="hex8" mat="2" name="Fluid">
      <elem id="2">5,6,7,8,9,10,11,12</elem>
    </Elements>
    <NodeSet name="solid_base">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="fluid_top">
      <n id="9"/><n id="10"/><n id="11"/><n id="12"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Solid" mat="1"/>
    <SolidDomain name="Fluid" mat="2"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix_base" type="zero displacement" node_set="solid_base">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="p_top" type="prescribed fluid dilatation" node_set="fluid_top">
      <value lc="1">1.0</value>
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
      <var type="fluid velocity"/>
      <var type="effective fluid pressure"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fluid_fsi": {
        "description": (
            "Strongly-coupled monolithic fluid-structure interaction "
            "(Module type='fluid-FSI'). The fluid mesh deforms with "
            "the solid via Arbitrary Lagrangian-Eulerian (ALE). The "
            "FSI interface is implicit — material id pairs solid + "
            "fluid blocks and FEBio resolves the coupling each "
            "Newton iteration. Used for arterial-wall hemodynamics, "
            "cardiac chamber dynamics, valve modeling, and "
            "compliant-vessel benchmarks."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric monolithic FSI solver",
        "materials": {
            "fluid-FSI": {
                "fluid": "Nested fluid material (rho, k, viscous)",
                "solid": "Nested ALE-mesh-motion solid (typically "
                         "very soft — E~1 — to track interface)",
            },
        },
        "pitfalls": [
            (
                "[Input] Module type MUST be 'fluid-FSI'. Using "
                "'fluid' alone gives an Eulerian (non-moving-mesh) "
                "fluid that ignores the solid; using 'solid' gives "
                "no fluid. Signal: fluid domain shows no mesh "
                "deformation despite solid moving; or `material "
                "type fluid-FSI not allowed in module fluid`. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] The ALE solid material inside the FSI "
                "fluid block should be VERY soft (E~1) — its only "
                "role is to define mesh motion. A stiff ALE solid "
                "resists fluid pressure and produces spurious "
                "interface tractions. Signal: a pressure-driven "
                "channel shows uniform velocity field instead of "
                "developing a Poiseuille profile because the ALE "
                "mesh refuses to follow the interface. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] FSI interface is implicit via material "
                "ID adjacency — node sharing between the two element "
                "blocks at the interface is REQUIRED. Disjoint "
                "meshes won't couple. Signal: FEBio diagnostic "
                "prints `FSI interface: 0 shared nodes between mat=1 "
                "and mat=2`; the fluid pressure has zero effect on "
                "the solid response. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] FSI is stiff — start with small dt and "
                "use SMOOTH ramping of pressure / velocity BCs. "
                "Sudden load application (LINEAR + small dt_0) "
                "produces high-frequency oscillations in the "
                "interface that take many steps to damp out. "
                "Signal: kinetic energy at the interface oscillates "
                "with amplitude > 50% of mean for the first ~50 "
                "steps before settling. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "fluid_fsi_3d_block": _fluid_fsi_3d_block,
}
