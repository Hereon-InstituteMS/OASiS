"""FEBio biphasic-FSI generators and knowledge.

FEBio Module type: 'biphasic-FSI'. Combines a biphasic solid (porous
matrix + interstitial fluid) with an adjacent free fluid domain — the
biphasic side fluxes interstitial fluid across the interface into the
free-fluid side. Used for blood-tissue interaction (perfused
myocardium, atherosclerotic plaque, cartilage-synovial fluid), drug
elution from porous stents, and any scenario where fluid leaves /
enters a porous tissue and continues as free flow.
"""


def _biphasic_fsi_3d_block(params: dict) -> str:
    """Porous tissue block (z<0.5) + free-fluid column (z>=0.5)
    sharing the z=0.5 interface. Free-fluid pressure drives flow
    across the interface; biphasic tissue resorbs / releases
    interstitial fluid.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    rho_f = params.get("density_fluid", 1.0)
    mu = params.get("viscosity", 0.01)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="biphasic-FSI"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="biphasic-FSI">
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
    <material id="2" type="biphasic-FSI">
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
    <Elements type="hex8" mat="1" name="Tissue">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <Elements type="hex8" mat="2" name="FreeFluid">
      <elem id="2">5,6,7,8,9,10,11,12</elem>
    </Elements>
    <NodeSet name="tissue_base">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="fluid_top">
      <n id="9"/><n id="10"/><n id="11"/><n id="12"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Tissue" mat="1"/>
    <SolidDomain name="FreeFluid" mat="2"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix_base" type="zero displacement" node_set="tissue_base">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="tissue_base"/>
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
      <var type="solid stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "biphasic_fsi": {
        "description": (
            "Coupled biphasic-tissue + free-fluid FSI (Module "
            "type='biphasic-FSI'). The biphasic side resorbs / "
            "releases interstitial fluid into the free-fluid side "
            "across a shared interface. Canonical for blood-tissue "
            "perfusion (myocardium, plaque), cartilage-synovial "
            "fluid interaction, drug elution from porous stents, "
            "and bioreactor scaffolds with media perfusion."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Monolithic non-symmetric biphasic-FSI solver",
        "materials": {
            "biphasic-FSI": {
                "fluid": "Nested free-fluid material (rho, k, "
                         "viscous)",
                "solid": "Nested ALE-mesh-motion solid (very soft, "
                         "E~1) for the free-fluid side",
            },
            "biphasic (paired)": "On the porous-tissue side, use a "
                                 "standard 'biphasic' material with "
                                 "nested <solid> and "
                                 "<permeability>.",
        },
        "pitfalls": [
            (
                "[Input] Module type MUST be 'biphasic-FSI' — NOT "
                "'biphasic' nor 'fluid-FSI'. Using 'biphasic' loses "
                "the free-fluid side; 'fluid-FSI' loses the "
                "interstitial-fluid coupling. Signal: with the "
                "wrong module, interstitial fluid does not cross "
                "the interface — porous-tissue pressure rises with "
                "no resorption pathway, or free-fluid pressure "
                "drops with no source. (Audit 2026-06-02.)"
            ),
            (
                "[Input] The free-fluid side uses the 'biphasic-FSI' "
                "material; the porous-tissue side uses the standard "
                "'biphasic' material. Mixing them up (biphasic-FSI "
                "on the tissue, biphasic on the fluid) flips the "
                "coupling direction and produces nonsense. Signal: "
                "interface flux direction reversed — fluid leaks "
                "out of the supposed-tissue side instead of the "
                "supposed-free-fluid side. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Like fluid-FSI, the ALE solid on the "
                "free-fluid side must be very soft (E~1) so the "
                "mesh follows interface motion. A stiff ALE solid "
                "resists fluid pressure and yields spurious "
                "interface tractions. Signal: pressure-driven "
                "flow across the interface stalls because the "
                "ALE mesh refuses to deform; tissue side stays "
                "rigid. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface continuity is enforced via "
                "shared nodes between biphasic + biphasic-FSI "
                "element blocks. Disjoint meshes won't couple. "
                "Signal: FEBio prints `biphasic-FSI interface: 0 "
                "shared nodes between mat=1 and mat=2`; the "
                "biphasic side equilibrates as if isolated. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "biphasic_fsi_3d_block": _biphasic_fsi_3d_block,
}
