"""FEBio plasticity generators and knowledge.

FEBio Module type: 'solid' with rate-independent plasticity materials.
J2 (von Mises) plasticity with isotropic + kinematic hardening, plus
specialised models: 'Hill orthotropic', 'plastic flow curve' (user-
defined isotropic hardening), and the reactive-plastic combination
material for cyclic / multi-scale work.

Common in biomechanics for cortical bone yielding, calcified tissue
post-yield response, surgical-tool plastic deformation, and metal
implants in orthopedic-load benchmarks.
"""


def _plasticity_3d_uniaxial(params: dict) -> str:
    """Uniaxial tension test on a hex8 cube — J2 plasticity with linear
    isotropic hardening. Bottom face fixed, top face pulled in z with
    prescribed displacement past the yield strain. Plastic flow
    activates above sigma_yield and follows the hardening modulus E_h.
    """
    E = params.get("E", 200000.0)
    nu = params.get("nu", 0.3)
    sig_y = params.get("yield_stress", 250.0)
    E_h = params.get("hardening_modulus", 1000.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>20</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <max_ups>0</max_ups>
    </solver>
  </Control>
  <Material>
    <material id="1" type="J2 plasticity">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
      <Y0>{sig_y}</Y0>
      <H>{E_h}</H>
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
    <bc name="pull" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">0.02</value>
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
      <var type="displacement"/>
      <var type="stress"/>
      <var type="elastic strain"/>
      <var type="plastic strain"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "plasticity": {
        "description": (
            "Rate-independent plasticity (J2 / von Mises and "
            "specialised models) on FEBio's solid module. Captures "
            "yielding, hardening, and residual strain after unload "
            "for cortical bone, metal implants, surgical-tool "
            "deformation, and any material exceeding its elastic "
            "limit."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Solid solver with stress-update return-mapping integration",
        "materials": {
            "J2 plasticity": {
                "E": "Young's modulus", "v": "Poisson's ratio",
                "Y0": "Initial yield stress (von Mises)",
                "H": "Linear isotropic hardening modulus (dY/dep)",
            },
            "Hill orthotropic plasticity": {
                "F, G, H, L, M, N": "Hill anisotropic yield "
                                    "coefficients in the material "
                                    "frame",
            },
            "plastic flow curve": {
                "elastic": "Nested elastic material",
                "Y0, Y1, ..., Yn at ep0, ep1, ..., epn": "User-"
                    "defined piecewise-linear hardening curve",
            },
        },
        "pitfalls": [
            (
                "[Numerical] Step size must resolve the yield "
                "transition. A single step jumping from below-"
                "yield to far-above-yield can mis-locate the "
                "plastic region. Signal: stress vs strain curve "
                "shows a kink at sigma > Y0 (yield is detected "
                "late); reducing step_size moves the kink to "
                "Y0 within ~1%. (Audit 2026-06-02.)"
            ),
            (
                "[Input] J2 plasticity uses Y0 (initial yield) + H "
                "(linear hardening modulus) — NOT yield_stress + "
                "hardening_modulus (those are the human names). "
                "Signal: parser warns `unknown material parameter "
                "yield_stress`, or computed yield happens at the "
                "wrong stress level because the parameter was "
                "silently defaulted to a built-in 0.0. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] H << E gives near-perfect plasticity "
                "(no hardening); H ~ E gives near-elastic (small "
                "permanent strain). The realistic range for "
                "engineering metals is H ~ 0.001 * E to 0.05 * E. "
                "Signal: H >> E in the catalog template produces "
                "essentially elastic response indistinguishable "
                "from the elasticity benchmark. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Cyclic loading needs kinematic "
                "(Bauschinger) hardening — pure isotropic J2 "
                "gives a symmetric yield surface that does not "
                "shift in the deviatoric plane and cannot "
                "reproduce reverse-yield softening seen in "
                "tension-compression tests. Signal: simulated "
                "tension/compression cycle gives near-identical "
                "yield stress on the return leg (~|Y0|) when "
                "experiments show reduced yield (~|Y0 - "
                "Bauschinger_shift|). (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "plasticity_3d_uniaxial": _plasticity_3d_uniaxial,
}
