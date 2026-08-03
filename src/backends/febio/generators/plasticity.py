"""FEBio plasticity generators and knowledge.

FEBio Module type: 'solid' with rate-independent plasticity materials.
The registered plasticity FEMATERIAL_ID factories on FEBio 4.12 are
'von-Mises plasticity' (E, v, Y, H), 'reactive plasticity' and
'reactive plastic damage' — verified against the binary's own `list`
dump on 2026-08-03. 'J2 plasticity', 'Hill orthotropic plasticity'
and 'plastic flow curve' are NOT registered and are rejected at
parse; they were removed from this file on 2026-08-03.

Common in biomechanics for cortical bone yielding, calcified tissue
post-yield response, surgical-tool plastic deformation, and metal
implants in orthopedic-load benchmarks.
"""


def _plasticity_3d_uniaxial(params: dict) -> str:
    """Uniaxial tension test on a hex8 cube — von-Mises plasticity with linear
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
      <qn_method type="full Newton"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="Material1" type="von-Mises plasticity">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
      <Y>{sig_y}</Y>
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
    <NodeSet name="fix_bottom">1,2,3,4</NodeSet>
    <NodeSet name="load_top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="Material1"/>
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
            "von-Mises plasticity": {
                "E": "Young's modulus", "v": "Poisson's ratio",
                "Y": "Initial yield stress (von Mises). NOT `Y0`.",
                "H": "Linear isotropic hardening modulus (dY/dep)",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d: <material id=\"1\" "
                    "name=\"M1\" type=\"von-Mises plasticity\">"
                    "<density>1</density><E>1000</E><v>0.3</v>"
                    "<Y>10</Y><H>10</H></material> reads SUCCESS "
                    "and runs to N O R M A L   T E R M I N A T I O N."),
            },
            "_falsified_2026_08_03": (
                "CORRECTION. This block previously listed three "
                "materials — `J2 plasticity`, `Hill orthotropic "
                "plasticity` and `plastic flow curve`. NONE of "
                "the three is registered in FEBio 4.12: they do "
                "not appear in the `list` factory dump under "
                "FEMATERIAL_ID, and a deck using any of them is "
                "rejected at parse with `tag \"material\" (line "
                "N) : invalid value for attribute \"type\"` "
                "(executed for `J2 plasticity` and `Hill "
                "orthotropic plasticity`). The registered "
                "plasticity FEMATERIAL_ID factories on this build "
                "are exactly: `von-Mises plasticity`, `reactive "
                "plasticity`, `reactive plastic damage`. Enumerate "
                "them yourself with "
                "`printf 'list\\nquit\\n' | febio4 -nosplash | "
                "grep FEMATERIAL_ID`."),
        },
        "pitfalls": [
            (
                "[Numerical] Step size must resolve the yield "
                "transition. A single step jumping from below-"
                "yield to far-above-yield can mis-locate the "
                "plastic region. Signal: stress vs strain curve "
                "shows a kink at sigma > Y (yield is detected "
                "late); reducing step_size moves the kink to "
                "Y within ~1%. (Audit 2026-06-02 — physics "
                "reasoning, NOT executed; only the parameter "
                "spelling was corrected to `Y` on 2026-08-03.)"
            ),
            (
                "[Input] The plasticity material is "
                "`von-Mises plasticity` and its yield parameter is "
                "`Y`. Signal: the wrong parameter name is a HARD "
                "parse error naming the tag, not a warning and not "
                "a silent default — `<yield_stress>250</yield_stress>` "
                "gives `tag \"yield_stress\" (line N) : "
                "unrecognized tag` and `Reading file ...FAILED!`. "
                "CORRECTED 2026-08-03 by execution on FEBio "
                "4.12.0: the previous entry claimed the material "
                "was `J2 plasticity`, the parameter was `Y0`, and "
                "the failure was a WARNING reading `unknown "
                "material parameter yield_stress` followed by a "
                "silent default to 0.0. All three are false. "
                "`J2 plasticity` is not a registered type; `Y0` is "
                "rejected with `tag \"Y0\" (line N) : unrecognized "
                "tag`; and the string `unknown material parameter` "
                "does not occur anywhere in the 4.12 binary or its "
                "libraries, so that Signal could never have "
                "matched. Nothing is ever silently defaulted here."
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
                "(Bauschinger) hardening — pure isotropic "
                "von-Mises "
                "gives a symmetric yield surface that does not "
                "shift in the deviatoric plane and cannot "
                "reproduce reverse-yield softening seen in "
                "tension-compression tests. Signal: simulated "
                "tension/compression cycle gives near-identical "
                "yield stress on the return leg (~|Y|) when "
                "experiments show reduced yield (~|Y0 - "
                "Bauschinger_shift|). (Audit 2026-06-02 — physics reasoning, NOT executed.)"
            ),
        ],
    },
}


GENERATORS = {
    "plasticity_3d_uniaxial": _plasticity_3d_uniaxial,
}
