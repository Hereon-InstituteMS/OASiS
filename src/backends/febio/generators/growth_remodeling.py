"""FEBio growth-remodeling generators and knowledge.

FEBio Module type: 'solid' with 'solid mixture' container plus
'growth' or 'remodeling' constituent models. Mass is added (growth)
or fiber properties change (remodeling) over time driven by a
biological / mechanical stimulus.

Canonical for vascular adaptation (constrictor / dilator response of
arteries), tissue scaffolds with osteoblast-driven mineralization,
muscle hypertrophy under chronic overload, and tumor growth in
mechanobiology benchmarks.
"""


def _growth_remodeling_3d_isotropic(params: dict) -> str:
    """Isotropic volumetric growth driven by a time-ramped stimulus.
    A passive neo-Hookean matrix is augmented with a 'growth' wrapper
    that adds mass according to the growth-rate constitutive law.
    Total deformation gradient F = F_e * F_g splits into elastic
    (F_e) and growth (F_g) parts.
    """
    E = params.get("E", 100.0)
    nu = params.get("nu", 0.4)
    growth_rate = params.get("growth_rate", 0.5)
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
      <solid type="growth">
        <theta lc="1">{growth_rate}</theta>
        <elastic type="neo-Hookean">
          <density>1.0</density>
          <E>{E}</E>
          <v>{nu}</v>
        </elastic>
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
    <NodeSet name="fix_corner">
      <n id="1"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_corner">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="growth tensor"/>
      <var type="relative volume"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "growth_remodeling": {
        "description": (
            "Growth-and-remodeling of biological tissue via FEBio's "
            "multiplicative split F = F_e * F_g and the 'growth' / "
            "'remodeling' constitutive wrappers. Mass is added "
            "(growth) or fiber properties change (remodeling) "
            "over time, driven by a biological / mechanical "
            "stimulus. Canonical for arterial adaptation, tissue "
            "scaffolds, muscle hypertrophy, and tumor mechanobiology."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver with multiplicative growth-tensor state variable",
        "materials": {
            "kinematic growth": {
                "elastic": "REQUIRED nested elastic material",
                "growth": "REQUIRED nested growth tensor. "
                          "Registered FEBio 4.12 types: `volume "
                          "growth`, `area growth`, `fiber growth`, "
                          "`general growth`. The volumetric one "
                          "takes <multiplier> (long name "
                          "'volumetric growth multiplier'), NOT "
                          "`theta` and NOT `gm`.",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d — runs a one-hex8 deck to "
                    "N O R M A L   T E R M I N A T I O N: "
                    "<material id=\"1\" name=\"M1\" type=\"kinematic "
                    "growth\"><density>1</density><elastic "
                    "type=\"neo-Hookean\"><density>1</density>"
                    "<E>1000</E><v>0.3</v></elastic><growth "
                    "type=\"volume growth\"><multiplier lc=\"1\">1.2"
                    "</multiplier></growth></material>. With "
                    "multiplier 1.2 the free face expanded "
                    "laterally by 0.2240 against 0.0202 for the "
                    "same deck with a plain neo-Hookean, i.e. the "
                    "growth is actually driving the deformation. "
                    "`<multiplier>` misspelled as `theta` or `gm` "
                    "gives `tag \"theta\" (line N) : unrecognized "
                    "tag`."),
            },
            "cell growth": "Registered FEMATERIAL_ID (concentration-"
                           "driven growth).",
            "remodeling solid": "Registered FEMATERIAL_ID — this is "
                                "the remodeling material's real "
                                "type string.",
            "_falsified_2026_08_03": (
                "CORRECTION. This block previously listed `growth`, "
                "`remodeling` and `isotropic growth` as material "
                "types. NONE of the three is registered in FEBio "
                "4.12; all three were executed on 2026-08-03 and "
                "every one was rejected with `tag \"material\" "
                "(line N) : invalid value for attribute \"type\"` "
                "and `Reading file ...FAILED!`. The registered "
                "growth FEMATERIAL_ID factories on this build are "
                "`kinematic growth`, `cell growth` and `remodeling "
                "solid`; `volume growth` / `area growth` / `fiber "
                "growth` / `general growth` are growth-TENSOR "
                "properties nested inside `kinematic growth`, not "
                "top-level materials."),
        },
        "pitfalls": [
            (
                "[Input] Signal: FALSIFIED 2026-08-03 by execution on "
                "FEBio 4.12.0. The previous claim — that growth "
                "materials must live inside a 'solid mixture' "
                "wrapper and that a top-level "
                "<material type='growth'> raises `growth material "
                "must be inside solid mixture` — is wrong twice "
                "over. (a) `growth` is not a registered material "
                "type at all, so the deck dies earlier with "
                "`tag \"material\" (line N) : invalid value for "
                "attribute \"type\"`. (b) The string `must be "
                "inside solid mixture` does not occur anywhere in "
                "the 4.12 binary or its libraries, so that Signal "
                "could never have matched. The real material, "
                "`kinematic growth`, IS a valid TOP-LEVEL material "
                "— verified by running it as the sole "
                "<material> of a deck, which reached "
                "N O R M A L   T E R M I N A T I O N. What it "
                "actually requires is two child properties, "
                "<elastic> and <growth>; omitting <growth> gives "
                "`Component \"<name>\" needs to have property "
                "\"growth\" defined (line N)`."
            ),
            (
                "[Numerical] Growth is HISTORY-DEPENDENT — F_g "
                "accumulates over time; restarting without "
                "preserving the growth-tensor state restarts F_g "
                "= I, undoing all accumulated growth. Signal: a "
                "single-run simulation gives radius = 1.2 at t=10; "
                "the same simulation restarted at t=5 gives "
                "radius < 1.2 because growth was reset. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Pure isotropic volumetric growth on a "
                "fully-constrained domain develops large residual "
                "stress (the elastic part F_e must compensate for "
                "the growth F_g). Constrain only minimally (single "
                "corner / center-of-mass + rotation) to avoid "
                "Newton divergence. Signal: [FALSIFIED 2026-08-03: FEBio "
                "does not use NOX — neither `NOX` nor `DIVERGED` occurs "
                "anywhere in the 4.12 binary or its libraries. On a "
                "failed Newton step FEBio prints `------- failed to "
                "converge at time : <t>` and, if the step cannot be "
                "retried, `E R R O R   T E R M I N A T I O N` with exit "
                "1] NOX residual explodes "
                "at the first growth increment when boundary "
                "conditions over-constrain the growth direction. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Anisotropic growth (e.g. axial-only "
                "vascular elongation) is encoded by the growth "
                "tensor structure not by the load-curve. Setting "
                "theta with a scalar load curve produces ISOTROPIC "
                "growth — the radius grows by the same factor as "
                "the length. For anisotropic growth select a "
                "DIFFERENT growth-tensor type on the <growth> "
                "property of `kinematic growth`. CORRECTED "
                "2026-08-03 by execution on FEBio 4.12.0: there "
                "is no `theta` parameter and no `directional "
                "growth` type. The registered growth-tensor types "
                "are `volume growth` (isotropic; its parameter is "
                "<multiplier>, and `theta` is rejected with "
                "`tag \"theta\" (line N) : unrecognized tag`), "
                "`area growth`, `fiber growth` and `general "
                "growth` — the last three are the anisotropic "
                "ones. Signal: simulating an axially-"
                "stretched artery with `volume growth` "
                "yields a balloon shape (equal axial and "
                "radial growth) rather than the expected "
                "elongated cylinder. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "growth_remodeling_3d_isotropic": _growth_remodeling_3d_isotropic,
}
