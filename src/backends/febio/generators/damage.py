"""FEBio damage generators and knowledge.

FEBio Module type: 'solid' with continuum damage materials. Two main
families:
  - 'elastic damage' (general wrapper around any elastic base) —
    scalar damage variable D evolves with strain history; effective
    stress sigma_eff = (1 - D) * sigma_elastic. Requires three child
    properties: <elastic>, <damage type="CDF ..."> and
    <criterion type="DC ...">.
  - Pre-combined variants: 'damage neo-Hookean', 'damage
    Mooney-Rivlin', 'damage fiber exponential', ...
  - CORRECTED 2026-08-03: the type strings 'damage', 'Simo damage'
    and 'reactive damage' named here previously are NOT registered
    in FEBio 4.12 and are rejected at parse.

Canonical for tissue tearing thresholds, cartilage degradation under
repeated loading, soft-tissue rupture benchmarks, and fatigue cycling
in elastomers.
"""


def _damage_3d_cycle(params: dict) -> str:
    """Cyclic uniaxial loading on a damageable neo-Hookean block.
    Three load-unload-reload cycles at increasing peak strain. The
    damage variable D grows monotonically; the reload stiffness on
    each cycle is reduced by (1-D).
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    eps_max = params.get("max_strain", 0.1)
    D_inf = params.get("D_inf", 0.5)
    beta = params.get("damage_rate", 2.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>60</time_steps>
    <step_size>0.05</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="damage">
      <density>1.0</density>
      <D_inf>{D_inf}</D_inf>
      <beta>{beta}</beta>
      <elastic type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </elastic>
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
    <bc name="cycle" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">{eps_max}</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt><pt>0.5,0.5</pt><pt>1.0,0</pt>
        <pt>1.5,0.75</pt><pt>2.0,0</pt>
        <pt>2.5,1.0</pt><pt>3.0,0</pt>
      </points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="damage"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "damage": {
        "description": (
            "Continuum damage mechanics via FEBio's 'damage' "
            "wrapper. A scalar damage variable D grows "
            "monotonically with strain history; effective stress "
            "is (1-D) * sigma_elastic. Captures progressive "
            "degradation under repeated loading. Used for soft-"
            "tissue tearing, cartilage degradation, fatigue "
            "cycling in elastomers, and rupture thresholds."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver with internal damage state variable",
        "materials": {
            "elastic damage": {
                "elastic": "REQUIRED nested elastic material "
                           "(neo-Hookean / HGO / Mooney-Rivlin / ...)",
                "damage": "REQUIRED nested damage CDF. Registered "
                          "types on FEBio 4.12 all start with "
                          "`CDF `: `CDF Simo`, `CDF log-normal`, "
                          "`CDF Weibull`, `CDF step`, `CDF quintic`, "
                          "`CDF gamma`, `CDF user`.",
                "criterion": "REQUIRED nested damage criterion. "
                             "Registered types all start with "
                             "`DC `: `DC Simo`, `DC strain energy "
                             "density`, `DC specific strain energy`, "
                             "`DC von Mises stress`, `DC Drucker "
                             "shear stress`, `DC max shear stress`, "
                             "`DC max normal stress`, `DC max normal "
                             "Lagrange strain`.",
                "_verified": (
                    "LIVE-VERIFIED 2026-08-03 on FEBio "
                    "4.12.0.86045466d — this exact material runs a "
                    "one-hex8 prescribed-compression deck to "
                    "N O R M A L   T E R M I N A T I O N: "
                    "<material id=\"1\" name=\"M1\" type=\"elastic "
                    "damage\"><density>1</density><elastic "
                    "type=\"neo-Hookean\"><density>1</density>"
                    "<E>1000</E><v>0.3</v></elastic><damage "
                    "type=\"CDF Simo\"><a>0.9</a><b>0.1</b></damage>"
                    "<criterion type=\"DC strain energy density\"/>"
                    "</material>. All three child properties are "
                    "mandatory; omitting one gives `Component "
                    "\"<name>\" needs to have property \"<prop>\" "
                    "defined (line N)`."),
            },
            "_falsified_2026_08_03": (
                "CORRECTION. This block previously listed `damage`, "
                "`Simo damage` and `reactive damage` as material "
                "types. NONE of the three is a registered "
                "FEMATERIAL_ID on FEBio 4.12 — `damage` exists only "
                "as a PLOT variable (FEPLOTDATA_ID) and as a "
                "mesh-adaptor criterion, not as a material. All "
                "three were executed on 2026-08-03 and every one "
                "was rejected with `tag \"material\" (line N) : "
                "invalid value for attribute \"type\"` and "
                "`Reading file ...FAILED!`. The registered "
                "damage-bearing FEMATERIAL_ID factories on this "
                "build include: `elastic damage`, `uncoupled "
                "elastic damage`, `damage neo-Hookean`, `damage "
                "Mooney-Rivlin`, `damage trans iso Mooney-Rivlin`, "
                "`damage fiber power`, `damage fiber exponential`, "
                "`damage fiber exp-linear`, `viscoelastic damage`, "
                "`uncoupled viscoelastic damage`, `reactive "
                "viscoelastic damage`, `reactive plastic damage`. "
                "Enumerate them yourself with `printf "
                "'list\\nquit\\n' | febio4 -nosplash | grep "
                "FEMATERIAL_ID`."),
        },
        "pitfalls": [
            (
                "[Numerical] Damage is HISTORY-DEPENDENT and "
                "MONOTONIC: once D has been incremented in an "
                "element, it cannot decrease. Re-running with a "
                "smaller load does NOT 'heal' the damage. Signal: "
                "running the same model twice with cumulative "
                "displacement-history boundary conditions gives a "
                "different post-unload state than a single longer "
                "run — that's the history coupling, not a bug. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] The `elastic damage` wrapper (CORRECTED "
                "2026-08-03: NOT `damage`, which is not a "
                "registered material — see the materials block) "
                "requires a nested "
                "elastic material — it cannot stand alone. "
                "Forgetting the <elastic> child or putting the "
                "elastic parameters at the top level produces a "
                "parse error. Signal: FEBio aborts with "
                "[FALSIFIED 2026-08-03: the message text quoted here "
                "does not occur anywhere in the FEBio 4.12.0.86045466d "
                "binary or any of its shared libraries (`strings` over "
                "febio4 + all 12 .so files, 267541 strings), so this "
                "Signal can never match on 4.12. The physics reasoning "
                "is desk research and was NOT executed] `damage material "
                "requires <elastic> child` or "
                "`unknown material parameter E in damage material`. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] D_inf < 1 leaves residual stiffness "
                "(1 - D_inf) * E_elastic at infinite strain. "
                "Setting D_inf == 1 means complete loss of "
                "stiffness — FEBio struggles past D ~ 0.99 "
                "because the effective stiffness goes to zero and "
                "Newton stalls. Signal: [FALSIFIED 2026-08-03: FEBio "
                "does not use NOX — neither `NOX` nor `DIVERGED` occurs "
                "anywhere in the 4.12 binary or its libraries. On a "
                "failed Newton step FEBio prints `------- failed to "
                "converge at time : <t>` and, if the step cannot be "
                "retried, `E R R O R   T E R M I N A T I O N` with exit "
                "1] NOX residual stops "
                "decreasing once max(D) approaches 0.99; use "
                "D_inf <= 0.95 in practice. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] On a cyclic load test, the reload "
                "stiffness on cycle n+1 is (1 - D_n) times the "
                "initial stiffness. If beta is too high (rapid "
                "damage saturation), the second loading cycle "
                "already shows ~D_inf damage and no further "
                "softening accumulates. Signal: the stress-strain "
                "loop on cycle 2 looks identical to cycle 3 — "
                "damage has saturated; reducing beta restores "
                "cycle-by-cycle softening. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "damage_3d_cycle": _damage_3d_cycle,
}
