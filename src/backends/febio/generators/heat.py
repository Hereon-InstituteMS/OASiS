"""FEBio heat-transfer generators and knowledge.

FEBio Module type: 'heat'. Steady-state and transient heat conduction
with isotropic Fourier conductivity.
"""


def _heat_3d_bar(params: dict) -> str:
    """Steady-state heat conduction in a 1x1x1 bar with
    Dirichlet temperatures on opposite faces."""
    k = params.get("conductivity", 1.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="heat"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
  </Control>
  <Material>
    <material id="1" type="isotropic Fourier">
      <density>1.0</density>
      <capacity>1.0</capacity>
      <conductivity>{k}</conductivity>
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
    <NodeSet name="cold_face">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="hot_face">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="cold" type="prescribed temperature" node_set="cold_face">
      <value lc="1">0.0</value>
    </bc>
    <bc name="hot" type="prescribed temperature" node_set="hot_face">
      <value lc="1">100.0</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
</febio_spec>
'''


KNOWLEDGE = {
    "heat": {
        "description": (
            "NOT AVAILABLE on FEBio 4.12 — this row is retained "
            "only to carry the falsification. FEBio 4.12.0 "
            "registers exactly ten modules (solid, biphasic, "
            "solute, multiphasic, fluid, fluid-FSI, "
            "multiphasic-FSI, fluid-solutes, thermo-fluid, "
            "polar fluid) and `heat` is not one of them. The "
            "shipped heat_3d_bar template CRASHES the solver. "
            "For conduction-type problems use another OASiS "
            "backend (4C, deal.II, FEniCSx, NGSolve all have "
            "verified heat templates); the closest FEBio "
            "physics is the `thermo-fluid` module, which solves "
            "heat transport in a compressible fluid, not "
            "conduction in a solid."),
        "input_format": "FEBio XML v4.0",
        "status": (
            "FALSIFIED 2026-08-03 by live execution on FEBio "
            "4.12.0.86045466d. Every element of the previous "
            "entry — the `heat` module, the `isotropic Fourier` "
            "material, the `prescribed temperature` BC — was "
            "checked against the binary's own factory registry "
            "(`printf 'list\\nquit\\n' | febio4 -nosplash`) and "
            "against real runs. None of them exist."),
        "materials": {},
        "pitfalls": [
            (
                "[Syntax] There is no `heat` module in FEBio "
                "4.12, and an unregistered <Module type=...> "
                "value is not diagnosed — it is a hard crash. "
                "FEBioModuleSection.cpp passes the string "
                "straight to FEModelBuilder::SetActiveModule() "
                "with no existence check. Signal: stdout stops "
                "mid-line at `Reading file <name>.feb ...` with "
                "no `SUCCESS!`, no `FAILED!`, no ERROR box and "
                "no .log file, and the process dies with SIGSEGV "
                "(exit status 139). Verified identically for "
                "type=\"heat\", type=\"biphasic-FSI\", and for "
                "mere case errors such as type=\"Solid\". A "
                "wrapper that only greps stderr for the word "
                "`error` sees a completely silent failure. "
                "(Live-verified 2026-08-03, FEBio 4.12.0; "
                "fixture scripts/tier2_fixtures/febio/"
                "unknown_module_type_segfaults.)"
            ),
            (
                "[Syntax] `isotropic Fourier` is not a "
                "registered material on FEBio 4.12 — the "
                "FEMATERIAL_ID factory list has 159 entries and "
                "contains no Fourier conduction law at all. "
                "Signal: `tag \"material\" (line N) : invalid "
                "value for attribute \"type\"` and `Reading file "
                "...FAILED!`, identical for the spellings "
                "`isotropic Fourier`, `Fourier` and "
                "`isotropic_Fourier` — the previous catalog "
                "entry's claim that only the underscore/short "
                "spellings fail is wrong, all three fail. "
                "(Live-verified 2026-08-03, FEBio 4.12.0.)"
            ),
            (
                "[Syntax] There is no `prescribed temperature` "
                "boundary condition in the FEBC_ID registry. The "
                "only temperature BCs are `thermo-fluid."
                "prescribed fluid temperature`, `thermo-fluid."
                "zero fluid temperature` and `thermo-fluid."
                "natural temperature`, all scoped to the "
                "thermo-fluid module and acting on a fluid "
                "temperature DOF, not a solid one. Signal: "
                "`tag \"bc\" (line N) : invalid value for "
                "attribute \"type\"`. (Live-verified 2026-08-03, "
                "FEBio 4.12.0.)"
            ),
        ],
    },
}


GENERATORS = {
    "heat_3d_bar": _heat_3d_bar,
}
