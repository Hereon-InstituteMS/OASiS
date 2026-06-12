"""FEBio viscoelasticity generators and knowledge.

FEBio Module type: 'solid' with viscoelastic material wrappers. Two
families:
  - 'uncoupled viscoelastic' — Prony-series deviatoric relaxation on
    an underlying nearly-incompressible elastic material
  - 'viscoelastic'           — coupled volumetric+deviatoric (full)

Common use cases: stress-relaxation tests on soft tissue (cartilage,
ligament, tendon), creep response, frequency-domain mechanical
testing in the time domain.
"""


def _viscoelasticity_3d_stress_relax(params: dict) -> str:
    """Uncoupled viscoelastic stress-relaxation test: hold a step
    displacement on the top face and observe the stress decay.

    Two-term Prony series on a neo-Hookean ground state. Run time long
    enough to capture both relaxation modes."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.499)
    g1 = params.get("g1", 0.4)
    t1 = params.get("t1", 0.5)
    g2 = params.get("g2", 0.3)
    t2 = params.get("t2", 5.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>50</time_steps>
    <step_size>0.2</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="uncoupled viscoelastic">
      <density>1.0</density>
      <g1>{g1}</g1>
      <t1>{t1}</t1>
      <g2>{g2}</g2>
      <t2>{t2}</t2>
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
    <bc name="step_load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>STEP</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>0.01,1</pt><pt>10,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
      <var type="relative volume"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "viscoelasticity": {
        "description": (
            "Time-dependent viscoelastic solid mechanics via FEBio's "
            "Prony-series viscoelastic material wrappers. Used for "
            "stress-relaxation tests on cartilage / ligament / "
            "tendon, creep response of soft tissue, and time-domain "
            "frequency-response analyses."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, transient DYNAMIC analysis",
        "materials": {
            "uncoupled viscoelastic": {
                "elastic": "Nested ground-state elastic material "
                           "(typically nearly-incompressible like "
                           "neo-Hookean with v=0.499)",
                "g1, g2, ...": "Prony coefficients (dimensionless "
                               "fractions; sum should be < 1 to "
                               "leave a non-zero elastic floor)",
                "t1, t2, ...": "Relaxation times (matching units of "
                               "the simulation time step)",
            },
            "viscoelastic": {
                "elastic": "Nested ground-state COUPLED elastic "
                           "material (not uncoupled)",
                "g1..gN, t1..tN": "Same as uncoupled",
            },
        },
        "pitfalls": [
            (
                "[Input] 'uncoupled viscoelastic' requires a nested "
                "uncoupled elastic material (neo-Hookean, Mooney-"
                "Rivlin, Ogden uncoupled, ...). Pairing it with a "
                "COUPLED elastic (e.g. 'St.Venant-Kirchhoff') is "
                "an error. Signal: FEBio aborts with "
                "`uncoupled viscoelastic requires uncoupled "
                "ground-state material` at parse time. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Sum of Prony coefficients g_i must be "
                "< 1 — otherwise the long-time response goes to "
                "zero (full relaxation) instead of a non-zero "
                "elastic floor. Signal: in a stress-relaxation "
                "test the stress decays to 0 instead of "
                "(1 - sum(g_i)) * initial_stress; visualizing the "
                "stress vs time curve shows no plateau. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Relaxation times t_i should span the "
                "physical range of interest (~0.1 * sim_time to "
                "~1.0 * sim_time). Setting all t_i << dt makes "
                "FEBio integrate the Prony series with vanishing "
                "memory effect; setting all t_i >> sim_time gives "
                "essentially purely elastic response. Signal: "
                "stress vs time plot looks identical to the "
                "purely-elastic reference (no decay) or to the "
                "fully-relaxed asymptote (instant drop). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Time integration uses a recursive "
                "convolution scheme that needs the FULL history; "
                "restarting from a checkpoint without preserving "
                "the internal state variables yields incorrect "
                "stress evolution at the restart step. Signal: "
                "stress curve has a visible jump at the restart "
                "time stamp compared to a single-run reference. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "viscoelasticity_3d_stress_relax": _viscoelasticity_3d_stress_relax,
}
