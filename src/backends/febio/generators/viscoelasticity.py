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
    c1 = params.get("c1", 1.0)
    k = params.get("bulk_modulus", 1000.0)
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
    <material id="1" name="Material1" type="uncoupled viscoelastic">
      <density>1.0</density>
      <g1>{g1}</g1>
      <t1>{t1}</t1>
      <g2>{g2}</g2>
      <t2>{t2}</t2>
      <elastic type="Mooney-Rivlin">
        <density>1.0</density>
        <c1>{c1}</c1>
        <c2>0.0</c2>
        <k>{k}</k>
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
    <logfile>
      <element_data data="sz;J" delim="," file="visco_relax.csv"/>
    </logfile>
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
                "g1, g2, ...": "Prony coefficients, dimensionless. "
                               "They scale the INSTANTANEOUS stiffness "
                               "UP from the ground state and do NOT set "
                               "the long-time plateau — the <elastic> "
                               "child sets that. sum(g_i) >= 1 is legal "
                               "and harmless, and no value is "
                               "range-checked (0, 1, 5 and -0.5 all "
                               "run). See the [Numerical] pitfall; the "
                               "old 'sum must be < 1' rule was "
                               "falsified by execution.",
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
                "[Input] `uncoupled viscoelastic` accepts ONLY an "
                "UNCOUPLED elastic child, and a coupled one is "
                "reported as a MISSING property rather than as a "
                "mismatch — so the message points at the wrong "
                "problem. "
                "WRONG: <material id=\"1\" name=\"M1\" "
                "type=\"uncoupled viscoelastic\"><density>1.0</density>"
                "<g1>0.5</g1><t1>1.0</t1>"
                "<elastic type=\"neo-Hookean\"><density>1</density>"
                "<E>1000</E><v>0.3</v></elastic></material> — "
                "`neo-Hookean` and `isotropic elastic` are COUPLED. "
                "RIGHT: <material id=\"1\" name=\"M1\" "
                "type=\"uncoupled viscoelastic\"><density>1.0</density>"
                "<g1>0.5</g1><t1>1.0</t1>"
                "<elastic type=\"Mooney-Rivlin\"><density>1</density>"
                "<c1>1</c1><c2>0</c2><k>1000</k></elastic></material>. "
                "Signal: `Component \"M1\" needs to have property "
                "\"elastic\" defined (line N)` and `Reading file "
                "...FAILED!`, where M1 is the material's own name. "
                "Executed with a coupled `neo-Hookean` child, a "
                "coupled `isotropic elastic` child, and NO child at "
                "all: all three give the byte-identical message, so "
                "the message cannot distinguish \"wrong kind of "
                "child\" from \"no child\". If you see it, check the "
                "child's kind before assuming the tag is missing. The "
                "Mooney-Rivlin form above runs to "
                "`N O R M A L   T E R M I N A T I O N`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The Prony coefficients g_i scale the "
                "INSTANTANEOUS stiffness UPWARD from the elastic "
                "ground state; they do NOT reduce the long-time "
                "plateau, and sum(g_i) >= 1 is neither forbidden nor "
                "harmful. FEBio's `viscoelastic` wrapper adds "
                "relaxing terms ON TOP of its <elastic> child, so the "
                "long-time response IS the child's response, always. "
                "WRONG: expecting the relaxed plateau to be "
                "(1 - sum(g_i)) * initial_stress, or picking g_i to "
                "set the plateau. "
                "RIGHT: pick the <elastic> child to set the plateau, "
                "and pick g_i to set how far ABOVE that plateau the "
                "instantaneous response sits. "
                "Signal: none — every variant runs clean, which is why "
                "this went uncaught. Measure it: ramp fast, then HOLD "
                "at constant stretch and run long compared with t_i, "
                "logging <element_data data=\"sz\"/>. Executed as a "
                "g1 sweep of 0.3, 0.7, 0.95, 1.0 and 1.5 on two "
                "meshes: the plateau stress came out IDENTICAL at "
                "every value of g1, and bit-identical to a pure "
                "neo-Hookean control deck with the same ground state, "
                "while the peak stress rose monotonically with g1. "
                "g1 is also NOT range-checked — 0, 1, 5 and even -0.5 "
                "all run to normal termination, so a non-physical "
                "value is entirely silent. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d. This "
                "FALSIFIES the previous claim that sum(g_i) >= 1 sends "
                "the long-time response to zero.)"
            ),
            (
                "[Numerical] A relaxation time t_i shorter than the "
                "LOAD RAMP makes the relaxation invisible: it has "
                "already finished before the ramp ends, so the "
                "recorded peak equals the plateau and the material "
                "behaves as if purely elastic. Too LONG a t_i hides "
                "the plateau instead, for the opposite reason — the "
                "run ends before the material has relaxed. Both "
                "produce a clean run. "
                "WRONG: t_i much smaller than the ramp duration, or "
                "much larger than t_end. "
                "RIGHT: t_i comparable to the ramp duration, with "
                "t_end at least several times t_i so the plateau is "
                "reached. "
                "Signal: none — measure the drop from peak to final "
                "stress. Executed on two meshes with a fixed ramp and "
                "t_i swept over six decades: at t_i three decades "
                "below the ramp the peak-to-plateau drop was zero to "
                "printed precision, two decades below it was about a "
                "percent, at t_i comparable to the ramp it reached "
                "tens of percent, and at t_i above t_end the measured "
                "drop fell again because the plateau had not yet been "
                "reached. Both meshes gave the same drop to four "
                "digits, so this is a time-integration property and "
                "not a discretization artefact. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] The relaxation is integrated by a recursive convolution that carries internal state per integration point, so the stress at a given time depends on the whole load history and not only on the current strain. "
                "WRONG: assuming a shorter or coarser run reproduces a longer one, or resuming a run without carrying the internal state across. "
                "RIGHT: run the full history in one job, and if you must restart, use FEBio's own dump/restart mechanism so the state travels with it. "
                "Signal: none from the solver — measure it yourself. Refining the TIME STEP alone does not change the answer, which is the reassuring half. On two meshes, sweeping the step count over a factor of eight at fixed t_end changed the final relaxed stress only in the seventh significant digit while the peak moved in the fourth — so the scheme is converged in time at ordinary step counts, and a discrepancy larger than that between two runs of the same history is NOT step-size error and should be investigated as a state-handling problem. "
                "STILL UNVERIFIED: the restart behaviour itself. Closing it needs a run interrupted and resumed through FEBio's dump-file mechanism (the -r restart flag plus a <Control> dump setting), compared against an uninterrupted reference of the same history. The claim is retained rather than softened, so that whoever runs that comparison can close it. "
                "(Time-step half executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
        ],
    },
}


GENERATORS = {
    "viscoelasticity_3d_stress_relax": _viscoelasticity_3d_stress_relax,
}
