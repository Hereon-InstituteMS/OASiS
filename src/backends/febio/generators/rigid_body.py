"""FEBio rigid-body generators and knowledge.

FEBio Module type: 'solid' with material type 'rigid body'. A rigid
body is a single element-block constrained to translate / rotate as a
single unit. Used for impactors, fixtures, articulating joints
(combined with rigid connectors), and contact-prescribed boundary
conditions.
"""


def _rigid_body_3d_pushdown(params: dict) -> str:
    """Rigid impactor (top) pushes down on a deformable block (bottom)
    via prescribed rigid-body translation. The deformable solid uses
    isotropic elastic material; the impactor uses rigid body.

    Demonstrates the FEBio idiom for prescribing motion to one body
    while letting another deform — central to indentation / impact /
    contact-mechanics benchmarks.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    dz = params.get("displacement", -0.1)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
    <material id="2" type="rigid body">
      <density>10.0</density>
      <center_of_mass>0.5,0.5,1.0</center_of_mass>
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
    <Elements type="hex8" mat="1" name="Deformable">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <Elements type="hex8" mat="2" name="Impactor">
      <elem id="2">5,6,7,8,9,10,11,12</elem>
    </Elements>
    <NodeSet name="base">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Deformable" mat="1"/>
    <SolidDomain name="Impactor" mat="2"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="base">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
  <Rigid>
    <rigid_bc name="impactor_x" type="rigid_fixed">
      <rb>2</rb>
      <Rx_dof>1</Rx_dof>
      <Ry_dof>1</Ry_dof>
      <Ru_dof>1</Ru_dof>
      <Rv_dof>1</Rv_dof>
      <Rw_dof>1</Rw_dof>
    </rigid_bc>
    <rigid_bc name="impactor_z" type="rigid_displacement">
      <rb>2</rb>
      <dof>Rz</dof>
      <value lc="1">{dz}</value>
    </rigid_bc>
  </Rigid>
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
      <var type="rigid body position"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "rigid_body": {
        "description": (
            "FEBio rigid-body material — a single material id whose "
            "elements all translate and rotate as one rigid unit. "
            "Used for impactors, fixtures, articulating joints, "
            "rigid-body contact prescription, and any body whose "
            "internal deformation is irrelevant to the analysis. "
            "Lives inside Module type='solid' but uses dedicated "
            "<Rigid> section instead of the standard <Boundary> "
            "block."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver, augmented with rigid-body DOFs (6 per rigid body)",
        "materials": {
            "rigid body": {
                "density": "Mass density (used for dynamics)",
                "center_of_mass": "Centre of mass coordinates (drives "
                                  "rotational dynamics; required for "
                                  "transient runs).",
            },
        },
        "pitfalls": [
            (
                "[Input] Rigid-body BCs go in a SEPARATE <Rigid> "
                "section, NOT in <Boundary>. Each rigid_bc "
                "references the material id via <rb>N</rb>, not "
                "a node set. Signal: putting "
                "`<bc type='prescribed displacement'>` on rigid-"
                "body nodes raises `cannot prescribe displacement "
                "on rigid body node — use Rigid section` or the BC "
                "is silently ignored. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Rigid-body DOF names are Rx/Ry/Rz "
                "(translation) and Ru/Rv/Rw (rotation), capitalized. "
                "Standard x/y/z DOF names are for FE nodes, not "
                "rigid bodies. Signal: a <dof>x</dof> in a "
                "rigid_displacement BC raises `invalid rigid DOF "
                "x — expected one of Rx,Ry,Rz,Ru,Rv,Rw`. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] center_of_mass is REQUIRED for dynamics "
                "(transient analyses). For purely static runs it "
                "can be omitted, but moments around an undeclared "
                "centre default to (0,0,0) and may give "
                "unphysical rotation. Signal: a transient rigid-"
                "body simulation aborts with `rigid body N missing "
                "center_of_mass` or the body rotates around (0,0,0) "
                "instead of its geometric centre. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Contact between rigid + deformable "
                "needs a <Contact> block (typically "
                "sliding-elastic). Without it the impactor passes "
                "through the deformable body. Signal: visualizing "
                "the run shows the rigid mesh overlapping the "
                "deformable mesh; no contact force / stress at "
                "the expected interface. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "rigid_body_3d_pushdown": _rigid_body_3d_pushdown,
}
