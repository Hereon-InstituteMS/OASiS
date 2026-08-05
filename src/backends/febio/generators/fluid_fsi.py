"""FEBio fluid-FSI generators and knowledge.

FEBio Module type: 'fluid-FSI'. Strongly-coupled monolithic fluid-
structure interaction: a deformable solid and an incompressible fluid
share a moving interface. The fluid mesh deforms with the solid
(ALE). Hallmark FEBio module for arterial-wall hemodynamics, cardiac
chamber modeling, and any compliant-vessel benchmark.
"""


def _fluid_fsi_3d_block(params: dict) -> str:
    """Deformable block carrying an internal fluid, using the `fluid-FSI`
    MATERIAL in the `fluid-FSI` MODULE, consolidated by a prescribed
    compression of its top face.

    The fluid velocity and the dilatation degree of freedom are fully
    constrained. That is required, not cosmetic: on a USE_MKL=OFF build
    the FSI family does not converge with the fluid velocity free. The
    earlier version of this template left it free and failed at the
    first step at every ALE stiffness from 1 to 1e6. See the [Solver]
    pitfall.
    """
    n = max(1, int(params.get("n", 4)))
    W = float(params.get("width", 0.2))
    L = float(params.get("height", 1.0))
    E = float(params.get("E", 100.0))
    rho_f = float(params.get("density_fluid", 1.0))
    mu = float(params.get("viscosity", 0.01))
    uz = float(params.get("compression", -0.05))
    steps = max(1, int(params.get("time_steps", 10)))
    dt = float(params.get("step_size", 0.02))

    def nid(i, j, k):
        return 1 + i + (n + 1) * (j + (n + 1) * k)

    nodes = [f'      <node id="{nid(i,j,k)}">'
             f'{i/n*W},{j/n*W},{k/n*L}</node>'
             for k in range(n + 1) for j in range(n + 1)
             for i in range(n + 1)]
    el, e = [], 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                e += 1
                c = (nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k),
                     nid(i, j+1, k), nid(i, j, k+1), nid(i+1, j, k+1),
                     nid(i+1, j+1, k+1), nid(i, j+1, k+1))
                el.append(f'      <elem id="{e}">'
                          + ",".join(map(str, c)) + "</elem>")
    bot = ",".join(str(nid(i, j, 0))
                   for j in range(n+1) for i in range(n+1))
    top = ",".join(str(nid(i, j, n))
                   for j in range(n+1) for i in range(n+1))
    sides = ",".join(str(nid(i, j, k))
                     for k in range(n+1) for j in range(n+1)
                     for i in range(n+1) if i in (0, n) or j in (0, n))
    alln = ",".join(str(nid(i, j, k))
                    for k in range(n+1) for j in range(n+1)
                    for i in range(n+1))
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="fluid-FSI"/>
  <Control>
    <analysis>DYNAMIC</analysis>
    <time_steps>{steps}</time_steps>
    <step_size>{dt}</step_size>
    <solver type="fluid-FSI">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
      <linear_solver type="bicgstab"/>
    </solver>
  </Control>
  <Material>
    <material id="1" name="FSIBlock" type="fluid-FSI">
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>0.0</v>
      </solid>
      <fluid type="fluid">
        <density>{rho_f}</density>
        <k>1e3</k>
        <viscous type="Newtonian fluid">
          <mu>{mu}</mu>
        </viscous>
      </fluid>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
{chr(10).join(nodes)}
    </Nodes>
    <Elements type="hex8" name="Part1">
{chr(10).join(el)}
    </Elements>
    <NodeSet name="bot">{bot}</NodeSet>
    <NodeSet name="top">{top}</NodeSet>
    <NodeSet name="sides">{sides}</NodeSet>
    <NodeSet name="all_nodes">{alln}</NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="FSIBlock"/>
  </MeshDomains>
  <Initial>
    <ic name="ef0" type="initial fluid dilatation" node_set="all_nodes">
      <value>0.0</value>
    </ic>
  </Initial>
  <Boundary>
    <bc name="fixbot" type="zero displacement" node_set="bot">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="confine" type="zero displacement" node_set="sides">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>
    </bc>
    <bc name="zeroef" type="zero fluid dilatation" node_set="all_nodes"/>
    <bc name="fluid_at_rest" type="zero fluid velocity" node_set="all_nodes">
      <wx_dof>1</wx_dof><wy_dof>1</wy_dof><wz_dof>1</wz_dof>
    </bc>
    <bc name="squash" type="prescribed displacement" node_set="top">
      <dof>z</dof>
      <value lc="1">{uz}</value>
      <relative>0</relative>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>{steps*dt},1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile>
      <node_data data="z;ef" delim="," file="fluid_fsi_nodes.csv"/>
    </logfile>
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
                "type fluid-FSI not allowed in module fluid` [FALSIFIED "
                "2026-08-03: this message text does not occur anywhere "
                "in the FEBio 4.12.0.86045466d binary or any of its "
                "shared libraries (`strings` over febio4 + all 12 .so "
                "files, 267541 strings), so this Signal can never match "
                "on 4.12. The physics reasoning is desk research and was "
                "NOT executed] . "
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
                "prints [FALSIFIED 2026-08-03: this message text does "
                "not occur anywhere in the FEBio 4.12.0.86045466d binary "
                "or any of its shared libraries (`strings` over febio4 + "
                "all 12 .so files, 267541 strings), so this Signal can "
                "never match on 4.12. The physics reasoning is desk "
                "research and was NOT executed] `FSI interface: 0 shared "
                "nodes between mat=1 "
                "and mat=2`; the fluid pressure has zero effect on "
                "the solid response. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] FSI is stiff — start with small dt and "
                "use SMOOTH ramping of pressure / velocity BCs. "
                "Sudden load application (LINEAR load_controller + "
                "small dt_0) produces high-frequency oscillations "
                "in the fluid_FSI interface that take many steps "
                "to damp out. Signal: the kinetic_energy logfile "
                "channel at the interface oscillates with "
                "amplitude > 50% of mean for the first ~50 NOX "
                "steps before settling. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "fluid_fsi_3d_block": _fluid_fsi_3d_block,
}
