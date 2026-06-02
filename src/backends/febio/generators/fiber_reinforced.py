"""FEBio fiber-reinforced generators and knowledge.

FEBio Module type: 'solid' with anisotropic fiber-reinforced materials.
Two main families:
  - 'fiber-exp-pow' / 'fiber-pow-linear' — single-family
  - 'Holzapfel-Gasser-Ogden' (HGO) — arterial-wall standard with two
    symmetric fiber families
  - 'transversely isotropic' — one fiber family on top of an
    isotropic matrix

Canonical for arterial wall mechanics, skeletal muscle, ligament,
tendon, anterior cruciate, myocardium, and any biological tissue with
preferred fiber direction.
"""


def _fiber_reinforced_3d_hgo(params: dict) -> str:
    """Holzapfel-Gasser-Ogden double-fiber-family arterial wall under
    uniaxial extension. Two symmetric fiber families at angle ±phi
    from the loading axis. Neo-Hookean isotropic matrix + fiber
    strain-energy term k1 * exp(k2 * E_f^2) - 1.
    """
    E_iso = params.get("E", 100.0)
    k1 = params.get("k1", 1.0)
    k2 = params.get("k2", 5.0)
    kappa = params.get("kappa", 0.1)
    phi_deg = params.get("fiber_angle_deg", 40.0)
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
    <material id="1" type="Holzapfel-Gasser-Ogden">
      <density>1.0</density>
      <c>{E_iso}</c>
      <k1>{k1}</k1>
      <k2>{k2}</k2>
      <kappa>{kappa}</kappa>
      <gamma>{phi_deg}</gamma>
      <k>1000.0</k>
      <mat_axis type="vector">
        <a>1,0,0</a>
        <d>0,1,0</d>
      </mat_axis>
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
      <value lc="1">0.3</value>
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
      <var type="fiber vector"/>
    </plotfile>
  </Output>
</febio_spec>
'''


KNOWLEDGE = {
    "fiber_reinforced": {
        "description": (
            "Anisotropic fiber-reinforced hyperelasticity via FEBio's "
            "HGO, transversely-isotropic, or single-family fiber "
            "materials. Used for arterial walls (Holzapfel-Gasser-"
            "Ogden two-family model), skeletal muscle, ligament, "
            "tendon, anterior cruciate, and myocardium."
        ),
        "input_format": "FEBio XML v4.0",
        "solver": "Standard solid solver (nonlinear) with anisotropic stress",
        "materials": {
            "Holzapfel-Gasser-Ogden": {
                "c": "Neo-Hookean matrix shear modulus",
                "k1": "Fiber exponential stiffness (force scale)",
                "k2": "Fiber exponent (controls strain-stiffening)",
                "kappa": "Dispersion parameter (0=perfectly aligned, "
                         "1/3=isotropic random)",
                "gamma": "Fiber angle (degrees) from local axis 'a'",
                "k": "Bulk modulus (penalises near-incompressibility)",
                "mat_axis": "Material orientation frame "
                            "(<a> = primary fiber axis, "
                            "<d> = secondary axis defining the plane)",
            },
            "transversely isotropic": {
                "fiber": "Nested fiber model (fiber-exp-pow, "
                         "fiber-pow-linear)",
                "matrix": "Nested isotropic matrix (neo-Hookean, "
                          "Mooney-Rivlin, ...)",
            },
        },
        "pitfalls": [
            (
                "[Input] mat_axis is REQUIRED to define the fiber "
                "orientation frame — without it FEBio defaults to "
                "the global coordinate system and the fibers point "
                "along x, which is almost never what you want for "
                "arterial-wall benchmarks. Signal: HGO stress-strain "
                "curve under uniaxial tension shows isotropic "
                "neo-Hookean response (no fiber strain-stiffening "
                "knee) because the fibers happen to be perpendicular "
                "to the loading axis. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] HGO fibers carry load only in TENSION "
                "— in compression the strain-energy term is gated by "
                "an IfPos(I4-1, ..., 0). A specimen under pure "
                "compression yields ~matrix-only response. Signal: "
                "stiffness in compression < stiffness in tension by "
                "factor ~ (1 + k1*k2*E_f^2 / c) — sometimes 10-100x "
                "for stiff fibers. This is physical, not a bug, but "
                "easy to mistake for one. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] kappa (dispersion) interacts with the "
                "fiber-strain-energy term: kappa=0 (perfect "
                "alignment) gives the sharpest response; kappa=1/3 "
                "(isotropic dispersion) makes HGO collapse to "
                "near-isotropic. Setting kappa > 1/3 is physically "
                "meaningless. Signal: parser may accept kappa=0.5 "
                "but the computed effective fiber direction "
                "differs from the input gamma by > 10%; verify by "
                "plotting the fiber stress contribution against an "
                "isotropic reference. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] HGO is intrinsically near-"
                "incompressible — set k (bulk modulus) at least "
                "~1000 * c to prevent volumetric locking. Signal: "
                "k too low gives visible volume change under "
                "uniaxial stretch (Jacobian J = det(F) deviates "
                "from 1 by > 1%); k too high gives cond(K) > 1e14 "
                "and Newton stalls. (Audit 2026-06-02.)"
            ),
        ],
    },
}


GENERATORS = {
    "fiber_reinforced_3d_hgo": _fiber_reinforced_3d_hgo,
}
