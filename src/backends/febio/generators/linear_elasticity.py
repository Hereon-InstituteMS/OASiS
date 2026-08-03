"""FEBio linear-elasticity generators and knowledge.

FEBio Module type: 'solid'. Material: 'isotropic elastic' (small strain).
"""


def _elasticity_3d_cube(params: dict) -> str:
    """Unit cube with prescribed-displacement uniaxial compression.

    Linear isotropic elastic material; STATIC analysis; FEBio v4.0 XML.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>0</T>
      <R>0</R>
      <Fc>0</Fc>
    </Constants>
  </Globals>
  <Material>
    <material id="1" name="Material1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
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
      <x_dof>1</x_dof>
      <y_dof>1</y_dof>
      <z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt>
        <pt>1,1</pt>
      </points>
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
    "linear_elasticity": {
        "description": "Linear elasticity with FEBio — isotropic elastic material",
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": "Newton-Raphson with direct linear solver",
        "materials": {
            "isotropic elastic": {"E": "Young's modulus (Pa)", "v": "Poisson's ratio"},
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "material constant 1", "c2": "material constant 2"},
        },
        "pitfalls": [
            (
                "[Input] FEBio uses lowercase 'v' for Poisson's "
                "ratio (NOT 'nu'). The 'nu' name is convention in "
                "FEniCSx / deal.II / NGSolve but FEBio's XML schema "
                "rejects it. Signal: the reader stops with "
                "`Reading file <name>.feb ...FAILED!` and an ERROR "
                "box reading `tag \"nu\" (line N) : unrecognized "
                "tag`; the element parameters of `isotropic "
                "elastic` are exactly <E> and <v>. "
                "CORRECTED 2026-08-03 by live run on FEBio "
                "4.12.0: the previously catalogued signal text "
                "(`unknown material parameter nu in isotropic "
                "elastic`) is NOT what FEBio emits — no such "
                "string exists in the binary, so the old entry "
                "could never have matched real output."
            ),
            (
                "[Input] Element connectivity is 1-indexed. Node "
                "id=0 does not exist in FEBio's XML schema; many "
                "mesh-conversion scripts (PyVista, meshio) "
                "default to 0-indexing and need an explicit +1 "
                "offset. Signal: rejected already in the <Nodes> "
                "block, before connectivity is even read, with "
                "`tag \"node\" (line N) : invalid value for "
                "attribute \"id\"` and `Reading file ...FAILED!`. "
                "CORRECTED 2026-08-03 by live run on FEBio "
                "4.12.0: the previously catalogued signal "
                "(`element references node 0 — node IDs must "
                "start at 1`) does not exist in the binary and "
                "names the wrong section — the failure is in "
                "<Nodes>, not in <Elements>."
            ),
            (
                "[Input] MeshDomains section is REQUIRED in v4.0 "
                "and links each domain to a material BY NAME: "
                "<SolidDomain name=... mat=\"<material name>\"> is "
                "resolved through FEModel::FindMaterial(name), which "
                "matches the material's name= attribute (NOT its "
                "numeric id). A leftover v3.x-style numeric "
                "mat=\"1\" on <SolidDomain> is rejected; the same "
                "mat= attribute left on the <Elements> tag is "
                "silently IGNORED in 4.0 (verified: decks with "
                "mat=\"1\", mat=\"M1\" and even mat=\"NONSENSE\" "
                "on <Elements> all run and give bit-identical "
                "displacements). "
                "Signal: `tag \"SolidDomain\" (line N) : invalid "
                "value for attribute \"mat\"` with `Reading file "
                "...FAILED!`. Dropping <MeshDomains> entirely, or "
                "merely placing it AFTER <Boundary>, does "
                "NOT produce that message — node sets are only "
                "resolvable once the domains are built, so the "
                "first <bc> fails instead with "
                "`tag \"bc\" (line N) : invalid value for "
                "attribute \"node_set\"`, which sends you hunting "
                "in the wrong section. The required order is "
                "Material, Mesh, MeshDomains, Boundary. "
                "(Live-verified 2026-08-03, FEBio 4.12.0.)"
            ),
            (
                "[Input] Every <material> must carry a name= "
                "attribute in the 4.0 schema, even though it also "
                "has id=. The name is what <MeshDomains> and the "
                "log refer to. Signal: `tag \"material\" (line N) "
                ": missing attribute \"name\"` and `Reading file "
                "...FAILED!` — this single omission is what blocks "
                "eight of the seventeen templates shipped in this "
                "backend from running at all. (Live-verified "
                "2026-08-03, FEBio 4.12.0.)"
            ),
            (
                "[Input] LoadData with a <load_controller> is "
                "needed for any BC value carrying an `lc=N` "
                "attribute, and the failure is NOT silent. Signal: "
                "the deck READS cleanly (`Reading file "
                "...SUCCESS!`) and only then aborts at model "
                "initialisation with `Invalid load curve ID`, exit "
                "status 1 — so a wrapper that only checks the "
                "reader line believes the deck is fine. "
                "CORRECTED 2026-08-03 by live run on FEBio "
                "4.12.0: the previously catalogued claim that the "
                "BC 'silently stays at its initial value with no "
                "ramping' is FALSE for 4.12 — nothing is solved at "
                "all."
            ),
            (
                "[Input] <Control> is not optional and its absence "
                "produces the single most dangerous FEBio outcome: "
                "a deck with no <Control> section READS "
                "successfully, prints NO error message of any "
                "kind, writes a non-empty .xplt (mesh only — 1353 "
                "bytes against 1629 for the same deck solved) "
                "plus every requested logfile CSV, and "
                "terminates. Signal: "
                "the CSVs contain ONLY a `*Step = 0` / `*Time = 0` "
                "block of zeros, the summary says `Number of time "
                "steps completed .... : 0`, and the banner is "
                "`E R R O R   T E R M I N A T I O N` with exit "
                "status 1. Never accept a FEBio run on the "
                "existence of output files; require the literal "
                "`N O R M A L   T E R M I N A T I O N` banner AND "
                "a non-zero completed-step count. (Live-verified "
                "2026-08-03, FEBio 4.12.0.)"
            ),
            (
                "[Input] <Control> requires a nested <solver "
                "type=\"...\"> whose type string is the module "
                "name (`solid` for the solid module). Signal: "
                "`Component \"\" needs to have property \"solver\" "
                "defined (line N)` at parse — the empty quotes are "
                "not a bug, the FEAnalysis object has no name. "
                "(Live-verified 2026-08-03, FEBio 4.12.0.)"
            ),
            (
                "[Discretization] A degenerate hex8 — the classic "
                "off-by-one that repeats a node id in the "
                "connectivity list — is NOT rejected. Signal: the "
                "run completes with `N O R M A L   T E R M I N A T "
                "I O N` and exit 0, the only hint is a WARNING box "
                "`1 isolated vertex removed.`, and the stresses "
                "are wrong: on a unit cube under uniaxial "
                "prescribed displacement, duplicating one node "
                "changed sx from 1.6447 to 2.0517 and broke the "
                "x/y symmetry (sy 1.6447 -> 1.4870). Any "
                "programmatically generated mesh must be checked "
                "for repeated node ids per element, because FEBio "
                "will happily integrate the collapsed element. "
                "(Live-verified 2026-08-03, FEBio 4.12.0; fixture "
                "scripts/tier2_fixtures/febio/"
                "degenerate_hex8_clean_wrong_result.)"
            ),
            (
                "[Input] Factory TYPE STRINGS (the `type=` "
                "attribute on <material>, <bc>, <Elements>, "
                "<solver>, <load_controller>, ...) are "
                "case-SENSITIVE, while enum PARAMETER VALUES "
                "(<analysis>, <symmetric_stiffness>, "
                "<interpolate>) are case-INSENSITIVE. Signal: "
                "`type=\"HEX8\"` gives `Invalid element type`, "
                "`type=\"Zero Displacement\"` gives `tag \"bc\" "
                "(line N) : invalid value for attribute \"type\"`, "
                "whereas `<analysis>static</analysis>` and "
                "`<analysis>Static</analysis>` both run and give "
                "bit-identical results to STATIC. Enumerate the "
                "exact strings with `printf 'list\\nquit\\n' | "
                "febio4 -nosplash`. (Live-verified 2026-08-03, "
                "FEBio 4.12.0.)"
            ),
            (
                "[Integration] FEBio's ERROR box wraps its text at "
                "71 columns, so a long diagnostic is split across "
                "lines inside the star frame and a fixed-string "
                "search for the whole sentence finds nothing. "
                "Signal: `grep -F 'The selected linear solver does "
                "not support the requested matrix format.'` "
                "returns 0 hits on a run that printed exactly "
                "that error, because the output reads `... the "
                "requested matrix` / `format.` on two lines. Match "
                "a short fragment that cannot wrap (`does not "
                "support the requested`), or strip the frame and "
                "re-join lines before matching. The `tag \"x\" "
                "(line N) : ...` family and the other short "
                "messages quoted in this catalog all fit on one "
                "line. (Live-verified 2026-08-03, FEBio 4.12.0.)"
            ),
        ],
    },
}


GENERATORS = {
    "linear_elasticity_3d_cube": _elasticity_3d_cube,
}
