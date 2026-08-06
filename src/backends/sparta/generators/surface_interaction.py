"""Gas-surface interaction: surf_collide models, surface tallies, surf_react."""

from ._common import output_idioms


def _circle_diffuse_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    2d flow past a circular cylinder with a fully accommodating diffuse wall
    and a per-surface-element energy-flux tally.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    vstream = params.get("vstream", 100.0)
    t_wall = params.get("t_wall", 300.0)
    acc = params.get("acc", 1.0)
    nrho = params.get("nrho", 1.0)
    fnum = params.get("fnum", 0.001)
    dt = params.get("dt", 1.0e-4)
    nsteps = params.get("nsteps", 500)
    return f"""\
# 2d flow past a cylinder with a diffuse wall - SPARTA DSMC
seed             12345
dimension        2
global           gridcut 0.0 nrho {nrho} fnum {fnum}
boundary         o r p
create_box       0 10 0 10 -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N O
mixture          air N O vstream {vstream} 0.0 0.0
# read_surf needs the grid to exist first
read_surf        data.circle
# 'diffuse' takes EXACTLY two arguments: wall temperature (K) then
# accommodation (0..1). 'specular' takes none and exchanges no energy.
surf_collide     wall diffuse {t_wall} {acc}
# every surface element must be bound to a collision model, or the run aborts
surf_modify      all collide wall
# WARNING, measured on this build: at the DEFAULT nrho 1.0 / fnum 0.001 (the
# non-dimensional numbers the upstream 'circle' example uses) this collide line
# is INERT — a 500-step run reports 'SurfColl occurs = 74282' but
# 'Collide occurs = 0 (0K)'. The gas-surface physics is real, the gas-phase
# collisions are not. Raise nrho (and fnum with it) before reading anything that
# depends on intermolecular collisions, and check ncoll in stats_style — this is
# exactly the Ncoll == 0 trap this physics' own pitfalls warn about.
collide          vss air air.vss
fix              in emit/face air xlo
# compute surf is ALWAYS a per-surf ARRAY -> consume it as c_ID[i]
compute          q surf all all etot
fix              fq ave/surf all 1 100 100 c_q[1]
# one input value -> f_ID is a per-surf VECTOR (no bracket)
compute          qtot reduce sum f_fq
timestep         {dt}
stats            100
stats_style      step np ncoll nscoll nscheck c_qtot
run              {nsteps}
"""


KNOWLEDGE = {
    "surface_interaction": {
        "description": "Gas-surface interaction: diffuse / specular / CLL wall "
                       "models, per-element surface tallies, surface reactions",
        "spatial_dims": [2, 3],
        "key_commands": {
            "read_surf": "read_surf <file> [group <ID>] [trans ...] [scale ...] "
                         "[rotate ...] [invert] [clip] — needs the grid first",
            "surf_collide": "surf_collide <ID> diffuse <Tsurf> <acc> | specular "
                            "[noslip] | cll <Tsurf> <acc_n> <acc_t> <acc_rot> "
                            "<acc_vib> | adiabatic | transparent | vanish | td "
                            "| impulsive | piston",
            "surf_modify": "surf_modify <group-ID|all> collide <sc-ID> [react "
                           "<sr-ID>]",
            "surf_react": "surf_react <ID> prob <file> | global <p_recomb> "
                          "<p_react> | adsorb ...",
            "group": "group <name> surf id <lo> <hi> — build a surface group so "
                     "different patches can carry different wall models",
            "compute surf": "compute <ID> surf <group-ID> <mix-ID> <values...> "
                            "— values include n nflux mflux press shx shy shz "
                            "ke erot evib etot; ALWAYS a per-surf array",
            "bound_modify": "bound_modify <face> collide <sc-ID> — attach a "
                            "wall model to a BOX face declared 's' in boundary",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-surf tally idiom", "dump format"),
        "pitfalls": [
            "[Syntax] 'surf_collide <ID> diffuse' takes exactly two arguments "
            "in the order Tsurf (K) then accommodation (0..1). A swap is "
            "caught only when the accommodation value falls outside [0,1]: "
            "'diffuse 0.5 300' aborts, but 'diffuse 0.9 0.3' is a perfectly "
            "legal 0.9 K wall and runs silently. "
            "Signal: 'ERROR: Illegal surf_collide diffuse command "
            "(../surf_collide_diffuse.cpp:50)' for an out-of-range "
            "accommodation, 'ERROR: Surf_collide tsurf <= 0.0 "
            "(../surf_collide.cpp:125)' for a non-positive temperature — and "
            "NOTHING at all when both numbers happen to be in range.",

            "[Physics] 'surf_collide <ID> specular' reverses the normal "
            "velocity component and therefore transfers no energy to the wall "
            "at all: the tallied per-surf etot collapses to floating-point "
            "round-off (some fifteen orders of magnitude below the same case "
            "with a diffuse wall) while the collision counts stay healthy, and "
            "the gas never equilibrates with the wall. specular takes no "
            "temperature argument, which is the tell. "
            "Signal: a wall energy-flux column at round-off magnitude while "
            "Nscoll is large; passing a temperature gives 'ERROR: Illegal "
            "surf_collide specular command "
            "(../surf_collide_specular.cpp:42)'.",

            "[Setup] Every surface element must be bound to a collision model "
            "before the first run, and a PARTIAL binding is caught too — "
            "binding only a surface group leaves the rest unassigned and the "
            "run aborts rather than quietly treating them as transparent. "
            "Signal: 'ERROR: <N> surface elements not assigned to a collision "
            "model (../surf.cpp:343)', where N is the count still unbound.",

            "[Setup] The surface commands have a strict dependency order: "
            "read_surf needs the grid, and surf_modify needs BOTH the surfaces "
            "and the surf_collide model to exist already. surf_collide itself "
            "is NOT tied to read_surf and may be declared earlier. "
            "Signal: 'ERROR: Cannot read_surf before grid is defined "
            "(../read_surf.cpp:73)'; 'ERROR: Surf_modify when surfs do not yet "
            "exist (../surf.cpp:227)'; 'ERROR: Could not find surf_modify "
            "sc-ID (../surf.cpp:230)' or 'ERROR: Could not find surf_modify "
            "sr-ID (../surf.cpp:260)' for a mistyped model ID.",

            "[Output] 'compute surf' ALWAYS produces a per-surf ARRAY, even "
            "with a single value and a single mixture group, so it must be "
            "consumed with a column index. Feeding the bare compute ID to fix "
            "ave/surf or to dump surf is an error, not a silently-wrong "
            "result. "
            "Signal: 'ERROR: Fix ave/surf compute does not calculate a "
            "per-surf vector (../fix_ave_surf.cpp:150)' or 'ERROR: Dump surf "
            "compute does not compute per-surf vector "
            "(../dump_surf.cpp:569)'. Use c_ID[1].",

            "[Output] The number of columns of a per-surf or per-grid compute "
            "is ngroup * nvalue, and ngroup is 1 for the built-in 'all' "
            "mixture, for any mixture you define without the 'group' keyword, "
            "AND for one defined with 'group <name>' — a named group is still "
            "one group. Only 'group SELF' or the built-in 'species' mixture "
            "give one group per species. So 'mixture air N O ...' yields ONE "
            "column, not one per species, and reading column 1 as 'the "
            "nitrogen flux' is silently wrong. Nothing in the log reports a "
            "mixture's group count. "
            "Signal: 'ERROR: Fix ave/surf compute array is accessed "
            "out-of-range (../fix_ave_surf.cpp:157)' or 'ERROR: Compute reduce "
            "compute array is accessed out-of-range "
            "(../compute_reduce.cpp:236)' when you ask for a second column.",

            "[Output] A raw 'compute surf' value written straight to 'dump "
            "surf' as c_ID[i] is the tally accumulated since the compute was "
            "last invoked, so its magnitude depends on the DUMP FREQUENCY and "
            "it is not time-averaged: many elements read exactly zero at any "
            "given snapshot. Route it through 'fix ave/surf' first. "
            "Signal: a dumped surface field where a large fraction of the "
            "elements are exactly 0 and the nonzero values scale with the dump "
            "interval; the fix-averaged field has far fewer zeros and "
            "converges as the run proceeds.",

            "[Setup] A surface file describing an OPEN curve or sheet (a "
            "half-body used with a symmetry plane) fails SPARTA's watertight "
            "test. Place the open endpoints exactly on a simulation-box face. "
            "The 'clip' keyword is NOT what makes it legal, and an earlier "
            "wording implied it was: measured, an open curve in the interior "
            "fails the watertight test with 'clip' exactly as it does without "
            "it, and a curve whose endpoints sit exactly on a box face is "
            "accepted either way. Adding clip to a curve that does not reach a "
            "face changes nothing. "
            "Signal: 'Watertight check failed' followed by the count of "
            "unmatched points. A surface pushed outside the box by trans / "
            "scale / rotate instead gives 'ERROR: <N> surface points are not "
            "inside simulation box (../surf.cpp:1689)'.",

            "[Setup] SPARTA opens every data file relative to the current "
            "working directory and OASiS stages deck references by BASENAME, "
            "but the SPARTA distribution ships several DIFFERENT geometries "
            "under the same name — a dozen example directories contain a file "
            "called 'data.circle', all but one identical and the odd one out "
            "(examples/ambi) a completely different body: a unit circle at the "
            "origin against a radius-three circle centred on (5,5), with "
            "twenty times as many points. A deck that says 'read_surf "
            "data.circle' can therefore be handed a surface that does not fit "
            "its box, and each of the two is rejected in the box the other was "
            "drawn for. "
            "Signal: 'ERROR: <N> surface points are not inside simulation box "
            "(../surf.cpp:1689)', or a run that starts cleanly with a surface "
            "in the wrong place. Prefer a uniquely-named surface file, or pass "
            "the file explicitly so it wins over the distribution copies.",

            "[Setup] A BOX face is not a surface unless you declare it one: "
            "'boundary' style 's' makes the face a surface, and it then needs "
            "'bound_modify <face> collide <sc-ID>'. Applying bound_modify to a "
            "face that is periodic or outflow is an error, so you cannot "
            "attach a wall temperature to a plain 'p' or 'o' face. "
            "Signal: 'ERROR: Box boundary not assigned a surf_collide ID "
            "(../domain.cpp:100)' when 's' is declared but not bound; 'ERROR: "
            "Bound_modify surf requires boundary be a surface "
            "(../domain.cpp:253)' when the face is not 's'.",
        ],
    },
}

GENERATORS = {
    "surface_interaction_circle_2d": _circle_diffuse_2d,
    "surface_interaction_2d": _circle_diffuse_2d,
}
