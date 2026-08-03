"""DSMC gas <-> solid conjugate heat transfer, in-code and via preCICE."""

from ._common import output_idioms


def _cht_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Self-consistent wall temperature: the tallied per-element energy flux
    drives a radiative-equilibrium wall temperature (fix surf/temp), which the
    diffuse wall model then reads back through a custom per-surf attribute.
    """
    vstream = params.get("vstream", 2634.1)
    t_free = params.get("t_free", 200.0)
    t_init = params.get("t_init", 100.0)
    emis = params.get("emis", 0.9)
    nrho = params.get("nrho", 4.247e19)
    fnum = params.get("fnum", 7.0e14)
    dt = params.get("dt", 3.5e-7)
    nevery = params.get("nevery", 200)
    nsteps = params.get("nsteps", 600)
    return f"""\
# DSMC gas <-> surface conjugate heat transfer - SPARTA
# Order matters: compute surf -> fix ave/surf -> fix surf/temp (which CREATES
# the custom per-surf attribute) -> surf_collide diffuse s_<name> -> surf_modify
seed             12345
dimension        2
global           nrho {nrho} fnum {fnum} gridcut 0.01
timestep         {dt}
boundary         o ro p
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1 block * * *
species          ar.species Ar
mixture          all vstream {vstream} 0.0 0.0 temp {t_free}
collide          vss all ar.vss
collide_modify   vremax 1000 yes
read_surf        circle.surf group 1
compute          q surf all all etot
fix              fq ave/surf all 1 {nevery} {nevery} c_q[1] ave one
# the SOURCE must be a per-surf VECTOR: f_<avesurf>, not the raw compute ID
fix              tw surf/temp all {nevery} f_fq {t_init} {emis} temperature
surf_collide     wall diffuse s_temperature 1.0
surf_modify      1 collide wall
fix              in emit/face all xlo
create_particles all n 0
compute          qtot reduce sum f_fq
stats_style      step np nscoll c_qtot
stats            {nevery}
dump             dsurf surf all {nevery} dump.wall id s_temperature f_fq
run              {nsteps}
"""


KNOWLEDGE = {
    "conjugate_heat_transfer": {
        "description": "DSMC gas <-> solid conjugate heat transfer: SPARTA "
                       "tallies the surface heat flux and reads back a wall "
                       "temperature, either through fix surf/temp in-code or "
                       "through preCICE against an FEM solid",
        "spatial_dims": [2, 3],
        "key_commands": {
            "compute surf": "compute <ID> surf <grp> <mix> etot — total energy "
                            "flux per surface element (per-surf ARRAY)",
            "fix ave/surf": "fix <ID> ave/surf <grp> <Nevery> <Nrepeat> "
                            "<Nfreq> c_<ID>[1] [ave one|running|window M]",
            "fix surf/temp": "fix <ID> surf/temp <grp> <Nevery> <f_source> "
                             "<Tinit> <emisurf> <custom-name> — CREATES the "
                             "custom per-surf attribute; 0 < emisurf <= 1",
            "surf_collide": "surf_collide <ID> diffuse s_<custom-name> <acc> — "
                            "reads the per-element temperature",
            "compute reduce": "compute <ID> reduce sum f_<avesurf> — a fix "
                              "ave/surf with ONE input is a per-surf VECTOR, "
                              "so no bracket",
            "dump surf": "dump <ID> surf all <N> <file> id s_<custom-name> "
                         "f_<avesurf> — writes the wall temperature field",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "coupling_notes": "For a two-code coupling, drive SPARTA from its "
                          "Python library (build with 'make mode=shlib "
                          "serial'), which exposes command / extract_global / "
                          "extract_compute / extract_variable only. There is "
                          "no per-surface scatter, so exchange a SCALAR (total "
                          "flux out, uniform wall temperature in) by "
                          "re-issuing a SPARTA equal-style variable for the "
                          "wall temperature each coupling window. Explicit "
                          "serial coupling is stable here because the solid's "
                          "thermal inertia damps the DSMC fluctuations.",
        "output_idioms": output_idioms("per-surf tally idiom", "dump format"),
        "pitfalls": [
            "[Setup] The command order is a hard dependency chain: fix "
            "surf/temp is what CREATES the custom per-surf attribute, so the "
            "surf_collide line that consumes it as s_<name> must come "
            "afterwards. Writing them in the intuitive order (wall model "
            "first) aborts — with exactly the same message you get if the fix "
            "is missing altogether, so the error does not tell you which "
            "mistake you made. "
            "Signal: 'ERROR: Surf_collide tsurf could not find custom "
            "attribute (../surf_collide.cpp:141)'.",

            "[Syntax] The <source> argument of fix surf/temp is BRACKET-driven, "
            "not compute-versus-fix: a bare id must name a per-surf VECTOR and "
            "a bracketed one must name a per-surf ARRAY. compute surf is "
            "always an array, so it must be given as c_<ID>[N] — the SPARTA "
            "doc page's own example passes a bare 'c_1' and does not work on "
            "this build, and 'c_<ID>[*]' fails the same way because the "
            "wildcard parses as index 0. A fix ave/surf with one input value "
            "is a vector, so it must be given bare as f_<ID>; f_<ID>[1] "
            "aborts. "
            "Signal: 'ERROR: Fix surf/temp compute does not compute per-surf "
            "vector (../fix_surf_temp.cpp:82)' for the bare-compute form, "
            "'ERROR: Fix surf/temp fix does not compute per-surf array "
            "(../fix_surf_temp.cpp:112)' for the over-indexed fix form.",

            "[Syntax] The emissivity argument is strictly inside (0, 1]. Zero "
            "is not 'no radiation', it is rejected — with emisurf = 0 the "
            "radiative-equilibrium temperature would be unbounded. "
            "Signal: 'ERROR: Fix surf/temp emissivity must be > 0.0 and <= 1 "
            "(../fix_surf_temp.cpp:125)'.",

            "[Output] A fix ave/surf fed ONE input value produces a per-surf "
            "VECTOR, referenced as f_ID with NO bracket; indexing it as f_ID[1] "
            "aborts. With two or more inputs it becomes an array read as "
            "f_ID[1], f_ID[2], ... The upstream compute surf feeding it is the "
            "opposite: always an array, so always indexed. "
            "Signal: 'ERROR: Compute reduce fix does not calculate a per-surf "
            "array (../compute_reduce.cpp:293)' when you index a "
            "single-value fix. Do not confuse it with line 280, which is the "
            "per-GRID twin of the same message ('... does not calculate a "
            "per-grid array') raised by a fix ave/grid.",

            "[Physics] The DSMC surface flux is statistically noisy and the "
            "wall temperature responds to it, so a short fix ave/surf window "
            "feeds noise straight into the wall temperature and the coupled "
            "system can oscillate without ever failing. Average over a window "
            "long compared with the DSMC fluctuation time and short compared "
            "with the solid's thermal time. "
            "Signal: the dumped s_<custom-name> field changes by a large "
            "fraction between consecutive fix surf/temp updates instead of "
            "drifting smoothly toward equilibrium.",

            "[Setup] SPARTA opens every data file relative to the CURRENT "
            "WORKING DIRECTORY, so a coupled run whose driver starts the "
            "participant in a fresh directory dies at setup unless the "
            "species, VSS and surface files were staged there first. "
            "Signal: 'ERROR on proc 0: Cannot open species file <name>' from "
            "particle.cpp at setup, before any stats line. The OASiS couple() "
            "path stages every file referenced by the deck into the "
            "participant work directory; pass task-specific files through the "
            "participant's data_files list so they win over the "
            "identically-named distribution examples.",

            "[Setup] A half-body surface used with a symmetry plane is an OPEN "
            "curve and fails the watertight test. Put the open endpoints "
            "exactly on a box face and read with 'read_surf <file> clip'. "
            "Signal: 'Watertight check failed' followed by the number of "
            "unmatched points.",
        ],
    },
}

GENERATORS = {
    "conjugate_heat_transfer_circle_2d": _cht_circle_2d,
    "conjugate_heat_transfer_2d": _cht_circle_2d,
}
