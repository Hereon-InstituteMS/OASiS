"""Particle-particle collisions (VSS) and internal-energy relaxation."""

from ._common import output_idioms


def _collisional_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisional argon in a 2d box, with the cell Knudsen number reported so the
    grid resolution is visible in the stats table.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e23)
    fnum = params.get("fnum", 6.0e11)
    temp = params.get("temp", 273.15)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Collisional (VSS) argon in a 2d box - SPARTA DSMC
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {temp}
global           nrho {nrho} fnum {fnum}
# 'vss' is the only collide style compiled into this build. The mixture named
# here must contain EVERY species loaded by 'species', and the .vss file must
# contain a line for every one of them.
collide          vss gas ar.vss
create_particles gas n 0
compute          tk temp
compute          nr grid all species nrho
compute          tg thermal/grid all all temp
fix              fnr ave/grid all 1 100 100 c_nr[*]
fix              ftg ave/grid all 1 100 100 c_tg[*]
compute          lam lambda/grid f_fnr[*] f_ftg lambda knall
compute          knmin reduce min c_lam[2]
timestep         {dt}
stats            100
stats_style      step np ncoll ncollave c_tk c_knmin
run              {nsteps}
"""


def _internal_energy_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Diatomic nitrogen relaxing from a cold rotational mode toward the
    translational temperature. Demonstrates that trot/tvib only evolve when a
    collide command is present.
    """
    nx = params.get("nx", 10)
    ny = params.get("ny", 10)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e11)
    temp = params.get("temp", 273.15)
    trot = params.get("trot", 100.0)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Rotational relaxation of N2 in a 2d box - SPARTA DSMC
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N2
# trot starts BELOW temp; collisions pull the two together
mixture          gas N2 vstream 0.0 0.0 0.0 temp {temp} trot {trot}
global           nrho {nrho} fnum {fnum}
collide          vss gas air.vss
create_particles gas n 0
compute          tg thermal/grid all all temp
compute          tr grid all all trot
compute          rtrans reduce ave c_tg[1]
compute          rrot reduce ave c_tr[1]
fix              keep ave/grid all 1 100 100 c_tg[*]
timestep         {dt}
stats            100
stats_style      step np ncoll c_rtrans c_rrot
run              {nsteps}
"""


KNOWLEDGE = {
    "collision_relaxation": {
        "description": "Particle-particle collisions with the VSS model plus "
                       "rotational / vibrational energy relaxation",
        "spatial_dims": [2, 3],
        "key_commands": {
            "collide": "collide vss <mixID> <file.vss> [relax variable] — vss "
                       "is the ONLY collide style compiled in this build",
            "collide_modify": "collide_modify [vremax <n> <yes|no>] [remain "
                              "<yes|no>] [vibrate <no|discrete|smooth>] "
                              "[rotate <no|smooth>] [ambipolar <yes|no>] "
                              "[nearcp <yes|no> <n>]",
            "mixture": "mixture <mixID> <species...> [temp T] [trot T] "
                       "[tvib T] [frac f] [group SELF]",
            "compute grid ... trot/tvib": "compute <ID> grid <grp> <mix> trot "
                                          "tvib — per-cell internal "
                                          "temperatures",
            "compute thermal/grid": "compute <ID> thermal/grid <grp> <mix> "
                                    "temp — per-cell translational temperature",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "vss_parameters": "The .vss file gives, per species, the reference "
                          "diameter (m), the viscosity exponent omega, the "
                          "reference temperature Tref (K) and the VSS scatter "
                          "parameter alpha. Those numbers are SI in the shipped "
                          "data files.",
        "output_idioms": output_idioms("per-grid tally idiom"),
        "pitfalls": [
            "[Syntax] 'collide vss <mixID>' without the .vss filename is a "
            "hard error, but a .vss file that simply does not list one of your "
            "species is caught only when the collision model is set up. "
            "Signal: 'ERROR: Illegal collide command (../collide_vss.cpp:47)' "
            "for the missing filename; 'ERROR on proc 0: Species <X> did not "
            "appear in VSS parameter file (../collide_vss.cpp:924)' when the "
            "file is the wrong one for your species (e.g. handing air.vss to "
            "an argon deck).",

            "[Syntax] The mixture named on the collide line must contain EVERY "
            "species loaded by the 'species' command — a subset is rejected, "
            "not silently ignored — and the check happens at the start of the "
            "FIRST RUN, not when the collide line is parsed, so a deck that "
            "never reaches 'run' never reports it. "
            "Signal: 'ERROR: Collision mixture does not contain all species "
            "(../collide.cpp:159)'; a mistyped mixture ID instead gives "
            "'ERROR: Collision mixture does not exist (../collide.cpp:155)'.",

            "[Physics] Without a collide command, create_particles gives every "
            "particle ZERO rotational and vibrational energy even when the "
            "mixture asks for trot/tvib, so internal-energy diagnostics read "
            "exactly zero for the whole run and no relaxation is possible. "
            "Signal: a Trot or Tvib column that is exactly 0 rather than "
            "noisy, from step 0 onward, while Np is large.",

            "[Syntax] collide_modify silently accepts only its documented "
            "keywords; a typo aborts with a generic message that does not name "
            "the offending keyword, so check the spelling against the compiled "
            "build rather than against a doc page for another version. "
            "Signal: 'ERROR: Illegal collide_modify command "
            "(../collide.cpp:1727)'.",

            "[Physics] ROTATIONAL relaxation is active by default but "
            "VIBRATIONAL relaxation is not: with no collide_modify the "
            "vibrational mode is inert, and both 'compute <ID> grid <grp> "
            "<mix> tvib' and 'compute <ID> tvib/grid <grp> <mix>' return "
            "exactly 0.0 for the entire run while Trot visibly relaxes in the "
            "same run. 'collide_modify vibrate smooth' switches it on and also "
            "seeds the mode from the mixture's tvib at creation — but ONLY if "
            "the collide_modify line comes BEFORE create_particles. That "
            "ordering is not checked and getting it wrong is completely "
            "silent: with create_particles first, Tvib is exactly 0 at step 0 "
            "and creeps up from there by collisions alone, staying orders of "
            "magnitude below the mixture's tvib for the whole run, with rc = 0 "
            "and no warning. An earlier wording said only 'Tvib is nonzero "
            "from step 0', which is true of one of the two orderings. "
            "'collide_modify vibrate discrete' is NOT the "
            "quiet no-op an earlier wording claimed: for any species whose "
            "vibdof is > 2 and which has no 'vibfile' on the species line it "
            "ABORTS at run setup with 'ERROR: Discrete vibrational info for "
            "species <X> not read in (../particle.cpp:177)' "
            "(particle.cpp skips the check only for vibdof <= 2). For a "
            "vibdof <= 2 species it is accepted and starts at 0, but it does "
            "not stay there — it is zero for the first stats blocks and then "
            "collisions populate the levels. Only a species with no "
            "vibrational mode at all reads "
            "identically 0 under every setting. "
            "Signal: a Tvib column that is exactly 0.0 on every stats line "
            "while the Trot column moves — rc = 0, no warning — which "
            "identifies the DEFAULT (no collide_modify) case; the 'discrete' "
            "mistake announces itself with the particle.cpp:177 abort "
            "instead.",

            "[Output] 'compute grid <grp> <mix> tvib' and 'compute tvib/grid "
            "<grp> <mix>' are different estimators and do not agree: on the "
            "same smooth-vibration run they differ by a factor of several. "
            "Pick one and use it consistently; do not compare a number from "
            "one against a number from the other. "
            "Signal: two vibrational-temperature columns in the same stats "
            "table whose ratio is roughly constant and far from 1.",
        ],
    },
}

GENERATORS = {
    "collision_relaxation_box_2d": _collisional_box_2d,
    "collision_relaxation_2d": _collisional_box_2d,
    "collision_relaxation_internal_energy_2d": _internal_energy_box_2d,
}
