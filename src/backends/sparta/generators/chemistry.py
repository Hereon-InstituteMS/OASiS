"""Gas-phase chemistry (TCE / QK) during DSMC collisions."""

from ._common import output_idioms


def _hot_air_box_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hot five-species air in a 3d box with TCE gas-phase chemistry. Starts as
    80/20 N2/O2 and dissociates; the species counts make the chemistry visible
    in the stats table.
    """
    n = params.get("n", 10)
    lx = params.get("lx", 1.0e-4)
    temp = params.get("temp", 20000.0)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e7)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Hot air in a 3d box with TCE gas-phase chemistry - SPARTA DSMC
seed             12345
dimension        3
boundary         rr rr rr
create_box       0 {lx} 0 {lx} 0 {lx}
create_grid      {n} {n} {n}
# EVERY species that can appear as a reaction product must be declared here,
# and the collide mixture must contain every declared species.
species          air.species N O N2 O2 NO
mixture          air N O N2 O2 NO
mixture          air N2 frac 0.8
mixture          air O2 frac 0.2
mixture          air N frac 0.0
mixture          air O frac 0.0
mixture          air NO frac 0.0
mixture          air vstream 0.0 0.0 0.0 temp {temp}
global           nrho {nrho} fnum {fnum}
collide          vss air air.vss
react            tce air.tce
create_particles air n 0
compute          cnt count N O N2 O2 NO
timestep         {dt}
stats            100
stats_style      step np ncoll nreact c_cnt[1] c_cnt[3] c_cnt[4]
run              {nsteps}
"""


KNOWLEDGE = {
    "chemistry": {
        "description": "Gas-phase chemical reactions (TCE / QK) evaluated "
                       "during DSMC collisions",
        "spatial_dims": [2, 3],
        "key_commands": {
            "react": "react tce|qk|tce/qk <reaction-file> — those three are the "
                     "styles compiled into this build",
            "react_modify": "react_modify [recomb <yes|no>] [rboost <factor>] "
                            "[compute_chem_rates <yes|no>]",
            "species": "species <file> <ID> ... — declare EVERY reactant and "
                       "product species",
            "mixture": "mixture <mixID> <species...> ; mixture <mixID> <sp> "
                       "frac <f> — fractions must sum to <= 1",
            "compute count": "compute <ID> count <species|mixture-group>... — "
                             "the cheapest way to watch composition change",
            "stats_style nreact": "nreact is the gas reactions on THAT "
                                  "timestep; nsreact is the surface ones",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms(),
        "pitfalls": [
            "[Syntax] 'react <style> <file>' never checks that the file is a "
            "reaction file. It consumes non-blank, non-comment lines in PAIRS "
            "(formula line, then coefficient line), so what you get depends on "
            "the line COUNT, not on the file's purpose: a complete pair of "
            "junk aborts loudly, but a file whose data lines run out mid-pair "
            "— any file with an odd number of data lines, which includes "
            "single-entry .vss and .species files — ends silently with the "
            "dangling line discarded and ZERO reactions loaded. A trailing "
            "junk line after otherwise valid reactions is dropped the same "
            "way. "
            "Signal: the count line '  style <style> #-of-reactions <N>' under "
            "the 'Gas reaction tallies:' block, printed after a run — that is "
            "the only place SPARTA states how many reactions it parsed. The "
            "loud forms are 'ERROR: Invalid reaction formula in file "
            "(../react_bird.cpp:641)', 'ERROR: Invalid reaction type in file "
            "(../react_bird.cpp:659)' and, for an unreadable path, 'ERROR on "
            "proc 0: Cannot open reaction file <f> (../react_bird.cpp:552)'. "
            "Do NOT use the end-of-run 'Gas reactions' counter as the test: "
            "SPARTA prints it in EVERY run, including runs with no react "
            "command at all.",

            "[Syntax] The react STYLE and the reaction FILE are not "
            "cross-checked: 'react qk <a tce file>' is accepted without error "
            "or warning and fires reactions at DIFFERENT rates than 'react tce "
            "<the same file>'. You cannot rely on SPARTA to tell you the model "
            "you asked for is the model you got. "
            "Signal: swap the style keyword and rerun — if the run still "
            "completes and Nreact still fills in, the style was never "
            "validated against the file, and the per-reaction tallies under "
            "'Gas reaction tallies:' will differ between the two.",

            "[Syntax] Mixture fractions are checked only for exceeding 1.0. A "
            "set of fractions that sums to LESS than 1 is accepted and the "
            "deficit is absorbed silently, so a deck that means 80/20 but "
            "writes 0.1/0.1 runs with a composition you did not choose. WHERE "
            "the deficit goes has two rules and you need both: species you left "
            "unset share it equally, AND the LAST species listed in the mixture "
            "absorbs whatever is still missing even when its own fraction was "
            "set explicitly, because init_fraction clamps the last entry of the "
            "cumulative array to 1.0 (mixture.cpp:278). So 0.1/0.1 over two "
            "species does not give 0.1/0.1 and does not give 0.5/0.5 — it gives "
            "roughly 0.09/0.91. An earlier wording said only that the remainder "
            "goes to the species you left unset, which does not explain the "
            "entry's own 0.1/0.1 example, where nothing is unset. "
            "Signal: 'ERROR: Mixture <ID> fractions exceed 1.0 "
            "(../mixture.cpp:225)' for the over-1 case, and nothing at all for "
            "the under-1 case — verify the realised composition with 'compute "
            "<ID> count <species...>' on step 0.",

            "[Physics] Chemistry needs collision energies above the activation "
            "energy, so at a modest temperature a correctly-configured TCE "
            "deck still reports Nreact = 0 for thousands of steps. A zero "
            "reaction count is therefore not by itself evidence of a "
            "misconfiguration. "
            "Signal: distinguish the two cases with the species counts, not "
            "with Nreact — if 'compute count' shows the product species "
            "strictly at 0 particles for the whole run AND Ncoll is large, "
            "check the setup; if the counts creep up, the chemistry is on and "
            "merely slow.",
        ],
    },
}

GENERATORS = {
    "chemistry_box_3d": _hot_air_box_3d,
    "chemistry_3d": _hot_air_box_3d,
}
