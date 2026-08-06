"""Static and dynamic grid adaptation."""

from ._common import output_idioms


def _adapt_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Static refinement around a surface (adapt_grid, one-shot) followed by
    dynamic refinement on particle count during the run (fix adapt).
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    vstream = params.get("vstream", 100.0)
    nrho = params.get("nrho", 1.0)
    fnum = params.get("fnum", 0.001)
    maxlevel = params.get("maxlevel", 3)
    dt = params.get("dt", 1.0e-4)
    nevery = params.get("nevery", 100)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Grid adaptation around a surface - SPARTA DSMC
seed             12345
dimension        2
global           gridcut 0.0 nrho {nrho} fnum {fnum}
boundary         o r p
create_box       0 10 0 10 -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N O
mixture          air N O vstream {vstream} 0.0 0.0
read_surf        data.circle
surf_collide     wall diffuse 300.0 1.0
surf_modify      all collide wall
# WARNING, measured on this build: at the DEFAULT nrho 1.0 / fnum 0.001 this
# collide line is INERT — a run reports 'SurfColl occurs = 73827' but
# 'Collide occurs = 0 (0K)'. Grid adaptation driven by a collision-derived
# quantity (cell Knudsen number, mean free path) is therefore meaningless at
# these numbers: with no collisions the lambda field is the 1e+20 sentinel.
# Raise nrho and fnum before adapting on anything but particle count.
collide          vss air air.vss
fix              in emit/face air xlo
# ONE-SHOT static refinement near the surface. 'adapt_grid' acts immediately,
# so any fix it references must already have produced output -- with 'surf'
# style there is no such dependency.
adapt_grid       all refine surf all 0.15 iterate 2
# DYNAMIC refinement during the run. The fix Nfreq and the adapt Nevery must
# match, and maxlevel caps the refinement depth.
compute          npc grid all all n
fix              fnp ave/grid all 1 {nevery} {nevery} c_npc[1]
fix              ad adapt {nevery} all refine coarsen value f_fnp 20.0 5.0 &
                 maxlevel {maxlevel}
timestep         {dt}
stats            {nevery}
stats_style      step np ncoll ngrid maxlevel
run              {nsteps}
"""


KNOWLEDGE = {
    "adaptive_grid": {
        "description": "Static (adapt_grid) and dynamic (fix adapt) grid "
                       "adaptation to resolve shocks, boundary layers and "
                       "surface geometry",
        "spatial_dims": [2, 3],
        "key_commands": {
            "adapt_grid": "adapt_grid <grp> refine|coarsen [coarsen|refine] "
                          "particle <rcount> <ccount> | surf <surfgrp> <ssize> "
                          "| value <c_ID[i]|f_ID[i]> <rval> <cval> | random "
                          "<rfrac> <cfrac> [iterate <n>] [maxlevel <n>] "
                          "[minlevel <n>] [thresh less|more less|more] "
                          "[combine sum|min|max] [cells <nx> <ny> <nz>]",
            "fix adapt": "fix <ID> adapt <Nevery> <grp> ... — same style "
                         "arguments as adapt_grid, applied during the run",
            "create_grid ... level": "create_grid Nx Ny Nz level <n> <region> "
                                     "<nx> <ny> <nz> — static hierarchical "
                                     "refinement at setup time",
            "balance_grid": "balance_grid rcb cell|part|time — rebalance cells "
                            "across MPI ranks after adaptation (no effect in a "
                            "serial build)",
            "stats_style": "add 'ngrid' and 'maxlevel' so the adaptation is "
                           "visible in the stats table",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-grid tally idiom"),
        "pitfalls": [
            "[Setup] 'adapt_grid ... value f_<fixID>' placed BEFORE any 'run' "
            "reads every cell as literal 0, because the fix has produced no "
            "output yet. With the DEFAULT 'thresh more less' that means no "
            "cell qualifies for refinement and the grid is left alone — but "
            "the zeros are not harmless: with a 'coarsen' action, or with "
            "'thresh less', the same command instead coarsens or refines the "
            "ENTIRE grid. 'adapt_grid ... particle ...' and 'adapt_grid ... "
            "value c_<computeID>[N]' both work correctly before a run; the "
            "trap is specific to a fix that has not yet fired. "
            "Signal: 'Adapting grid ...' followed by '  no grid adaptation "
            "performed' instead of the '<N> cells refined, <M> cells "
            "coarsened' line. That message is AMBIGUOUS — it is also printed "
            "when the fix has data but no cell crosses the threshold — so "
            "confirm by moving the adapt_grid line after a short run.",

            "[Syntax] The bracket on an adapt_grid or fix adapt 'value' source "
            "must match its shape: a fix ave/grid with ONE input value is a "
            "per-grid VECTOR and must be written 'value f_<ID>' with no "
            "bracket, while a compute grid is always an ARRAY and must be "
            "written 'value c_<ID>[N]'. This is checked before and after a "
            "run alike, so it is easy to mistake for the "
            "fix-not-yet-computed trap above. "
            "Signal: 'ERROR: Adapt fix does not calculate a per-grid array "
            "(../adapt_grid.cpp:484)'.",

            "[Numerical] Refinement is unbounded unless capped: every level "
            "multiplies the cell count and divides the particles per cell, and "
            "below one particle per cell every per-cell diagnostic (including "
            "the one driving the adaptation) becomes noise, which can drive "
            "further refinement. Use 'maxlevel'. "
            "Signal: the Ngrid and Maxlevel columns keep climbing while Np is "
            "flat; check 'compute reduce min' on a per-cell particle count and "
            "require it to stay >= 1.",

            "[Setup] 'fix adapt' can only act on steps where the fix it reads "
            "has produced output, so its Nevery must be a multiple of that "
            "fix's Nfreq — and the stats interval that prints any dependent "
            "quantity is subject to the same rule. The failure is a mid-run "
            "abort right after the step-0 stats line, not a setup error. There "
            "are THREE different messages depending on who reads the fix, and "
            "the one for the adapt itself is easy to miss — an earlier wording "
            "listed only the other two, so a guard built from it would not fire "
            "on the mistake the entry is about. "
            "Signal: 'ERROR: Fix for adapt not computed at compatible time "
            "(../adapt_grid.cpp:502)' when the adapt Nevery is not a multiple "
            "of the fix Nfreq; 'ERROR: Stats and fix not computed at compatible "
            "times (../stats.cpp:203)' when stats_style prints f_<ID> directly; "
            "'ERROR: Fix used in compute reduce not computed at compatible time "
            "(../compute_reduce.cpp:805)' when a compute reduce reads it.",

            "[Physics] Adapting on particle count refines where particles "
            "already are, which is not where the gradients are: in a "
            "hypersonic case that means the post-shock region, not the shock "
            "itself. Adapting on the cell Knudsen number from 'compute "
            "lambda/grid ... knall' targets under-resolution directly. "
            "Signal: after adaptation the cell Knudsen minimum from 'compute "
            "reduce min c_lam[2]' has not improved even though Ngrid has "
            "grown several-fold — the refinement went to the wrong cells.",

            "[Setup] 'balance_grid' distributes cells over MPI ranks; on a "
            "serial build it runs and moves nothing, so it is not a remedy for "
            "an unbalanced particle load within one rank. "
            "Signal: the line 'Balance grid migrated 0 cells' — it is printed "
            "for every balance_grid call, and a 0 there in a run you expected "
            "to rebalance means either a serial build or an already-balanced "
            "decomposition.",
        ],
    },
}

GENERATORS = {
    "adaptive_grid_circle_2d": _adapt_circle_2d,
    "adaptive_grid_2d": _adapt_circle_2d,
}
