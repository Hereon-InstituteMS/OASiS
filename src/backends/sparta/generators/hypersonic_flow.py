"""Hypersonic rarefied flow over a body: inflow, shock, surface heating."""

from ._common import output_idioms


def _hypersonic_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hypersonic argon over a 2d circle, with a diffuse wall, a surface
    energy-flux tally and Knudsen-driven grid adaptation.
    """
    vstream = params.get("vstream", 2634.1)
    t_free = params.get("t_free", 200.0)
    t_wall = params.get("t_wall", 500.0)
    nrho = params.get("nrho", 4.247e19)
    fnum = params.get("fnum", 7.0e14)
    dt = params.get("dt", 3.5e-7)
    nsteps = params.get("nsteps", 600)
    nevery = params.get("nevery", 200)
    return f"""\
# Hypersonic argon over a 2d circle with Kn-driven grid adaptation - SPARTA
seed             12345
dimension        2
global           nrho {nrho} fnum {fnum} gridcut 0.01
timestep         {dt}
# xlo/xhi outflow, ylo reflecting (symmetry line), yhi outflow, z periodic
boundary         o ro p
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1 block * * *
species          ar.species Ar
mixture          all vstream {vstream} 0.0 0.0 temp {t_free}
collide          vss all ar.vss
collide_modify   vremax 1000 yes
read_surf        circle.surf group 1
surf_collide     wall diffuse {t_wall} 1.0
surf_modify      1 collide wall
fix              in emit/face all xlo
create_particles all n 0
# surface heat flux (per-surf array -> index it)
compute          q surf all all etot
fix              fq ave/surf all 1 {nevery} {nevery} c_q[1] ave one
compute          qtot reduce sum f_fq
# cell Knudsen number, used both as a resolution check and to drive adaptation
compute          nr grid all all nrho
compute          tg thermal/grid all all temp
fix              fg ave/grid all 1 {nevery} {nevery} c_nr[*] c_tg[*] ave one
compute          lam lambda/grid f_fg[1] f_fg[2] lambda knall
fix              ad adapt {nevery} all refine coarsen value c_lam[2] 2.0 4.5 &
                 combine min thresh less more cells 2 2 1
stats_style      step np nattempt ncoll nscoll ngrid maxlevel c_qtot
stats            {nevery}
run              {nsteps}
"""


KNOWLEDGE = {
    "hypersonic_flow": {
        "description": "Hypersonic rarefied flow over a body: bow shock, "
                       "surface pressure and heat flux, DSMC",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix emit/face": "fix <ID> emit/face <mixID> <face...> [n <Np>] "
                             "[nevery <N>] [perspecies yes|no] [subsonic <P> "
                             "<T|NULL>] [region <regID>]",
            "collide_modify": "collide_modify vremax <N> <yes|no> — the "
                              "upstream hypersonic decks all set this",
            "read_surf": "read_surf <file> group <ID> [trans/scale/rotate/clip]",
            "compute surf": "compute <ID> surf <grp> <mix> press shx shy etot "
                            "— surface pressure, shear and heat flux",
            "fix adapt": "fix <ID> adapt <Nevery> <grp> refine coarsen value "
                         "c_lam[2] <rthresh> <cthresh> combine min thresh less "
                         "more cells 2 2 1",
            "compute lambda/grid": "compute <ID> lambda/grid <nrho-src> "
                                   "<temp-src> lambda knall — column 2 is "
                                   "Kn_cell, the natural adaptation criterion",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-surf tally idiom", "per-grid tally idiom"),
        "pitfalls": [
            "[Setup] 'fix emit/face' cannot inject through a PERIODIC face. "
            "The inflow face must be declared 'o' (outflow) or a surface in "
            "the 'boundary' command, and this is checked only when the fix is "
            "defined, so it fires late in a long deck. "
            "Signal: 'ERROR: Cannot use fix emit/face on periodic boundary "
            "(../fix_emit_face.cpp:182)'.",

            "[Physics] An emit face does NOT hold the domain at the freestream "
            "density. In a box whose other faces are outflow, particles leave "
            "faster than one face injects and the particle count falls, "
            "silently, for the whole run — the flow you are sampling is "
            "thinner than the freestream you specified. "
            "Signal: the Np column decreases monotonically over the run "
            "instead of levelling off. Either drive the case to steady state "
            "and check Np has flattened, or emit on every inflow face.",

            "[Physics] Surface heat flux and pressure need statistical steady "
            "state, which for a hypersonic case is several flow-through times. "
            "Sampling the first stats block gives a number that is wrong by a "
            "large factor while the run looks entirely healthy. "
            "Signal: the tallied surface quantity is still monotonically "
            "changing between successive fix ave/surf outputs; only when "
            "successive windows agree within the seed-to-seed spread is it "
            "converged.",

            "[Numerical] The bow shock and the wall boundary layer are far "
            "thinner than the freestream mean free path scale, so a uniform "
            "grid sized on freestream conditions is far too coarse there. "
            "SPARTA does not warn: the run is clean and the shock is simply "
            "smeared. "
            "Signal: 'compute <ID> lambda/grid ... knall' + 'compute reduce "
            "min' gives a cell Knudsen number below 1 somewhere in the domain; "
            "drive 'fix adapt' from that column rather than from particle "
            "count.",

            "[Numerical] 'fix adapt' driven by a per-grid COMPUTE (c_ID[i]) "
            "re-evaluates it each adaptation step, but driven by a FIX (f_ID) "
            "it can only act on steps where that fix has produced output. Set "
            "the fix Nfreq and the adapt Nevery to the same value. "
            "Signal: 'ERROR: Fix used in compute reduce not computed at "
            "compatible time (../compute_reduce.cpp:805)' or 'ERROR: Stats and "
            "fix not computed at compatible times (../stats.cpp:203)' — both "
            "abort mid-run, right after the step-0 stats line.",

            "[Setup] Over-refinement is unbounded unless you cap it: each "
            "refinement level multiplies the cell count, and the particle "
            "count per cell falls with it until every per-cell diagnostic is "
            "noise. Use the 'maxlevel' keyword. "
            "Signal: the Ngrid and Maxlevel columns keep climbing between "
            "stats lines while Np is flat — cells are being split without new "
            "particles to fill them.",
        ],
    },
}

GENERATORS = {
    "hypersonic_flow_circle_2d": _hypersonic_circle_2d,
    "hypersonic_flow_2d": _hypersonic_circle_2d,
}
