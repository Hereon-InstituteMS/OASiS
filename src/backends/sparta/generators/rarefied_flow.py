"""Rarefied / free-molecular flow: SPARTA decks and knowledge."""

from ._common import output_idioms


def _free_molecular_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisionless thermal argon in a 2d box with specularly reflecting walls.
    There is deliberately NO collide command: this is the Kn -> infinity limit.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e11)
    temp = params.get("temp", 273.15)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Free-molecular (collisionless) argon in a 2d box - SPARTA DSMC
# No 'collide' command => Ncoll is 0 by construction. That is the definition
# of this regime, not a defect; check Ncoll to prove it.
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {temp}
# global MUST precede create_particles, else 'Created 0 particles'
global           nrho {nrho} fnum {fnum}
create_particles gas n 0
# compute temp is the KINETIC temperature (bulk motion NOT removed);
# with vstream 0 it coincides with the thermal temperature.
compute          tk temp
timestep         {dt}
stats            100
stats_style      step np ncoll c_tk
run              {nsteps}
"""


def _fourier_channel_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisional argon between a cold (ylo) and a hot (yhi) fully accommodating
    diffuse wall — the standard wall-bounded conduction test. Reports the
    per-cell thermal temperature profile and the cell Knudsen number.
    """
    nx = params.get("nx", 8)
    ny = params.get("ny", 40)
    lx = params.get("lx", 2.0e-5)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e23)
    fnum = params.get("fnum", 2.0e11)
    t_cold = params.get("t_cold", 300.0)
    t_hot = params.get("t_hot", 1000.0)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 2000)
    return f"""\
# 2d Fourier channel: argon between a cold ylo wall and a hot yhi wall.
# Both y faces are 'surface' boundaries ('s'), each bound to its own
# surf_collide model. x is periodic, so this is a pure 1d conduction problem.
seed             12345
dimension        2
boundary         p ss p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {t_cold}
global           nrho {nrho} fnum {fnum}
# a 'diffuse' wall needs BOTH a temperature and an accommodation coefficient,
# in that order; 'specular' takes no temperature and exchanges no energy.
surf_collide     cold diffuse {t_cold} 1.0
surf_collide     hot  diffuse {t_hot} 1.0
bound_modify     ylo collide cold
bound_modify     yhi collide hot
collide          vss gas ar.vss
create_particles gas n 0
# temperature profile: per-cell thermal temperature, time-averaged
compute          tg thermal/grid all all temp
fix              ftg ave/grid all 10 20 200 c_tg[*]
compute          tmin reduce min f_ftg
compute          tmax reduce max f_ftg
# resolution check: cell Knudsen number must be >= 1 for the cells to be
# smaller than the local mean free path
compute          nr grid all species nrho
fix              fnr ave/grid all 10 20 200 c_nr[*]
compute          lam lambda/grid f_fnr[*] f_ftg lambda knall
compute          knmin reduce min c_lam[2]
timestep         {dt}
stats            200
stats_style      step np ncoll c_tmin c_tmax c_knmin
dump             dgrid grid all 1000 dump.profile id yc f_ftg
run              {nsteps}
"""


KNOWLEDGE = {
    "rarefied_flow": {
        "description": "Rarefied / free-molecular gas flow (high Knudsen "
                       "number) solved with DSMC simulator particles",
        "spatial_dims": [2, 3],
        "regime": "Kn = lambda/L. Kn >> 1 is free-molecular (drop the collide "
                  "command); Kn ~ 0.01-1 is the transitional regime DSMC is "
                  "built for; Kn << 0.01 is continuum and a Navier-Stokes "
                  "solver is cheaper and more accurate.",
        "key_commands": {
            "global": "global nrho <n> fnum <F> [gridcut <d>] — nrho is real "
                      "number density (1/m^3), fnum is real particles per "
                      "simulation particle",
            "create_grid": "create_grid Nx Ny Nz [level <n> <bounds> nx ny nz]",
            "create_particles": "create_particles <mixID> n 0 [region <regID>]",
            "compute temp": "compute <ID> temp — GLOBAL kinetic temperature, "
                            "streaming velocity NOT removed",
            "compute thermal/grid": "compute <ID> thermal/grid <grp> <mix> "
                                    "temp — per-cell temperature with the "
                                    "per-cell mean velocity removed",
            "compute lambda/grid": "compute <ID> lambda/grid <nrho-src> "
                                   "<temp-src> lambda knall — column 1 is the "
                                   "mean free path, column 2 is Kn_cell",
            "compute dt/grid": "compute <ID> dt/grid <grp> <tfrac> <cfrac> "
                               "<tau> <temp> <usq> <vsq> <wsq> — recommended "
                               "per-cell timestep",
            "compute reduce": "compute <ID> reduce min|max|ave|sum "
                              "c_X[i]|f_X[i] — the only way to get a per-grid "
                              "or per-surf quantity into stats_style",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "unit_systems": "SI by default (m, kg, s, K). 'units cgs' switches the "
                        "constants but does NOT convert your input files.",
        "output_idioms": output_idioms("per-grid tally idiom", "boundary tally idiom"),
        "pitfalls": [
            "[Numerical] Grid cells much larger than the local mean free path "
            "run cleanly and give an over-diffusive answer, and a "
            "particle-count check does not catch it — a coarse grid has MORE "
            "particles per cell, not fewer. The cell Knudsen number is the "
            "quantity that must be checked. "
            "Signal: 'compute <C> lambda/grid <nrho-src> <temp-src> lambda "
            "knall' + 'compute reduce min c_C[2]' returns a value below 1 "
            "(cells larger than a mean free path). SPARTA never checks this "
            "for you and never warns.",

            "[Numerical] compute lambda/grid and compute dt/grid read from a "
            "fix ave/grid that has produced no output yet, so on the first "
            "stats line they return SPARTA's no-data sentinel, not a physical "
            "value. Reading the step-0 row gives a mean free path of 1e+20 m "
            "and a cell Knudsen number of 1e+25. "
            "Signal: an absurd 1e+20-scale value in the lambda or Kn column on "
            "the first stats line only; it becomes physical from the first "
            "output step of the fix onward. Never sample diagnostics at "
            "step 0.",

            "[Numerical] A timestep larger than the mean collision time runs "
            "cleanly and inflates transport — SPARTA has no CFL check and "
            "never compares your 'timestep' with its own recommendation. "
            "'compute <D> dt/grid ...' computes a recommended per-cell "
            "timestep and reports the SAME value in a well-resolved run and in "
            "one using a hundred times too large a step. "
            "Signal: reduce compute dt/grid with 'compute reduce min' and "
            "assert your timestep is below it; a run whose recommended dt sits "
            "far below the dt you set exits 0 regardless.",

            "[Numerical] An fnum that leaves well under one simulation "
            "particle per cell still runs, still reports a plausible collision "
            "rate, and turns every PER-CELL diagnostic into garbage — mean "
            "free paths tens of orders of magnitude too large. The collapse is "
            "specifically at sub-particle-per-cell occupancy; around one "
            "particle per cell the diagnostics are still sane. "
            "Signal: 'compute <C> grid all all n' + 'compute reduce min "
            "c_C[1]' — require the cell MINIMUM to be >= 1, not just the "
            "average, before trusting any per-cell quantity.",

            "[Physics] 'compute <ID> temp' sums the kinetic energy of every "
            "particle WITHOUT subtracting any mean velocity, so it reports "
            "T_thermal + m*|vstream|^2/(dim*kB) — a translational temperature "
            "inflated by the bulk motion. The inflation is proportional to "
            "SPECIES MASS and to the SQUARE of the stream speed, so it is "
            "enormous for a heavy species at hypersonic speed and negligible "
            "for a light species or a slow flow; it is independent of "
            "dimension and of whether collisions are on. It is also purely "
            "translational: for a molecular gas it drifts as translation "
            "exchanges energy with rotation. "
            "Signal: put 'compute <ID> temp' and a reduced 'compute <ID> "
            "thermal/grid <grp> <mix> temp' in the same stats_style; a gap of "
            "about m*|vstream|^2/(dim*kB) between them means the first column "
            "is not a thermal temperature. With vstream 0 the two agree.",

            "[Numerical] compute thermal/grid removes the per-cell MEAN "
            "velocity, so it only removes the bulk motion to the extent the "
            "cell resolves the flow. Across an unresolved velocity gradient "
            "the sub-cell shear variance is counted as thermal motion and the "
            "reported temperature reads high; refining the grid drives it "
            "down toward the true value. "
            "Signal: the cell-averaged thermal/grid temperature FALLS "
            "substantially when you refine the grid at fixed physics — a "
            "resolved field is grid-independent, so a moving answer means the "
            "cells are straddling the shear.",

            "[Numerical] compute thermal/grid is biased LOW per timestep "
            "when cells hold few particles: it subtracts the per-cell SAMPLE "
            "mean, which costs one degree of freedom, and SPARTA writes "
            "exactly 0 K into any cell holding one particle or none. A "
            "'compute reduce ave' then averages those hard zeros in, so at a "
            "couple of particles per cell the reported cell-average can be a "
            "THIRD of the temperature you set — much worse than the naive "
            "one-degree-of-freedom estimate. Time-averaging through 'fix "
            "ave/grid' largely removes it, because the effective count is the "
            "particle count times the number of samples. "
            "Signal: an instantaneous cell-averaged thermal/grid temperature "
            "far below 'compute temp' in a gas at rest, which climbs back "
            "toward it when you lower fnum or route the compute through a fix "
            "ave/grid. Note the first tens of steps also fall on their own as "
            "the near-uniform fill from create_particles relaxes to Poisson "
            "occupancy — that is not physics either.",

            "[Physics] A homogeneous equilibrium box is worthless as a "
            "timestep or grid-resolution test: the collision statistics of a "
            "gas at rest are almost unchanged by a ten-times-too-large "
            "timestep or a ten-times-too-coarse grid, because there is no "
            "gradient for the errors to act on. Both errors show up only in a "
            "gradient-driven quantity. "
            "Signal: a dt or grid study whose Ncoll / Ncollave columns are flat "
            "is measuring nothing — move the study to a wall-bounded or "
            "flow-driven case before drawing a conclusion.",

            "[Physics] Wall and boundary fluxes in a driven run need several "
            "flow-through times; the first stats block is not an answer and "
            "the run looks perfectly healthy throughout the transient. "
            "Signal: at steady state the two opposed wall fluxes are equal and "
            "opposite. Tally both and use that balance as the convergence "
            "test, not the wall-clock or the step count.",
        ],
    },
}

GENERATORS = {
    "rarefied_flow_box_2d": _free_molecular_box_2d,
    "rarefied_flow_2d": _free_molecular_box_2d,
    "rarefied_flow_free_molecular_2d": _free_molecular_box_2d,
    "rarefied_flow_channel_2d": _fourier_channel_2d,
}
