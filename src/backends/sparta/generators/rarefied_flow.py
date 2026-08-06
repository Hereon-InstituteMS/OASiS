"""Rarefied / free-molecular flow: SPARTA decks and knowledge."""

from ._common import output_idioms


def _free_molecular_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisionless thermal argon in a 2d box with specularly reflecting walls.
    There is deliberately no collide command.

    The DEFAULTS ARE NOT IN THE FREE-MOLECULAR LIMIT and must be changed
    before the name of this template is true of the run. At nrho 7.07043e22
    with lx = ly = 1e-4 m the VHS mean free path for argon is ~1.9e-5 m, i.e.
    Kn_box ~ 0.19 — transitional. Measured on the installed build: adding the
    single line 'collide vss gas ar.vss' to this exact deck yields
    'Collide attempts = 6212' and 'Collide occurs = 4893' over the default
    500 steps with 1000 particles. Omitting collide at these numbers is
    therefore suppressing ~4.9 real collisions per particle, not modelling a
    collisionless gas. For a genuine Kn >> 1 run lower nrho (and fnum with
    it) by several orders of magnitude, or enlarge the box.
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
# No 'collide' command => Ncoll is 0 by construction, so Ncoll == 0 here proves
# nothing about the regime. Kn_box at the DEFAULT nrho/box above is only ~0.19:
# these numbers are transitional, not collisionless. Lower nrho and fnum by
# several decades (or enlarge the box) before calling a run free-molecular.
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
            "value. The sentinel is BIG = 1e+20 (compute_lambda_grid.cpp:37), "
            "written into lambda whenever a cell's number density is zero; the "
            "Kn column then carries lambda/cell-size, so its magnitude depends "
            "on your cell size (1e+24 at a 1e-4 m cell, 1e+25 at 1e-5 m). "
            "Step 0 is NOT the only exposure: any cell that holds no particles "
            "keeps the sentinel on EVERY stats line, so on a grid refined past "
            "one particle per cell the maximum stays at 1e+20 forever while "
            "the minimum looks healthy. "
            "Signal: an absurd 1e+20-scale value in the lambda or Kn column. "
            "Never sample diagnostics at step 0, and reduce with BOTH 'compute "
            "reduce min' and 'compute reduce max' — a max still pinned at "
            "1e+20 after the fix has produced output means empty cells, not a "
            "warm-up transient.",

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
            "T_thermal + m*|vstream|^2/(3*kB) — a translational temperature "
            "inflated by the bulk motion. The inflation is proportional to "
            "SPECIES MASS and to the SQUARE of the stream speed, so it is "
            "enormous for a heavy species at hypersonic speed and negligible "
            "for a light species or a slow flow. The divisor is 3*kB in BOTH "
            "2d and 3d — compute_temp.cpp normalises by 3, not by 'dimension', "
            "because even a 2d run carries three velocity components — so it "
            "is independent of dimension and of whether collisions are on. It "
            "is also purely "
            "translational: for a molecular gas it drifts as translation "
            "exchanges energy with rotation. "
            "Signal: put 'compute <ID> temp' and a reduced 'compute <ID> "
            "thermal/grid <grp> <mix> temp' in the same stats_style; a gap of "
            "about m*|vstream|^2/(3*kB) between them means the first column "
            "is not a thermal temperature. With vstream 0 the two agree.",

            "[Numerical] compute thermal/grid removes the per-cell MEAN "
            "velocity, so it only removes the bulk motion to the extent the "
            "cell resolves the flow. Across an unresolved velocity gradient "
            "the sub-cell shear variance is counted as thermal motion and the "
            "reported temperature reads high; refining the grid drives it "
            "down toward the true value. "
            "Signal: the cell-averaged thermal/grid temperature falls when you "
            "refine the grid at fixed physics — but by a FEW PERCENT, not "
            "'substantially' as an earlier wording had it. Measured on a "
            "Couette slab with walls translating at plus and minus 1000 m/s, "
            "well above the thermal speed, a tenfold refinement moved it about "
            "four and a half percent. That follows from the mechanism and is "
            "not a property of this case: the sub-cell velocity spread is "
            "(dU/dy)*dy, so its variance enters as the SQUARE of the cell size "
            "and in only ONE of three velocity components. Two things make "
            "that small number unusable unless you control for them. First, "
            "read it through a 'fix ave/grid', never instantaneously: "
            "refining lowers the per-cell occupancy and the low-occupancy bias "
            "of compute thermal/grid pushes the reading DOWN as well, in the "
            "same direction, so an instantaneous comparison cannot tell the "
            "two apart. Second, run more than one seed and ask whether the "
            "coarse and fine CLUSTERS separate rather than whether two numbers "
            "differ — measured here the seed spread was about a percent at the "
            "coarse grid, so a single pair gives you only a few times the "
            "noise. Removing the wall motion entirely collapses the difference "
            "to a fraction of a percent and the clusters overlap, which is the "
            "control that shows the residual really is the shear.",

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

            "[Output] 'fix <F> ave/time Nevery Nrepeat Nfreq ...' enforces two "
            "constraints that a reader cannot tell apart from the message: "
            "Nfreq must be an exact multiple of Nevery, AND Nevery*Nrepeat "
            "must not exceed Nfreq. Both violations abort with the SAME "
            "generic text, which names neither rule and does not print the "
            "three numbers, so the fix is to check both by hand. The window "
            "the fix averages is the LAST Nrepeat samples ending at each "
            "Nfreq step, not the whole Nfreq interval — 'ave/time 10 5 1000' "
            "averages five samples out of the final fifty steps of every "
            "thousand and ignores the other 950. "
            "Signal: 'ERROR: Illegal fix ave/time command "
            "(../fix_ave_time.cpp:129)' for either violation. Widen the window "
            "with Nrepeat, not with Nfreq. (Verified 2026-08-07)",

            "[Output] A fix's output can only be printed on steps where the "
            "fix has fresh data, and SPARTA checks this the STRICT way: "
            "'stats N' must be a multiple of the fix's Nfreq, so sampling the "
            "stats table MORE often than the averaging window is a hard error, "
            "not a repeated value. 'stats 50' against Nfreq 100 aborts; 'stats "
            "200' against Nfreq 100 runs. 'compute reduce' reading a fix "
            "raises the same objection with its own message. The check fires "
            "at the start of the run, after setup has printed, so a deck can "
            "look healthy for several screens before it stops. "
            "Signal: 'ERROR: Stats and fix not computed at compatible times "
            "(../stats.cpp:203)', and from the reduce path 'ERROR: Fix used in "
            "compute reduce not computed at compatible time', which "
            "compute_reduce.cpp raises from three different call sites (lines "
            "778, 805 and 832) for a global, a per-grid and a per-surf input — "
            "so match the text, not the line. (Verified 2026-08-07)",

            "[Output] The FIRST stats row of a run prints every f_<ID> of a "
            "fix ave/* as a literal 0, because no averaging window has closed "
            "yet. There is no warning and no sentinel value — the zero is "
            "indistinguishable from a real measurement of zero, and it is the "
            "row an agent reading the top of the table sees first. The same "
            "zero reappears at the start of every subsequent 'run' command. "
            "Signal: the f_<ID> column reads exactly 0 on the step-0 row and "
            "jumps to a physical value on the first row at or after Nfreq. "
            "Discard the first row; if you need a time series on disk instead "
            "of in the table, add 'file <name>' to the fix ave/time line, "
            "which writes one row per Nfreq step. (Verified 2026-08-07)",

            "[Output] 'compute <ID> property/grid <grid-group> <attrs>' is the "
            "exception to SPARTA's per-grid shape rule: with ONE attribute it "
            "produces a per-grid VECTOR read as c_ID with no bracket, and only "
            "with two or more does it produce an ARRAY read as c_ID[i]. "
            "'compute grid' and 'compute surf' are ALWAYS arrays, so the "
            "bracket convention you learn on those is wrong here and the two "
            "mistakes have opposite messages. It also holds GEOMETRY ONLY — "
            "id, proc, xlo/ylo/zlo, xhi/yhi/zhi, xc/yc/zc, vol — so any flow "
            "quantity asked of it is rejected; nrho and temperature come from "
            "'compute grid'. 'compute property/surf' takes a SURF group and "
            "the analogous element geometry (id, v1x..v3z, xc/yc/zc, area, "
            "normx/normy/normz). "
            "Signal: 'ERROR: Compute reduce compute does not calculate a "
            "per-grid array (../compute_reduce.cpp:232)' when you bracket a "
            "one-attribute property/grid, 'ERROR: Compute reduce compute does "
            "not calculate a per-grid vector (../compute_reduce.cpp:229)' when "
            "you do not bracket a multi-attribute one or a compute grid, "
            "'ERROR: Invalid keyword in compute property/grid command "
            "(../compute_property_grid.cpp:84)' for a flow quantity, and "
            "'ERROR: Invalid compute property/grid field for 2d simulation "
            "(../compute_property_grid.cpp:54)' for zlo/zhi/zc in 2d. "
            "(Verified 2026-08-07)",

            "[Output] 'compute reduce' is the ONLY bridge from a per-grid, "
            "per-surf or per-particle compute to a number in the stats table; "
            "stats_style accepts a compute only if it produces a global "
            "scalar. So 'compute ke/particle' — the per-particle kinetic "
            "energy, and the natural input to a velocity-distribution check — "
            "cannot be printed directly and has to be reduced (min, max, sum, "
            "ave) or histogrammed. 'replace <col1> <col2>' needs min or max "
            "mode and two DIFFERENT input columns, so it cannot be used on a "
            "single-value reduce. "
            "Signal: 'ERROR: Stats compute does not compute scalar "
            "(../stats.cpp:678)' when a per-particle or per-surf compute is "
            "named in stats_style, and 'ERROR: Illegal compute reduce command "
            "(../compute_reduce.cpp:168)' for a replace pair that names one "
            "column or a column past the end. (Verified 2026-08-07)",

            "[Setup] A 'region' is a SELECTOR, not geometry. It reflects "
            "nothing, blocks nothing and creates no surface: a region defined "
            "in a deck and never named by another command changes the run not "
            "at all — same particle count, same collision count, same exit "
            "count, step for step. Only the commands that take a region "
            "argument see it (create_particles, create_grid, adapt_grid, the "
            "fix emit family, fix ave/histo, dump particle). To obstruct a "
            "flow you need read_surf plus a surf_collide model. Styles are "
            "block, cylinder, sphere, plane, union and intersect; block takes "
            "six bounds and accepts INF and EDGE; union and intersect take the "
            "COUNT of sub-regions first ('region u union 2 a b'); 'side out' "
            "inverts the selection. In a 2d run a sphere and a z-cylinder of "
            "the same radius select exactly the same cells. "
            "Signal: there is NO signal for an unused region — the run is "
            "identical to one without it, which is why this has to be checked "
            "by reading the deck. The typo cases do speak: 'ERROR: "
            "Unrecognized region style (../domain.cpp:471)' for a style that "
            "does not exist, and 'ERROR: Create_particles region does not "
            "exist (../create_particles.cpp:122)' for a name that was never "
            "defined. (Verified 2026-08-07)",

            "[Setup] An external body force needs THREE lines that agree, and "
            "getting two of them right leaves a run that is silently "
            "force-free. 'fix <ID> field/grid <ax> <ay> <az>' (or "
            "field/particle) names GRID-style variables for field/grid and "
            "PARTICLE-style variables for field/particle, written as BARE "
            "names with no 'v_' prefix, with NULL for an unused component. The "
            "fix on its own does nothing at all: the mover only consults it "
            "after 'global field grid <fix-ID> <Nevery>' or 'global field "
            "particle <fix-ID>' activates it, and that first argument is a FIX "
            "ID, not a number — upstream's own example names its fix '1', so "
            "the line reads 'global field grid 1 0' and copies as if 1 were a "
            "flag. Without the global line the deck runs to completion with "
            "the particle statistics bit-identical to a deck with no field. "
            "The simpler alternative is 'global field constant <magnitude> "
            "<fx> <fy> <fz>', which needs no fix and no variable. "
            "Signal: for the silent case there is NO signal — compare a "
            "temperature or mean kinetic energy against the same deck with the "
            "fix deleted and see whether anything moved. The loud cases are "
            "'ERROR: External field fix ID not found (../update.cpp:221)' when "
            "the global line names something that is not a fix, 'ERROR: "
            "Variable for fix field/grid is invalid style "
            "(../fix_field_grid.cpp:105)' for an equal-style variable where a "
            "grid-style one is required, and 'ERROR: Variable name for fix "
            "field/grid does not exist (../fix_field_grid.cpp:103)' when the "
            "name carries a 'v_' prefix. (Verified 2026-08-07)",

            "[Numerical] 'fix <ID> dt/reset <Nevery> <c_ID|f_ID> <weight> "
            "<resetflag>' can change the global timestep underneath a running "
            "simulation, and the last argument decides whether it does. With "
            "resetflag 0 the fix only COMPUTES a recommended timestep and "
            "publishes it as f_ID; with 1 or 2 it WRITES it into the global "
            "timestep, so the dt you set with the 'timestep' command is gone "
            "after the first Nevery steps and every subsequent result belongs "
            "to a step size you did not choose — a per-cell recommendation on "
            "a near-equilibrium box can be several times the value a user "
            "picked. <weight> in [0,1] is an exponential smoothing factor on "
            "the change, not a safety factor. The recommendation itself comes "
            "from 'compute dt/grid', so it inherits that compute's dependence "
            "on a fix ave/grid that has to have produced output first. "
            "Signal: put 'dt' in stats_style. A constant Dt column means "
            "resetflag 0 or no fix at all; a Dt column that steps at multiples "
            "of Nevery means the timestep is being rewritten, and the "
            "convergence study you thought you were running is not one. "
            "(Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "rarefied_flow_box_2d": _free_molecular_box_2d,
    "rarefied_flow_2d": _free_molecular_box_2d,
    "rarefied_flow_free_molecular_2d": _free_molecular_box_2d,
    "rarefied_flow_channel_2d": _fourier_channel_2d,
}
