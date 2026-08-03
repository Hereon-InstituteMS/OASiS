"""Particle injection from box faces or surfaces (inflow boundaries)."""

from ._common import output_idioms


def _emit_channel_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Continuous inflow through the xlo face of an otherwise open 2d box.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-2)
    ly = params.get("ly", 1.0e-2)
    vstream = params.get("vstream", 1000.0)
    temp = params.get("temp", 300.0)
    nrho = params.get("nrho", 1.0e20)
    fnum = params.get("fnum", 1.0e13)
    dt = params.get("dt", 1.0e-7)
    nsteps = params.get("nsteps", 1000)
    return f"""\
# Continuous inflow through a box face - SPARTA DSMC
# The emit face must NOT be periodic; 'o' (outflow) or a surface face is fine.
seed             12345
dimension        2
global           gridcut 0.0 nrho {nrho} fnum {fnum}
boundary         o o p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N2
# the emit rate follows THIS mixture: its nrho (falling back to global nrho),
# its vstream and its temp, times the face area, divided by fnum
mixture          air N2 vstream {vstream} 0.0 0.0 temp {temp}
collide          vss air air.vss
fix              in emit/face air xlo
timestep         {dt}
stats            100
stats_style      step np ncoll nexit
run              {nsteps}
"""


KNOWLEDGE = {
    "particle_emission": {
        "description": "Particle injection / emission from box faces or "
                       "surfaces — the DSMC inflow boundary condition",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix emit/face": "fix <ID> emit/face <mixID> <face...> [n <Np>] "
                             "[nevery <N>] [perspecies yes|no] [region <regID>] "
                             "[modulate v_<name>] [subsonic <P> <T|NULL>]",
            "fix emit/surf": "fix <ID> emit/surf <mixID> <group-ID> [n <Np>] "
                             "[normal yes|no] [nevery <N>] [perspecies ...]",
            "fix emit/face/file": "fix <ID> emit/face/face/file <mixID> <face> "
                                  "<file> <Nx> <Ny> — spatially varying inflow",
            "mixture": "mixture <mixID> <species...> nrho <n> vstream <vx> "
                       "<vy> <vz> temp <T> — these set the emitted flux",
            "boundary": "the emit face must be 'o' or 's'; 'p' is rejected",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("boundary tally idiom"),
        "pitfalls": [
            "[Setup] 'fix emit/face' cannot be attached to a PERIODIC face — "
            "particles already re-enter there, so an emit fix would "
            "double-count. The check happens when the fix is defined. "
            "Signal: 'ERROR: Cannot use fix emit/face on periodic boundary "
            "(../fix_emit_face.cpp:182)'.",

            "[Numerical] The emitted flux is set by the MIXTURE, not by "
            "'global nrho': an 'nrho' keyword on the emit mixture silently "
            "overrides the global value for every emitted and created "
            "particle. Two densities in one deck do not conflict-check. "
            "Signal: the domain fills to a steady Np far from nrho*V/fnum; "
            "grep every mixture line for 'nrho' before trusting an inflow "
            "density.",

            "[Physics] Emitting on ONE face of an otherwise outflow box does "
            "not hold the domain at the freestream: particles leave through "
            "the other faces faster than one face injects, so Np falls "
            "throughout the run with rc = 0 and no warning. This is the "
            "difference between an inflow boundary and an initial condition. "
            "Signal: the Np column decreases monotonically instead of "
            "levelling off, while Nexit stays nonzero. Either seed the domain "
            "with create_particles as well, or emit on every inflow face.",

            "[Physics] A mixture with vstream 0 still emits — the thermal "
            "flux through a face is nonzero even for a gas at rest — but the "
            "inflow rate is then set by the thermal speed alone and is several "
            "times lower than the drift flux you probably intended. The deck "
            "runs and the domain fills, only slowly. "
            "Signal: the Np column plateaus far below nrho*V/fnum for the "
            "domain; compare a run with the vstream you meant against one with "
            "vstream 0 — if they differ by a large factor, the emit mixture's "
            "vstream is what you got wrong.",

            "[Numerical] 'fix emit/face ... n <Np>' overrides the flux-derived "
            "count entirely and injects Np particles per insertion step per "
            "face regardless of density, area or velocity — the same override "
            "trap as 'create_particles n', and the steady-state particle count "
            "then scales linearly with your n. It also cannot be combined with "
            "the default per-species insertion. "
            "Signal: 'ERROR: Cannot use fix emit/face n > 0 with perspecies yes "
            "(../fix_emit_face.cpp:100)' if you just add n; once you add "
            "'perspecies no' it is silent, and the steady Np is proportional to "
            "n rather than to nrho.",

            "[Setup] 'fix emit/surf' emits from a SURFACE group, so the "
            "surfaces must already have been read and grouped; the group is "
            "checked by name when the fix is defined. Use 'normal yes' if you "
            "want the flux directed along the element normal rather than drawn "
            "from the mixture's streaming velocity. "
            "Signal: 'ERROR: Fix emit/surf group ID does not exist "
            "(../fix_emit_surf.cpp:61)'.",
        ],
    },
}

GENERATORS = {
    "particle_emission_channel_2d": _emit_channel_2d,
    "particle_emission_2d": _emit_channel_2d,
}
