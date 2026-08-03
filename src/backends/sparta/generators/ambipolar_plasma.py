"""Weakly-ionized (ambipolar) DSMC: electrons carried along with their ions."""

from ._common import output_idioms


def _ambipolar_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hypersonic ionizing air over a cylinder with the ambipolar approximation:
    electrons are attached to their parent ion rather than moved on their own
    (much shorter) timescale.
    """
    vstream = params.get("vstream", 12500.0)
    temp = params.get("temp", 217.63)
    nrho = params.get("nrho", 2.6404e20)
    fnum = params.get("fnum", 1.0e15)
    t_wall = params.get("t_wall", 615.0)
    dt = params.get("dt", 1.0e-8)
    nsteps = params.get("nsteps", 1000)
    return f"""\
# Weakly ionized air over a cylinder, ambipolar approximation - SPARTA DSMC
seed             12345
dimension        2
boundary         o ro p
global           gridcut 0.01
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1
global           fnum {fnum}
# a mixture-level nrho OVERRIDES global nrho for everything drawn from it
mixture          species nrho {nrho} vstream {vstream} 0.0 0.0 temp {temp}
# every ion AND the electron must be declared for fix ambipolar to find them
species          air.species N2 O2 N O NO N2+ O2+ N+ O+ NO+ e
# the mixture used for inflow/creation must EXCLUDE the electron: ambipolar
# electrons are created with their ion, never injected independently
mixture          species copy noelectron
mixture          noelectron delete e
mixture          noelectron N2 frac 0.8
mixture          noelectron O2 frac 0.2
# circle.surf has a UNIQUE basename in the distribution; 'data.circle' does
# NOT — eleven example directories ship different geometries under that name.
read_surf        circle.surf group 1
surf_collide     wall diffuse {t_wall} 1.0
surf_modify      1 collide wall
fix              ambi ambipolar e N+ N2+ NO+ O+ O2+
collide          vss species air.vss relax variable
collide_modify   vremax 1000 yes vibrate discrete rotate smooth
collide_modify   ambipolar yes
react            tce air.tce
create_particles noelectron n 0
fix              in emit/face noelectron xlo
compute          cnt count species
compute          tk temp
timestep         {dt}
stats            250
# ionisation is slow: expect the ion counts to stay at 0 for hundreds of steps
stats_style      step np nattempt ncoll nsreact c_tk c_cnt[6] c_cnt[8]
run              {nsteps}
"""


KNOWLEDGE = {
    "ambipolar_plasma": {
        "description": "Weakly ionized flow in the ambipolar approximation: "
                       "each electron is carried with its parent ion instead "
                       "of being moved on the electron timescale",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix ambipolar": "fix <ID> ambipolar <e-species> <ion-species...> "
                             "— every named species must already be declared",
            "collide_modify": "collide_modify ambipolar yes — required for the "
                              "collision routines to treat the attached "
                              "electrons",
            "mixture copy/delete": "mixture <new> copy <old> ; mixture <new> "
                                   "delete <species> — the standard way to "
                                   "build an electron-free inflow mixture",
            "react": "react tce <file> — the ionisation channels live in the "
                     "reaction file, so ambipolar without react produces no "
                     "ions at all",
            "compute count": "compute <ID> count species — per-species counts; "
                             "the only practical way to watch ionisation",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms(),
        "pitfalls": [
            "[Setup] Every species named on the 'fix ambipolar' line — the "
            "electron and each ion — must already be declared by a 'species' "
            "command, and the check is by name, so a typo in an ion name is "
            "reported the same way as a genuinely missing species. "
            "Signal: 'ERROR: Fix ambipolar species does not exist "
            "(../fix_ambipolar.cpp:44)'.",

            "[Setup] The mixture used for create_particles and for the inflow "
            "must NOT contain the electron species: ambipolar electrons are "
            "created together with their ion and never injected on their own. "
            "Build the inflow mixture with 'mixture <new> copy <old>' followed "
            "by 'mixture <new> delete e'. "
            "Signal: an electron count from 'compute <ID> count species' that "
            "is nonzero at step 0, before any ionising collision has occurred.",

            "[Physics] Ionisation is a high-threshold channel, so a correctly "
            "configured ambipolar deck reports zero ions and zero electrons "
            "for many hundreds of steps before the first ion appears — the "
            "ion counts creep up long before the electron count does. A zero "
            "count early in the run is not evidence of a broken setup. "
            "Signal: watch the per-species columns from 'compute count', not "
            "Nreact; declare the setup broken only if the ion counts are still "
            "identically 0 once Ncoll has accumulated over the whole run.",

            "[Setup] 'fix ambipolar' and 'collide_modify ambipolar yes' are "
            "accepted independently and NEITHER warns about the other's "
            "absence — the ambipolar collision path is only switched on by the "
            "collide_modify line, so a deck with the fix alone is syntactically "
            "perfect and physically incomplete. "
            "Signal: there is no error and no warning at all. Verify by "
            "watching 'compute <ID> count species': once ions appear, the "
            "electron count must track the total ion count. If ions accumulate "
            "and the electron column stays at 0, the collide_modify line is "
            "missing.",

            "[Setup] 'surf_react <ID> prob <file>' validates every probability "
            "in the file against the species list, so a surface-reaction file "
            "written for a different species set fails at read time, not at "
            "run time. "
            "Signal: 'ERROR: Surface reaction probability for a species > 1.0 "
            "(../surf_react_prob.cpp:287)' — usually it means the file's "
            "column layout does not line up with your declared species.",
        ],
    },
}

GENERATORS = {
    "ambipolar_plasma_circle_2d": _ambipolar_circle_2d,
    "ambipolar_plasma_2d": _ambipolar_circle_2d,
}
