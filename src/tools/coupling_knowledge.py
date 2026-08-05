"""Coupling knowledge served to agents: the `couple` contract + one complete,
runnable participant script per backend.

Why this module exists
----------------------
A usability probe drove the live tools and measured the coupling corpus at
~15 kB for all nine backends together, none of it varying by backend, while a
single-code payload for one physics is ~36 kB. Two backends were named; seven
were not. Worse, it documented the DEPRECATED `coupled_solve` enum and told the
agent to write "a new subdomain script generator" — a private function no agent
can call — while the contract of the general `couple` tool existed only inside
that tool's own docstring.

The rules this module follows, because the consumer is a small model that will
not infer and will not hunt for a second payload:

  * load-bearing first — the contract before the theory;
  * COMPLETE runnable scripts, never fragments to assemble;
  * required vs optional stated in words, not implied;
  * every per-backend payload repeats the contract essentials, so landing on it
    directly is still enough to work from;
  * no absolute host paths: the interpreter/binary for a backend is whatever
    `discover(query='list')` reports for it on this install;
  * nothing here is an answer to a problem — no exact solutions, no measured
    error or order tables. Behaviour of the code, yes; results, no.
"""
from __future__ import annotations

from pathlib import Path

_PARTICIPANT_DIR = Path(__file__).resolve().parents[2] / "data" / "coupling_participants"


def _script(name: str) -> str:
    """The participant script SHIPPED WITH OASiS, served verbatim.

    The script is a file rather than a string literal on purpose: the file is
    the artefact that gets executed in the test suite, so the text an agent is
    served and the text that was proven to run cannot drift apart.
    """
    p = _PARTICIPANT_DIR / f"participant_{name}.py"
    if not p.is_file():
        return (f"[OASiS] participant script for '{name}' is missing from the "
                f"install (expected data/coupling_participants/{p.name}).")
    return p.read_text()

# ══════════════════════════════════════════════════════════════════════════
# CORE — served by knowledge(topic='coupling') with no solver
# ══════════════════════════════════════════════════════════════════════════

_CONTRACT = '''\
## 1. THE PARTICIPANT CONTRACT — this is the whole interface

`couple` runs a partitioned fixed-point iteration over N>=2 participants. A
participant is ANY runnable command. Every iteration, for every participant,
the driver does exactly three things:

  1. writes `<work_dir>/imports.json`
  2. runs your `command` with `cwd = <work_dir>`
  3. reads `<work_dir>/exports.json`

Nothing else is passed. No arguments, no environment, no stdin. Your script
gets its problem definition from constants written into the script itself.

REQUIRED of every participant script:
  * read `imports.json`. It is written EVERY iteration and is exactly `{}` on
    iteration 1, so you MUST have a fallback initial value for whatever you
    import. It is never deleted, so a stale one from a previous attempt is
    still there when you run the script by hand — delete it first;
  * write `exports.json` before exiting;
  * export the SAME number of points, in the SAME order, on EVERY iteration,
    and keep `normal_fluxes` consistently present or consistently absent. The
    driver relaxes export vectors element by element; a changed length is
    caught and reported, but it ends the run;
  * write `exports.json` LAST, only after the solve succeeded. The driver
    checks that the file EXISTS; it does NOT check your exit code. A complete
    file written before a later crash is accepted as a result (a TRUNCATED one
    is caught, as bad JSON).

NOT done for you (these are the four things agents get wrong):
  * the driver does NOT copy your script into `work_dir` — write the script
    file into `work_dir` yourself, then name it bare in `command`;
  * the driver does NOT interpolate between the two meshes — each participant
    maps the partner's samples onto its own interface points;
  * the driver does NOT convert units, sign conventions or field names;
  * `work_dir` must be an ABSOLUTE path, and a relative one is REJECTED — it
    would resolve against the server's own directory, which you cannot see.

## 2. imports.json / exports.json — the exact shapes

`imports.json` is a dict keyed by PARTNER NAME:

    {"<partner name>": {<InterfaceData>}, ...}

only containing the partners you listed in `imports_from`. `exports.json` is
ONE InterfaceData object, not a dict of them:

    {"field_name":    "temperature",          # REQUIRED key, free-form value
     "coordinates":   [[x0,y0], [x1,y1], ...],# REQUIRED — YOUR interface points
     "values":        [v0, v1, ...],          # REQUIRED — one per point
     "normal_fluxes": [q0, q1, ...],          # optional, one per point
     "n_points":      21}                     # optional label, never read

THREE KEYS ARE REQUIRED: `field_name`, `coordinates`, `values`. `field_name`
is a free-form label whose VALUE is never interpreted, but leaving the KEY out
is a hard error reported as `bad exports.json`. `n_points` is ignored entirely.

`normal_fluxes` is optional — but if EITHER side omits it, the interface
conservation check cannot run at all, and the result says so. Supply it
whenever a flux exists; it is the only guard against a coupling that converges
to a non-conservative answer.

Nothing validates the SHAPES. `values` of a different length than
`coordinates`, or a flat `coordinates` list, is accepted and gives a wrong
answer quietly. Check them in your own script.

BOTH `values` AND `normal_fluxes` are relaxed and BOTH enter the convergence
residual. Exporting a large-magnitude flux next to a small-magnitude value
makes the residual mostly about the flux; that is usually what you want for a
Dirichlet-Neumann coupling, but know that it happens.

## 3. How you call it

    couple(participants='[
      {"name": "left",  "command": ["<interpreter>", "participant_left.py"],
       "work_dir": "/abs/path/run/left",  "imports_from": ["right"],
       "timeout": 900},
      {"name": "right", "command": ["<interpreter>", "participant_right.py"],
       "work_dir": "/abs/path/run/right", "imports_from": ["left"],
       "timeout": 900}]',
      max_iter=60, tol=1e-8, accelerator="constant", theta=0.5,
      critic_approved=True)

  * `name` — how the partner finds this participant's data inside imports.json.
  * `command` — argv list. Get the interpreter or binary for a backend from
    `discover(query='list')`, which prints it per backend on THIS install.
    Never hard-code an interpreter path from an example.
  * `work_dir` — absolute; created if missing; your script must already be in it.
  * `imports_from` — names of the partners whose exports this one consumes.
    Omit a name and that participant simply never sees that partner's data.
  * `timeout` — seconds per participant call (default 3600). A compiled code
    that hangs will otherwise stall the whole coupling.
  * `data_files` is NOT supported by this tool — an extra key in the spec is
    silently ignored. Copy every mesh/species/config file your solver opens
    into `work_dir` yourself before calling `couple`.
  * `theta` must be in (0, 1]; `accelerator` must be exactly "constant" or
    "aitken". Both are rejected with an error message if not.

Returns JSON with `converged`, `iterations`, `residual`, `history`, per-
participant `exports`, a `validation` block, and OASiS's `verification` /
`trustworthy_result` verdict. Two things about it:
  * the `exports` returned are the RELAXED blend the driver holds, not the last
    raw output of your solver. On a converged run the difference is below the
    tolerance; on a failed run they are a mixture of two iterations, which is
    one more reason not to report a non-converged run as a result;
  * `trustworthy_result` is false until an independent critic review for THIS
    exact set of arguments is on record. `critic_approved=True` on its own does
    nothing — OASiS looks the review up rather than believing the flag. Call
    `submit_critic_review(solver="couple", coupling_args=<the same arguments as
    a JSON object>, findings=<what the critic concluded>)` first.

A run that did not converge is reported as FAILURE — never report its numbers
as a result.
'''

_DRIVER_BEHAVIOUR = '''\
## 4. How the iteration actually behaves — read before choosing theta

THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL. Inside one iteration, every participant
reads the PREVIOUS iteration's exports. Participant B does not see the export
participant A produced moments earlier in the same iteration. Ordering the
participants differently changes nothing.

THE DRIVER RELAXES EVERY PARTICIPANT. Each participant's own export vector is
blended with its own previous export:

    export_relaxed = (1 - theta) * export_previous + theta * export_new

so a two-participant loop applies relaxation TWICE per cycle, once on each
side. This is why a linear Dirichlet-Neumann coupling that textbooks solve in
one Gauss-Seidel step converges only geometrically here.

RESIDUAL — AND WHY IT HAS A FLOOR. The reported residual compares each
participant's RAW new export against its PREVIOUS RELAXED one, normalised by
the raw magnitude, stacked over all participants. It is not a physical error.
Iteration 1 has nothing to compare against and is recorded as NaN.

The consequence matters: even a participant whose raw output has ALREADY
stopped changing still shows a residual falling like (1-theta)^k, purely
because the relaxed value is still catching up to the raw one. So with a
CONSTANT theta the residual's decay rate is bounded by (1-theta) per iteration
once the raw output has settled, and the iteration count you need is roughly

    log(tol / d0) / log(1 - theta)

where d0 is the INITIAL mismatch measured the way the residual is — relative to
the field magnitude. Take d0 = 1 (a 100%-wrong start) and tol=1e-8 gives about
27 iterations at theta=0.5, 52 at theta=0.3, 83 at theta=0.2; at tol=1e-6,
about 20 / 39 / 62. Those are a SIZING GUIDE, not a lower bound: a field with a
large offset — temperatures around 300 K whose interface value is wrong by a few
K — starts at d0 of a few percent and reaches tol in measurably fewer iterations
than the d0=1 figure. Evaluate the expression for your own theta and tol instead
of reusing a number, then give max_iter generous headroom, because
under-budgeting looks exactly like a physics failure: a run that stops at
max_iter=20 with theta=0.3 never had a chance. (With accelerator="aitken" theta
moves, so the rate moves with it, but the same mechanism is there.)

`accelerator`: **the default, "aitken", is also the safer one — reach for
"constant" to DIAGNOSE, not as your first choice.**
  * "aitken" — theta adapts per participant, starting from the theta you pass,
    clamped into [0.05, 1.0]. THIS IS THE DEFAULT AND YOU SHOULD NORMALLY KEEP
    IT. Measured across conductance ratios rho from 1/4 to 9 and theta from 0.1
    to 1.0 on this driver, Aitken matched or beat a constant theta almost
    everywhere, and in a quarter of those settings it converged to the right
    interface value where the SAME constant theta diverged by tens of orders of
    magnitude. It is the main thing protecting you from a theta chosen too
    large. It is not magic: at a strongly unbalanced ratio no accelerator
    rescues a bad split — see the role-swapping advice below.
  * "constant" — theta fixed at exactly what you passed. Predictable and
    reproducible, which makes it the right tool for working out what the
    iteration is doing, and unforgiving: above the stability limit for your rho
    it diverges instead of adapting. One case was found where a constant theta
    converged and Aitken did not: a very unbalanced split (rho = 9) at the one
    theta that works there, where Aitken reached the correct interface value but
    was still marginally above tol at the iteration budget. That is the
    exception, not the rule; if "aitken" stalls, raise max_iter first, then try
    the same theta constant, and only then touch the physics.

### Choosing theta — this maps to the real `theta` parameter

For a two-participant Dirichlet-Neumann split the iteration is linear in the
interface unknowns, and the driver's Jacobi+relaxation loop has amplification
factor  sqrt((1-theta)^2 + rho*theta^2), where

    rho = (interface conductance of the DIRICHLET-side subdomain)
          / (interface conductance of the NEUMANN-side subdomain)

and "interface conductance" is material coefficient / distance from the
interface to that subdomain's own outer boundary (k/d for conduction, EA/L for
a bar, and so on — it is the subdomain's stiffness as seen from the interface).
That gives three facts you can act on:

  * the best theta is  **theta ~ 1 / (1 + rho)** — 0.5 when the two sides are
    balanced, smaller when the Dirichlet side is the stiffer one, larger when it
    is the softer one. This is the one number in this section worth computing
    before you run anything: swept over rho from 1/4 to 9, the fastest constant
    theta was 1/(1+rho) at EVERY ratio, and at the unbalanced ones it was the
    ONLY constant theta that converged at all rather than diverging;
  * theta = 1.0 NEVER converges at rho = 1 and DIVERGES for rho > 1. It is not
    a "no relaxation, exact for linear problems" setting on this driver;
  * convergence is fastest when the DIRICHLET side is the SOFTER / LESS
    CONDUCTIVE subdomain. If a coupling converges too slowly to be practical,
    SWAP WHICH SIDE IS DIRICHLET before doing anything else — that replaces rho
    by 1/rho and is usually a bigger win than any theta.

Observed on this driver, running real two-code couplings:
  * at rho = 1, theta = 0.5 converges and theta = 1.0 oscillates forever
    WITHOUT blowing up — the interface value simply never settles;
  * at rho = 4, theta = 0.5 with a CONSTANT accelerator DIVERGES — the interface
    values run away by many orders of magnitude and the conservation check fires
    — while theta = 0.2 converges. Nothing warns you in advance: a diverging
    coupling looks like a converging one for the first few iterations. Note the
    default "aitken" survives this particular case; do not read the constant-
    theta stability limit as a property of the tool's default;
  * for one asymmetric split, the SAME problem with the SAME tolerance failed
    to converge inside the iteration budget with the stiff subdomain on the
    Dirichlet side, and converged comfortably inside it once the two roles were
    swapped and theta set to 1/(1+rho) for the new rho. Choosing the side is
    the cheapest tuning knob you have, and it is free — it costs one edit to
    each script's `SIDE` and `T_OUTER`.

  | Symptom                                       | Do this                    |
  | first try, know nothing                       | estimate rho, set theta=1/(1+rho), keep accelerator="aitken" |
  | first try, cannot estimate rho at all         | theta=0.5, keep accelerator="aitken" |
  | residual falls steadily but slowly            | keep theta, raise max_iter; then swap which side is Dirichlet |
  | residual flat or oscillating in sign          | halve theta                |
  | residual GROWING, values exploding            | halve theta, and check the flux sign convention (section 5) |
  | want to see what the iteration is doing       | same theta, accelerator="constant" — reproducible, no adaptation |
  | converged, but you want it faster             | put Dirichlet on the softer subdomain, theta = 1/(1+rho) |

There is no theta that makes this driver converge in one step for a two-code
Dirichlet-Neumann split. Budget tens to hundreds of iterations and set max_iter
and the per-participant `timeout` accordingly.

If you need genuine Gauss-Seidel sub-iteration inside a time window, that is
what `couple_precice` with a `serial-implicit` scheme provides — see
`knowledge(topic='precice')`.
'''

_SIGNS = '''\
## 5. Interface flux: TWO different quantities, two different signs

Confusing these is the mistake that produces a converged coupling which OASiS
then stamps NOT VERIFIED, with nothing in the output explaining why.

**(1) The BC VALUE you APPLY in the receiving code — the SAME number.**
Let subdomain A have outward normal n_A at the interface. The flux density A
loses through the interface is

    q_out = -k * dT/dn_A                            [W/m^2 in 2D]

B's outward normal at the same interface points the other way, and B adds the
Neumann datum to its weak form as `+ integral(g * v) ds_interface`. The two
sign flips cancel:

    g_B = q_out,A            -- hand B exactly the number A computed

In 4C, `DESIGN LINE NEUMANN` `VAL` is that same quantity and takes it directly.

**(2) The `normal_fluxes` array you EXPORT — OPPOSITE numbers.**
Each participant exports the flux through the interface with respect to ITS OWN
outward normal. Those normals are anti-parallel, so on a conservative interface

    integral(normal_fluxes_A) + integral(normal_fluxes_B)  ~  0

That is what OASiS's conservation check tests. Export both sides with the same
sign and a CORRECT coupling fails it: you get `Interface flux NOT balanced` and
a NOT VERIFIED verdict on a coupling that converged perfectly well.

In one line: **apply the same number, export opposite numbers.**

EXPORT A FLUX DENSITY, AND ALWAYS SHIP `coordinates` WITH IT. The two sides of
a partitioned coupling normally sample the interface differently — that is the
point of partitioning. With `coordinates` present the check integrates the
density along the interface (over arclength for a curve, as a mean for a
surface, component-wise for a vector traction), which is independent of how
many points each side used. Without them it can only sum the raw arrays, and
that only agrees when both sides happen to use the same number of points.

Two more things about the check, so its verdicts are readable:
  * if EITHER side omits `normal_fluxes` it reports `Interface conservation was
    NOT CHECKED` rather than passing silently. A run with no conservation
    evidence is not the same as one that passed;
  * a genuinely near-zero net flux (a symmetric profile) is NOT reported as
    unbalanced — the comparison is floored by the flux magnitudes, so float
    noise on two cancelling integrals cannot manufacture a failure.
'''

_SIDES_TABLE = '''\
## WHICH SIDE EACH BACKEND CAN TAKE

"Dirichlet side" = imports a field VALUE, applies it as an essential BC on the
interface, exports the resulting interface FLUX.
"Neumann side" = imports a FLUX, adds it to the weak-form RHS on the interface,
exports the resulting interface VALUE.

Established by running a real two-code coupling in each role on this install,
not copied from a docstring:

| Backend    | Dirichlet | Neumann | Participant is           | Proven how |
|------------|-----------|---------|--------------------------|------------|
| FEniCSx    | yes       | yes     | a Python script          | coupled to 4C, NGSolve, scikit-fem, both roles |
| 4C         | yes       | yes     | Python wrapper + YAML    | coupled to FEniCSx, both roles |
| NGSolve    | yes       | yes     | a Python script          | coupled to FEniCSx and scikit-fem, all four role/position combinations |
| scikit-fem | yes       | yes     | a Python script          | coupled to FEniCSx and NGSolve, all four role/position combinations |
| DUNE-fem   | yes       | yes     | a Python script          | coupled to FEniCSx and deal.II, both roles |
| deal.II    | yes       | yes     | Python wrapper + C++ exe | coupled to FEniCSx and DUNE-fem, both roles |
| FEBio      | yes       | yes     | Python wrapper + XML     | FEBio-to-FEBio, both roles — ELASTICITY, not heat: FEBio 4 has no heat module |
| Kratos     | yes*      | yes*    | a Python script          | coupled to FEniCSx, both roles — NOT reproducible here: see the asterisk |
| SPARTA     | yes*      | NO      | Python wrapper + deck    | coupled to a thermal shell; the Neumann role is impossible, and the residual cannot beat the Monte-Carlo noise |

Every UNSTARRED "yes" means a real two-code coupling was run in that role on
THIS install and CONVERGED; it is not copied from a tool docstring. The two
starred rows are weaker, and the difference matters before you plan around them:

  * KRATOS was proven in a SEPARATE Kratos install, not in OASiS's interpreter
    here. A core-only Kratos has no thermal element, so
    `import KratosMultiphysics.ConvectionDiffusionApplication` is the thing to
    test first — if it fails, the conduction participant cannot run at all on
    this machine, whatever the table says.
  * SPARTA's Dirichlet role ran end to end and the physics agreed, but `couple`
    reported FAILURE: a Monte-Carlo residual has a noise floor. "yes" here means
    the ROLE is possible, NOT that you will get a converged run out of it.

Note in particular that the DEPRECATED `coupled_solve` docstring lists 4C on the
Neumann side only — that limitation belongs to its own fixed generators, not
to 4C.

Nothing in the driver is specific to heat conduction. A backend that can (a)
impose a prescribed field on a boundary, (b) impose a flux/traction on a
boundary and (c) report both at interface points, can take either side for
whatever physics it solves. Call `knowledge(topic='coupling', solver='<name>')` for
any of them: it returns a complete runnable participant script for that backend
and says plainly what was proven and what was not.

BOTH SIDES MUST SOLVE THE SAME PHYSICS. The driver moves numbers; it does not
translate a temperature into a displacement. Coupling a heat code to a
structural code is a THERMO-STRUCTURAL problem and needs a real transfer
relation in the participant scripts (see `knowledge(topic='tsi')`).
'''

_SIDES = _SIDES_TABLE.replace("## WHICH SIDE", "## 6. WHICH SIDE", 1)


def coupling_sides_table() -> str:
    """The side table on its own, for discover(query='coupling')."""
    return _SIDES_TABLE

_FAILURES = '''\
## 7. FAILURE MODES, and what each one actually means

| What you see                                        | Cause                        |
|-----------------------------------------------------|------------------------------|
| `participant X wrote no exports.json (rc=...)`      | your script died. The stderr tail is in the message — read it. Run the script standalone in its work_dir first. |
| `participant X bad exports.json`                    | malformed JSON, a TRUNCATED file, or a missing required key. All three of `field_name`, `coordinates`, `values` must be present. |
| `participant X changed its export size from N to M` | that participant exported a different number of points (or dropped `normal_fluxes`) between iterations. Usually a mesh or interface-detection step that depends on the imported data. Fix the participant; the driver cannot relax a changing vector. |
| `participant X timed out`                           | raise `timeout` in the participant spec, or coarsen that side's mesh. |
| `did not converge to tol=... in N iters`            | in order: is max_iter above the (1-theta) floor in section 4; is theta right for this conductance ratio; would swapping which side is Dirichlet help; do the two decks actually agree on units and material. NOT a result. |
| residual stuck at O(1), oscillating                 | theta too large, or a SIGN error — the Neumann side is pushing the flux the wrong way. |
| residual GROWING, values exploding                  | theta above the stability limit for this conductance ratio. Halve it. See section 4. |
| converged, but `Interface flux NOT balanced`        | most often both sides exported `normal_fluxes` with the same sign (section 5). Otherwise: different units on the two sides, or one side exporting an INTEGRATED flux where the other exports a DENSITY. |
| `Interface conservation was NOT CHECKED`            | one side exported no `normal_fluxes`, so nothing verifies conservation. The run is not wrong, it is unguarded. Export the flux on both sides. |
| `NOT COUPLED: no participant lists imports_from`     | nobody was given a partner's name, so nothing was exchanged. The run is not a coupling. |
| `ONE-WAY: participant(s) X list no imports_from`    | X never sees its partners. A note, not a failure — a master/slave coupling really does look like this. If you meant a two-way coupling, that is the bug. |
| `NOT COUPLED: no participant's export changed`      | converged at iteration 2 with an exactly zero residual. Both participants returned their initial guess: they are not reading `imports.json`, or they are reading it under the wrong partner name. THIS IS THE MOST CONVINCING WRONG RESULT THE TOOL CAN PRODUCE — it looks like an instant, perfect convergence. |
| `non-finite export values at iter N` (a warning)    | your solve diverged; the run CONTINUES to max_iter, so look for this in `validation` rather than expecting it to stop. Usually a subdomain with no essential BC anywhere, which is singular. |
| converged to a plausible but wrong answer           | check `n_points` from each side on iteration 1: an empty or wrongly located interface set gives a well-behaved solution of the wrong problem. Then check that both sides use the same units and the same interface coordinate. |

FIRST THING TO DO WHEN A COUPLING FAILS: delete `imports.json` from each
work_dir and run each participant script by hand there. That exercises the
iteration-1 fallback path and tells you whether the failure is in the physics
or in the handshake. Nearly every coupling failure is visible there.

SECOND THING: check that the interface really is shared. Print each side's
`n_points` and the first and last coordinate. Two subdomains that do not touch,
or touch at a coordinate one of them rounds differently, produce exactly the
"converged to the wrong answer" symptom.
'''


def _index(names: list[str]) -> str:
    rows = "\n".join(f"  knowledge(topic='coupling', solver='{n}')" for n in names)
    return f'''\
## 8. PER-BACKEND PARTICIPANT SCRIPTS — one call each, complete and runnable

Each of these returns a COMPLETE participant script for that backend plus the
traps specific to it. Copy it into the participant's `work_dir`, edit the
marked block, run it once by hand, then call `couple`.

{rows}

preCICE instead of this driver:  knowledge(topic='precice', solver='<name>')
4C-native thermo-structural:     knowledge(topic='tsi')
'''


_BACKEND_ORDER = ["fenics", "fourc", "ngsolve", "skfem", "dune", "dealii",
                  "febio", "kratos", "sparta"]


_RECAP = '''\
## The contract, in case you landed here first

Your script runs in `work_dir` with no arguments. It reads `imports.json`
(`{partner_name: InterfaceData}` — ABSENT or empty on iteration 1, so it must
have a fallback) and writes `exports.json` (ONE InterfaceData:
`{"field_name","n_points","coordinates","values","normal_fluxes"}`). Export the
same number of points in the same order every iteration, and write
`exports.json` last — the driver takes its existence as proof of success and
never looks at your exit code. The driver does NOT copy your script into
`work_dir` and does NOT interpolate between the two meshes.

Dirichlet side = imports a VALUE, exports the resulting FLUX.
Neumann side  = imports a FLUX, exports the resulting VALUE.
Apply the partner's flux number UNCHANGED; export your own flux with respect to
YOUR outward normal, so the two sides' fluxes carry opposite signs.

Full contract, relaxation guidance and failure modes: `knowledge(topic='coupling')`.
'''


def _payload(title: str, sides: str, script_name: str, launch: str,
             traps: str, extra: str = "") -> str:
    return (f"# Coupling participant: {title}\n\n"
            f"## Sides this backend can take\n\n{sides}\n\n"
            f"{_RECAP}\n"
            f"## COMPLETE PARTICIPANT SCRIPT — copy verbatim, edit the marked "
            f"block only\n\n```python\n{_script(script_name)}```\n\n"
            f"## Launching it\n\n{launch}\n"
            f"## {title}-specific traps\n\n{traps}\n{extra}")


_RIGHT_BLOCK = """\
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.6, 1.1      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 0.4      # same y-extent as the partner
IFACE_X   = 0.6           # SAME interface coordinate as the partner
K         = 1.5           # this subdomain's own material
F_SRC     = 0.0
T_OUTER   = 300.0         # Dirichlet on ITS outer boundary (here x = 1.1)
NX, NY    = 18, 12        # its own mesh — deliberately NOT the partner's
T_INIT    = 310.0
Q_INIT    = 0.0
"""

_LAUNCH_PY = '''\
1. Make TWO copies of the script, one per subdomain, and write each into that
   participant's own `work_dir` (an absolute path). Name them whatever you
   reference in `command` — `participant_left.py` and `participant_right.py`
   below. The driver does not copy anything for you.
2. {STEP2}
3. Run each copy BY HAND in its own directory first, with no `imports.json`
   present (delete a leftover one from an earlier attempt — the driver never
   removes it). Each must print its interface line and write `exports.json`.
   A coupling cannot repair a participant that never ran.
4. {INTERP}
5. Have a critic review the setup, then put the review on record — the flag
   alone is NOT enough, OASiS looks the review up rather than believing you:

```
submit_critic_review(solver="couple",
                     coupling_args='{{"participants": "<the same JSON string>",
                                     "max_iter": 60, "tol": 1e-8,
                                     "accelerator": "constant", "theta": 0.5}}',
                     findings="<what the critic checked and concluded>")
```

6. Then couple. The `coupling_args` above must match these arguments exactly,
   or the review does not bind and the result comes back NOT VERIFIED:

```
couple(participants='[
  {{"name":"left","command":["<interpreter>","participant_left.py"],
   "work_dir":"/abs/run/left","imports_from":["right"],"timeout":900}},
  {{"name":"right","command":["<interpreter>","participant_right.py"],
   "work_dir":"/abs/run/right","imports_from":["left"],"timeout":900}}]',
  max_iter=60, tol=1e-8, accelerator="constant", theta=0.5, critic_approved=True)
```
'''.replace("{RIGHT}", _RIGHT_BLOCK)

# Step 4 differs for the four backends whose participant is a Python WRAPPER
# around a separate binary (4C, FEBio, SPARTA, deal.II). For those, `command`
# must run PYTHON and the binary path goes into a CONSTANT INSIDE the script.
# "Get the interpreter or binary from discover(query='list')" is the wrong
# instruction there: what discover prints for those backends IS the binary, and
# a model that follows step 4 verbatim puts the solver binary in `command`,
# where it cannot run a Python wrapper.
#
# This was previously done with `_LAUNCH_PY.replace(<literal>, ...)` at four
# call sites. ALL FOUR WERE DEAD: the literals wrapped their lines at a
# different point than the text (and 4C's was left from an older wording
# entirely), so every wrapper backend served the generic step 4 and not one of
# them said `command` runs the wrapper. `str.replace` cannot fail, so nothing
# reported it. Step 4 is a NAMED FIELD now, so a stale note raises at import
# instead of going quiet, and `test_wrapper_backends_say_the_command_runs_the
# _wrapper` asserts on the SERVED text rather than on the call.
_INTERP_GENERIC = (
    "Get the interpreter from `discover(query='list')`: it prints one\n"
    "   line per backend with the exact interpreter path on THIS install.\n"
    "   Never copy an interpreter path out of an example.")

# The RULES step 2 states are the same whatever the physics; only the concrete
# block changes. Kept separate so a bespoke step 2 can reuse them.
_TWO_BLOCK_RULES = '''\

   The rules the two blocks must satisfy, whatever your real numbers are:
   one copy has `SIDE="dirichlet"` and the other `SIDE="neumann"`; each
   `PARTNER` is the other's `name`; both have the SAME `IFACE_X`; the two
   x-extents meet at `IFACE_X` and do not overlap; each subdomain keeps at
   least one Dirichlet boundary of its own, or its problem is singular and the
   solve blows up. The two meshes need NOT match.'''


def _step2_block(block: str, what: str = "the placeholder problem") -> str:
    return ("The script AS SHIPPED is the LEFT / Dirichlet side. In the second "
            f"copy,\n   replace the edit block with the complementary side. For "
            f"{what} that is\n   exactly:\n\n```python\n{block}```\n"
            + _TWO_BLOCK_RULES)


# Step 2 embedded the CONDUCTION right-side block into every backend's payload.
# For the three backends whose shipped script does not solve that problem it
# named constants that do not exist in the script the agent had just been given
# (FEBio has no K/T_OUTER/T_INIT/F_SRC — it is elasticity; Kratos has no
# SIDE/IFACE_X/Y0,Y1/NX,NY/F_SRC/Q_INIT; SPARTA has none of nine of them). For
# SPARTA it also flatly contradicted the same payload's own headline, which says
# the Neumann role is IMPOSSIBLE, by handing over a `SIDE="neumann"` block.
# Verified by execution: applying the served block to the served FEBio script
# fails on the first key. So step 2 is per-backend now.
_STEP2_DEFAULT = _step2_block(_RIGHT_BLOCK)

# FEBio's script is the ELASTIC analogue, so its complementary block is in
# displacement/modulus, not temperature/conductivity. These constant names are
# the ones the shipped FEBio script actually defines; the pairing was run as a
# real FEBio-to-FEBio coupling and converged with non-matching meshes.
_RIGHT_BLOCK_FEBIO = """\
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.5, 1.0      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 1.0      # same cross-section as the partner
Z0, Z1    = 0.0, 0.1      # same cross-section as the partner
IFACE_X   = 0.5           # SAME interface coordinate as the partner
E_MOD     = 2250.0        # this subdomain's own material
NU        = 0.3
U_OUTER   = 1.0e-4        # prescribed u_x on ITS outer face (here x = 1.0)
NX, NY    = 12, 6         # its own mesh — deliberately NOT the partner's
U_INIT    = 5.0e-5
Q_INIT    = 0.0
"""

# Kratos ships a DIRICHLET-side script with no `SIDE` switch, so there is no
# "flip SIDE" edit to make: the Neumann side is a code change, described under
# the traps. SPARTA cannot take the Neumann side at all.
_STEP2_KRATOS = ('''\
The shipped script is the DIRICHLET side and has NO `SIDE` switch, so
   the second participant is NOT this script with one constant flipped. Two
   ways to build the pair, and the first is what was actually run:

   (a) PAIR IT WITH ANOTHER BACKEND. Take the Neumann side from a backend whose
       script has a `SIDE` switch — `knowledge(topic="coupling",
       solver="fenics")` is the lightest — and edit only its block. Kratos was
       proven against FEniCSx this way, in both roles.
   (b) MAKE A KRATOS NEUMANN SIDE. Copy the script and change the interface
       condition as described under the traps below: do NOT `Fix(TEMPERATURE)`
       on the interface nodes; set `FACE_HEAT_FLUX` from the partner's
       `normal_fluxes` and create the interface `ThermalFace` conditions.
       Nothing else — mesh, material, export — changes.

   For its own edit block, the second participant's subdomain must start where
   the first ends (`X0` = the first copy's `X1`), keep the same `H`, carry its
   own `K` and its own outer `T_OUTER`, and name the partner in `PARTNER`. Each
   subdomain must keep one Dirichlet boundary of its own or it is singular.''')

_STEP2_SPARTA = ('''\
THERE IS NO SECOND COPY OF THIS SCRIPT. SPARTA is a DIRICHLET-side
   participant only — no surface-collision model accepts a prescribed heat
   flux — so it cannot take the complementary role, and a `SIDE="neumann"`
   edit of this script does not exist.

   The partner is a NEUMANN-side participant in another backend: it imports
   SPARTA's exported `normal_fluxes` as its interface flux and exports the wall
   temperature SPARTA imports. `knowledge(topic="coupling", solver="fenics")`
   gives such a script; set its `SIDE="neumann"`, put its interface at the wall
   SPARTA's surface file describes, and make sure both sides agree on units —
   SPARTA works in SI with energy flux per unit area.

   Read the stochasticity note below BEFORE you size `NRUN`/`NAVE`: this
   coupling was run end to end here and `couple` reported FAILURE on the
   residual even though the physics agreed, because a Monte-Carlo estimate has
   a noise floor the residual cannot fall below.''')


def _launch_py(interp: str = _INTERP_GENERIC, step2: str = "") -> str:
    """The launch section, with steps 2 and 4 written for this backend."""
    for field in ("{INTERP}", "{STEP2}"):             # guard the guard
        if field not in _LAUNCH_PY:
            raise AssertionError(f"_LAUNCH_PY lost its {field} field")
    return (_LAUNCH_PY.replace("{INTERP}", interp)
                      .replace("{STEP2}", step2 or _STEP2_DEFAULT))


def _interp_wrapper(binary: str, const: str, extra: str = "") -> str:
    """Step 4 for a wrapper participant: python in `command`, binary in a const."""
    return (f"The `command` runs the WRAPPER, so the interpreter is a plain\n"
            f"   Python with numpy — NOT the {binary} binary. The {binary} BINARY\n"
            f"   path goes into `{const}` INSIDE the script; that is what\n"
            f"   `discover(query='list')` prints for this backend.{extra}")


def coupling_core() -> str:
    return (
        "# Cross-code coupling with OASiS — `couple`\n\n"
        "## 0. WHICH TOOL, IN ONE PARAGRAPH\n\n"
        "`couple(participants, ...)` is THE tool. It is physics-agnostic: you write "
        "one self-contained solver script per subdomain, and OASiS runs the "
        "fixed-point iteration, the relaxation, the convergence-or-fail and the "
        "conservation check. `couple_precice(...)` is the alternative when each side "
        "is a real preCICE participant and you want preCICE's mapping and implicit "
        "schemes. `coupled_solve(...)` is DEPRECATED — it only reproduces a fixed enum "
        "of benchmark problems on a hard-coded unit square and cannot express your "
        "problem; do not start there.\n\n"
        + _CONTRACT + "\n" + _DRIVER_BEHAVIOUR + "\n" + _SIGNS + "\n"
        + _SIDES + "\n" + _FAILURES + "\n" + _index(_BACKEND_ORDER)
    )


# ══════════════════════════════════════════════════════════════════════════
# PER-BACKEND — served by knowledge(topic='coupling', solver=...)
# ══════════════════════════════════════════════════════════════════════════

def _fenics() -> str:
    return _payload(
        "FEniCSx (dolfinx)",
        "**Either side.** Both were run as real cross-code couplings against a "
        "second code on this install: FEniCSx as the Dirichlet side against 4C "
        "on Neumann, and FEniCSx as the Neumann side against 4C on Dirichlet. "
        "Both converged with non-matching interface meshes.",
        "fenics", _launch_py(),
        '''\
* `LinearProblem` in dolfinx 0.9+ REQUIRES `petsc_options_prefix`. Omit it and
  the constructor raises before anything is solved.
* An interface Dirichlet BC from partner data is a `fem.Function` whose array
  you fill at the interface DOFs, then `fem.dirichletbc(function, dofs)` — the
  two-argument form. The scalar form `fem.dirichletbc(value, dofs, V)` cannot
  carry a per-node profile.
* For the Neumann side you need `meshtags` + a subdomain `ds` measure. A bare
  `ufl.ds` applies the flux to the WHOLE boundary, including the outer
  Dirichlet edge, which is silently wrong rather than an error.
* `dmesh.meshtags` wants the facet indices SORTED. Unsorted indices are
  accepted and then tag the wrong facets.
* `V.tabulate_dof_coordinates()` gives the DOF coordinates that
  `x.array` is indexed by — use those, not the mesh geometry nodes, or the
  interface values land on the wrong entries for anything above P1.
* The exported flux is an L2 projection of `-K*S*grad(T)[0]` onto the same CG1
  space, so it is defined at exactly the interface DOFs. Do not finite-difference
  towards an "adjacent" node: on an unstructured triangle mesh the nearest
  interior node is not normal to the interface.
* Run FEniCSx participants serially (one MPI rank). Under `mpirun` each rank
  would write its own `exports.json` over the others.''')


def _fourc() -> str:
    return _payload(
        "4C Multiphysics",
        "**Either side.** Both were run as real cross-code couplings against "
        "FEniCSx on this install and both converged with non-matching interface "
        "meshes. Note this contradicts the deprecated `coupled_solve` docstring, "
        "which lists 4C on the Neumann side only — that limitation belongs to "
        "the legacy tool's own generators, not to 4C.",
        "fourc",
        _launch_py(_interp_wrapper(
            "4C", "FOURC_BIN",
            extra="\n   That Python needs numpy + meshio (OASiS's own has both). Put\n"
                  "   4C's dependency lib directory in `FOURC_LD` if the binary does\n"
                  "   not find its libraries by itself.")),
        '''\
* `PROBLEMTYPE: "Scalar_Transport"` with `TIMEINTEGR: "Stationary"` is the
  conduction problem. Element line is `TRANSP QUAD4 ... MAT 1 TYPE Std` in a
  `TRANSPORT ELEMENTS` section — `SOLID QUAD4` is a structural element and will
  be rejected against `MAT_scatra`.
* A SPATIALLY VARYING boundary datum is `VAL: [1.0]` together with
  `FUNCT: [1]`, where `FUNCT1` holds a `SYMBOLIC_FUNCTION_OF_SPACE_TIME`. 4C
  multiplies the two. There is no table form, so the imported samples must be
  fitted to an expression; the script uses a least-squares polynomial in y.
  RAISE `FIT_DEG` if your interface profile is not close to polynomial — a
  bad fit is a silently wrong boundary condition, not an error.
* The same `VAL x FUNCT` rule holds for `DESIGN LINE NEUMANN CONDITIONS`, and
  4C's Neumann `VAL` is exactly the flux quantity the partner exported. Hand it
  over unchanged.
* VTU output lands in `out-vtk-files/scatra-<step>-<rank>.vtu`. THE TRAILING
  NUMBER IS THE MPI RANK, NOT THE STEP. Sorting on the last number returns
  `scatra-00000-0.vtu`, which is the INITIAL CONDITION — an all-zero field that
  looks like a converged solve of a trivial problem. Parse the FIRST number.
* The scalar field is named `phi_1`, never `temperature`.
* The flux field `flux_domain_phi_1` only appears if you set
  `CALCFLUX_DOMAIN: "diffusive"` in `SCALAR TRANSPORT DYNAMIC`. It is the
  diffusive flux vector `-D grad(phi)`; project it on your outward normal.
  Computing the flux yourself from `phi_1` differences is unnecessary and worse.
* A 4C VTU repeats every node once per element (QUAD4 -> 4 copies of each
  node). Collapse duplicates by coordinate before exporting, or `n_points` is
  four times too large and changes with the mesh.
* `DLINE-NODE TOPOLOGY` must list the nodes of every design line you reference.
  A `DESIGN LINE ... CONDITIONS` entry with `E: 2` and no DLINE 2 topology is
  accepted and does nothing.
* The 4C binary needs its dependency libraries on `LD_LIBRARY_PATH`; put that
  directory in `FOURC_LD` if the binary does not find them by itself.''')


def _ngsolve() -> str:
    return _payload(
        "NGSolve",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against scikit-fem, with non-matching interface meshes — "
        "and all converged.",
        "ngsolve", _launch_py(),
        '''\
* TWO CONSECUTIVE `gfu.Set(value, definedon=mesh.Boundaries(...))` CALLS CANCEL
  EACH OTHER. The second `Set` zeroes what the first wrote outside its own
  region, so setting the outer Dirichlet value and then the interface value
  loses one of them — and the run finishes with a plausible, WRONG answer, not
  an error. Put every Dirichlet value into ONE `mesh.BoundaryCF({...})`.
* `Integrate(grad(gfu)[0], mesh, definedon=mesh.Boundaries("..."))` RETURNS
  EXACTLY 0.0. An H1 GridFunction's gradient has no boundary trace. Use the
  reaction/residual method — `a.mat * gfu.vec - f.vec` restricted to the
  interface DOFs — which is also the more accurate flux.
* `SplineGeometry.AddRectangle(..., bcs=(...))` names the edges in the order
  bottom, right, top, left. Getting that order wrong silently puts the
  interface condition on the wrong edge.
* Netgen meshing is DETERMINISTIC across separate processes for the same
  geometry and `maxh`, which is what makes "the same number of interface
  points every iteration" hold — each coupling iteration is a fresh process
  that re-meshes from scratch. Do not introduce anything mesh-size-dependent
  that varies with the imported data.
* At `order=1` the H1 DOFs coincide with the mesh vertices, so
  `NodeId(VERTEX, i)` gives you a stable interface indexing. At higher order
  you must locate the interface DOFs properly instead.
* Solve with `a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")` applied
  to the residual after setting the Dirichlet data — the direct
  `gfu.vec.data = a.mat.Inverse() * f.vec` form ignores your BCs.''')


def _skfem() -> str:
    return _payload(
        "scikit-fem",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against NGSolve, with non-matching interface meshes — and "
        "all converged.",
        "skfem", _launch_py(),
        '''\
* This is the lightest participant of the set: pure Python, numpy + scipy, no
  compilation and no JIT. If you are prototyping a coupling and do not care
  which code solves a side, start here.
* `MeshTri.init_tensor(x, y)` builds the structured subdomain mesh directly
  from your NX/NY, so the interface node set is exactly predictable.
* `facets_satisfying(...)` defaults to `boundaries_only=False` in current
  scikit-fem, so it can return INTERIOR facets that happen to satisfy your
  predicate. That is harmless when the interface is a domain edge (as here) and
  wrong the moment it is not — pass `boundaries_only=True` when in doubt.
* Applying the imported value: put it into the solution vector at the interface
  DOFs and pass those DOFs as `D` to `condense`. Applying an imported flux: a
  `LinearForm` assembled on a `FacetBasis` restricted to the interface facets.
* Getting the outgoing flux: the residual `K @ x - f` at the CONSTRAINED
  interface DOFs is the consistent nodal flux and needs no differentiation.
  Divide by the tributary length if you export a density, and keep whichever
  convention you chose consistent with the partner.
* scikit-fem gives you the assembled matrices, so it is the easiest backend in
  which to check a coupling by hand: assemble the two subdomains and the
  monolithic problem and compare. That is the strongest verification available
  for a partitioned coupling and it costs a few lines here.''')



# ══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════

_BACKENDS = {
    "fenics": _fenics, "fenicsx": _fenics, "dolfinx": _fenics,
    "fourc": _fourc, "4c": _fourc,
    "ngsolve": _ngsolve,
    "skfem": _skfem, "scikit-fem": _skfem, "scikitfem": _skfem,
}

_ALIAS_CANON = {"fenics": "fenics", "fenicsx": "fenics", "dolfinx": "fenics",
                "fourc": "fourc", "4c": "fourc", "ngsolve": "ngsolve",
                "skfem": "skfem", "scikit-fem": "skfem", "scikitfem": "skfem",
                "dune": "dune", "dune-fem": "dune", "dunefem": "dune",
                "dealii": "dealii", "deal.ii": "dealii",
                "febio": "febio", "kratos": "kratos", "sparta": "sparta"}


def coupling_knowledge(solver: str = "") -> str:
    """knowledge(topic='coupling', solver=...) — core payload, or one backend."""
    key = (solver or "").strip().lower()
    if not key:
        return coupling_core()
    fn = _BACKENDS.get(key)
    if fn is None:
        known = ", ".join(sorted(set(_ALIAS_CANON.get(k, k)
                                      for k in _BACKENDS)))
        return (f"# No coupling participant pattern for solver='{solver}'\n\n"
                f"That name is not one OASiS ships a participant script for. "
                f"The ones it does: {known}.\n"
                f"The general contract below applies to EVERY backend, so you "
                f"can still write a participant for '{solver}' from it — the "
                f"driver needs only a command that reads imports.json and "
                f"writes exports.json.\n\n" + coupling_core())
    return fn()


# ══════════════════════════════════════════════════════════════════════════
# preCICE — served by knowledge(topic='precice', solver=...)
# ══════════════════════════════════════════════════════════════════════════

_PRECICE_CORE = '''\
# preCICE coupling with OASiS — `couple_precice`

## 0. WHEN TO USE IT INSTEAD OF `couple`

`couple` is a file handshake: OASiS starts your solver once per iteration and
moves JSON between the runs. It needs nothing from the solver but a script.

`couple_precice` is the library path: both codes run CONCURRENTLY, stay alive
for the whole simulation, and exchange data through preCICE at every time
window. Use it when you need
  * a transient coupling where restarting the solver each iteration is absurd,
  * implicit sub-iteration inside each time window (true Gauss-Seidel), or
  * preCICE's mesh mapping between genuinely non-matching surfaces.
It costs: every participant must be able to `import precice` (or link
libprecice) IN ITS OWN INTERPRETER. That is the gate — a backend whose
interpreter cannot import precice cannot use this path at all, no matter what
physics it solves.

## 1. WHAT YOU SUPPLY vs WHAT OASiS GENERATES

You supply, as JSON strings:

    participants = [{"name":  "Left",              # preCICE participant name
                     "mesh":  "Left-Mesh",         # the mesh IT provides
                     "writes":["Temperature"],     # data names it writes
                     "reads": ["Heat-Flux"],       # data names it reads
                     "command":["<interpreter>","participant_left.py"]}, ...]
    data      = [{"name":"Temperature","type":"scalar"},
                 {"name":"Heat-Flux","type":"scalar"}]      # or "vector"
    exchanges = [{"data":"Temperature","from":"Left","to":"Right"},
                 {"data":"Heat-Flux",  "from":"Right","to":"Left"}]
    work_dir  = "/abs/path"        scheme = "serial-explicit" | "serial-implicit"
                                            | "parallel-explicit" | "parallel-implicit"
    dimensions = 2                 max_time = 10.0        time_window = 1.0

OASiS writes `work_dir/precice-config.xml` for you, containing:
  * one `<data:scalar|vector>` per entry in `data`;
  * one `<mesh>` per participant, listing every data name it writes or reads;
  * one `<participant>` per participant, with `<provide-mesh>`, a
    `<receive-mesh from=...>` for every partner whose data it reads,
    `<write-data>` / `<read-data>`, and a **`nearest-neighbor` read mapping**;
  * `<m2n:sockets>` for each exchanging pair, `exchange-directory="."`;
  * the `<coupling-scheme:...>` with `<time-window-size>`, `<max-time>`,
    `<participants first=... second=...>` and one `<exchange>` per entry.

For an IMPLICIT scheme it additionally emits, with values you CANNOT currently
change through the tool:
  * `<max-iterations value="20" />`
  * `<relative-convergence-measure limit="1e-6" />` on the FIRST exchanged data
  * `<acceleration:aitken>` with `<initial-relaxation value="0.5" />`
If you need different acceleration, a different mapping (`nearest-projection`,
`rbf`) or a different tolerance, write `precice-config.xml` yourself and launch
the participants yourself — `generate_precice_config` supports them, the tool
does not pass them through.

HARD LIMITS OF THE GENERATED CONFIG, know them before you design the coupling:
  * EXACTLY TWO PARTICIPANTS. `<participants first= second= >` names only the
    first two. A third participant is emitted into the XML and then REJECTED by
    preCICE — every process dies with `Participant "X" is not configured for
    coupling scheme`. A real N-way coupling needs `<coupling-scheme:multi>`,
    which this tool does not generate.
  * FOR `serial-implicit`, `exchanges[0]` MUST BE THE FIELD WRITTEN BY THE
    SECOND PARTICIPANT. OASiS silently makes `exchanges[0]` both the
    convergence measure and the acceleration datum, and preCICE only allows
    second-to-first data there: get the order wrong and both sides abort with
    `only data exchanged from the second to the first participant can be used
    for acceleration`.
  * NO EXCHANGE IS MARKED `initialize="true"`, so `requires_initial_data()`
    always returns False and the first participant READS ZERO in the first time
    window. Make your first window one your solver can survive with a zero
    incoming field, or ramp it.
  * MAPPING IS READ-DIRECTION, `consistent`, nearest-neighbor only. There is no
    conservative mapping, so a FLUX or FORCE exchanged across non-matching
    meshes is NOT conserved by the mapping. Exchange intensive quantities where
    you can.
  * every participant runs with `cwd = work_dir` — the SAME directory. Two
    scripts writing the same output filename overwrite each other, and two
    couplings sharing a work_dir clash on the socket files and on
    `precice-config.xml`. One work_dir per coupling run.
  * `converged` in the returned JSON is EXIT CODES plus preCICE's own
    per-window verdict read back from `precice-<name>-iterations.log`. It is
    still not a check on the VALUES: the orchestrator never sees the exchanged
    fields. Read the participant logs, and have each participant print and
    check its own numbers.

## 2. THE PARTICIPANT LOOP — the same skeleton in every code

```python
import precice, numpy as np

p = precice.Participant("<NAME>", "precice-config.xml", 0, 1)   # rank 0 of 1
coords = np.array([[x0, y0], [x1, y1], ...])      # YOUR interface points
vid = p.set_mesh_vertices("<MESH_NAME>", coords)  # same mesh name as in the spec

if p.requires_initial_data():                     # only for *-implicit / initialize
    p.write_data("<MESH_NAME>", "<WRITE_DATA>", vid, initial_outgoing)
p.initialize()

while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():           # IMPLICIT: save state
        saved = solver.save_state()
    dt = p.get_max_time_step_size()
    incoming = p.read_data("<MESH_NAME>", "<READ_DATA>", vid, dt)
    outgoing = solver.advance(dt, incoming)       # YOUR solve, using `incoming` as a BC
    p.write_data("<MESH_NAME>", "<WRITE_DATA>", vid, outgoing)
    p.advance(dt)
    if p.requires_reading_checkpoint():           # IMPLICIT: redo this window
        solver.restore_state(saved)
p.finalize()
```

CHECKPOINTS ARE MANDATORY FOR IMPLICIT SCHEMES. `serial-implicit` and
`parallel-implicit` sub-iterate each time window. A participant that ignores
`requires_writing_checkpoint()` / `requires_reading_checkpoint()` does NOT hang
— it ABORTS with `The required actions write-iteration-checkpoint are not
fulfilled`, exit code 255, on both sides. For `*-explicit` both calls always
return False, so the same loop is correct there too: write it once, with the
checkpoint calls, always.

## 3. LAUNCH TRAPS — these are what actually goes wrong

  * BOTH PARTICIPANTS MUST RUN AT THE SAME TIME. Each blocks inside
    `initialize()` until the other connects. Starting them one after the other,
    or under a single `mpirun` over both files, hangs until the timeout.
    `couple_precice` starts them concurrently for you; if you launch by hand,
    background the first one.
  * `libprecice` must be loadable, and `LD_LIBRARY_PATH` is NOT optional:
    without it `import precice` fails with
    `libprecice.so.3: cannot open shared object file`. OASiS reads
    `$PRECICE_LIB_DIR` (it has a built-in default) and prepends it for the
    participant processes it launches; set that variable if the library lives
    elsewhere, and use `extra_env` for anything else a participant needs.
  * `pyprecice` must match `libprecice`'s major version. A mismatch shows up as
    an ImportError or an immediate segfault, not as a helpful message.
  * The participant NAME, the MESH name and the DATA names in your script must
    match the spec EXACTLY, character for character. preCICE aborts on a
    mismatch, but only after both sides have connected.
  * A stale `precice-run/` directory in `work_dir` from a killed run makes the
    next run hang on connect. Delete it before re-running.
  * A PARTICIPANT THAT NEVER STARTS MAKES THE OTHER BLOCK FOREVER. preCICE has
    no connect timeout. If one `command` is wrong — bad interpreter, missing
    module — the partner sits in `initialize()` until OASiS's `timeout` fires,
    and because the orchestrator waits on the participants one after another
    the real wall-clock cost is N x timeout. Run each participant's command by
    hand once (it will block at initialize; that is the correct symptom) before
    coupling.
  * `mapping="rbf"` generates INVALID XML — preCICE v3 needs a
    `<basis-function:*>` child that the generator does not emit. And
    `nearest-projection` needs `set_mesh_edges` / `set_mesh_triangles` calls in
    the participant, which the pattern below does not make. Stay on
    nearest-neighbor unless you write the config yourself.
  * preCICE writes its own INFO log to each participant's stdout and OASiS
    returns only a short tail, so your solver's own prints are usually NOT in
    the returned `logs`. Write your diagnostics to a file in `work_dir`.
  * `set_mesh_vertices` expects an (N, dimensions) array. Passing 3-column
    coordinates to a `dimensions=2` config is a shape error at initialize time.

## 4. WHAT preCICE DOES THAT `couple` DOES NOT

  * mesh mapping between non-matching interfaces (nearest-neighbor here;
    nearest-projection and RBF exist in preCICE but the tool does not expose
    them), whereas with `couple` each participant interpolates for itself;
  * implicit sub-iteration WITHIN a time window with Aitken or IQN acceleration
    — genuine Gauss-Seidel, which the `couple` driver's Jacobi loop cannot do;
  * transient couplings without restarting a solver per iteration.
And what `couple` does that preCICE does not: it works with a code that has no
preCICE adapter at all, which on this install is most of them.

## 5. ADAPTER REALITY CHECK

preCICE's own adapter ecosystem (OpenFOAM, CalculiX, SU2, FEniCS via
`fenicsprecice`, deal.II via a community adapter) is separate from whether a
backend can be driven as a plain participant here. What matters for
`couple_precice` is only whether `import precice` works in that backend's
interpreter and whether you can drive that backend's time loop from Python.
Per-backend verdicts: `knowledge(topic='precice', solver='<name>')`.
'''


def precice_knowledge(solver: str = "") -> str:
    """knowledge(topic='precice', solver=...) — core payload, or one backend.

    The `solver` argument used to be accepted and dropped, so every backend got
    byte-identical output while `couple_precice`'s docstring promised
    "Each backend's preCICE participant pattern is available via
    knowledge(topic='precice', solver=...)".
    """
    key = (solver or "").strip().lower()
    canon = _ALIAS_CANON.get(key, key)
    if not canon:
        return _PRECICE_CORE
    entry = _PRECICE_BY_BACKEND.get(canon)
    if entry is None:
        known = ", ".join(_BACKEND_ORDER)
        return (f"# preCICE on solver='{solver}'\n\nNo per-backend preCICE note "
                f"for that name. Backends covered: {known}.\n\n" + _PRECICE_CORE)
    return (f"# preCICE participant: {entry['title']}\n\n"
            f"**Verdict on this install: {entry['verdict']}**\n\n"
            f"{entry['body']}\n\n---\n\n{_PRECICE_CORE}")


# Per-backend preCICE verdicts. Every CAN was established by running a real
# two-participant coupling through OASiS's own preCICE orchestrator on this
# install; every CANNOT was established by looking for a preCICE entry point in
# the installed code and not finding one.
_PRECICE_BY_BACKEND = {
    "fenics": {
        "title": "FEniCSx (dolfinx)",
        "verdict": "CAN — proven by a real coupled run",
        "body": '''\
`import precice` works in the same interpreter as `dolfinx`, and a FEniCSx
participant was coupled to a second code through `couple_precice` end to end.

  * You do NOT need `fenicsprecice`. The raw `precice` bindings are enough and
    are what was proven here; the adapter is a convenience layer, not a
    requirement.
  * `LD_LIBRARY_PATH` must include the preCICE lib directory for THIS
    interpreter too — pass it through `extra_env` if you launch by hand.
  * Interface vertices: use `V.tabulate_dof_coordinates()` rows on the
    interface, take the first `dimensions` columns, and keep that order fixed
    for the whole run — preCICE indexes by the vertex ids `set_mesh_vertices`
    returned.
  * Applying the incoming field: fill a `fem.Function` at the interface DOFs
    and use the two-argument `fem.dirichletbc(function, dofs)`; for a Neumann
    datum use `meshtags` plus a subdomain `ds` measure, never bare `ufl.ds`.
  * Getting the outgoing flux: L2-project `-K*grad(T)[n]` and read it at the
    interface DOFs, or use the residual/reaction of the constrained system.
    Do not finite-difference toward a "neighbouring" node on a triangle mesh.
  * Run one MPI rank per participant unless you have configured preCICE for
    parallel participants.''',
    },
    "ngsolve": {
        "title": "NGSolve",
        "verdict": "CAN — proven by a real coupled run",
        "body": '''\
`import precice` works in the interpreter that carries NGSolve, and an NGSolve
participant was coupled to a second code end to end.

TWO NGSolve-SPECIFIC SILENT-WRONG TRAPS, both found by running:
  * TWO CONSECUTIVE `gfu.Set(value, definedon=mesh.Boundaries(...))` CALLS
    CANCEL EACH OTHER. The second `Set` zeroes what the first wrote outside its
    own region, so a participant that sets an outer Dirichlet value and then an
    interface value ends up with one of them gone — and the run completes with
    a plausible, wrong answer. Put every Dirichlet value into ONE
    `mesh.BoundaryCF({...})` and `Set` once.
  * `Integrate(grad(gfu)[0], mesh, definedon=mesh.Boundaries("..."))` RETURNS
    EXACTLY 0.0. An H1 GridFunction's gradient has no boundary trace, so this
    silently reports zero flux. Use the reaction/residual method instead: form
    `a.mat * gfu.vec - f.vec` and sum it over the boundary DOFs.
  * Applying an incoming flux is a `LinearForm` term `g * v * ds("interface")`,
    with `g` a `CoefficientFunction`; build it by fitting or interpolating the
    incoming samples, since preCICE hands you values at YOUR vertices.''',
    },
    "skfem": {
        "title": "scikit-fem",
        "verdict": "CAN — proven by a real coupled run",
        "body": '''\
`import precice` works in the interpreter that carries scikit-fem, and a
scikit-fem participant was coupled to a second code end to end. It is the
lightest participant of all of them: pure Python, no compilation, no JIT.

  * Interface vertices come straight out of `mesh.p` — plain numpy, so keeping
    a fixed order is trivial. Slice to the first `dimensions` columns.
  * Applying the incoming value: `skfem.condense(K, f, x=x, D=dirichlet_dofs)`
    with the incoming values written into `x` at those DOFs.
  * Applying an incoming flux: assemble a `LinearForm` over the interface
    `FacetBasis` with the incoming density as the load.
  * Getting the outgoing flux: the residual `K @ x - f` restricted to the
    constrained interface DOFs is the consistent nodal flux, which is both
    easier and more accurate than differentiating the solution.''',
    },
    "dune": {
        "title": "DUNE-fem",
        "verdict": "CAN — proven by a real coupled run",
        "body": '''\
A DUNE-fem participant was coupled end to end. Two install-level facts decide
whether it works at all:

  * `precice` and `dune.fem` must be importable in ONE interpreter. If DUNE
    lives in its own conda environment without `pyprecice`, you do not have to
    install anything into it: put the site-packages of the interpreter that HAS
    `pyprecice` on `PYTHONPATH` and the preCICE lib on `LD_LIBRARY_PATH`, and
    the DUNE interpreter imports both. Verify with a one-line
    `python -c "import precice, dune.fem"` BEFORE coupling — a failed import
    leaves the partner blocking in `initialize()` with no error.
  * DUNE-fem JIT-COMPILES EACH DISTINCT SCHEME, which takes on the order of a
    minute. Build the scheme ONCE, BEFORE `precice.Participant(...)`, or the
    partner blocks on connect while you compile. Make the coupled boundary
    datum a `dune.ufl.Constant` (or a discrete function) and MUTATE `.value`
    each window instead of rebuilding the scheme, or you pay that compile on
    every coupling iteration.''',
    },
    "dealii": {
        "title": "deal.II",
        "verdict": "CAN, as a C++ participant — proven by a real coupled run",
        "body": '''\
deal.II has no Python API, so the participant is a compiled C++ executable that
links `libprecice` directly. One was built and coupled to a Python participant
end to end.

  * Build through CMake with `DEAL_II_SETUP_TARGET` and add
    `target_link_libraries(<target> <path-to>/libprecice.so)`. A hand-rolled
    `g++ -I<dealii>/include` does NOT work — it fails on deal.II's own bundled
    headers.
  * `DEAL_II_DIR` must point at the BUILD/INSTALL tree that contains
    `lib/cmake/deal.II/deal.IIConfig.cmake`. Pointing it at the source
    checkout silently falls back to whatever system deal.II exists, which is
    usually a different, older version.
  * At run time the executable needs the preCICE lib directory on
    `LD_LIBRARY_PATH`; pass it via `extra_env`.
  * The C++ API mirrors the Python one: `precice::Participant`,
    `setMeshVertices`, `readData`, `writeData`, `advance`,
    `requiresWritingCheckpoint` / `requiresReadingCheckpoint`.
  * Community deal.II preCICE adapters exist upstream, but none is needed for
    this: linking the library directly is what was proven here.''',
    },
    "kratos": {
        "title": "Kratos Multiphysics",
        "verdict": "CAN — proven by a real coupled run, with a caveat about WHICH Kratos",
        "body": '''\
A Kratos participant was coupled to a second code end to end. The install is
the hard part:

  * A Kratos wheel can import cleanly on one host and be unusable on another —
    one on this class of host fails at import with a `GLIBC` version error from
    its bundled shared objects. If that happens, use a Kratos built from source
    and set BOTH `PYTHONPATH` to the install root and `LD_LIBRARY_PATH` to its
    `libs` directory, via `extra_env`.
  * A core-only Kratos has NO `ConvectionDiffusionApplication` and therefore no
    thermal element at all. Check `import KratosMultiphysics.<App>` for every
    application your participant needs BEFORE coupling.
  * Kratos drives its own time loop, so the preCICE loop wraps
    `InitializeSolutionStep()` / `SolveSolutionStep()` /
    `FinalizeSolutionStep()`, and the checkpoint save/restore is a copy of the
    solution-step variables on the interface nodes.
  * Kratos also has its own CoSimulation application. That is a different,
    Kratos-internal coupling path; it is not what `couple_precice` drives.''',
    },
    "sparta": {
        "title": "SPARTA (DSMC)",
        "verdict": "CAN, but ONLY with RTLD_DEEPBIND — proven by a real coupled run",
        "body": '''\
A SPARTA DSMC participant was coupled to a solid conduction participant end to
end. There is exactly one way to load it, and every other way segfaults:

  * `libsparta.so` DEFINES ITS OWN `MPI_*` STUB SYMBOLS and links no real MPI.
    `import precice` pulls a real MPI into the global symbol namespace, and
    SPARTA's stub calls are then interposed by it, so SPARTA crashes inside
    `PMPI_Type_size` with a segfault — MPI was never initialised. Loading
    SPARTA first instead fails the other way, with a missing `libmpi`.
    `from mpi4py import MPI` first does not help.
  * THE FIX: load SPARTA's library yourself with deep binding, so its own
    symbols win:

```python
import ctypes, os
mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
lib = ctypes.CDLL("<path to>/libsparta.so", mode=mode)
```

  * SPARTA is a Monte-Carlo code. Its interface output carries statistical
    noise, so an IMPLICIT scheme's convergence measure may never be met even
    though the physics is fine. An explicit scheme with enough sampling per
    window is the honest choice; if you use implicit, set the tolerance above
    the sampling noise and say so.
  * A SPARTA surface can take a prescribed TEMPERATURE
    (`surf_collide ... diffuse`) but NOT a prescribed heat flux — no
    surface-collision model accepts one. SPARTA is a Dirichlet-side
    participant only.''',
    },
    "fourc": {
        "title": "4C Multiphysics",
        "verdict": "CANNOT — 4C has no preCICE entry point",
        "body": '''\
There is no preCICE support in 4C. Searching the installed 4C source tree for
`precice` returns nothing, the built binary contains no preCICE symbols, and it
links no preCICE library. This is not a configuration problem you can fix from
the outside: adding preCICE to 4C means writing and building an adapter into
4C's own source.

USE `couple` INSTEAD. 4C works well as a participant in OASiS's file-handshake
driver — it has been run there on BOTH the Dirichlet and the Neumann side of a
cross-code coupling. Call `knowledge(topic='coupling', solver='fourc')` for a
complete runnable 4C participant.''',
    },
    "febio": {
        "title": "FEBio",
        "verdict": "CANNOT (as installed) — no preCICE in the binary, no Python API",
        "body": '''\
The installed FEBio binary contains no preCICE symbols and FEBio ships no
Python module, so there is nothing to call `precice` from. FEBio runs an XML
deck to completion and exits; it does not expose a time loop.

A preCICE-enabled FEBio would have to be a compiled FEBio plugin using FEBio's
own callback interface. That is a real route, but nothing of the kind is built
here, so this is UNVERIFIED, not supported.

USE `couple` INSTEAD. FEBio participates fine in OASiS's file-handshake driver
as an XML-writing / log-parsing wrapper. Call
`knowledge(topic='coupling', solver='febio')`.''',
    },
}


def _febio() -> str:
    return _payload(
        "FEBio",
        "**Either side — but NOT for heat.** FEBio 4 has no heat module "
        "(FEBioHeat was removed upstream and survives only as a plugin), so a "
        "conduction participant is impossible here. The shipped script solves "
        "the exact linear analogue instead: a uniaxial-strain elastic bar, "
        "where displacement plays the role of temperature and the P-wave "
        "modulus the role of conductivity. Both roles were run as a real "
        "FEBio-to-FEBio coupling on this install, with non-matching meshes, "
        "and converged.",
        "febio", _launch_py(_interp_wrapper("FEBio", "FEBIO"),
                            _step2_block(_RIGHT_BLOCK_FEBIO,
                                         "the placeholder ELASTIC problem")),
        '''\
* NO SCRIPTING API. FEBio is XML-in, plot/log-out. A participant is therefore a
  wrapper: write a complete `.feb` deck with the imported data baked in, run
  `febio4 -i deck.feb`, parse the ASCII logfile, write exports.json.
* AN UNKNOWN `<Module type>` SEGFAULTS — it does not produce an error message.
  FEBio 4's registered modules are solid, biphasic, solute, multiphasic, fluid,
  fluid-FSI, fluid-solutes, multiphasic-FSI, thermo-fluid and polar fluid. A
  deck asking for a heat module dies with SIGSEGV while reading the file, which
  looks exactly like a corrupted deck.
* PER-POINT interface data is possible and is the whole reason this works:
  - a prescribed field is `<MeshData><NodeData name="..." node_set="...">` plus
    `<bc type="prescribed displacement"><value type="map">...`;
  - a prescribed traction is `<MeshData><SurfaceData data_type="vec3">` plus
    `<surface_load type="traction"><traction type="map">...`.
  In BOTH, `lid` is the 1-based index INTO THAT SET, not the node or face id.
  Getting that wrong silently applies the right numbers to the wrong places.
* Do not build XML numbers with `repr()` or an f-string `!r`: a numpy 2 scalar
  stringifies as `np.float64(0.0)` and FEBio rejects the deck.
* `febio4` has NO `-h`/`--help` flag; passing one is a fatal error. The real
  options are `-i -o -r -s -d -p -g1 -g2 -config -noconfig -import -noappend
  -nosplash -silent`.
* Check for `N O R M A L   T E R M I N A T I O N` in stdout before trusting the
  logfile — FEBio can exit 0 after writing a partial log.
* The elastic analogue of the flux convention: export
  `q_out = -(sigma . n_own)_x`, so the two sides carry opposite signs, and the
  Neumann side applies the partner's number unchanged as a traction.''')


def _sparta() -> str:
    return _payload(
        "SPARTA (DSMC)",
        "**DIRICHLET-TYPE ONLY, and it will not pass the convergence check.** "
        "SPARTA imports a wall TEMPERATURE and exports the energy flux the gas "
        "deposits, which is exactly the Dirichlet role. It CANNOT take the "
        "Neumann role: no SPARTA surface-collision model accepts a prescribed "
        "heat flux — `diffuse`, `cll`, `td` and `impulsive` all take a "
        "temperature, and `fix surf/temp` derives one from SPARTA's OWN "
        "computed flux, not from an imported one. A full coupling to a thermal "
        "shell was run on this install: the physics agreed across the interface "
        "and the interface energy balance closed, but `couple` still reported "
        "FAILURE, because the residual cannot fall below the Monte-Carlo "
        "sampling noise. Read the stochasticity note below before using it.",
        "sparta", _launch_py(_interp_wrapper(
            "SPARTA", "SPARTA",
            extra="\n   Copy the surf / species / vss files into `work_dir` yourself:\n"
                  "   `couple` has no `data_files`, so nothing else puts them there."),
            _STEP2_SPARTA),
        '''\
* STOCHASTICITY IS THE HEADLINE. DSMC output is a Monte-Carlo estimate. Its
  sampling noise does NOT shrink as the coupling iterates, so the driver's
  relative residual has a FLOOR at the noise level and `tol` below that floor
  can never be met — the run ends as "did not converge", which is honest, not a
  bug. Set `tol` above the noise, or accept an explicit non-iterated exchange,
  and say which you did. With a FIXED seed the runs are bit-reproducible, which
  makes an apparently converging residual possible even when the physics has
  not settled; that is more dangerous than the noise, not less.
* YOU MUST STAGE THE DATA FILES. `couple` does not copy anything. Put the surf,
  species and vss files in `work_dir` yourself, or the deck dies with
  `Cannot open species file ...` from inside SPARTA.
* PER-ELEMENT interface data goes in through a custom surf attribute:
  `custom surf create tsurf float 0 file tsurf.in 1 tsurf` and then
  `surf_collide 1 diffuse s_tsurf 1.0`. That is the only route to a spatially
  varying wall temperature.
* Reading the flux back: `compute ... surf all all etot` +
  `fix ... ave/surf ...` + a surf `dump`. THE DUMP MUST REFERENCE `f_1`, NOT
  `f_1[1]` — a `fix ave/surf` with a single input column is a per-surf VECTOR
  and the bracketed form aborts with `Dump surf fix does not compute per-surf
  array`.
* `read_surf` and `read_grid` files of the same name are DIFFERENT geometries.
  Staging the wrong one is accepted and gives a wrong answer.
* If you also want SPARTA under preCICE, its shared library needs
  `RTLD_DEEPBIND` or it segfaults against preCICE's MPI — see
  `knowledge(topic='precice', solver='sparta')`.''')


def _kratos() -> str:
    return _payload(
        "Kratos Multiphysics",
        "**Either side — but NOT in OASiS's own interpreter on this install.** "
        "Kratos is not importable where OASiS runs here, so its participant was "
        "proven in a separate Kratos install: both the Dirichlet and the "
        "Neumann role were run against FEniCSx and both converged with "
        "non-matching interface meshes. The script below is the DIRICHLET side; "
        "the Neumann side is the same script with the interface condition "
        "changed, described under the traps.",
        "kratos", _launch_py(step2=_STEP2_KRATOS),
        '''\
* CHECK THE INSTALL FIRST, IT IS THE USUAL FAILURE. `import KratosMultiphysics`
  can fail at import with a GLIBC version error from the bundled shared
  objects even though the package installed cleanly. If that happens, use a
  Kratos built from source and point `PYTHONPATH` at its install root and
  `LD_LIBRARY_PATH` at its `libs` directory.
* A CORE-ONLY KRATOS HAS NO THERMAL ELEMENT.
  `import KratosMultiphysics.ConvectionDiffusionApplication` must succeed
  before a conduction participant can work at all. Test that one line by hand
  before writing anything else.
* The thermal problem is configured through a `ConvectionDiffusionSettings`
  object placed in `mp.ProcessInfo[CONVECTION_DIFFUSION_SETTINGS]`, with
  `SetUnknownVariable(TEMPERATURE)`, `SetDiffusionVariable(CONDUCTIVITY)`,
  `SetVolumeSourceVariable(HEAT_FLUX)` and
  `SetSurfaceSourceVariable(FACE_HEAT_FLUX)`. Every one of those variables must
  also be added with `AddNodalSolutionStepVariable` BEFORE the nodes are
  created, or they silently do not exist on the nodes.
* DIRICHLET SIDE (this script): write the imported temperature into each
  interface node and `node.Fix(TEMPERATURE)`.
  NEUMANN SIDE: do NOT fix the interface nodes; instead set `FACE_HEAT_FLUX` on
  them from the partner's exported `normal_fluxes` and create the interface
  `ThermalFace` conditions so that flux is assembled. Everything else, the
  mesh, the material and the export, is unchanged.
* Kratos also ships a CoSimulation application. That is Kratos's own internal
  multi-physics coupling, unrelated to `couple`; do not mix the two.
* If a participant needs a `params.json` or any other file, stage it into
  `work_dir` yourself — `couple` copies nothing.''')


# Registered after their definitions (the dispatch table is declared earlier so
# that it sits next to the alias map it mirrors).
_BACKENDS.update({"febio": _febio, "sparta": _sparta, "kratos": _kratos})


def _dune() -> str:
    return _payload(
        "DUNE-fem",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against deal.II, with non-matching interface meshes — and "
        "all converged.",
        "dune", _launch_py(),
        '''\
* JIT COMPILATION IS THE THING THAT WILL BITE YOU. DUNE-fem compiles each
  distinct UFL form on first use, and a cold cache can take a minute or more
  per form. The participant runs as a FRESH PROCESS every coupling iteration,
  so if the FORM TEXT changes with the imported data you pay that compile on
  every iteration and the coupling appears to hang.
  THE RULE: keep the form structurally CONSTANT. Put the imported interface
  data into a discrete function's dof vector (`gfun.as_numpy[...] = ...`) or a
  `dune.ufl.Constant`, never into the form's text. Then only the first
  iteration is slow.
* `0 * v * dx` FOLDS TO A DOMAINLESS UFL ZERO and fails with "integral is
  missing an integration domain". Wrap a zero source in
  `dune.ufl.Constant(0.0)` rather than writing the literal.
* THERE IS NO `tabulate_dof_coordinates`. Get the dof-to-coordinate map by
  interpolating the coordinate functions into the same space and reading
  `.as_numpy` — `space.interpolate(x[0]).as_numpy` gives the x of every dof, in
  dof order.
* `structuredGrid` CARRIES NO BOUNDARY IDS. Select the outer and interface
  boundaries with a coordinate predicate,
  `conditional(lt(abs(x[0] - X_IFACE), 1e-8), 1, 0)`, and use the same
  indicator both for the Dirichlet BC and to MASK the Neumann `ds` term. An
  unmasked `ds` puts the interface flux on the whole boundary.
* The default `galerkin(..., solver="cg")` projection used to recover the
  interface flux runs at a loose linear tolerance, so the exported flux carries
  small solver noise. It is far below a 1e-8 coupling tolerance but it is not
  machine precision; tighten the scheme's linear-solver parameters before
  asking for a much tighter coupling tolerance.
* DUNE usually lives in its own conda environment. Use the interpreter
  `discover(query='list')` reports for it, not OASiS's own.''')


def _dealii() -> str:
    return _payload(
        "deal.II",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against DUNE-fem, with non-matching interface meshes — and "
        "all converged.",
        "dealii", _launch_py(_interp_wrapper(
            "deal.II", "DEALII_EXE",
            extra="\n   BUILD THE SOLVER FIRST (see the traps below): the wrapper runs\n"
                  "   an executable that does not exist until you have built it, and\n"
                  "   `DEALII_EXE` is the path to YOUR build, not to a deal.II install.")),
        '''\
* THE PARTICIPANT IS TWO FILES: a compiled C++ solver and a thin Python
  wrapper. The wrapper converts imports.json into the solver's plain-text input
  file, runs the executable, and converts its output into exports.json. OASiS
  ships the C++ source and a CMakeLists next to this script
  (`heat_iface_dealii.cc`, `CMakeLists.txt` in the same directory the payload
  came from). Build it once:

```
cmake -S <dir with the .cc and CMakeLists> -B <build dir> \\
      -DDEAL_II_DIR=<deal.II build or install tree> -DCMAKE_BUILD_TYPE=Release
make -C <build dir> -j8
```

* `DEAL_II_DIR` MUST BE THE BUILD OR INSTALL TREE that contains
  `lib/cmake/deal.II/deal.IIConfig.cmake`. Pointing it at the source checkout
  silently configures against whatever system deal.II happens to exist, which
  is usually an older version, and the build then fails in confusing ways.
* DO NOT try to compile with a bare `g++ -I<dealii>/include`. deal.II's bundled
  headers (Kokkos and friends) are only found through CMake's
  `DEAL_II_SETUP_TARGET`.
* PASS THE PROBLEM THROUGH THE INPUT FILE, not through recompilation. The
  shipped solver reads side, geometry, conductivity, mesh size and the imported
  interface samples from one text file, so a coupling iteration is a re-run,
  not a rebuild.
* THE NODAL FLUX MUST BE AVERAGED OVER BOTH ADJACENT CELLS. Assembling the
  interface flux cell by cell and writing it into a per-node array is
  last-writer-wins, which silently biases every interior interface node toward
  one cell. Accumulate and divide by the count.
* Neumann side: assemble `+ integral(g * v) ds` over the interface FACES only,
  selected by the boundary id you set from the interface coordinate. deal.II
  will happily integrate over every boundary face if you do not restrict it.''')


_BACKENDS.update({"dune": _dune, "dune-fem": _dune, "dunefem": _dune,
                  "dealii": _dealii, "deal.ii": _dealii})
