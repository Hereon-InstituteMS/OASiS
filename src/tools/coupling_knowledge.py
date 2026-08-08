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
  * write `exports.json` LAST, only after the solve succeeded, AND EXIT 0. The
    driver requires both: a missing file ends the run, and so does a non-zero
    exit code even when a complete `exports.json` is sitting there — a solver
    that diverges commonly writes its last iterate and then aborts, and
    coupling on that output produced a converged-looking result built on a
    crashed participant. A TRUNCATED file is caught separately, as bad JSON.
    So do not use a non-zero exit to signal anything but failure.

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
  * `data_files` — a list of ABSOLUTE paths to files your solver opens (mesh,
    species, surface, config). They are copied into `work_dir` once, before the
    iteration starts, and a path that does not exist is a LOUD setup error
    rather than a solver dying mid-iteration on 'Cannot open ...'. Declaring a
    file here also binds it into the critic review, so rewriting it after the
    review invalidates the approval. You may still copy files in yourself;
    `data_files` is the supported way and the one that gets both of those.
  * `theta` must be in (0, 1]; `accelerator` must be exactly "constant" or
    "aitken". Both are rejected with an error message if not.
  * `noise_replicates` — ONLY when a participant is a Monte-Carlo / sampled
    estimator (DSMC, a stochastic solver, anything whose answer to the same
    question differs run to run). Set it to 2 or more and the driver runs every
    participant that many times on the SAME imports BEFORE iterating, measures
    the residual between independent replicates, and judges convergence against
    max(tol, that floor). See section 4a. It costs
    `noise_replicates` extra solves per participant. Leave it at 0 for
    deterministic solvers — where it does nothing anyway.
  * `noise_floor` — declare a floor you established yourself instead of (or on
    top of) measuring one. `noise_block` — how many consecutive residuals must
    AVERAGE below the criterion before the run stops; only in effect when a
    non-zero floor is.

Returns JSON with `converged`, `iterations`, `residual`, `history`, per-
participant `exports`, a `validation` block, and OASiS's `verification` /
`trustworthy_result` verdict — plus `noise_floor`, `tol_effective`,
`stopped_at_noise_floor` and `noise_notes` whenever a floor was in play. Two
things about it:
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
the field magnitude. EVALUATE IT FOR YOUR OWN theta AND tol; it is three
keystrokes and there is no table to look up:

    theta=0.5, tol=1e-8, d0=1  ->  log(1e-8)/log(0.5)  ~  27 iterations

That is a value of the formula, not something anybody observed, and it is an
ORDER-OF-MAGNITUDE SIZING GUIDE rather than a lower bound — a field with a large
offset, temperatures around 300 K whose interface value is wrong by a few K,
starts at a d0 of a few percent and gets there in fewer. What matters is the
shape: the count grows without limit as theta shrinks, so give max_iter generous
headroom, because under-budgeting looks exactly like a physics failure — a run
that stops at max_iter=20 with a small theta never had a chance. (With
accelerator="aitken" theta moves, so the rate moves with it, but the same
mechanism is there.)

`accelerator`: **the default, "aitken", is also the safer one — reach for
"constant" to DIAGNOSE, not as your first choice.**
  * "aitken" — ONE theta for the whole interface state, recomputed every
    iteration, starting from the theta you pass and clamped into [0.05, 1.0].
    There is no per-participant and no per-field theta: Aitken's derivation is
    for a single sequence extrapolated from the composite fixed-point map, and
    giving each participant its own theta relaxes the two halves of one coupled
    system by different amounts — on a Dirichlet-Neumann split that drove the
    two thetas apart to opposite clamps and made the iteration diverge where a
    constant theta converged. The two fallback paths inside the update — the
    first iteration, where there is no previous residual, and a degenerate
    denominator — hold the previous theta, clamped into the same [0.05, 1.0].
    THIS IS THE DEFAULT AND YOU SHOULD NORMALLY KEEP IT. Measured across conductance ratios rho from 1/4 to 9 and theta from 0.1
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

WHAT "AITKEN" MEANS HERE. The update is the classical Aitken dynamic-relaxation
recurrence on the global interface residual r_k = G(x_k) - x_k:

    theta_k = -theta_{k-1} * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2

with the result clamped into [0.05, 1.0]. It IS given the previous RESIDUAL,
which is the operand the derivation calls for — an earlier version of this
driver handed it the previous RAW EXPORT instead, which makes theta an
arbitrary number inside the clamp with no relation to the iteration, and the
only symptom was slower convergence on correct setups. Two things still do not
follow from the textbook derivation: the clamp is not part of it, and the
convergence RATE is not guaranteed, because this is a vector fixed point
extrapolated by one scalar. Everything stated above is what this implementation
was measured to do. If you need a specific acceleration scheme rather than this
one, drive the coupling with `couple_precice` and a `serial-implicit` scheme.

### Choosing theta — this maps to the real `theta` parameter

For a two-participant Dirichlet-Neumann split the iteration is linear in the
interface unknowns, and the driver's Jacobi+relaxation loop has amplification
factor  sqrt((1-theta)^2 + rho*theta^2), where

    rho = (interface conductance of the DIRICHLET-side subdomain)
          / (interface conductance of the NEUMANN-side subdomain)

and "interface conductance" is material coefficient / distance from the
interface to that subdomain's own outer boundary (k/d for conduction, EA/L for
a bar, and so on — it is the subdomain's stiffness as seen from the interface).
That gives four facts you can act on:

  * the best theta is  **theta ~ 1 / (1 + rho)** — 0.5 when the two sides are
    balanced, smaller when the Dirichlet side is the stiffer one, larger when it
    is the softer one. This is the one number in this section worth computing
    before you run anything: swept over rho from 1/4 to 9, the fastest constant
    theta was 1/(1+rho) at EVERY ratio;
  * the amplification factor above is below 1 exactly when
    **theta < 2 / (1 + rho)**, which is TWICE the optimum. So what converges is
    an INTERVAL, not a point: everything below double the value you just
    computed. The interval NARROWS as the split gets more unbalanced, which is
    why a coarse sweep at a strongly unbalanced ratio can turn up only one value
    that works — the interval got smaller than the spacing, not the method.
    Above the limit the iteration diverges, and halving theta always brings you
    back inside it;
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

### 4a. A STOCHASTIC PARTICIPANT — the residual has a floor and `tol` cannot cross it

If ANY participant is a Monte-Carlo or otherwise sampled estimator — a DSMC
code, a stochastic solver, anything that answers the same question slightly
differently each time it is asked — then read this before you set `tol`.

The residual is the CHANGE in the export vector between iterations. A sampled
participant changes its export every run whether or not the physics moved, so
the residual cannot fall below the size of that scatter, however well the
coupling has converged. A `tol` underneath the floor is unreachable BY
CONSTRUCTION: the run always ends "did not converge", on a coupling that is
right. Sizing `tol` by guesswork does not fix it — too tight and every run fails,
too loose and you have declared victory at a number you cannot defend.

`noise_replicates` MEASURES the floor instead. The driver runs every participant
that many times on the SAME imports and evaluates its own residual expression
across the replicates, so the floor is the residual a perfectly converged run
would still report. Convergence is judged against max(tol, floor), over a block
mean of the last few residuals so that one lucky dip into the noise cannot end
the run. The result carries `noise_floor` and `stopped_at_noise_floor`.

USE 4 OR MORE, not the minimum of 2. The floor is itself an estimate and a small
one is a bad estimate: three replicates give three samples, and the same
coupling can measure a floor several times larger or smaller from one attempt to
the next purely from that scatter. The driver adds a note below six samples.

THE MEASUREMENT HAPPENS TWICE, and the two are not the same number. Before the
loop there is no previous relaxed vector to compare against, so all that can be
measured is the scatter BETWEEN independent answers — a LOWER BOUND, because the
loop compares against a lagged relaxed average carrying its own accumulated
noise, and because noise that has propagated through a partner over earlier
iterations is not in it. That bound is used to avoid a pointless long run. If the
loop still finishes un-converged, the floor is re-measured with the participants
in their FINAL state, against the very vector the loop compares to — that one is
the residual the loop actually reports, measured — and the verdict is re-judged.
The re-measurement is paid only on failure.

THREE THINGS THAT FOLLOW, and they are the whole discipline:
  * READ `noise_floor` BEFORE APPLYING ANY TOLERANCE TO THE RESULT. A grading,
    acceptance or agreement tolerance tighter than the floor is measuring the
    sampler, not the coupling;
  * a floor of EXACTLY ZERO from a Monte-Carlo participant means its SEED IS
    FIXED. The run is repeatable, not converged, and a residual that falls under
    a fixed seed is no evidence it would fall under another. `noise_notes` says
    so explicitly. Vary the seed between runs if you want the floor to mean
    anything;
  * the floor answers "can this sampler still see the iteration moving", NOT
    "has the iteration finished". A physics drift smaller per iteration than the
    sampling noise but accumulating over many of them is invisible to it.
    Increase the sampling (which lowers the floor) if you need to see smaller
    steps.

For a deterministic solver the replicates come back bit-identical, the floor is
zero, max(tol, 0) is tol and nothing about the run changes. There is no reason
to set it there, and no harm if you do.

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
| Kratos     | yes       | yes     | a Python script          | Neumann side coupled to the real 4C binary against an analytic reference; Dirichlet side against FEniCSx — but in ITS OWN interpreter, see below |
| SPARTA     | yes       | not natively | Python wrapper + deck | coupled to a thermal shell and CONVERGED once the residual is judged against its measured Monte-Carlo noise floor; no flux BC exists, only an indirect radiative-equilibrium route |

Every "yes" above means a real two-code coupling was run in that role on THIS
install and CONVERGED; it is not copied from a tool docstring. Two rows carry
conditions you must know before planning around them:

  * KRATOS runs in ITS OWN INTERPRETER, not the one OASiS itself runs in. The
    coupling to 4C was driven with the Kratos participant under a separate
    Python — which is exactly what the `command` field is for, so this is a
    configuration fact and not a limitation. A core-only Kratos has no thermal
    element, so `import KratosMultiphysics.ConvectionDiffusionApplication` in
    THAT interpreter is the thing to test first: if it fails, the conduction
    participant cannot run there whatever the table says. Note also that
    `discover(query='list')` probes Kratos in OASiS's own interpreter, so it can
    report Kratos unavailable on a machine where the coupling works.
  * SPARTA is a Monte-Carlo code, so its residual has a floor and a `tol` under
    that floor can never be met. That used to make every SPARTA coupling report
    FAILURE even when the physics agreed. Pass `noise_replicates=5` and the
    driver measures the floor and judges against it — see section 4a. Without
    it, expect "did not converge" on a correct run.

Note in particular that the DEPRECATED `coupled_solve` docstring lists 4C on the
Neumann side only — that limitation belongs to its own fixed generators, not
to 4C.

EVERY ROW ABOVE EXCHANGES ONE SCALAR. A VECTOR interface — displacement and
traction, velocity and force, anything with components — is a separate claim
with a separate table, three pairs behind it (FEniCSx <-> scikit-fem,
FEniCSx <-> deal.II, NGSolve <-> scikit-fem), its own participant scripts
(`participant_*_elastic.py`) and five rules that a scalar coupling does not
need. Read section 6a before writing one; four of those five converge cleanly
to the wrong answer if you get them wrong.

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

_VECTOR = '''\
## 6a. VECTOR INTERFACES — displacement/traction, velocity/force, any n_comp

Everything above exchanges ONE SCALAR. A vector interface is the primitive
under TSI, FSI and contact, and the driver already carries it: `values` and
`normal_fluxes` may be (N,) OR (N, n_comp), the residual is taken per
component, and the conservation check balances each component on its own.
Five things change on your side, and four of them converge beautifully to the
wrong answer if you get them wrong.

**1. MAP EACH COMPONENT SEPARATELY.** The driver does no interpolation — the
non-matching interface is the participant's job — and `np.interp` takes only a
1-D `fp`. One call over a flattened (N, 2) array interleaves the components:
right length, clean convergence, every number wrong. Loop over components.

**2. THE TRACTION SIGN.** Export

        q_out = -(sigma . n_own)                  n_own = your outward normal

the SAME convention the scalar participants use for heat (q_out = -k dT/dn).
The two sides' exports then CANCEL componentwise — which is what makes the
balance check a conservation statement — and the NEUMANN side applies the
partner's numbers UNCHANGED (`L += inner(g, v) * ds`), because the natural
boundary term of the elasticity weak form is +(sigma . n_own) . v. Exporting
the raw traction instead flips the sign the Neumann side applies.

**3. THE INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.** A
node where the interface meets a constrained outer face is a constrained node
in the un-split problem, so it must stay constrained in BOTH subproblems.
Handing it to the interface leaves it free on the Neumann side: that
subproblem is still well posed, still converges, and lands a few percent off.
Measured on the shipped participants — 4.7% in the interface displacement and
28% in the interface traction, on a run whose residual reached 1e-10 and whose
interface balanced.

**4. theta = 1/(1 + MAX_c rho_c), not 1/(1 + rho).** rho is a ratio PER
COMPONENT, and the two differ unless the subdomains share a Poisson ratio
(M/mu = 2(1-nu)/(1-2nu) depends on nu alone). The driver's Jacobi amplification
for component c is sqrt((1-theta)^2 + rho_c theta^2), below one only while
theta < 2/(1+rho_c), so the LARGEST rho binds. Whenever rho_max > 1 + 2 rho_min,
theta from the smaller one DIVERGES on the other component while the first
settles — and the global residual reports only "did not converge".

**5. A FREE (traction) BOUNDARY NEXT TO THE INTERFACE OPENS THE SPECTRUM.**
Bending compliance scales as L^3 against L^1 for the axial one, so a subdomain
with free faces is far softer in some interface modes than a 1-D conductance
estimate suggests. Measured: the same split had a Steklov spectrum of
[0.049, 2.07] with free faces and [0.25, 0.64] with constrained ones, and only
the second is mesh-independent. Estimate rho, then verify by running.

### What has been established, and by which fixture

| Pair                   | What was run | Evidence |
|------------------------|--------------|----------|
| FEniCSx <-> scikit-fem | 3 arrangements covering all four (role, position) combinations | fixture coupling/vector_pair_fenics_skfem |
| FEniCSx <-> deal.II    | 2 arrangements; each code in both positions and both roles | fixture coupling/vector_pair_fenics_dealii |
| NGSolve <-> scikit-fem | 3 arrangements covering all four (role, position) combinations | fixture coupling/vector_pair_ngsolve_skfem |

Read that column literally. "All four combinations" means the four
(role, position) pairs were each exercised across the arrangements — NOT that
every backend was run in all four on its own. Each backend in each pair is
proven in three of the four; the fourth for that backend is covered by its
partner.

Each runs plane-strain elasticity split by a straight interface, exchanging a
2-component displacement and a 2-component traction on non-matching interface
meshes, and asserts componentwise: continuity of both displacement components,
equilibrium of both traction components, per-component conservation, and
agreement with BOTH a closed form and an un-split monolithic solve. Shipped
participant scripts: `participant_{fenics,skfem,ngsolve,dealii}_elastic.py`
(the deal.II one needs `elast_iface_dealii` built from the same CMake tree).
No other backend has a vector participant yet — 4C, DUNE-fem, FEBio, Kratos
and SPARTA are scalar-only here, and that is an absence of evidence rather
than a demonstrated inability.

### The one thing that does NOT hold

**The exported TRACTION is not trustworthy near the ends of the interface**
when the interface meets a constrained boundary. That corner is a
Dirichlet-Neumann corner for the Neumann subproblem, which carries a stress
singularity the monolithic problem does not have. Measured over a 4x
refinement: the coupled displacement converges (1.22e-02 -> 2.34e-03 against
the monolithic solve) while the exported traction at the interface end gets
WORSE (2.11x -> 2.51x the true value). Refinement does not fix it, because it
is not a discretisation error. Use the displacement channel for the answer,
check traction equilibrium on the interface INTERIOR, and do not read the
end-node traction as a result. Fixture:
coupling/vector_traction_recovery_at_the_interface_ends.
'''

_SIDES = (_SIDES_TABLE.replace("## WHICH SIDE", "## 6. WHICH SIDE", 1)
          + "\n" + _VECTOR)


def coupling_sides_table() -> str:
    """The side table on its own, for discover(query='coupling')."""
    return _SIDES_TABLE

# ══════════════════════════════════════════════════════════════════════════
# FAILURE MODES — ONE source, rendered two ways
# ══════════════════════════════════════════════════════════════════════════
#
# These rows were a hand-written markdown table and nothing else, which cost
# them their only route in: every other family of knowledge in OASiS ships its
# symptoms in the `[Category] ... Signal: ...` shape, and `knowledge(signal=...)`
# is how a post-execution critic gets from a symptom to the entry that explains
# it. Coupling had no pitfall list at all, so a coupling failure could not be
# routed to the very rows that name it — the reader had to already know to ask
# for `topic='coupling'` and then read 40 kB.
#
# So the rows live HERE, once, and both surfaces are generated from them:
# the table in the core payload, and a searchable pitfall list. A row cannot
# appear in one and be missing from the other.
#
# `observations` is the part that makes the search work at all. The `seen`
# string is what the TOOL prints; a person arriving with a problem types what
# THEY saw, in their own words, with no mechanism in it — "the two sides stopped
# agreeing", "it converged but the answer is wrong". A substring match against
# tool output never finds those. Each row therefore carries plain-language
# phrasings of the same symptom, and the matcher scores against them too.

_FAILURE_ROWS: list[dict] = [
    {
        "cat": "Handshake",
        "seen": "`participant X wrote no exports.json (rc=...)`",
        "means": "your script died. The stderr tail is in the message — read "
                 "it. Run the script standalone in its work_dir first.",
        "observations": ["the participant produced no output file",
                         "one side crashed", "my script exited and nothing "
                         "was written", "the run stops on the first iteration",
                         "the run aborted partway through with an mpi error",
                         "one program never starts at all",
                         "the handshake never completes",
                         "a participant died and took the coupling with it"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X bad exports.json`",
        "means": "malformed JSON, a TRUNCATED file, or a missing required key. "
                 "All three of `field_name`, `coordinates`, `values` must be "
                 "present.",
        "observations": ["the export file cannot be read",
                         "the driver rejects the file my script wrote",
                         "a key is missing from the exported data",
                         "exports.json is missing a key the driver requires",
                         "my exported json has different keys than expected"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X changed its export size from N to M`",
        "means": "that participant exported a different number of points (or "
                 "dropped `normal_fluxes`) between iterations. Usually a mesh "
                 "or interface-detection step that depends on the imported "
                 "data. Fix the participant; the driver cannot relax a "
                 "changing vector.",
        "observations": ["the number of interface points changes between "
                         "iterations", "one side sometimes exports the flux "
                         "and sometimes does not", "the mesh is rebuilt every "
                         "iteration"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X timed out`",
        "means": "raise `timeout` in the participant spec, or coarsen that "
                 "side's mesh.",
        "observations": ["one side takes far too long", "the coupling hangs "
                         "on one participant", "the solver is stuck and never "
                         "finishes", "it hangs and I have to kill it",
                         "nothing happens for a very long time",
                         "the run deadlocks at startup",
                         "one participant waits forever for the other",
                         "it just hangs with no error"],
    },
    {
        "cat": "Relaxation",
        "seen": "`did not converge to tol=... in N iters`",
        "means": "in order: is max_iter above the (1-theta) floor in section 4; "
                 "is theta right for this conductance ratio; would swapping "
                 "which side is Dirichlet help; do the two decks actually agree "
                 "on units and material. NOT a result. If any participant is a "
                 "Monte-Carlo / sampled estimator, see the stochastic entry "
                 "below before touching any of that.",
        "observations": ["it ran out of iterations", "the residual never got "
                         "small enough", "the coupling will not finish",
                         "it is stuck and will not finish",
                         "it stops at the iteration limit",
                         "it never reaches the tolerance I asked for",
                         "convergence is painfully slow",
                         "the sub-iterations hit the maximum count every step",
                         "it takes far too many outer iterations"],
    },
    {
        "cat": "Relaxation",
        "seen": "residual stuck at O(1), oscillating",
        "means": "theta too large, or a SIGN error — the Neumann side is "
                 "pushing the flux the wrong way.",
        "observations": ["the residual goes up and down and never settles",
                         "the interface value flips back and forth",
                         "the answer bounces between two values",
                         "the interface oscillates in a checkerboard pattern",
                         "the transferred load has the wrong sign",
                         "the traction looks mirrored",
                         "the structure moves the opposite way to the load",
                         "the flux is pushed the wrong way",
                         "it ping-pongs between the participants and makes no "
                         "progress",
                         "the two sides chase each other and never settle"],
    },
    {
        "cat": "Relaxation",
        "seen": "residual GROWING, values exploding",
        "means": "theta above the stability limit for this conductance ratio. "
                 "Halve it. See section 4.",
        "observations": ["the numbers keep getting bigger",
                         "the solution blows up", "the interface temperature "
                         "runs away", "the temperatures are becoming enormous",
                         "the values are growing without limit",
                         "huge unphysical numbers", "the answer diverges",
                         "it is unstable",
                         "it will not stabilise even with heavy "
                         "under-relaxation",
                         "the amplitude grows every step until it crashes",
                         "the displacement grows every coupling iteration "
                         "until it overflows",
                         "it goes unstable when one side is much stiffer or "
                         "much more conductive than the other"],
    },
    {
        "cat": "Conservation",
        "seen": "converged, but `Interface flux NOT balanced`",
        "means": "most often both sides exported `normal_fluxes` with the same "
                 "sign (section 5). Otherwise: different units on the two "
                 "sides, or one side exporting an INTEGRATED flux where the "
                 "other exports a DENSITY.",
        "observations": ["the two sides stopped agreeing",
                         "the two sides do not agree about the flux",
                         "each side reports a different amount of heat",
                         "energy is not conserved across the interface",
                         "what leaves one subdomain does not enter the other",
                         "the forces I send over do not add up to the same "
                         "total on the other side",
                         "the load transfer is not conservative",
                         "the total force before and after mapping differs",
                         "the energy at the interface grows every step",
                         "there is a flux imbalance at the interface",
                         "the lift one code computes is not the lift the "
                         "other one feels",
                         "the load one side applies is not the load the other "
                         "side receives",
                         "the transfer is not conservative"],
    },
    {
        "cat": "Conservation",
        "seen": "`Interface conservation was NOT CHECKED`",
        "means": "one side exported no `normal_fluxes`, so nothing verifies "
                 "conservation. The run is not wrong, it is unguarded. Export "
                 "the flux on both sides.",
        "observations": ["nothing checked the conservation",
                         "the result says the balance was not checked",
                         "there is no flux in my export"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`NOT COUPLED: no participant lists imports_from`",
        "means": "nobody was given a partner's name, so nothing was exchanged. "
                 "The run is not a coupling.",
        "observations": ["nothing was exchanged between the codes",
                         "the two runs are independent of each other",
                         "nothing seems to be passed between the solvers",
                         "no data is transferred between the participants",
                         "neither side reacts to the other"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`ONE-WAY: participant(s) X list no imports_from`",
        "means": "X never sees its partners. A note, not a failure — a "
                 "master/slave coupling really does look like this. If you "
                 "meant a two-way coupling, that is the bug.",
        "observations": ["only one side reacts to the other",
                         "one participant never changes",
                         "the coupling only goes one way",
                         "only one of the two codes actually moved",
                         "one program does nothing while the other updates",
                         "the feedback from one side is missing"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`NOT COUPLED: no participant's export changed`",
        "means": "converged at iteration 2 with an exactly zero residual. Both "
                 "participants returned their initial guess: they are not "
                 "reading `imports.json`, or they are reading it under the "
                 "wrong partner name. THIS IS THE MOST CONVINCING WRONG RESULT "
                 "THE TOOL CAN PRODUCE — it looks like an instant, perfect "
                 "convergence.",
        "observations": ["it converged immediately",
                         "it converged on the second iteration",
                         "the residual was exactly zero",
                         "it converged far too fast",
                         "it converged on the very first step which seems too "
                         "good to be true",
                         "the answer is just my initial guess",
                         "the export did not change between iterations",
                         "the exported numbers are identical every iteration",
                         "the log prints the same number every iteration",
                         "the second program receives nothing",
                         "the field I send is not arriving"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`non-finite export values at iter N` (a warning)",
        "means": "your solve diverged; the run CONTINUES to max_iter, so look "
                 "for this in `validation` rather than expecting it to stop. "
                 "Usually a subdomain with no essential BC anywhere, which is "
                 "singular.",
        "observations": ["I am getting nan in the output",
                         "nan appeared in the results",
                         "the values became infinite",
                         "inf and nan in the exported field",
                         "one subdomain has no boundary condition",
                         "the subdomain problem is singular",
                         "it crashed with nan after a few coupling steps",
                         "the displacements went to infinity"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "converged to a plausible but wrong answer",
        "means": "check `n_points` from each side on iteration 1: an empty or "
                 "wrongly located interface set gives a well-behaved solution "
                 "of the wrong problem. Then check that both sides use the same "
                 "units and the same interface coordinate.",
        "observations": ["it converged but the answer is wrong",
                         "everything looks fine but the number is wrong",
                         "the result is plausible and still wrong",
                         "the answer looks reasonable but does not match my "
                         "hand calculation",
                         "the number disagrees with the analytical solution",
                         "the two subdomains may not be touching",
                         "the interface is in the wrong place",
                         "there is a visible gap between the two subdomains",
                         "my interface meshes do not line up exactly",
                         "the two interfaces are misaligned",
                         "the two sides have different numbers of nodes on "
                         "the interface",
                         "the transferred field is spiky at the boundary "
                         "between the two meshes",
                         "the two surfaces do not coincide",
                         "each code is fine on its own but the coupled "
                         "outcome is nonsense"],
    },
    # ── rows below are not in the original table. Each is behind a fixture. ──
    {
        "cat": "Silent-wrong",
        "seen": "converged, balanced, EMPTY `validation`, and still wrong",
        "means": "a UNIT MISMATCH between the two decks passes every check the "
                 "tool has. Conservation is a property of the fixed point and "
                 "holds whatever units the boundary data is in, so the balance "
                 "closes, the residual falls and the validation block stays "
                 "empty while the interface value is out by tens of percent. "
                 "NOTHING INSIDE `couple` CAN SEE THIS. The only detector is an "
                 "independently computed answer for the same quantity — a "
                 "monolithic re-solve of the un-split problem, or a closed form "
                 "— compared against the converged interface state.",
        "observations": ["it converged but the answer is wrong",
                         "every check passed and the number is still wrong",
                         "the answer does not match my hand calculation or a "
                         "reference solution",
                         "the validation block is empty and I do not trust it",
                         "one deck is in celsius and the other in kelvin",
                         "the two sides use different units",
                         "the value is off by a factor of a thousand",
                         "it came out a thousand times too large",
                         "the field arrives scaled by some random factor",
                         "the magnitude is wrong by a clean power of ten",
                         "the coupled outcome is offset by a constant "
                         "everywhere",
                         "everything is shifted by the same amount",
                         "the deformation is about half of what the "
                         "experiment shows",
                         "the outcome disagrees with a published measurement",
                         "the residual goes down but the physics is wrong",
                         "one field is transferred and the receiving code "
                         "produces garbage"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "a wrong material on BOTH sides moves the flux and NOT the "
                "interface value",
        "means": "if both subdomains carry the same wrong factor on their "
                 "material coefficient, the CONDUCTANCE RATIO is unchanged, so "
                 "the interface value lands exactly where it should and only "
                 "the flux is wrong. Checking the primary interface quantity — "
                 "the temperature, the displacement — CANNOT detect it, and "
                 "neither can the balance. Check the FLUX against an "
                 "independent answer as well as the value.",
        "observations": ["the interface temperature is right but the heat flow "
                         "is wrong", "the value is right and the flux is not",
                         "the material may be wrong on both sides"],
    },
    {
        "cat": "Stochastic",
        "seen": "`did not converge` with the residual FLAT rather than falling",
        "means": "a Monte-Carlo participant (DSMC, any sampled estimator) "
                 "returns a slightly different export every time it is asked "
                 "the same question, so the residual has a FLOOR at the size of "
                 "that scatter and a `tol` underneath it can never be met. The "
                 "run is right and the verdict is useless. Pass "
                 "`noise_replicates=5` (four or more \u2014 the floor is itself an "
                 "estimate and three samples is a bad one): `couple` then runs each "
                 "participant that many times on the SAME imports, measures the "
                 "residual between independent replicates, and judges "
                 "convergence against max(tol, that floor) over a block mean. "
                 "It returns `noise_floor`, and ANY tolerance you or a grader "
                 "then apply must be at least that floor. A floor measured as "
                 "exactly zero on a Monte-Carlo code means the SEED IS FIXED: "
                 "the run is repeatable, not converged.",
        "observations": ["the residual stops falling and stays there",
                         "the residual plateaus",
                         "it never converges no matter how many iterations",
                         "one side is a particle code",
                         "my solver gives a slightly different answer every "
                         "time", "monte carlo noise", "the physics looks right "
                         "but it reports failure",
                         "a rerun gives a different number",
                         "repeating the identical run does not repeat"],
    },
]


def _failure_table() -> str:
    rows = "\n".join(f"| {r['seen']} | {r['means']} |" for r in _FAILURE_ROWS)
    return (
        "| What you see | Cause |\n"
        "|--------------|-------|\n" + rows)


def coupling_pitfall_entries() -> list[str]:
    """The failure rows in the `[Category] ... Signal: ...` shape the rest of
    the catalog uses, so `knowledge(topic='coupling', signal=...)` can route a
    symptom to one. Generated from `_FAILURE_ROWS`, never maintained twice."""
    out = []
    for r in _FAILURE_ROWS:
        obs = "; ".join(r["observations"])
        out.append(f"[Coupling][{r['cat']}] {r['means']} "
                   f"Signal: {r['seen']} — in plain words: {obs}.")
    return out


# ══════════════════════════════════════════════════════════════════════════
# SEARCH — routing a described SYMPTOM to the entry that explains it
# ══════════════════════════════════════════════════════════════════════════
#
# The query is what a person SAW; the entries are what the tool PRINTS and what
# the author of the entries thought the symptom sounded like. Those two
# vocabularies overlap far less than either author expects, and a self-check by
# the person who wrote the entries measures almost nothing — they share a
# vocabulary with themselves by construction.
#
# THIS SCORER IS THE SECOND ONE. The first was a plain token overlap divided by
# the query length, and a hostile third-party pass with a hundred queries it
# wrote before reading anything measured it at ~62% recall and ~52% top-1. Three
# faults, all structural, all fixed here:
#
#   1. DIVIDING BY THE QUERY LENGTH PENALISED DETAIL. `nan` found the right
#      entry; `solver crashed with nan after the third coupling step` found
#      nothing, because one match out of seven tokens fell under the threshold.
#      That is backwards: a user who types more has told you more. The score is
#      now driven by the BEST evidence in the query (`core`), with coverage only
#      breaking ties, so extra words can never remove a hit.
#   2. EVERY TOKEN COUNTED THE SAME, so `interface`, `iteration`, `step` and
#      `mesh` — which appear in most entries — decided the ranking, and the
#      wordiest entry became a magnet that topped queries about energy balance
#      and about flux imbalance. Tokens are IDF-weighted against the entry set
#      now, so a word that appears everywhere carries nothing and `energy` beats
#      `interface`.
#   3. THE ENTRIES SPEAK THERMAL AND HALF THE USERS SPEAK MECHANICS. Every
#      force / load / traction / pressure / mapping phrasing missed, including
#      the two highest-value entries in the table. Hence SYNONYMS, plus a much
#      wider set of `observations` on the rows themselves.
#
# A fourth, cheap: a stem that is not in the vocabulary is retried as a
# five-character prefix, so `oscilating` still reaches `oscillating`.

_STOP = {
    "the", "and", "but", "for", "with", "that", "this", "was", "were", "are",
    "its", "it", "is", "not", "you", "your", "have", "has", "had", "did",
    "does", "from", "into", "out", "off", "get", "got", "when", "what", "why",
    "how", "all", "any", "one", "two", "both", "same", "each", "still", "just",
    "only", "very", "some", "there", "then", "than", "them", "they", "their",
    "which", "who", "will", "would", "can", "could", "should", "about", "run",
    "runs", "ran", "code", "codes", "side", "sides", "thing", "things", "make",
    "made", "see", "saw", "look", "looks", "like", "way", "ways", "use", "used",
    "problem", "result", "results", "answer", "answers", "give", "gives",
    "solver", "solvers", "simulation", "case", "model", "keep", "keeps",
    # Generic English that a blind tester caught ACTING AS THE DECIDING TOKEN:
    # `much` alone returned the stability-limit row (it appears in "much
    # stiffer"), `slightly` returned the Monte-Carlo row, `second` collided
    # "the second program" with "the second iteration", and `condition`
    # returned the singular-subdomain row from the words "boundary condition".
    # An intensifier or an ordinal is not evidence about a failure mode.
    "much", "many", "slightly", "little", "second", "third", "condition",
    "seem", "seems", "actually", "really", "quite", "rather", "even",
}

# Query vocabulary -> entry vocabulary. Every group below was written from a
# recorded MISS, not from imagination: these are the words engineers actually
# used for symptoms the table already covered.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # a structural engineer's word for the thing the table calls a flux
    "force": ("flux", "conserv"), "forc": ("flux", "conserv"),
    "load": ("flux",), "traction": ("flux",), "pressure": ("flux",),
    "stress": ("flux",), "heat": ("flux",),
    "transfer": ("flux", "exchang"), "mapping": ("interpolat", "point"),
    "conservative": ("conserv", "balanc"), "conservation": ("conserv", "balanc"),
    "imbalance": ("balanc", "conserv"), "sum": ("balanc", "conserv"),
    "total": ("balanc", "conserv"),
    # runaway
    "explod": ("blow", "grow", "stabil"), "explode": ("blow", "grow", "stabil"),
    "explosion": ("blow", "grow"), "infinity": ("infinit",),
    "overflow": ("infinit", "grow"), "unstable": ("stabil", "blow"),
    "instabil": ("stabil", "blow"), "unstabl": ("stabil", "blow"),
    "runaway": ("grow", "blow"), "diverg": ("blow", "grow"),
    "enormou": ("enormou", "huge", "grow"), "huge": ("huge", "grow"),
    "bigger": ("grow",), "amplitude": ("oscillat", "grow"),
    "checkerboard": ("oscillat",), "bounc": ("oscillat", "flip"),
    "flip": ("flip", "oscillat"), "wobbl": ("oscillat",),
    # a stalled process
    "deadlock": ("hang", "stuck"), "freeze": ("hang", "stuck"),
    "frozen": ("hang", "stuck"), "wait": ("hang", "stuck"),
    "stall": ("hang", "stuck", "plateau"), "hung": ("hang",),
    # geometry that does not meet
    "gap": ("touch", "meet"), "misalign": ("touch", "meet", "place"),
    "align": ("touch", "meet", "place"), "overlap": ("touch", "meet"),
    "coincid": ("touch", "meet"),
    # magnitude errors, which is what a unit mismatch looks like
    "factor": ("unit", "celsiu", "kelvin"), "scale": ("unit",),
    "scaled": ("unit",), "thousand": ("unit",), "million": ("unit",),
    "magnitude": ("unit",), "convert": ("unit",), "conversion": ("unit",),
    # sign
    "mirror": ("sign", "opposit"), "opposit": ("sign", "opposit"),
    "revers": ("sign", "opposit"), "backward": ("sign", "opposit"),
    "inward": ("sign",), "outward": ("sign",), "negativ": ("sign",),
    # the iteration
    "subiteration": ("iteration",), "substep": ("iteration",),
    "sweep": ("iteration",), "relaxation": ("relaxation", "theta"),
    "underrelaxation": ("relaxation", "theta"),
    "stochast": ("carlo", "sampl", "nois"), "random": ("carlo", "sampl", "nois"),
    "particl": ("carlo", "sampl", "nois"), "dsmc": ("carlo", "sampl", "nois"),
    "plateau": ("plateau", "stop", "fall"),
    "nonsens": ("wrong",), "garbage": ("wrong",), "rubbish": ("wrong",),
    "bogus": ("wrong",), "implausibl": ("wrong",),
}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _toks(s: str) -> list[str]:
    import re as _re
    return [_stem(w) for w in _re.findall(r"[a-zA-Z_]{3,}", s.lower())
            if w not in _STOP]


def _hay(r: dict) -> str:
    """Everything an entry can be matched against — INCLUDING its category.
    Leaving the category out meant that typing the word the row is filed under,
    `conservation` or `relaxation`, matched nothing at all."""
    return " ".join([r["cat"], r["seen"], r["means"], *r["observations"]])


_SEARCH_INDEX: dict = {}


def _search_index() -> dict:
    """Entry token sets plus an IDF weight per token, built once."""
    if _SEARCH_INDEX:
        return _SEARCH_INDEX
    import math
    sets = [set(_toks(_hay(r))) for r in _FAILURE_ROWS]
    n = len(sets)
    df: dict[str, int] = {}
    for s in sets:
        for tok in s:
            df[tok] = df.get(tok, 0) + 1
    scale = math.log(n + 1)
    idf = {tok: math.log((n + 1) / (d + 1)) / scale for tok, d in df.items()}
    prefixes: dict[str, set[str]] = {}
    for tok in df:
        if len(tok) >= 6:
            prefixes.setdefault(tok[:5], set()).add(tok)
    _SEARCH_INDEX.update({"sets": sets, "idf": idf, "prefixes": prefixes,
                          "vocab": set(df)})
    return _SEARCH_INDEX


# A prefix rescue is a GUESS, so it is worth less than a real match, and it is
# never allowed to be the ONLY evidence. `install` shares five characters with
# `instant`, so a conda question was answered with "it converged at iteration 2
# with an exactly zero residual" — a confidently wrong answer, which is the
# worst thing this search can produce. So a prefix match is discounted AND
# cannot supply `core`: it can lift an entry another token already reached, and
# nothing more. That is enough for the case it exists for, a typo inside a
# sentence whose other words are right.
_PREFIX_WEIGHT = 0.8

# An entry must match at least one query token this informative. Coverage alone
# must never be enough: `wrong`, `converging`, `interface` and `mesh` appear
# across most of the table, and an entry selected by those has been selected by
# nothing. This is what stops a one-word query from returning five rows.
#
# Both constants were chosen by SWEEPING them against two independent blind
# query sets — one from each hostile tester — and taking the point where recall
# on the first set is unchanged and refusals are highest. They trade precision
# against recall directly and cannot be tuned to fix both.
_CORE_MIN = 0.55

# The verbatim-containment bonuses are strong evidence and were firing on any
# query short enough to be a substring of somebody's observation — `wrong` is
# inside eight of them. They need a phrase, not a word.
_PHRASE_MIN_CHARS = 12

# WHAT AN UNRECOGNISED WORD COSTS. A word the table has never heard of used to
# be scored at the MAXIMUM weight in the coverage denominator, so every extra
# word an engineer typed pushed the right entry down — `handshake` returned all
# four handshake rows and `handshake never completes` returned nothing. That is
# the exact fault this scorer was rewritten to remove, reintroduced through the
# denominator, and a second blind tester found it by typing sentences instead of
# phrases. An unknown word is weak evidence of anything, so it now costs little.
_UNKNOWN_W = 0.12


def _synonyms_for(tok: str) -> tuple[str, ...]:
    """SYNONYM keys are written as stems, but `_stem` strips only -ing/-ed/-s
    and friends, so `wobble` never became `wobbl` and the entry it pointed at
    was unreachable from the most natural spelling of the word. Try the token,
    its stem, and its e-less form."""
    for k in (tok, _stem(tok), tok[:-1] if tok.endswith("e") else tok):
        hit = SYNONYMS.get(k)
        if hit:
            return hit
    return ()


def _expand(tok: str, vocab: set[str], prefixes: dict) -> dict[str, float]:
    """The entry-vocabulary tokens this query token can stand for, each with the
    confidence it carries."""
    out = {tok: 1.0}
    for s in _synonyms_for(tok):
        out.setdefault(s, 1.0)
    if tok not in vocab and len(tok) >= 6:
        for cand in prefixes.get(tok[:5], set()):
            out.setdefault(cand, _PREFIX_WEIGHT)
    return out


# Above this, an entry is returned. Chosen so that a query whose only match is a
# word appearing across most of the table — `wrong`, `converging`, `interface` —
# falls below it, while a single informative word on its own clears it.
_THRESHOLD = 0.60


def coupling_signal_search(signal: str, limit: int = 5) -> list[str]:
    """Rank the coupling failure entries against a free-text symptom.

    Score per entry:
      3.0   the whole query appears verbatim in the entry
      2.0   the query and one of the entry's plain-language observations
            contain each other
      0.7 * core  + 0.3 * coverage
            `core` is the IDF weight of the single most informative query token
            the entry matches, so extra words cannot take a hit away. `coverage`
            is the share of the query's weight the entry explains, and breaks
            ties towards the entry that accounts for more of what was said; an
            unrecognised word carries almost no weight there, or coverage would
            punish detail through the back door.
    """
    idx = _search_index()
    idf, vocab, prefixes = idx["idf"], idx["vocab"], idx["prefixes"]
    raw = _toks(signal)
    if not raw:
        return []
    # De-duplicate but keep the expansion per distinct token.
    expanded = {t: _expand(t, vocab, prefixes) for t in dict.fromkeys(raw)}
    # An unmatched token still costs coverage: it is part of what was said.
    denom = sum(max((idf.get(e, _UNKNOWN_W) * c for e, c in exp.items()),
                    default=_UNKNOWN_W)
                for exp in expanded.values()) or 1.0
    low = signal.lower().strip()
    phrase = len(low) >= _PHRASE_MIN_CHARS
    scored = []
    for i, (r, entry) in enumerate(zip(_FAILURE_ROWS, coupling_pitfall_entries())):
        toks = idx["sets"][i]
        core = 0.0
        got = 0.0
        for exp in expanded.values():
            hits = [(idf.get(e, 0.0) * c, c) for e, c in exp.items()
                    if e in toks]
            if not hits:
                continue
            w, conf = max(hits)
            got += w
            if conf >= 1.0:                 # a guess never becomes the evidence
                core = max(core, w)
        score = 0.7 * core + 0.3 * (got / denom) if core >= _CORE_MIN else 0.0
        if phrase and low in _hay(r).lower():
            score += 3.0
        if phrase:
            for o in r["observations"]:
                if low in o.lower() or o.lower() in low:
                    score += 2.0
                    break
        if score > _THRESHOLD:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


_FAILURES = '''\
## 7. FAILURE MODES, and what each one actually means

''' + _failure_table() + '''

Reach these from a SYMPTOM instead of reading the table:
`knowledge(topic='coupling', signal='<what you actually saw>')` — describe the
observation in your own words, no mechanism needed, e.g.
`signal='it converged but the answer is wrong'`.

AND THERE IS A SECOND INDEX, which you want when there is NOTHING to paste.
The table above is what this tool PRINTS. The failures that cost the most print
nothing at all — a unit mismatch, a wrongly applied interface sign, a
participant that never reads its imports, a lossy mapping — and those live as
corpus entries whose Signal clauses say outright that you would see nothing,
rather than inventing a symptom you would not:

    knowledge(topic='pitfalls', solver='coupling', signal='<what you saw>')
    knowledge(topic='pitfalls', solver='coupling', physics='silent_wrong')

The groups are silent_wrong, participant_contract, verification_limits,
capability_limits and fsi. They are also the entries the project's own coverage
criterion counts for coupling, so they are the ones with fixtures behind them.
The `fsi` group is the fluid-structure-specific set — the traction sign, the
added-mass instability, the ALE mesh, and the two modes in which a one-way or
sign-flipped FSI passes every conservation check there is.

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

FLUID-STRUCTURE INTERACTION is its own pattern rather than a backend — a
vector traction one way, a displacement the other, and the fluid consuming it
by moving its mesh:

  knowledge(topic='coupling', solver='fsi')

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
same number of points in the same order every iteration, write `exports.json`
last, and EXIT 0 — the driver requires the file AND a zero exit code, so a
solver that writes its last iterate and then aborts ends the run instead of
being coupled on. The driver does NOT copy your script into `work_dir` and does
NOT interpolate between the two meshes.

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

   Read the stochasticity note below BEFORE you size `NRUN`/`NAVE`: a
   Monte-Carlo estimate has a noise floor the residual cannot fall below, so a
   plain `tol` makes `couple` report FAILURE on a coupling whose physics agrees.
   Set `SEED_MODE = "vary"` and pass `noise_replicates=5`.''')


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



def _fsi() -> str:
    """knowledge(topic='coupling', solver='fsi') — the FSI pattern.

    Not a backend. It is served through the same dispatch because an agent with
    an FSI problem asks for FSI, not for "the FEniCSx participant plus the
    scikit-fem participant plus the three things nobody wrote down".
    """
    return (
        "# Partitioned fluid-structure interaction with `couple`\n\n"
        "## What FSI needs that a scalar coupling does not\n\n"
        "Everything in `knowledge(topic='coupling')` still applies — the "
        "participant contract, the JSON shapes, the relaxation. FSI adds four "
        "things, and the two scripts below are one worked pair that has all "
        "four.\n\n"
        "1. **The exchange is VECTOR-valued.** `values` is an (N, 2) or (N, 3) "
        "list: a traction one way, a displacement the other. The driver handles "
        "that already — it relaxes the flattened vector and reports a residual "
        "PER COMPONENT in `block_residuals`, which matters here because "
        "traction and displacement differ by many orders of magnitude and a "
        "single global norm would let the small one move freely.\n"
        "2. **The two participants exchange DIFFERENT quantities.** A "
        "Dirichlet-Neumann heat coupling passes temperature one way and flux "
        "the other; FSI passes traction one way and displacement the other, and "
        "the fluid consumes the displacement by MOVING ITS MESH (ALE), not by "
        "setting a boundary value. That mesh motion is the entire "
        "structure-to-fluid direction. If it is missing you have a rigid-wall "
        "flow solve dressed as FSI, and every force-side check will pass.\n"
        "3. **The traction sign has to be decided once, in writing.** See the "
        "section below; getting it wrong in both places is silent.\n"
        "4. **The iteration can be unstable for physical reasons.** Added mass: "
        "with an incompressible fluid and a light structure the unrelaxed "
        "iteration diverges, and a smaller time step makes it worse. "
        "`knowledge(topic='pitfalls', solver='coupling', physics='fsi')` has "
        "the criterion and what to do.\n\n"
        "## THE SIGN, ONCE\n\n"
        "```\n"
        "  fluid exports  values = t = sigma_f . n_s = -sigma_f . n_f\n"
        "                        = THE LOAD ON THE STRUCTURE\n"
        "  structure applies it DIRECTLY as a Neumann traction, no sign change\n"
        "```\n"
        "`n_f` is the fluid's outward normal on the interface and `n_s = -n_f` "
        "the structure's. Cauchy's `t(n) = sigma.n` is the traction exerted BY "
        "the material `n` points INTO, so `sigma_f . n_f` is what the structure "
        "does to the fluid and the load on the structure is its negative. Check "
        "against hydrostatics before you trust an implementation: `sigma_f = "
        "-p I` with `p > 0` gives `t = +p n_f`, the fluid pushing the wall away "
        "from itself.\n\n"
        "Both participants ALSO export `normal_fluxes`, each with respect to "
        "ITS OWN outward normal, so the driver's conservation check can run and "
        "the two must sum to zero. Be clear about what that check then buys: it "
        "measures the FORCE TRANSFER across two non-matching interface "
        "discretisations. It cannot see a convention flipped on both sides.\n\n"
        "## Interface parametrisation\n\n"
        "Both sides send `coordinates` in the UNDEFORMED positions of their own "
        "interface nodes, and the displacement carries the motion. The "
        "interface is a material surface, so that parametrisation is fixed; "
        "sending deformed coordinates makes each side interpolate against "
        "something that moves with the answer.\n\n"
        "## THE FLUID PARTICIPANT — copy verbatim, edit the marked block only\n\n"
        f"```python\n{_script('fsi_fluid_fenics')}```\n\n"
        "## THE STRUCTURE PARTICIPANT — copy verbatim, edit the marked block only\n\n"
        f"```python\n{_script('fsi_solid_skfem')}```\n\n"
        "TWO MORE STRUCTURE PARTICIPANTS ship with the same contract, and both "
        "have been run as real coupled FSI on this install: "
        "`participant_fsi_solid_fenics.py` (FEniCSx, same interpreter as the "
        "fluid) and `participant_fsi_solid_fourc.py` (4C — a plain-Python "
        "WRAPPER that writes an inline-mesh 4C deck, runs the binary and reads "
        "the VTU back, so it runs under an ordinary python with numpy and "
        "meshio, not under 4C). Running the pair once with each is how you find "
        "out whether an answer depends on one structure code. Two things worth "
        "knowing before you use the 4C one: its WALL QUAD4 is bilinear and "
        "shear-locks in bending, so a plate a few elements thick comes out too "
        "stiff and the mesh has to be refined until that bias is below whatever "
        "you are asserting; and a spatially varying surface traction goes in as "
        "VAL x FUNCT with FUNCT a symbolic expression, where a POLYNOMIAL FIT "
        "is a silently wrong boundary condition rather than an error — the "
        "shipped participant emits the piecewise-linear interpolant exactly, "
        "using `heaviside`, and reports the deviation every iteration.\n\n"
        "## Wiring them\n\n"
        "```python\ncouple(participants=json.dumps([\n"
        '  {"name":"fluid","command":["<fenicsx python>","participant_fluid.py"],\n'
        '   "work_dir":"/abs/fluid","imports_from":["solid"]},\n'
        '  {"name":"solid","command":["<python>","participant_solid.py"],\n'
        '   "work_dir":"/abs/solid","imports_from":["fluid"]}]),\n'
        "  max_iter=60, tol=1e-9, accelerator='aitken', theta=0.5,\n"
        "  monolithic=json.dumps({...}), probe=True)\n```\n\n"
        "`probe=True` is not optional for FSI. The interface-sensitivity probe "
        "is the ONLY check that separates a real two-way coupling from a fluid "
        "that never moves its mesh.\n\n"
        "## Checking it, in the order the checks are worth anything\n\n"
        "1. **Run each participant by hand first**, with no `imports.json`, and "
        "check the fluid's net interface force against a configuration you can "
        "compute. For a straight channel with the wall held rigid that is plane "
        "Poiseuille: pressure drop `12*mu*U_mean*L/H^2`, so the net normal "
        "force on the wall is `dp*L/2` and the net tangential force is "
        "`6*mu*U_mean/H*L`. This is the check that catches a wrong sign or a "
        "wrong unit, and it is the only one here that does not share code with "
        "the coupling.\n"
        "2. **Both directions, by suppression.** Re-run with the fluid's mesh "
        "motion switched off. If the converged deflection does not change, the "
        "coupling is one-way, whatever it is called. Report the size of the "
        "change; it is the only quantitative evidence that the reverse "
        "direction carries physics.\n"
        "3. **Interface equilibrium and kinematic continuity, COMPONENTWISE.** "
        "Net force out of the fluid against net force into the structure, and "
        "the displacement the fluid imposed against the structure's own. A "
        "coupling that drops the tangential component passes every check that "
        "only looks at totals.\n"
        "4. **A reference solve.** `fsi_reference_newtonkrylov.py` in the same "
        "directory re-solves the coupled interface equation by Newton-Krylov "
        "and writes `monolithic.json`, so it plugs straight into "
        "`couple(monolithic=...)`. It establishes that the ITERATION found the "
        "root — read its docstring for what it cannot establish, which is "
        "anything wrong INSIDE a participant, because it drives the same "
        "scripts.\n"
        "5. **An independent code.** For FSI that means a native monolithic FSI "
        "solver on the same geometry. 4C has one "
        "(`PROBLEMTYPE: Fluid_Structure_Interaction` with a monolithic "
        "`COUPALGO`). Nothing above substitutes for it.\n\n"
        "## Traps\n\n"
        "`knowledge(topic='pitfalls', solver='coupling', physics='fsi')` — the "
        "traction sign, added mass, the one-way FSI that passes everything, the "
        "whole-convention flip that even a reference re-solve agrees with, the "
        "inverted ALE mesh that hangs instead of failing, and why the traction "
        "recovered from the structure's own stress field is not an equilibrium "
        "check.\n")


# ══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════

_BACKENDS = {
    "fenics": _fenics, "fenicsx": _fenics, "dolfinx": _fenics,
    "fourc": _fourc, "4c": _fourc,
    "fsi": _fsi, "fluid-structure": _fsi, "fluid_structure": _fsi,
    "ngsolve": _ngsolve,
    "skfem": _skfem, "scikit-fem": _skfem, "scikitfem": _skfem,
}

_ALIAS_CANON = {"fenics": "fenics", "fenicsx": "fenics", "dolfinx": "fenics",
                "fsi": "fsi", "fluid-structure": "fsi",
                "fluid_structure": "fsi",
                "fourc": "fourc", "4c": "fourc", "ngsolve": "ngsolve",
                "skfem": "skfem", "scikit-fem": "skfem", "scikitfem": "skfem",
                "dune": "dune", "dune-fem": "dune", "dunefem": "dune",
                "dealii": "dealii", "deal.ii": "dealii",
                "febio": "febio", "kratos": "kratos", "sparta": "sparta"}


def coupling_knowledge(solver: str = "", signal: str = "") -> str:
    """knowledge(topic='coupling', solver=..., signal=...) — the core payload,
    one backend's participant script, or the failure entries matching a symptom.

    `signal` is checked FIRST and on its own: someone arriving with a broken run
    wants the two entries that explain it, not 40 kB with them somewhere inside.
    """
    sig = (signal or "").strip()
    if sig:
        hits = coupling_signal_search(sig)
        if hits:
            body = "\n\n".join(f"* {h}" for h in hits)
            return (f"# Coupling failure entries matching {sig!r}\n\n{body}\n\n"
                    f"---\nThese are the coupling failure modes whose symptom "
                    f"matches what you described. The full contract, the "
                    f"relaxation guidance and the whole failure table are in "
                    f"`knowledge(topic='coupling')`; a complete runnable "
                    f"participant script for one backend is "
                    f"`knowledge(topic='coupling', solver='<name>')`.")
        # Deliberately NOT followed by the whole core payload. Returning 40 kB
        # on a miss makes a failed search indistinguishable from a hit for
        # anything that only checks whether some word came back — which is
        # exactly how a fixture asserting on this went green against a mutant
        # that searched for nonsense.
        return (f"# No coupling failure entry matches {sig!r}\n\n"
                f"Nothing in the coupling failure table matches that symptom. "
                f"Try describing it differently — what the RESULT looked like "
                f"rather than what you think caused it. The whole table is "
                f"section 7 of `knowledge(topic='coupling')`.")
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

For an IMPLICIT scheme it additionally emits:
  * `<max-iterations value="20" />`        — argument `max_iterations`
  * `<relative-convergence-measure limit="1e-6" />` on the exchanged data
                                           — argument `convergence_tol`
  * `<acceleration:aitken>` with `<initial-relaxation value="0.5" />`
                                           — argument `relaxation`
and the read mapping takes `mapping` (`nearest-neighbor` |
`nearest-projection`). Those four ARE forwarded — this note used to say they
were not, which was true of an earlier version of the tool and false of this
one. Checked by generating a config with non-default values and reading them
back out of the XML.

What is still NOT reachable through the tool: a different acceleration TYPE and
an `rbf` mapping. `generate_precice_config` takes both — `acceleration={"type":
"aitken"|"IQN-ILS", "data": ..., "mesh": ...}` and `mapping="rbf"` — and
`couple_precice` passes neither. For those, or for anything
`<coupling-scheme:multi>` needs, call `generate_precice_config` yourself and
launch the participants yourself.

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


# Per-backend preCICE verdicts, and WHICH FIXTURE ESTABLISHES EACH ONE.
#
# This comment used to say "Every CAN was established by running a real
# two-participant coupling through OASiS's own preCICE orchestrator on this
# install" while only TWO of the seven CANs had one. The other five were
# written from an import check or from nothing: the strong fixture's own
# docstring downgrades NGSolve and DUNE to the import GATE, and deal.II, Kratos
# and SPARTA had no preCICE fixture at all — the shipped deal.II participant is
# a `couple` file-handshake wrapper and neither it nor its CMakeLists mentioned
# preCICE, so a reader greping for a preCICE-linked participant found nothing
# and the boldest sentence in the file was the false one.
#
# Every verdict below now NAMES the fixture that establishes it, so the claim
# is checkable rather than merely asserted:
#
#   scikit-fem, FEniCSx
#       scripts/tier2_fixtures/coupling/precice_can_verdicts_proven_by_a_real_run
#   NGSolve, DUNE-fem, deal.II, Kratos
#       scripts/tier2_fixtures/coupling/precice_can_verdicts_for_the_other_four
#   SPARTA
#       scripts/tier2_fixtures/coupling/sparta_precice_load_order_and_coupled_run
#
# Each of those runs a REAL two-participant coupling through the registered
# `couple_precice` tool and checks the EXCHANGED FIELDS — against a closed form
# for the seven FEM pairs, against SPARTA run standalone at the same wall
# temperature for the DSMC one. None of them accepts `converged` as evidence.
#
# Every CANNOT (4C, FEBio) was established by looking for a preCICE entry point
# in the installed code and not finding one — that is a weaker kind of evidence
# than a run, and it is labelled as such in those two entries.
_PRECICE_BY_BACKEND = {
    "fenics": {
        "title": "FEniCSx (dolfinx)",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_proven_by_a_real_run)"),
        "body": '''\
`import precice` works in the same interpreter as `dolfinx`, and a FEniCSx
participant was coupled to a second code through `couple_precice` end to end:
FEniCSx on the Neumann side against a scikit-fem Dirichlet side, non-matching
interface meshes, serial-implicit, interface temperature and flux checked
against a closed form.

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
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
`import precice` works in the interpreter that carries NGSolve, and an NGSolve
participant was coupled to a second code end to end: NGSolve on the Neumann
side against a scikit-fem Dirichlet side, non-matching interface meshes,
serial-implicit, interface temperature and flux checked against a closed form.
NGSolve shares the OASiS venv with scikit-fem here, so that pair is one
interpreter — the coupling is real, but it is not what proves preCICE spans
separate environments; the FEniCSx, DUNE and Kratos pairs are.

TWO NGSolve-SPECIFIC SILENT-WRONG TRAPS, both found by running, and both
avoided in the participant that fixture couples:
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
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_proven_by_a_real_run, and "
                    "as the partner in precice_can_verdicts_for_the_other_four)"),
        "body": '''\
`import precice` works in the interpreter that carries scikit-fem, and a
scikit-fem participant was coupled to a second code end to end — in BOTH roles,
Dirichlet and Neumann, and against four different partner codes. It is the
lightest participant of all of them: pure Python, no compilation, no JIT, which
is why it is the standing partner the other backends' couplings are checked
against.

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
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
A DUNE-fem participant was coupled end to end: DUNE on the Neumann side, in its
own conda environment, against a scikit-fem Dirichlet side in the OASiS venv,
non-matching interface meshes, serial-implicit, interface temperature and flux
checked against a closed form. Two install-level facts decide whether it works
at all:

  * `precice` and `dune.fem` must be importable in ONE interpreter. If DUNE
    lives in its own conda environment without `pyprecice`, you do not have to
    install anything into it: put the site-packages of the interpreter that HAS
    `pyprecice` on `PYTHONPATH` and the preCICE lib on `LD_LIBRARY_PATH`, and
    the DUNE interpreter imports both. Verify with a one-line
    `python -c "import precice, dune.fem"` BEFORE coupling — a failed import
    leaves the partner blocking in `initialize()` with no error.
    That whole-site-packages form works HERE because the DUNE environment has
    no package the OASiS venv would shadow badly. It is not universal: see the
    Kratos entry, where the same recipe shadows the good install and has to be
    narrowed to a directory of symlinks.
  * DUNE-fem JIT-COMPILES EACH DISTINCT SCHEME. Measured on a cold cache here
    that is MINUTES, not the "about a minute" this note used to claim — the two
    schemes of a Dirichlet-Neumann heat participant took about seven. Build and
    COMPILE the scheme ONCE, BEFORE `precice.Participant(...)` — a throw-away
    `scheme.solve(...)` is what actually triggers the compile, so constructing
    the scheme is not enough — or the partner blocks on connect while you wait.
    Make the coupled boundary datum a `dune.ufl.Constant` (or a discrete
    function) and MUTATE it each window instead of rebuilding the scheme, or
    you pay that compile on every coupling iteration.''',
    },
    "dealii": {
        "title": "deal.II",
        "verdict": ("CAN, as a C++ participant — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
deal.II has no Python API, so the participant is a compiled C++ executable that
links `libprecice` directly. One is SHIPPED here —
`data/coupling_participants/precice_heat_dealii.cc`, with an optional CMake
target beside the `couple` participant's — and it was built and coupled to a
scikit-fem Python participant end to end, non-matching interface meshes,
serial-implicit, interface temperature and flux checked against a closed form.
Start from that file rather than from this description.

  * Build through CMake with `DEAL_II_SETUP_TARGET` and add
    `target_include_directories(<target> PRIVATE <precice>/include)` plus
    `target_link_libraries(<target> <path-to>/libprecice.so)`. A hand-rolled
    `g++ -I<dealii>/include` does NOT work — it fails on deal.II's own bundled
    headers — and the preCICE flags must go ON TOP of `deal_ii_setup_target`,
    not instead of it.
  * Guard the target on `find_library(precice)`. A hard `find_package` breaks
    the `couple` participant's build on an install with no preCICE, which is
    most of them.
  * `ldd <exe> | grep libprecice` is the one-line check that the thing you
    built is actually a preCICE participant. Nothing else in the tree tells you
    — a deal.II wrapper driven by the file-handshake `couple` driver looks
    identical from the outside and links no preCICE at all.
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
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four), with "
                    "a caveat about WHICH Kratos"),
        "body": '''\
A Kratos participant was coupled to a second code end to end: Kratos on the
Neumann side, in its own interpreter, against a scikit-fem Dirichlet side in
the OASiS venv, non-matching interface meshes, serial-implicit, interface
temperature and flux checked against a closed form. The install is the hard
part, and it is harder than for any other backend here:

  * A Kratos wheel can import cleanly on one host and be unusable on another —
    one on this class of host fails at import with a `GLIBC` version error from
    its bundled shared objects. If that happens, use a Kratos built from source
    and set BOTH `PYTHONPATH` to the install root and `LD_LIBRARY_PATH` to its
    `libs` directory, via `extra_env`.
  * THE PYTHONPATH RECIPE THAT WORKS FOR DUNE DOES NOT WORK HERE, and it fails
    in two different ways at once. Pointing `PYTHONPATH` at the WHOLE
    site-packages of the interpreter that has `pyprecice`:
      - shadows the good Kratos, because `PYTHONPATH` is searched BEFORE the
        target interpreter's own site-packages, so a broken `KratosMultiphysics`
        sitting in the preCICE interpreter wins and dies at import; and
      - breaks `cyprecice`, which is a compiled extension built against ONE
        numpy ABI — import it next to a different numpy and it fails with
        `numpy.core.multiarray failed to import`.
    What works is a NARROW shim: a directory of symlinks to exactly `precice`,
    `cyprecice` (package and `.so`), `numpy`, `numpy.libs` and `mpi4py`, and
    that directory on `PYTHONPATH`. preCICE then gets the numpy it was built
    against and Kratos keeps everything else of its own.
  * PROBE THE WHOLE GATE, not half of it. `import KratosMultiphysics` alone
    picks the wrong interpreter: this host has a system Python that imports
    Kratos fine and is 3.8, so it cannot load a cp312 `pyprecice` at all — it
    passes a Kratos-only probe and then dies inside the coupling on a numpy
    C-extension error. Probe
    `import precice, KratosMultiphysics, KratosMultiphysics.<App>` in ONE
    command, with the shim already on `PYTHONPATH`.
  * A core-only Kratos has NO `ConvectionDiffusionApplication` and therefore no
    thermal element at all. Check `import KratosMultiphysics.<App>` for every
    application your participant needs BEFORE coupling.
  * On the NEUMANN side the incoming flux density goes on as `FACE_HEAT_FLUX`
    on `ThermalFace2D2N` conditions built along the interface — that is
    ConvectionDiffusion's surface-source route, declared with
    `settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)`. Setting the nodal
    variable WITHOUT creating the conditions does nothing at all: there is then
    no boundary integral to carry it, and the participant silently solves an
    insulated problem.
  * Kratos drives its own time loop, so the preCICE loop wraps
    `InitializeSolutionStep()` / `SolveSolutionStep()` /
    `FinalizeSolutionStep()`, and the checkpoint save/restore is a copy of the
    solution-step variables on the interface nodes.
  * Kratos also has its own CoSimulation application. That is a different,
    Kratos-internal coupling path; it is not what `couple_precice` drives.''',
    },
    "sparta": {
        "title": "SPARTA (DSMC)",
        "verdict": ("CAN — proven by a real coupled run (fixture: "
                    "sparta_precice_load_order_and_coupled_run); IN-PROCESS "
                    "only with RTLD_DEEPBIND"),
        "body": '''\
A SPARTA DSMC participant was coupled to a solid conduction participant end to
end: rarefied argon past a cylinder against a lumped thermal shell, ten time
windows, serial-explicit, and the coupled wall temperature checked against
SPARTA run STANDALONE at uniform wall temperatures bracketing it.

RUN SPARTA AS A SUBPROCESS. That is what the coupled run does and what the
shipped participant does: one `spa_serial -in <deck>` invocation per time
window, with the wall temperature written into the deck's
`custom surf ... file` and the flux read back out of a surf dump. Two separate
reasons, and neither goes away:

  * the SPARTA Python library exposes `command` / `extract_global` /
    `extract_compute` / `extract_variable` and NO per-surf scatter, so an
    in-process participant can exchange a SCALAR and nothing more, while the
    deck carries a per-element field;
  * in-process, SPARTA and preCICE fight over MPI symbols — see below.

IF YOU DRIVE IT IN PROCESS ANYWAY, there is exactly one load order that works.
All four were run:

  * `import precice` first, then the stock `sparta.py` wrapper — SEGFAULT.
    `libsparta.so` DEFINES ITS OWN `MPI_*` STUB SYMBOLS and links no real MPI;
    `import precice` pulls a real MPI into the global symbol namespace, SPARTA's
    stub calls are interposed by it, and SPARTA dies inside `PMPI_Type_size`
    with MPI never initialised.
  * `from mpi4py import MPI` first, then preCICE, then SPARTA — SEGFAULT, same
    frame. It does not help.
  * SPARTA first through the stock wrapper (which uses `RTLD_GLOBAL`) — fails
    the other way: `import precice` then dies with
    `ImportError: libmpi.so.12: cannot open shared object file`.
  * THE ONE THAT WORKS: load SPARTA's library yourself with deep binding and
    LOCAL visibility, so its own symbols win inside it and preCICE's inside
    preCICE:

```python
import ctypes, os
mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
lib = ctypes.CDLL("<path to>/libsparta.so", mode=mode)
import precice                       # only now
```

  * SPARTA is a Monte-Carlo code. Its interface output carries statistical
    noise, so an IMPLICIT scheme's convergence measure may never be met even
    though the physics is fine. An explicit scheme with enough sampling per
    window is the honest choice; if you use implicit, set the tolerance above
    the sampling noise and say so. `couple_precice` reports an explicit scheme
    as UNMEASURED rather than converged, which is the right verdict here — so
    grade a DSMC coupling on its fixed point against standalone runs at the
    same wall temperature, not on anything the orchestrator returns.
  * USE A NEW SEED EACH WINDOW. With a fixed seed the run is bit-reproducible
    and a fixed-point iteration can look converged when only the RNG is
    repeating.
  * A SPARTA surface can take a prescribed TEMPERATURE
    (`surf_collide ... diffuse`) but there is NO surface-collision style that
    accepts a prescribed heat flux, so treat SPARTA as a Dirichlet-side
    participant. A flux can only be imposed indirectly, by converting it to a
    radiating-equilibrium temperature through `fix surf/temp` — see
    `knowledge(topic='coupling', solver='sparta')` for the route and its
    caveats.''',
    },
    "fourc": {
        "title": "4C Multiphysics",
        "verdict": ("CANNOT — 4C has no preCICE entry point (established by "
                    "ABSENCE, not by a run: fixture "
                    "precice_absent_in_fourc_and_febio)"),
        "body": '''\
There is no preCICE support in 4C. Searching the installed 4C source tree for
`precice` returns nothing, the built binary contains no preCICE symbols, and it
links no preCICE library. This is not a configuration problem you can fix from
the outside: adding preCICE to 4C means writing and building an adapter into
4C's own source.

Note what kind of evidence that is. A CAN here is backed by a coupling that
RAN; this CANNOT is backed by not finding an entry point, which is weaker and
is why it is worded as absence.

USE `couple` INSTEAD. 4C works well as a participant in OASiS's file-handshake
driver — it has been run there on BOTH the Dirichlet and the Neumann side of a
cross-code coupling. Call `knowledge(topic='coupling', solver='fourc')` for a
complete runnable 4C participant.''',
    },
    "febio": {
        "title": "FEBio",
        "verdict": ("CANNOT (as installed) — no preCICE in the binary, no "
                    "Python API (established by ABSENCE, not by a run: "
                    "fixture precice_absent_in_fourc_and_febio)"),
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
        "**DIRICHLET-TYPE IN PRACTICE, and it will not pass the convergence "
        "check.** SPARTA imports a wall TEMPERATURE and exports the energy flux "
        "the gas deposits, which is exactly the Dirichlet role. There is NO "
        "native flux boundary condition: of the nine `surf_collide` styles "
        "(`adiabatic`, `cll`, `diffuse`, `impulsive`, `piston`, `specular`, "
        "`td`, `transparent`, `vanish`) the four that take a thermal datum — "
        "`diffuse`, `cll`, `td`, `impulsive` — all take a TEMPERATURE, and none "
        "accepts a prescribed heat flux.\n\n"
        "A flux CAN still be imposed INDIRECTLY, and it is worth knowing the "
        "route exists before you conclude the coupling is impossible. "
        "`fix surf/temp` converts a per-surface energy flux into a per-surface "
        "temperature through the gray-body Stefan-Boltzmann law "
        "`q = sigma*emisurf*T^4`, i.e. `T = (q/(sigma*emisurf))^(1/4)`, and it "
        "takes that flux from ANY per-surf compute or fix — SPARTA's own doc "
        "says \"SPARTA does not check that the specified compute/fix calculates "
        "an energy flux\". So an IMPORTED flux reaches it: write the partner's "
        "flux to a file, load it with `custom surf ... file` into a custom "
        "per-surf vector, wrap that vector in `fix ave/surf s_<name>` (which "
        "sets `per_surf_flag`), and hand that fix to `fix surf/temp`.\n\n"
        "READ THE CAVEAT BEFORE USING IT. That is NOT a Neumann condition. It "
        "prescribes the temperature that would RADIATE the imported flux, so it "
        "constrains the wall temperature, not the gas-side flux, and it drags in "
        "an emissivity that has nothing to do with your coupling. The flux "
        "SPARTA then reports is free to differ from the one you imposed; making "
        "the two agree is a feedback loop you have to close yourself, and it is "
        "NOT what the shipped script does. This route was established by reading "
        "the SPARTA source and docs on this install and has NOT been run here — "
        "treat it as available, not as proven. The shipped script and everything "
        "below are the Dirichlet role.\n\n"
        "A full coupling to a thermal "
        "shell was run on this install: the physics agreed across the interface "
        "and the interface energy balance closed. With a plain `tol` it still "
        "reported FAILURE, because the residual cannot fall below the "
        "Monte-Carlo sampling noise; with `noise_replicates` set it converges "
        "against the measured floor and reports that floor. Read the "
        "stochasticity note below before using it.",
        "sparta", _launch_py(_interp_wrapper(
            "SPARTA", "SPARTA",
            extra="\n   List the surf / species / vss files in the participant's\n"
                  "   `data_files` (absolute paths). `couple` stages them into\n"
                  "   `work_dir` before the first iteration and fails loudly on a\n"
                  "   missing one, instead of SPARTA dying on 'Cannot open ...'."),
            _STEP2_SPARTA),
        '''\
* STOCHASTICITY IS THE HEADLINE, AND `couple` HAS A SWITCH FOR IT. DSMC output
  is a Monte-Carlo estimate. Its sampling noise does NOT shrink as the coupling
  iterates, so the driver's relative residual has a FLOOR at the noise level and
  a `tol` below that floor can never be met — the run ends as "did not
  converge", which is honest and useless.
  PASS `noise_replicates=5` (four or more; the floor is itself an estimate and
  three samples is a bad one). The driver then runs each participant that many
  times on the same imports, MEASURES the residual across independent
  replicates, and judges convergence against max(tol, that floor)
  over a block mean. It returns `noise_floor`; every tolerance you or anyone
  else later applies to the result must be at least that. Do NOT guess a `tol`
  "above the noise" instead — a guessed threshold is a number you cannot defend
  and it is the same act as tuning until it passes. Section 4a of
  `knowledge(topic='coupling')` has the details and the limits.
  SET `SEED_MODE = "vary"` FIRST. With a FIXED seed the runs are
  bit-reproducible, the measured floor comes out exactly zero, and an apparently
  converging residual is possible even when the physics has not settled — that
  is more dangerous than the noise, not less. `noise_notes` in the result says
  so when the floor measures zero.
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
