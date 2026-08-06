# Consolidating the branches

Measured 2026-08-06 against base `8cb1d79`, by actually merging into a scratch
worktree rather than reasoning about it.

## The scale

| branch | commits | files changed |
|---|---|---|
| knowledge/fenics-verify | 60 | 390 |
| feature/anti-fabrication | 52 | 79 |
| knowledge/setup-and-portability | 37 | 44 |
| knowledge/4c-extraction | 35 | 571 |
| knowledge/dealii-verify | 35 | 205 |
| knowledge/coupling-revision | 35 | 92 |
| knowledge/ngsolve-skfem-verify | 34 | 247 |
| feature/coupling-robustness | 30 | 39 |
| knowledge/febio-extraction | 30 | 181 |
| knowledge/kratos-sparta | 26 | 284 |
| knowledge/dune-extraction | 14 | 98 |
| knowledge/purge-eval-contamination | 6 | 21 |

394 commits, roughly 2250 changed files. **290 distinct non-fixture files**; the
rest are per-backend fixture directories, which cannot collide with each other
by construction.

## The conflict surface is small

Merging all twelve in sequence into a scratch worktree:

    clean: 2      feature/anti-fabrication, knowledge/ngsolve-skfem-verify
    conflicting: 10, between 1 and 11 files each — about 40 files in total

Forty files out of 2250. The work is overwhelmingly additive.

Files touched by three or more branches, which is where the conflicts live:

    10  scripts/run_tier2_fixtures.py
     9  tests/test_fixtures_cannot_pass_vacuously.py
     6  src/tools/consolidated.py
     5  src/tools/knowledge.py
     5  scripts/verify_signal_clauses.py
     4  validation/* (transcripts and ledgers)

`run_tier2_fixtures.py` looks alarming at ten branches, but there are only
**six distinct versions** of it and the three largest branches already share an
identical blob — the two real fixes (`--write-results` was parsed and never
read; interpreter resolution assumed a `.venv` inside the repo) propagated by
hand as each branch hit them.

## What the conflicts actually are

Nearly all of them are **the same correction made twice, independently**. Two
branches purged the same contaminated payload and wrote equivalent replacement
text:

    HEAD  "Exercised live on dune-fem 2.10 over a four-level refinement
           sequence. Expect the coarsest pair at higher order to come in
           below the asymptotic rate..."

    THEM  "Theory for a conforming Lagrange space of order k...
           These are the ASYMPTOTIC rates — the coarsest levels of a sweep
           are commonly pre-asymptotic..."

Both are right, and both say the same thing. That is the easiest class of
conflict there is: pick either side on the merits, do not re-derive.

## Order to merge in

Infrastructure first, so the gates exist before the content they judge arrives:

1. `feature/anti-fabrication` — merges clean. Brings the gates: contamination,
   format contract, discoverability, fixture keys, wall-clock, quoted
   diagnostics, plus the retrieval layer and the claim-attributed coverage
   metric. Merging it first means every later branch is judged on arrival.
2. `knowledge/purge-eval-contamination` — the other contamination purge. Take it
   next while its conflicts are only against the base, and prefer whichever
   wording is more specific about the MECHANISM.
3. `knowledge/setup-and-portability` — touches `server.py` and `consolidated.py`;
   land it before the per-backend branches pile onto the same files.
4. The per-backend knowledge branches, smallest first: dune, febio, kratos-sparta,
   ngsolve-skfem-verify, dealii-verify, fenics-verify, 4c-extraction. Each one's
   fixtures live in its own directory and cannot collide; only its knowledge
   edits can.
5. `feature/coupling-robustness` and `knowledge/coupling-revision` last. They
   both touch `src/tools/knowledge.py` and hold the flagship, so they deserve
   the most careful eyes and the least merge pressure.

## Rules for resolving

- **Never re-derive a measurement to settle a conflict.** Both sides were
  executed; pick the clearer text.
- **Prefer the side that states the mechanism** over the side that states a
  number, and never resurrect a measured number that a purge removed — the
  contamination gate will fail the merge, which is the point.
- **Re-run the gates after every single merge**, not at the end. A merge that
  reintroduces a leaked payload is silent, and finding it after twelve merges
  means bisecting twelve merges.
- **Check fixture keys after merging any two backends.** A collision marks BOTH
  fixtures FAILED in the runner, and seven working FEniCSx fixtures were sitting
  red before anyone noticed.

## A trap worth recording

The first attempt at this ran the scratch worktree on the PortableSSD, which is
exFAT and cannot store git's permission bits. Every file therefore appeared
modified, and every merge refused with "local changes would be overwritten" —
which reads exactly like twelve genuine conflicts. Merge testing must happen on
a real filesystem.

The attempt before that used `git merge-tree --write-tree`, which does not exist
in git 2.25 (it arrived in 2.38). It reported CONFLICT for all twelve with an
empty file list. An empty conflict list is the tell that the tool failed, not
that the merge did.

---

# A checker that was built, failed four times, and withdrawn

Two agents independently found knowledge entries written in another library's
vocabulary — skfem entries saying "a pressure GridFunction" and "applying a
DirichletBC", where `hasattr(skfem, "GridFunction")` is False and the 12.x API
is `condense` / `enforce` / `penalize`. An agent following those reaches for a
symbol that does not exist and gets an AttributeError unrelated to the pitfall.

That looked mechanically checkable, so it got a checker. It was wrong four
times, each time in the direction of accusing correct text:

1. **A hard-coded ownership table.** 47 hits, 12 real. `dune.fem.GridFunction`
   and `dune.ufl.DirichletBC` both EXIST — DUNE shares UFL with FEniCSx and has
   its own GridFunction — and `condense` is core deal.II
   (`AffineConstraints::condense`). 35 correct entries would have been rewritten.
2. **CamelCase treated as an API signal.** 576 hits on skfem alone, flagging
   CRITICAL, Safer, Downstream, LUMPED, Underlying, Quadrature.
3. **A trailing paren treated as a call.** `get_quadrature(RefTet, 9) WORKS (45
   points...)` and `Krylov-Uzawa (skfem example 30)` both put an ordinary word
   before a paren. Also flagged `NotImplementedError`, a Python builtin, and
   every entry that CORRECTLY says a symbol is absent — "Don't reach for a
   skfem.NewtonSolver, it doesn't exist" is the knowledge being right.
4. **Dependencies excluded.** `meshio._exceptions.WriteError` is real; the probe
   searched only skfem's own modules.

After all four fixes it still reported 115 on NGSolve: `Curve`, `Cylinder`,
`Glue`, `gfT.Set`. Every one real. `Curve` is a METHOD on `Mesh`, `Cylinder`
lives in `netgen.occ`, and `gfT.Set` is a method on an instance. A module-level
`hasattr` sweep cannot see any of those, and no amount of pattern tuning fixes
that — the approach is unsound for object-oriented APIs, which is all of them.

**Withdrawn rather than shipped.** A gate at 115 false accusations would be
switched off within a day, and it would take the trustworthy gates with it.

What survives is the finding itself, which two agents obtained the right way: by
executing the entry and watching the symbol not exist. Cross-library vocabulary
is real, it has been found in both directions, and it is worth checking during
any knowledge pass — by hand, on the entries being touched, not by a sweep.

---

# The defect class, stated exactly

Independently reproduced 2026-08-06, in seven lines of scipy:

    saddle system with a one-dimensional pressure nullspace
      MatrixRankWarning : NONE
      isfinite(x).all() : True

    structurally singular matrix
      MatrixRankWarning : fires
      isfinite(y).all() : False

Two skfem entries (`hydraulic_resistance#2`, `navier_stokes#3`) told agents to
guard the Stokes pressure nullspace by catching `MatrixRankWarning: Matrix is
exactly singular`, with `np.isfinite` as the backstop. **Both halves are blind
to the case they were written for.** The velocity is not merely finite — it
matches the pinned solution to 5.5e-14. Only the pressure LEVEL is arbitrary,
which is precisely why neither check can see anything.

The warning is real. It fires on a structurally zero pivot and on an
inf-sup-violating equal-order pair, and in both the result is non-finite, which
is the case `isfinite` was the right partner for. So it is a SUFFICIENT signal
of rank deficiency and never a NECESSARY one.

This is the shape every backend keeps producing, and it is worth naming because
it is not "the knowledge is wrong":

  * the CAUTION is correct — an unpinned Stokes system really does have an
    undetermined pressure level;
  * the MECHANISM is correct — the block really is rank deficient;
  * the OBSERVABLE is not produced.

An agent following such an entry writes the prescribed guard, the guard stays
silent, and the silence is read as success. The entry does not fail to help; it
manufactures confidence. That is why a fixture must assert on something the run
actually emits, and why "no measured number in the knowledge" and "the Signal
must be observable" are the same rule seen from two sides.

The corrected entries now give an observable that IS present: the nullity of the
condensed block, or two solves with different pins — identical velocity,
pressure differing by a constant.

Sibling instances of the same shape found the same day:

    scipy cg on an indefinite system   returns info=1000, raises nothing
    NGSolve CG on a genuinely indefinite matrix   converges in 47 iterations
                                                  with a block preconditioner
    dolfinx failing solve              prints ZERO characters on fd 1
    4C beams/contact/FBI               run to completion where the entry
                                       promised an abort
    SPARTA ambipolar_plasma:3          reports "broken" exactly when the deck
                                       is correct

---

# The largest open risk, sized

440 entries across the corpus state a solver failure as their ONLY observable —
"Newton diverges", "CG raises", "the run aborts", "NaN appears" — with nothing
in the entry acknowledging the failure might be silent:

    fourc 132   fenics 72   kratos 54   ngsolve 46   skfem 42
    dune 32     dealii 30   febio 26    coupling 4   sparta 2

These are candidates, not proven defects. But the sample that has been RUN is
not encouraging. Every instance executed so far falsified, in five independent
codebases:

    skfem hyperelasticity#2   "Newton diverges"  -> converges to 8e-16 silently
                                                    onto a different field
    skfem stokes#0            "cg returns info!=0" -> info=0 on all 9 RHS tried
    NGSolve dg_methods#6      "CG raises"        -> exhausts its cap, returns a
                                                    field with rel. error 208
    NGSolve contact#3         flipped normals    -> Newton status 0 while the
                                                    penalty drives bodies together
    deal.II dg_transport::4   "GMRES NoConvergence
                               at step 0-1, NaN"  -> converged in 3 steps,
                                                    last_value 0, no exception
    deal.II advection_dg::1   "L2 norm diverges" -> error keeps falling, rate
                                                    stays above 2, CG converges
    4C fbi#0, ehl#0, fpsi#5   "aborts with X"    -> runs to completion, exits 0,
                                                    passes every result test
    FEniCSx (whole backend)   quoted console text -> a FAILING dolfinx solve
                                                    prints ZERO characters

Two independent codebases produced the SAME wrong claim about DG penalty and
central flux (NGSolve dg_methods#3/#4, deal.II advection_dg::1 and
dg_advection_reaction::0) and the same true behaviour on measurement. That
suggests the errors are inherited from shared folklore rather than invented per
backend, which is worth knowing: the same wrong sentence will be in other
people's documentation too.

WHY THIS CANNOT BE GATED, only measured
---------------------------------------
Whether a solver actually fails is a property of the run, not of the text. No
static check can distinguish "Newton diverges" (true here, false there) from
"Newton diverges" (never). The only instrument is execution, which is exactly
what a tier-2 fixture is — so the fixture programme IS the remedy, and this
number is the estimate of how much of it remains meaningful.

The practical rule for anyone writing or reviewing an entry: if the Signal names
a failure, the entry must also say what happens when the failure does NOT
appear. Where the honest answer is "it converges cleanly and every internal
check passes", that sentence is the most valuable one in the entry, because it
tells the agent its guards cannot help and it must compare against a reference.

---

# Two failure modes that look alike and are opposites

A deal.II pass separated these by measurement, and the distinction refines advice
that had been circulating in this project in half-form. Newton on a
hyperelasticity problem, same mesh, same load:

    wrong TANGENT (K_geo sign error)   converges quadratically
                                       to the SAME answer, 2.2e-16 away
                                       cost: one extra step (16 vs 15)

    wrong STRESS  (residual is wrong)  converges quadratically
                                       to a DIFFERENT answer, 7.3% away
                                       cost: FEWER steps (12 vs 15)

**A wrong tangent costs iterations, not accuracy. A wrong residual costs
accuracy, not iterations.** The framework differentiates whatever residual it is
handed, so a consistently-differentiated wrong stress converges beautifully onto
the wrong solution — and it converges FASTER, because the fabricated problem
happens to be easier.

The practical consequence is sharp: an agent watching iteration counts to detect
a stress error will see the count go DOWN. The catalog entry this corrects
claimed the tangent error is "off by 2x" and blamed the wrong bug for the wrong
symptom.

Two other backends found the halves independently, which is why they were being
conflated:

    skfem hyperelasticity#2   PK2 stress in the PK1 slot: entry says "Newton
                              diverges"; measured, converges to 8e-16 onto a
                              field 0.45%/1.43% off at 5%/20% stretch
    NGSolve nonlinear_elasticity#3   factor-2 hand-coded P: entry says linear
                              convergence; measured, QUADRATIC convergence to a
                              wrong displacement. Linear convergence needs a
                              tangent inconsistent with the residual — which is
                              the OTHER bug

A third measurement completed it, same instrument and same problem, with
backtracking enabled in both runs (without it the inconsistent tangent does not
converge slowly — it diverges, and the claim is about a RATE):

    consistent tangent          observed order 1.993, 16 steps
                                residuals squaring: 5175 -> 69.6 -> 0.995
                                -> 2.1e-4 -> 1.2e-11
    geometric term dropped      observed order 0.9996 — first order to four
                                digits — 29 steps, near-constant contraction,
                                SAME answer to 2.8e-13

So the full statement is:

    wrong TANGENT  -> right answer, convergence order 2 -> 1, MORE steps
    wrong STRESS   -> wrong answer, order stays 2,        FEWER steps

An agent watching iteration counts catches the first and is ACTIVELY MISLED by
the second. Neither bug announces itself, and the only reliable detection is
comparing the ANSWER against a reference — never watching the solver.

---

# Cannot verify here, versus not done

Some claims cannot be executed on this machine at all. That is a different
statement from "nobody got to it", and the paper must not blur them — a reader
who finds a gap will ask which it was.

Measured from the installs themselves, not from documentation:

    deal.II   config.h:  DEAL_II_WITH_MPI      undef
                         DEAL_II_WITH_P4EST    undef
                         DEAL_II_WITH_PETSC    undef
                         DEAL_II_WITH_TRILINOS undef
                         DEAL_II_WITH_SUNDIALS undef
                         DEAL_II_WITH_ADOLC    undef
              -> parallel_poisson (7 claims) unreachable; the SUNDIALS half of
                 time_dependent_heat#1 unreachable; the AD half of
                 hyperelasticity#7 unreachable

    FEBio     binary carries no MKL and no Pardiso
              -> the claims about those solver paths unreachable

    4C        binary carries no ArborX and no preCICE
              -> beam_interaction geometric search (2 claims) unreachable;
                 preCICE coupling unreachable, which the coupling fixtures
                 already pin as a NEGATIVE capability claim rather than a gap

Excluding what cannot be verified, deal.II sits at 103/122 = 84% of its
reachable surface rather than 76% of its nominal one.

A WARNING ABOUT THIS COLUMN, learned the expensive way. 4C reported four claims
as blocked and **three of those reports were wrong**:

    beam_interaction:3/:4   reported ArborX-blocked. FOUR_C_WITH_ARBORX=OFF only
                            rules out `bounding_volume_hierarchy`, which exactly
                            8 of ~1974 upstream decks select — verified by
                            counting them. The default bruteforce_with_binning
                            runs fine; the right deck takes two seconds and
                            yields 13 coupled segments.
    multiscale:5            reported as needing a stopwatch. It needed a
                            COUNTABLE observable instead — macro elements per
                            rank, 8/24/32 where even splits are 2-3-3.
    pasi:2                  reported as having no single-run observable. It had
                            one.

The agent's own diagnosis: "I generalised from one failing deck and stopped
looking." A blocked claim is a claim nobody will revisit, so it costs more than
an uncovered one — an uncovered claim is visibly work remaining, while a blocked
claim looks settled. The bar for this column must therefore be higher than for
any other: name the exact flag or missing library, and show the check that reads
it, as deal.II's sentinel fixture does.

TWO REFINEMENTS FOUND BY AGENTS, both worth keeping:

`parallel_poisson::0` is actually REACHABLE despite the six around it, because
it is a claim about what happens with the features OFF: the distributed header
compiles, MPI_InitFinalize reports one rank, and
`parallel::distributed::Triangulation`'s constructor is `= delete`d, so the
failure is a compile error with no link error and no runtime exception. A
compile-outcome fixture covers it. Six blocked, not seven.

And the evidence for the blockage is itself in the tree: the sentinel fixture
`install_feature_flags_visible` reads those config flags, so "cannot verify
here" is an executed observation rather than an assertion in a report. That is
the right pattern for every environment limit — the claim that something is
impossible on this host should be as verifiable as the claims that are.

---

# The flagship argument, in its strongest form

Two coupling mutations, both verified by running the harness, both converging
cleanly. They are the reason the verification gate compares against a reference
instead of trusting the coupling's own checks.

**Unit mismatch** — one participant in Celsius, one in Kelvin:

    converged             34 iterations
    residual              7.782e-09
    flux balance          6.463e-09
    validation entries    0
    balance warnings      0
    interface T           61.8% from the closed form
    interface q          1365.8% from the closed form
    monolithic check      fires

Every internal check green, the answer badly wrong.

**Consistent-conductivity mutation** — both participants 25% away from the
conductivity the closed form is computed from. This one is strictly harder:

    converges                             yes
    the two sides agree at the interface  yes
    flux balance                          untouched
    interface TEMPERATURE                 DOES NOT CHANGE AT ALL
    interface flux                        off by the same 25%

The temperature is unmoved because the conductance RATIO is preserved — both
subdomains moved together. So a reviewer checking the primary output sees the
right number. Only the flux moves, and only the closed form catches it.

That is the argument. A partitioned coupling can be internally perfect and
solving a different problem, and every quantity a practitioner would naturally
check can agree. Convergence, residual, interface agreement, flux balance and
even the interface temperature are all consistent with a wrong answer at once.

The same conclusion arrives from three independent directions today: 440 entries
whose only observable is a solver failure that mostly does not occur; a wrong
Newton residual converging FASTER onto a wrong answer; and these two couplings.
Watching the solver never suffices. Comparing the answer against a reference
does.

---

# A second checker withdrawn, and the hazard it failed to catch

THE HAZARD IS REAL. `src/backends/fenics/backend.py` records that for 17 of
FEniCSx's physics `src/tools/deep_knowledge.py` is CANONICAL, and the matching
`KNOWLEDGE` dicts in `src/backends/fenics/generators/*.py` are "DEAD CODE and
may have drifted — do NOT edit them and expect a behaviour change."

I briefed an agent to correct three falsified entries "in src/backends/fenics/".
Two of the three live in the dead half. It read the ordering comment, noticed,
and put the corrections where they are served — which is the only reason that
pass changed anything. A whole verification pass was one wrong assumption away
from editing files nobody reads, and that failure mode is silent and cheerful:
the edit applies, the tests pass, the diff looks right, the payload is untouched.

THE CHECKER FOR IT DID NOT WORK. It compared, per physics name, the pitfall
count written in a generator file against the count actually served. On its
first real run it reported `boundary_conditions: served 6, generator file 3` —
and that is not a divergence. `boundary_conditions` is a SUB-KEY that recurs
under many different physics in the same file, each with its own short list;
the checker's `max()` took the largest of those (3) and compared it against a
served list (6) that is the union across physics. Two different things with the
same name.

Withdrawn rather than tuned. Comparing counts cannot distinguish "the same list
in two places" from "two different lists that share a key", and the fix would
need a structural model of how each backend assembles its knowledge — which is
precisely the thing that differs per backend and defeated the AST reader in the
coverage metric earlier.

WHAT TO DO INSTEAD, since the hazard stands: name the canonical file in the
brief. That is now the rule for any knowledge-correction task — establish where
the served copy lives BEFORE editing, by reading the backend's own source-of-
truth note or by asking the registry, and say so explicitly in the instruction.
The agent that caught this did exactly that unprompted, and its report opens
with the scope correction rather than burying it.

That is two checkers withdrawn this session (the other: cross-library
vocabulary, unsound for object-oriented APIs) against eight that shipped. Both
withdrawals came from running the checker against real data and reading the
first hit rather than the count.

---

# Self-checked retrieval overstates reach

A pass that rewrote 69 catalog entries verified every one was still findable:
69 queries, 20 misses on the first attempt — exactly the failure mode where the
words a query would match get deleted along with the wrong number — then
restored the vocabulary and reached 69/69.

That work was real and the restoration was right. But the check has a ceiling:
the queries were written by the same agent that wrote the entries, so they share
its vocabulary. I ran four queries of my own against the corrected catalog and
one missed:

    "Newton converged but the displacement is wrong"  -> 3 hits, correct entry
    "the energy grows every step"                     -> 3 hits, correct entry
    "CG returned but the field is far too large"      -> 1 hit
    "the field stopped changing after the first step" -> NO MATCH

The missed entry (ngsolve time_dependent_ns#2) is well written and explains the
mechanism exactly: an Inverse built on FreeDofs writes zero into every
constrained row, so `gfu.vec.data = inv * rhs` DELETES the prescribed boundary
velocity. My query described the CONSEQUENCE an agent would actually observe —
the field stops changing — in words the entry never uses.

So "all my queries hit" means the entries are consistent with their author's
mental model, not that they are reachable from a symptom an agent would paste.
The two differ most exactly where it matters: an agent queries from what it SAW,
an author writes from what they UNDERSTOOD.

The practical rule, and it costs almost nothing: retrieval checks for a
correction pass should come from someone who did not write the corrections.
Failing that, phrase every query as an observation with no mechanism in it —
"the field stopped changing", "the answer got worse when I refined", "it ran but
the number is wrong" — never as a description of the cause.

---

# A passing fixture that asserted a false statement

This is the failure the whole programme is built to prevent, found inside the
programme itself, and it deserves to be recorded plainly.

`stokes_lbb_and_pressure_constraints` (DUNE) printed and asserted

    zero_pressure_entry_claim_not_reproduced=True

The claim IS reproduced. The fixture built its inlet mask by interpolating
`[x[0], x[1], 0]` and slicing the PRESSURE leg, which returns the constant-0
third component — so every pressure dof tested as lying on x = 0, and the
"inlet maximum" it compared against was simply the global maximum. The fixture
ran, passed, and certified the opposite of the truth.

Counting `scheme.dirichletBlocks` instead settles it: both spellings constrain
the same 264 velocity dofs at 8x8, but `None` constrains **0** pressure dofs
while `0` constrains **68**, and the boundary pressure is exactly 0.0 versus
2.0. Fixed, re-executed, sentinel replaced with `zero_entry_pins_the_pressure`.

WHY IT MATTERS MORE THAN ANY SINGLE WRONG ENTRY. A wrong catalog entry misleads
an agent. A wrong FIXTURE misleads everyone downstream, permanently, and wears
the badge that is supposed to mean "executed". Coverage counted it. The mutation
harness would have killed it — the mutation removes the pathology and the
assertion changes — so even mutation evidence does not catch an assertion that
is internally consistent and externally false.

Nothing in the gate stack detects this. The only thing that caught it was an
agent re-reading its own probe and asking what the mask actually selected. That
is not a process that can be automated, and pretending otherwise would be the
same error one level up.

What CAN be said: the failure needed a derived quantity (a mask built by
interpolation and sliced by leg) standing in for the thing being measured. Where
a fixture can count the thing directly — dirichletBlocks, dof counts, non-zeros
— it should, and this fixture now does.

---

# Four ways a fixture key misreports coverage

All four were found by execution this session, each in a different backend, and
each produced a number nobody could have caught by reading.

**1. The key points at nothing.** `structural_dynamics:7` where the list ends at
6; `stokes:7` where it ends at 3. The fixture is real and defends nothing while
counting as coverage. Gated by `test_fixture_keys_point_at_real_claims`.

**2. Two fixtures share one key.** The runner marks BOTH failed, so seven
working FEniCSx fixtures sat red for bookkeeping reasons, and two Kratos IGA
fixtures likewise. Gated by the same test.

**3. Both agents withdraw the duplicate.** Two agents each found the other's
fixture for the same claim and each deleted their own — turning a visible
collision into a silent GAP. Caught only because both mentioned it in their
reports. Not gated: nothing distinguishes "withdrawn because duplicate" from
"never written".

**4. The key names a real claim through the wrong axis.** SPARTA attaches 10
UNIVERSAL_PITFALLS to every physics row, where they occupy slots 9-18. Six
fixtures covered all ten and declared them `universal:<n>` — correct, and
invisible to a counter walking `rarefied_flow`, which read 7/19 instead of
19/19. SPARTA measured 76% and was actually at 98.7%. Verified independently:
each of slots 9-18 appears in all 10 physics rows.

The fix was to carry BOTH keys — `universal:n` and its positional alias
`rarefied_flow:9+n` — with the coverage gate collapsing aliases onto the
identity key so nothing is double-counted, and deduplicating by FIXTURE NAME so
the collision rule is not weakened: one fixture under two conventions is one
owner, two fixtures still collide. Verified after the change: 84 claims keyed,
0 real collisions, all 10 aliases owned by their identity key's fixture.

CHECKED FOR ELSEWHERE, and it is confined. Three backends have `covers` entries
naming a physics different from their `physics` field — dune (poisson_mms,
navier_stokes, maxwell, mixed_methods), dealii (dg_transport) and sparta. For
dune and dealii every one of those IS a registered physics with its own pitfall
list, so they are legitimate multi-claim fixtures that the identity-keyed metric
already counts correctly. SPARTA's `universal` was the only axis that exists
solely as a shared block appended to other rows, and therefore the only one a
positional walk could miss.

WHAT THESE HAVE IN COMMON. Every one is a disagreement between two ways of
naming the same claim, and in every case both namings were defensible. That is
why they survive review: nobody wrote anything wrong. The metric has to be told
which naming is canonical, and where two exist it has to be told how they map —
which is exactly what the `covers` list is for, and why an unchecked `covers`
list would be worse than none.

And the direction of error is not consistent. Case 1 inflates, case 4 deflates
by 23 points, cases 2 and 3 corrupt without moving the number predictably. A
coverage figure is only as good as the key hygiene beneath it, and that hygiene
is now three gates plus one thing no gate catches.

---

# Asserting on a stochastic code without pinning noise

SPARTA is direct-simulation Monte Carlo, so every quantity carries statistical
scatter and the obvious fixture — "the value is X" — is either pinned noise or a
tolerance wide enough to pass anything. The design that solved it is worth
copying wherever a code is stochastic, adaptive, or otherwise not bitwise
reproducible:

  * measure the noise floor from ONE seed's late windows;
  * make every assertion against a DIFFERENT seed;
  * assert a RATIO against that floor, never an absolute value.

Wall-flux fixture: seed B's late windows all inside 3x seed A's floor, both
first windows more than 10x outside. Surface-flux: floor 0.54%, seed B's ten
late windows all inside 3x, its first window 26% and tens of times outside. No
measured value is pinned anywhere, and the assertion is dimensionless.

The shear fixture goes further and removes a CONFOUND rather than tolerating it.
The occupancy bias of `rarefied_flow:6` pushes the same direction as the effect
under test, so the fixture routes through `fix ave/grid` to eliminate it, then
runs three seeds per grid and asks only whether the clusters SEPARATE. They do,
with no overlap. Its mutation is the best control in the tree: zeroing both wall
velocities collapses the signal from 4.55% to 0.54% and the clusters stop
separating — which proves the residual is the shear and not a grid artifact.

That last step is the one usually skipped. A mutation that removes the pathology
shows the fixture responds to something; a mutation that removes the pathology
AND lands on the independently measured noise floor shows it responds to the
right thing.

---

# Mutation evidence is not uniformly persisted

1256 fixtures exist, all parse, all carry a runnable artefact. But the evidence
that each DETECTS its pathology — the mutation verdict, which is the only thing
separating a fixture from a script that happens to pass — is recorded in three
different ways and, for five backends, not at all.

Measured across all worktrees, taking the best version of each fixture:

    surface     fixtures   re-runnable control
    fourc            373       354   95%   `_mutation` field in fixture.json
    fenics           200       183   92%   T2_MUTATE env hook in the source
    dealii           119       104   87%   T2_MUTATE env hook
    coupling          18        18  100%   `_mutation` field
    skfem            144         0    0%
    kratos           131         0    0%
    ngsolve          115         0    0%   (48 describe it in _comment prose)
    febio             76         0    0%
    dune              41         0    0%
    sparta            35         0    0%

Every one of those agents reported its fixtures mutation-proven, and their
reports contain the specific mutation and the expectation it removed. There is
no reason to doubt the runs happened. The problem is that for 401 fixtures the
evidence lives only in a session transcript: nobody can re-run it, and a
reviewer asking "show me this fixture detects what it claims" gets prose.

THE T2_MUTATE HOOK IS THE BETTER CONVENTION and it was not mine — FEniCSx and
deal.II arrived at it independently. The control lives inside the fixture and is
switched by an environment variable, so re-running the proof is one command and
needs no harness that stages files, no parent-directory copying, and none of the
three staging defects that silently voided mutation verdicts earlier today.
A `_mutation` JSON field describes an edit somebody must reapply; a T2_MUTATE
hook IS the edit, already written and already tested.

This is not a claim that 401 fixtures are unsound. It is a claim that their
soundness is currently unauditable, which for this project is the same class of
problem as an unverified knowledge claim: the assertion may well be true and
nothing in the tree lets a reader check it.
