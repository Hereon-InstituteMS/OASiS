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

---

# Would the merged corpus pass the contamination gate? Measured.

Both audits flagged that "0 contamination hits" is branch-local: the gate lives
on two branches and reports green there, while the campaign branches — where the
knowledge actually lives — serve the exact class the gate was written against.
Neither audit could say what the MERGED corpus would do, because it exists
nowhere. Measured now, in a throwaway worktree off the base:

    all campaign branches, scanned individually        226 hits
    merged, taking knowledge/purge-eval-contamination
    FIRST and then seven campaign branches              20 hits

**Purge fixes 206 of 226.** Its work is done and simply has not propagated —
which is the argument for merging it first, before anything else.

The residual 20 were entirely on `knowledge/ngsolve-skfem-verify`, and the
reason is exact: that is the one branch which merged CLEANLY, so purge's
corrections never had occasion to touch it. Clean merges hide unfixed
contamination; conflicted ones surface it. That is the opposite of the intuition
and worth remembering during consolidation.

Those 20 are now fixed at source (commit `c1340c79`): four served MMS
convergence tables plus one served manufactured solution. Six remain on that
branch, four of them in `kratos/curved_mms.py` and
`dealii/poisson_mixed_bc.py` — both already fixed on `feature/anti-fabrication`
earlier the same day. The same correction now exists on three branches and has
landed on one.

Merge order confirmed by this probe: purge first, then the campaign branches;
conflicts are 1-4 files each and are purge's decontamination against the same
file's contaminated version, which resolves by taking the purged side and
re-applying any genuinely new content on top.

---

# The last merge, and the class of defect it exposed

`knowledge/coupling-revision` was deliberately left for last, and correctly:
it and `feature/coupling-robustness` rewrote the same iteration loop, the same
`_aitken`, and the same `couple()` for different reasons. The two rewrites turn
out to be COMPLEMENTARY where they touch code and CONTRADICTORY where they
touch text, and separating those two is most of the work.

## The code merged by layering, once the question was asked the right way

The conflict looks like two competing loop bodies. It is not.
coupling-robustness's loop is the same loop plus evidence collection
(`returncodes`, per-block residuals, responsiveness digests, the graph, the
sensitivity probe) plus four refusals (non-zero exit, empty export, changed
export length, unknown partner name). coupling-revision's `_invoke()` is the
loop body as it stood BEFORE any of that, factored into a function so the
noise-floor measurement could drive participants down the same path.

So the resolution is not a side. Keep coupling-robustness's loop whole, keep
`_invoke` for what it was factored out for, and make `_invoke` reproduce the
loop's observable behaviour (`sort_keys`, stale-export deletion, non-zero-exit
refusal) so the floor really is a floor for the residual the loop reports.
Taking `_invoke` for the loop — which is what "pick a side" would have done —
deletes every refusal silently.

`tests/test_coupling_robustness.py` decides the `_aitken` half outright:
`test_aitken_formula_matches_the_hand_computation` pins the recurrence against
a hand computation and `test_aitken_is_given_the_residual_and_not_the_raw_export`
pins `theta['residual_norm']`. There is nothing to weigh.

## FIVE served statements were true on one branch and false on the other

This is the shape worth recording, because nothing in the gate stack sees it
and it is not "the knowledge is wrong". Each of these was measured, correct,
and carefully written on `knowledge/coupling-revision` — and each describes a
driver that `feature/coupling-robustness` had already replaced:

    served claim                              measured on the merged driver
    ---------------------------------------   ---------------------------------
    "does NOT check your exit code"           refused: "participant A exited
    (in two places)                           with code 7 ... output of a
                                              FAILED run"
    "theta adapts per participant"            199 _aitken calls over 200
    (on BOTH branches)                        iterations with 2 participants
    "two fallback paths floor it at 0.1"      theta_prev=0.07 comes back 0.07
    "NOT the textbook recurrence — feeds      it is handed res_prev;
     the previous RAW EXPORT"                 residual_norm 6.1e-06, not the
                                              size of the solution
    "`data_files` is NOT supported by this    the file arrives in work_dir and
     tool, silently ignored"                  the participant sees it

Two of those five were the corrections the merge brief specifically asked to
preserve. They could not be preserved, because preserving them would ship a
false statement — which is the point: **a correction is only true against the
code it was measured on, and a merge changes the code.** Any branch that
CORRECTS documentation is carrying a hidden dependency on its own tree, and
that dependency is invisible in the diff. The text and the code it describes
conflict semantically while merging cleanly.

The tell is cheap and should be routine: for every doc correction a branch
carries, re-run the measurement on the merged tree before believing it. Five
for five failed here.

## Two fixtures asserted the pre-fix behaviour

Same mechanism one level down, and worse, because a fixture wears the badge
that means "executed". `driver_invariants_the_contract_asserts` asserted
`participant_exited_nonzero_and_was_still_accepted=True` and
`data_files_key_staged_the_file=False`; `aitken_clamp_bounds_theta` asserted
`theta_adapts_per_participant=True`. All three were true when written.

Turning an assertion round is not enough — the recorded MUTATION usually stops
discriminating at the same time. The old exit-code mutation (write the export
under another name) killed the old assertion because the run then failed; it
cannot kill the new one, because the run fails either way. The replacement
flips `sys.exit(7)` to `sys.exit(0)`, which isolates the exit code as the cause.
**When a fixture's direction reverses, its control has to be re-derived, not
re-pointed.**

## A dead mechanism found by asking who calls it

`knowledge.py` carried `_coupling_failure_modes()`, added on
coupling-robustness to append the 32-entry failure index to the coupling guide.
Its call site was the inline payload string, and that string had been replaced
by the `coupling_knowledge` module — so the helper sat there calling nothing,
and its docstring still described a call that no longer happened. That is how
the loss stayed invisible: the function reads as live.

Restoring the append was the obvious repair and the wrong one.
`test_knowledge_tool_output_matches_the_payload_function` requires the tool to
serve exactly what `coupling_knowledge()` returns, "not a second copy", so the
append fails a gate — and on the numbers it deserves to: the block is 38 kB
against a 35 kB guide, so `knowledge(topic='coupling')` would more than double
and carry TWO symptom corpora for an agent holding one error message. The guide
now carries a 900-byte cross-reference instead, inside the payload function, and
each corpus stays in one place.

## A behaviour difference the merge does not cause and must not hide

Measured on the skfem conduction pair, rho = 4, tol = 1e-4, max_iter = 200:

    accelerator="constant", theta=0.2   CONVERGED, 83 iters, 9.4e-05
    accelerator="constant", theta=0.5   not converged, 8.2e-01 (above the
                                        stability limit 2/(1+rho) = 0.4)
    accelerator="aitken",   theta0=0.5  not converged, STALLS at 3.7e-03
    accelerator="aitken",   theta0=0.2  not converged, STALLS at 2.4e-03

The Aitken arms do not diverge — the interface value is 3.1e-04 from the closed
form — the residual stops falling, with theta on the 0.05 floor for 40 of 199
adaptations. The per-participant Aitken this replaced converged this same case
inside the same budget, which is why `aitken_clamp_bounds_theta` was written
against it.

This is a property of `ffb5d1c6`, not of this merge: `_aitken` and the loop are
taken from coupling-robustness unchanged. It is recorded here because the served
sweep ("Aitken matched or beat a constant theta almost everywhere, measured
across rho from 1/4 to 9") was measured on the OTHER driver, and one cell of it
does not reproduce. The sweep has not been re-run.

## Where this merge left the numbers

The ten coupling / verification / participant test files, run serially with the
suite's own interpreter, before (`ffb5d1c6`) and after:

    before   3 failed, 282 passed, 3 skipped, 22 subtests passed
    after    3 failed, 286 passed, 3 skipped, 22 subtests passed

Re-run once more after the four verdict-channel fixes below, unchanged:
3 failed, 286 passed, 3 skipped, 22 subtests passed, and `diff` of the two
failure lists is empty.

The failure SET is byte-identical and all three are pre-existing: they are
`test_signal_verification.py`'s checks against `scan_results/tier2_results.json`,
which carries no `fixture_fingerprint` and is therefore treated as stale — which
is that gate working, not a regression. The +4 are the stochastic-branch tests
`knowledge/coupling-revision` adds to `test_coupling_driver.py`, all passing.

Fixtures re-run individually after the merge, each passing:
`driver_invariants_the_contract_asserts` (mutations 3/3 KILLED),
`aitken_clamp_bounds_theta`, `failure_table_reachable_from_a_symptom`,
`reference_solutions_unreachable_through_the_tools`,
`sparta_no_native_flux_bc`, `precice_absent_in_fourc_and_febio`,
`sides_table_backed_by_runs`. Plus the four static fixture-hygiene gates
(1349 passed, 1 skipped) over the whole fixture tree including the six new
coupling fixtures.

## The one collision neither branch could have seen

Everything above was found by reading the two sides. This one was only found by
running the merged tool, and it is the most instructive thing in the merge.

`knowledge/coupling-revision` puts "CONVERGENCE IS AT THE NOISE FLOOR, NOT AT
tol" into `CouplingResult.warnings`. On its own branch that is harmless, because
`couple()` there decided trustworthiness with a keyword filter:

    checks_ok = r.converged and not any(
        ("NOT CONVERGED" in w or "non-finite" in w or "NOT balanced" in w
         or "NOT COUPLED" in w) for w in val)

and the noise message contains none of those four. `feature/coupling-robustness`
deleted that filter deliberately — it was silently discarding the findings of
seven other checks — and replaced it with `checks_ok = r.converged and not val`.
Both changes are right. Together they stamp NOT VERIFIED on every correct
stochastic coupling, measured:

    converged        True
    noise_floor      1.039e-02
    tol_effective    1.039e-02
    verification     NOT VERIFIED — the coupling did not converge, or failed
                     one of OASiS's silent-wrong checks

Neither branch's tests could catch it. coupling-revision's assert on the driver
result, one level below `couple()`. coupling-robustness's never set
`noise_replicates`, because the argument does not exist there.

**A merge can be wrong in a way that neither side is wrong, and no test on
either branch covers.** The only instrument is running the merged surface with
both features switched on at once — which is a short list per merge and should
be written down before merging, not discovered after.

The repair is a CHANNEL, not a filter: `CouplingResult.criterion_notes`. The
criterion a run was held to is neither provenance nor a finding, and the two
existing lists are both wrong for it. It is reported in `checks_not_run`, which
is always printed inside the verdict and never flips it. Handed over as its own
list rather than pattern-matched back out of `warnings`, so a reworded message
cannot silently start or stop flipping a verdict — which is the failure the
keyword filter had.

Re-running after that fix found the SAME defect a second time, in a different
check: `check_residual_blocks` was still judged against the requested `tol`, so
a run held to 8.3e-03 was faulted for blocks "still changing by more than
1.0e-08 relative". It now takes `tol_effective`, as `check_convergence` beside
it already did. **One instance of this shape is rarely the only one** — every
check that takes a tolerance has to be asked which tolerance it means.

After both: `validation` empty, and with a review on record the same run comes
back VERIFIED with the floor named in its coverage section.

### The same shape three times, in three different checks

Once found, it kept being found. Every check that judges a coupling had been
written assuming the residual means what `tol` says it means:

    check_convergence          quoted `tol` in a NOT CONVERGED message on a run
                               judged at the floor  (fixed on the branch itself)
    the criterion notice       sat in `warnings`, which `couple` copies into the
                               findings list  -> NOT VERIFIED
    check_residual_blocks      compared blocks against `tol`, so a run held to
                               8.3e-03 was faulted for blocks moving by more
                               than 1.0e-08  -> NOT VERIFIED
    check_interface_sensitivity  faulted the participant for being stochastic,
                               in a branch whose own message says "hidden state
                               ... OR IT IS STOCHASTIC"  -> NOT VERIFIED

Each was found only by running the merged tool again after fixing the previous
one, because one finding is enough to stamp NOT VERIFIED and the first one masks
the rest. **The rule that would have found all four at once: when a feature
changes what a number MEANS, every consumer of that number is a candidate, and
the list of consumers is enumerable — grep for the parameter.**

The last of the four is the interesting one, because the honest fix is not to
silence the check but to move it to the channel that says what it could not
decide. With a MEASURED floor, "the response cannot be told apart from run-to-
run drift" is a true statement about the instrument rather than about the
coupling, and the thing that CAN decide it is the monolithic comparison — which
is what the coverage note now says. The half that must not weaken does not: a
participant whose export does not move at all is still a finding, floor or no
floor. That took a second pass, because signal = 0 also satisfies
`signal <= noise * margin` and the first guard routed it to coverage.

### And the fourth, which the real coupling caught after the synthetic one passed

Handing `check_residual_blocks` the effective tolerance instead of `tol` looked
like the same fix as the one beside it, passed on a synthetic noisy pair, and
was wrong. The real SPARTA coupling said so:

    block(s) gas.normal_fluxes=1.00e+00 are still changing by more than
    6.9e-01 relative, while the global norm reports convergence

The driver measures the floor of ONE statistic — the global L2 residual over the
stacked export vector. A block residual is the WORST ENTRY-WISE relative change
of one block. A DSMC surface flux has entries near zero whose relative change
between samples is order 1, so the second floor is far larger than the first and
nothing relates them. **A measured floor belongs to the statistic it was
measured on and transfers to no other**, which is easy to write down afterwards
and was not obvious while fixing four instances of "judged against the wrong
tolerance" in a row.

There is therefore no threshold that means anything here under a floor, and
choosing one to make a fixture pass would be the softening the branch exists to
refuse. The check reports coverage instead: scale masking has NOT been ruled out
on this run, it would take a per-block floor nothing measures, the per-block
numbers are returned for the reader, and the decisive check remains an
independent reference.

Twice now the honest answer has been "say what could not be decided" rather than
"decide it anyway". Both times the alternative was a number chosen to make a
fixture pass, and both times the coverage note is more informative than the
verdict would have been.

WHAT WAS DELIBERATELY NOT TOUCHED. `check_interface_balance`,
`check_interface_flux_profile`, `check_monolithic_consistency` and
`check_interfaces_are_the_same_surface` also take tolerances, and none of them
was changed. They judge conservation, geometry and agreement with a reference —
quantities the residual criterion says nothing about. The rule is about
consumers of the RESIDUAL, and the enumeration of those is complete:
`check_convergence`, `check_residual_blocks`, `check_interface_sensitivity`, and
the criterion notice itself.

### What finally settled it

`stochastic_noise_floor_makes_dsmc_gradable` — the real SPARTA-to-shell coupling
that is the whole reason the branch exists — is the instrument that found the
third and fourth instances and the one that confirms the repair. Run against the
merged tree it FAILED twice before it passed, each time on a different check and
each time on the same underlying mistake, and both failures were invisible to
the synthetic noisy pair that passed at every step. **A synthetic participant
reproduces the mechanism; only the real one reproduces the statistics.** The
DSMC surface flux has near-zero entries and a genuinely broad sensitivity
scatter, and neither is something a two-line `random()` participant has.

It passes now, with the arms it was written with, plus one new assertion that is
the point of all four fixes: `noise_aware_validation_stayed_empty=True`. A
correct coupling judged at its measured floor must leave the findings block
empty, because a non-empty one is a NOT VERIFIED verdict, and that verdict on a
correct stochastic coupling is precisely what the branch was written to remove.

---

# What is NOT done, stated exactly

## The consolidation is not finished — four branch tips are not ancestors

Checked with `git merge-base --is-ancestor <branch> HEAD` after this merge:

    merged:      coupling-revision, coupling-robustness, 4c-extraction,
                 ngsolve-skfem-verify, kratos-sparta, dune-extraction,
                 setup-and-portability, purge-eval-contamination
    NOT merged:  knowledge/fenics-verify        1 commit  (execution ledger)
                 knowledge/dealii-verify        1 commit  (execution ledger)
                 knowledge/febio-extraction     1 commit  (execution ledger)
                 feature/anti-fabrication      10 commits

The three ledger commits are the same additive shape as the one already taken
from coupling-revision. `feature/anti-fabrication` is real outstanding work,
and one of its commits adds a GATE — `tests/test_fixtures_carry_a_mutation_
control.py`, which does not exist in this tree — that would judge every fixture
in the merged corpus, including the six coupling fixtures this merge brings.
That gate should land before anyone reports a corpus-wide mutation figure.

Two of its ten commits also edit `docs/CONSOLIDATION.md` (one adds 114 lines),
so they will conflict with this file. Resolve by keeping both sections: they
describe different findings and neither supersedes the other.

## The Aitken comparative claim is not backed on this driver

`aitken_beats_constant_on_an_unbalanced_ratio` FAILS against the merged tree
(`aitken_reached_closed_form` and `aitken_beat_constant` both missing), for the
same reason as the rho=4 stall recorded above: at rho=6 with theta=0.5 the ONE
global theta no longer rescues the split that the per-participant theta did.
The fixture is left failing rather than retuned, because retuning it would be
choosing a ratio that makes the served sentence true instead of measuring
whether it is.

What the served knowledge still claims, and what is now behind it:

    "Measured across conductance ratios rho from 1/4 to 9 and theta from 0.1
     to 1.0 on this driver, Aitken matched or beat a constant theta almost
     everywhere, and in a quarter of those settings it converged to the right
     interface value where the SAME constant theta diverged"

That 40-cell sweep was run on the PER-PARTICIPANT accelerator, which no longer
exists. Two cells have been re-measured on the merged one and neither
reproduces. The sentence should not be trusted until the sweep is re-run, and
re-running it is the outstanding work — not editing the sentence to match two
cells, which would be the same mistake in the other direction.
