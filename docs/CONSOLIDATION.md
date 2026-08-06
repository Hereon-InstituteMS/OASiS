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

So all three agree once separated: neither bug announces itself, and the only
reliable detection is comparing the ANSWER against a reference, never watching
the solver.
