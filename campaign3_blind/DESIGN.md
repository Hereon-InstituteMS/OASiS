# Campaign 3 — blind evaluation. Pre-registered design

Frozen before any evaluation run. Motivated by Christian Cyron's review point
that a tool must not contain methods or information specific to the very test
cases used to assess it, and by the decision to go further than that: the
agent under test never sees an exact solution at all.

## 1. The separation

Three roles, strictly separated:

| role | sees the exact solution? | what it does |
|---|---|---|
| problem builder (offline, this repo) | YES — it is the only place one exists | derives and verifies the source term, emits a blind task and a sealed key |
| OASiS-running agent | **NEVER** | solves the stated boundary value problem and writes out its solution fields |
| grader (offline, after the fact) | YES — opens the sealed key | computes the true error and the observed convergence order |

Consequences:

* The agent's deliverable is no longer a convergence rate. It cannot compute
  one, because it has no reference. It writes its **solution values with node
  coordinates**, per mesh level, as CSV. All error and order computation
  happens on the grading side.
* The agent's own convergence judgement is the **general mesh-independence
  heuristic**: refine, and check whether the solution stops changing, both in
  a global norm and at individual points. This works for any problem, with or
  without an analytical solution, and contains nothing specific to the test
  cases. It is reported as `MESH_INDEPENDENCE` and is graded as a *claim*, not
  used as truth.
* Because the agent reports fields rather than a number, fabricating a good
  result is far harder than before: a plausible convergence rate can be
  invented, a converging sequence of fields cannot.

## 2. What is enforced, and what is not (corrected after adversarial review)

Enforced automatically by the builder, verified in code:

1. **Source-term verification.** Substituting the exact solution and the derived
   source term into the strong form gives a residual that simplifies to exactly
   zero. Independently re-derived for all ten instances by a second reviewer.
2. **Leak check on the task text** for the sealed expression and for the words
   "exact solution", "manufactured", "MMS", and similar.

Enforced by the GRADER (this is where the real protection lives):

3. **The grader owns the evaluation set.** The task prescribes a fixed,
   cell-centred probe grid independent of the agent's mesh, and a submission is
   rejected unless it reports exactly those points. This blocks norm dilution
   with known-zero boundary nodes, node clustering in a shrinking boundary
   layer, dropping bad nodes as NaN, and submitting fewer levels than
   prescribed — all four were demonstrated to score CORRECT before this change.
4. **Process-level execution evidence.** Only solver-produced artefacts count;
   the runner's echo of the prompt is excluded by name, because every prompt
   names its backend and the previous gate therefore passed on a directory
   containing nothing but the task text.
5. **Decay quality.** The error sequence must be monotone with R² ≥ 0.98.

NOT enforced, stated honestly:

* **There is no eigenfunction check in the code.** An earlier version of this
  document claimed one; it was never implemented. Non-eigenfunction fields are
  currently a drafting convention, not a gate.
* **`verify_outer_boundary` and `verify_interface` exist but are not wired into
  the build.** The properties they check were verified independently and do
  hold for all ten instances, but the build does not enforce them.
* **The blinding is not cryptographic.** With a printed closed-form source term
  and low-parameter manufactured fields, an adversary recovered all ten sealed
  solutions in under four seconds of symbolic algebra. Two instances (B1, D3)
  leak by naked inspection. Blinding here means the agent is not handed the
  solution and must produce full fields; it does not mean recovery is
  infeasible. Closing this properly requires numerically specified source terms
  with high-parameter random fields — open work.
* **The key seal is not a sandbox.** `chmod 000` does not stop a process
  running as the same user, and the builders holding the solutions in plaintext
  sit above the agent's working directory. A real sandbox is required before
  results from this campaign can be called blind.

## 3. Coupled problems — the rigour upgrade

Previously coupled instances had no reference at all and were graded on a
plausibility band. Now the manufactured-solution discipline is applied to
coupled problems as well. The global domain is split between two codes, and a
single manufactured field is by construction the exact solution of both
subproblems because it satisfies, all verified symbolically:

* the strong form in each subdomain,
* continuity of the field across the interface,
* continuity of the normal flux across the interface,

which are precisely the transmission conditions a partitioned scheme iterates
on. A coupled run is therefore graded on a **true convergence order**, exactly
as rigorously as a single-code run.

Existing band-graded coupled instances (for example the rarefied-gas case,
which has no closed form) are retained as a minority, so the claim keeps its
breadth.

### The geometry is deliberately not the one OASiS ships a solver for

Every coupled instance was originally the unit square split at x = 1/2. That is
exactly the geometry `coupled_solve` hard-codes — its own docstring describes "a
fixed enum of benchmark problems on a hardcoded unit-square split at x=0.5". An
instance on that geometry would have measured whether the tool contains the test
case: the OASiS arm would receive a pre-built solver for the graded problem
while the bare arm writes a partitioned scheme from scratch. That is the same
objection the blind redesign exists to answer — evaluation specifics living
inside the tool — and it would have been the easiest thing in the paper to
attack.

The coupled domain is now the rectangle (0, 3/2) × (0,1), split at x = 3/5 (and
the box (0, 3/2) × (0,1) × (0,1) in 3D). `coupled_solve` cannot serve it, so
both arms must use a general coupling path. Nothing is weakened: the
transmission conditions hold at any interior interface, so the
manufactured-solution construction is indifferent to where the split sits. The
two-material profile's interface coefficients are now solved symbolically from
the three physical requirements rather than hand-derived, because hand-derived
constants are how a construction silently stops satisfying its own transmission
conditions when someone moves the interface.

A general coupling orchestrator is a legitimate capability and is not what is
being removed here. What is removed is a pre-solved instance of the graded
problem.

### The grader takes the geometry from the key

`subdomain_bounds` previously defaulted to the unit square split at 1/2. Moving
the geometry made it silently wrong: it would have built its probe grid over the
wrong region and compared a solution against points outside the subdomain that
produced it, reporting a number that looked like a result. Coupled keys now
carry `extent_a`/`extent_b`, written from the same constants the problem is
built with, and a key without them makes the grader stop rather than guess.

Single-code problems still use the grader's unit-domain default. That is correct
for B1–B7, all of which genuinely live on the unit square or unit cube — but it
is now a **constraint, not a coincidence**: a single-code instance placed on any
other domain must carry its extents in the key first, or it will be graded on
the wrong probe grid.

## 4. Problem set (built and verified)

Single-code, blind:

| id | code | physics | dim | order |
|---|---|---|---|---|
| B1 | NGSolve | anisotropic diffusion, constant SPD tensor | 2 | 2 |
| B2 | deal.II | variable-coefficient diffusion | 2 | 2 |
| B3 | FEniCSx | nonlinear diffusion, a(u) = 1 + u²/2 | 2 | 2 |

Coupled, blind, graded on true convergence order:

| id | codes | physics | dim | order |
|---|---|---|---|---|
| D1 | FEniCSx + NGSolve | domain-decomposed diffusion | 2 | 2 |
| D2 | FEniCSx + deal.II | domain-decomposed diffusion | 3 | 2 |
| D3 | FEniCSx + Kratos | two-material heat conduction, conductivity jump 1:4 | 2 | 2 |

| D4 | FEniCSx + deal.II | domain-decomposed linear elasticity (vector) | 2 | 2 |

All four on (0, 3/2) × (0,1) (× (0,1) in 3D) split at x = 3/5 — see above for
why that is not the unit square.

D1–D3 were diffusion three times over, so the coupled set exercised one operator
and called it a suite. **D4 is vector-valued**: two fields cross the interface
and the transmission condition is the traction σ·n rather than a scalar flux,
which is a different thing for a partitioned scheme to get right. Both
components of the traction are checked, because a construction continuous in the
normal component but not in shear would fail to converge for a reason the
problem statement never states — which looks like the agent's fault and is ours.
The shear traction at the interface is verified NONZERO, so this cannot be
passed by treating it as two scalar problems. Elasticity is also the one physics
all nine backends support, making it the fairest cross-code comparison
available.

To be extended (3D single-code, elasticity, mixed formulations, further
pairings) before the campaign launches.

## 5. Grading

Order of checks, integrity strictly before any comparison against truth:

1. `COULD_NOT_COMPLETE` declared → honest incomplete.
2. No usable solution files → failed.
3. No execution evidence for a named code (for coupled: for either code) →
   fabricated.
4. True error per mesh level: the sealed exact solution is evaluated at the
   agent's own node coordinates; the error norm is the root mean square of the
   nodal errors, which is proportional to the L2 norm on the prescribed uniform
   meshes and therefore has the same order.
5. Observed order: mean of log2 of successive error ratios. The task
   prescribes exact halving, so this estimator is unbiased and needs no
   mesh-size estimate. (Validated: a synthetic second-order solve grades
   1.996, a first-order solve 0.996.)
6. Correct if the observed order lies within the per-instance tolerance of the
   theoretical order and inside the plausibility band; otherwise confidently
   wrong.

The agent's own `MESH_INDEPENDENCE` verdict is recorded alongside but never
used as truth. Comparing it against the true error yields a secondary result:
whether the general heuristic is a reliable proxy when no exact solution
exists.

## 6. Shielding and data safety

* Sealed keys live in `keys/` and are made unreadable for the duration of every
  run; the grader unseals them afterwards.
* Runs are written to fresh directories under `runs/`; nothing is ever
  overwritten.
* Every run records tokens in and out, wall time, and tool calls, so cost and
  energy are computed from measured usage.
* The trajectory leak audit additionally scans for any appearance of a sealed
  solution in the agent's context; a hit invalidates the run.

---

# Amendment 1 — 2026-08-07. Blindness audit, three rebuilt instances, key custody

Made before any evaluation run (the campaign had produced zero results), so
nothing is invalidated and no result was seen before these changes.

## 1. Three of the eleven instances disclosed their solution

A symbolic leak gate (`blind_eval.leakgate`, in the OASiS repo, so it is under
version control — this directory is not) was run against all eleven problems.
Three failed:

| id | how it leaked | severity |
|---|---|---|
| B1 | one source term **as printed** is `12*pi**2 * u_exact` | naked-eye |
| D3 | two/three printed terms sum to `pi**2 * u_A` and `4*pi**2 * u_B` | naked-eye |
| B2 | `4*pi**2 * u_exact` survives expansion, not visible in the printed form | one `expand()` |

§2 of this document admits B1 and D3 ("two instances leak by naked inspection").
**B2 was not known.** It was found by the gate, not by inspection.

All three share one cause: **the hidden field is an eigenfunction of the
operator in at least one direction**, so applying the operator returns a
multiple of the field, which then sits in the published source term. B1 and B2
use `cos(2 pi x)` / `sin(pi x) sin(pi y)`; D3 uses a `sin(pi y)` transverse
profile, and `-k d²/dy²` of it is `k pi²` times itself.

The eigenfunction factors were replaced with profiles that still vanish on the
boundary — so `u = 0 on the boundary` remains the honest, zero-disclosure
boundary datum — but are not eigenfunctions. Physics, backend, element, mesh
sequence, theoretical order and geometry are unchanged; only the hidden field
moved. Rebuilt by `scripts/blind_rebuild_leaky.py`, which refuses to write
unless the residual is exactly zero, the boundary trace vanishes, and the gate
is clean. All eleven now pass.

## 2. The "NOT enforced" list in §2 is now shorter

* **The eigenfunction check now exists.** §2 said "there is no eigenfunction
  check in the code … a drafting convention, not a gate." There is one, it is
  stronger than a proportionality test (it catches `f = c*u + remainder`, which
  is how B1, B2 and D3 actually leaked), and it is what found B2.
* **Interface conditions are verified non-trivially for D3.** Note that for
  D1, D2 and D4 the existing `verify_transmission` / `verify_traction_continuity`
  calls pass the **same** field and the **same** material to both sides, so
  `jump_u=0; jump_flux=0` is a tautology that would hold for any input. That is
  not a defect in the manufactured fields (a single global field does satisfy
  the transmission conditions); it is a defect in the *evidence*. D3 is the only
  instance where the check has content, and after the rebuild it still does.
* **`verify_interface` is dead code and is also self-defeating** — it computes
  `u.subs(x, x_if) - u.subs(x, x_if)`, which is zero by construction. It is
  never imported. It should be deleted rather than left to look like a control.

## 3. What is still NOT enforced, stated honestly

* **Blinding is still not cryptographic.** Replacing the eigenfunction factors
  removes recovery *by inspection*, which is what B1 and D3 offered. It does not
  make recovery infeasible: a closed-form source term with a low-parameter
  manufactured field can still be inverted by assuming an ansatz and matching
  coefficients. Closing that needs numerically specified source terms with
  high-parameter fields, and that remains open work. What can now be said is
  narrower and true: **no instance hands over its solution to a reader, and none
  is one division away.**
* **The seal is still not a sandbox.** `run_blind.py` gives the agent
  `run_bash` and a `read_file` that honours absolute paths, with no namespace,
  chroot or path allowlist. `chmod 000` stops an agent that wanders; it cannot
  stop one that decides to attack, because the agent runs as the directory's
  owner. This is now mitigated rather than solved — see §4.

### The builders were the real hole, and it was wide open

§6 of the original design says sealed keys are "made unreadable for the duration
of every run". They were not — `keys/` was `drwxrwxr-x` when this audit began —
but that is the smaller problem. **`build_problems.py`, `build_extra.py` and
`build_coupled.py` hold every hidden field as a Python literal, were mode 644,
and sit two directories above the agent's working directory** (`runs/<cell>/work`).
The agent has `run_bash` and an absolute-path `read_file`. `cat ../../build_extra.py`
returns eight of the eleven current solutions.

Encrypting `keys/` while leaving those readable is custody theatre. §2 of the
original document notes the exposure — "the builders holding the solutions in
plaintext sit above the agent's working directory" — and nothing acted on it.
The seal now covers the builder sources, `preflight` fails if they are readable,
and `scripts/blind_keys.py exposure` sweeps the whole agent-reachable tree.

That sweep needed a **structural** rule, not a string search: a builder writes
`x * (1 - x) * y * (1 - y) * sp.cos(...)` while the key stores
`x*y*(1-x)*(1-y)*cos(2*pi*x)` — different bytes, same function — so grepping the
tree for the sealed strings finds nothing and reports safety.

## 4. Key custody: three controls, none trusted alone

1. **Not reachable.** `keys/` is `chmod 000` during runs, and unreadability is
   verified by *executing* reads from a separate process rather than asserting
   it. (`keys/` was `drwxrwxr-x` — plainly readable — when this audit started.)
2. **Not readable.** Keys are AES-256-GCM encrypted under a passphrase that is
   on no disk: not in a script, not in an env file, not in argv. It is typed
   interactively at grading time. Demonstrated: an agent that defeats control 1
   — which it can, it owns the directory — finds only ciphertext.
3. **Not changeable after the fact.** A timestamped SHA-256 commitment over
   every key file is committed to the OASiS repo *before* the campaign runs, at
   `data/blind_key_commitment.json`. The solutions are never published; the
   hashes prove they were fixed in advance, which answers "how do we know you
   did not derive the solutions after seeing the results?" checkably instead of
   on trust. The commitment is over the **plaintext**, so a third party handed a
   key file can verify it with `scripts/blind_keys.py verify` (hashlib and json
   only, no dependencies).

`run_blind.py`'s `keys_are_sealed()` returns True when `keys/` is **missing or
empty**, so a deleted keys tree reads as sealed and the campaign would start
with nothing to grade against. Absence is not a seal. `blind_eval.keyvault.is_sealed`
requires the directory to exist *and* be unlistable.

## 5. Convergence order no longer requires the key

§5 derives the observed order from the **true** error, so no order could be
computed at all until the keys were unsealed. Per the owner's instruction that
convergence be tested "by heuristics and mesh halving", `scripts/blind_grade.py`
adds a first phase that derives the order by Richardson mesh halving from the
agent's own reported values, with no exact solution anywhere. Phase 2 opens the
key and computes the true error as before.

Running both is strictly better than either. A disagreement between the key-free
and key-based orders means the sealed solution, the problem statement, or the
run is wrong — a class of error neither phase catches alone. Verified: a
synthetic second-order run over 1024 probe points returns 2.0000 with the keys
still sealed.

### Analysis artefacts leak keys too

The exposure sweep, pointed at the agent-scratch tree rather than the campaign
directory, found **seven plaintext copies of live solutions** outside `keys/`:
sub-agent transcripts that had quoted key files, a `keys_snapshot.py`, its
`__pycache__`, and a previous agent's `attack/repro/keys/B7/key.json`. None was
in the campaign tree; all were readable by any process running as this user.

Custody has to cover the scratch space that tooling writes, not just the
directory the design names. `scripts/blind_keys.py exposure --root <dir>` takes
a root for exactly this reason. Purged; the sweep now reports zero.

---

# Amendment 2 — 2026-08-07. The coupled set was graded on quantities that cannot fail

Made before any evaluation run. The campaign has still produced zero results, so
nothing is invalidated and no result was seen before these changes.

Two findings arrived independently and turned out to be the same defect seen
from two directions.

## 1. The interface verification was a tautology

Amendment 1 §2 already recorded it as a note: for D1, D2 and D4 the
`verify_transmission` / `verify_traction_continuity` calls pass the **same
field** and the **same material** to both sides, so `jump_u = 0; jump_flux = 0`
holds for any input whatever. What Amendment 1 did not do is act on it. It is
not a weak check. It is not a check.

D3 was the only instance with content, because its conductivity jumps 1:4 and
its two subdomain fields are genuinely different functions that agree only ON
the interface. That is the whole construction principle, and it is now the rule
rather than the exception:

1. different materials on the two subdomains,
2. `u_A` chosen freely,
3. `u_B` **forced** by both interface conditions — continuity and flux
   equilibrium. With `k_A != k_B` the normal derivatives must differ, so `u_B`
   is necessarily a different function, and continuity becomes a constraint the
   construction has to satisfy rather than a statement about one function
   evaluated twice,
4. sources derived symbolically per subdomain,
5. everything verified by substitution — symbolically to exact zero **and**
   numerically in floating point to machine precision.

D3's hand-solved quadratic is generalised by a **flux potential**. For a
coefficient that factorises as `k(x,y) = kappa(x) * lambda(y)`, put
`eta' = 1/kappa`, `zeta' = 1/lambda` and take `u = U(eta(x), zeta(y))`. Then `u`
and `k du/dn` are continuous across every material line for *any* `U`, including
where two lines meet. This buys three things the bespoke construction cannot:
hidden fields drawn at random, several interfaces at once, and interfaces in
both coordinate directions meeting at a point. The vector instance cannot use it
— a traction is not linearised by one potential — and keeps its own
construction, in which matching the normal traction forces `lambda` to be shared
so the whole contrast lives in `mu`.

**The verification now refuses to certify a tautology.** Every transmission
check returns `VACUOUS`, not `PASS`, when the two sides carry the same
expression or the same material, or when the interface value or flux is
identically zero. A harness that cannot notice it is checking nothing is how
this survived review the first time. Each check also ships an antidote: perturb
the construction and require the check to fail.

## 2. The graded quantity could not move

Independently: of 30 coupling fixtures, 21 use `heat_arrangement` and 2 contain
any vector construct at all, and a conductivity mutation left every internal
check green with the answer wrong.

The mechanism is exact. In the Dirichlet-driven split conduction problem the
band-graded coupled cells use, the interface temperature is
`(c_l T_l + c_r T_r) / (c_l + c_r)` with `c = k / L`, so it depends on the two
conductivities **only through their ratio**. Scaling both leaves it exactly
invariant. Measured:

| both conductivities scaled by | interface T moved | interface flux moved | ±20% band on T |
|---|---|---|---|
| 2.0 | 0.0000% | 100% | PASSES |
| 5.0 | 0.0000% | 400% | PASSES |
| 0.25 | 0.0000% | 75% | PASSES |

A ±20% band on interface temperature cannot separate a correct coupling from one
with 100% wrong material data. The flux catches every one of them.

### What the mutations showed about the primary grade

`scripts/mutate_coupling.py` runs real partitioned Dirichlet-Neumann solves,
correct and mutated, at three mesh levels, and measures every candidate grading
quantity. On the two-material instance (`k_A = 1`, `k_B = 4`):

| variant | order from mesh halving | true-error order | two-sided flux jump | interface T moved |
|---|---|---|---|---|
| correct | 1.838 | 1.884 | 1.6e-14 | — |
| wrong transmitted quantity | 1.865 | 0.112 | **3.00** | 29.6% |
| reversed interface mapping | 1.847 | 0.405 | **0.75** | 0.000% |
| sign error on the flux | 1.844 | 0.002 | **2.00** | 54.4% |
| receiver's coefficient 25% wrong | 1.847 | 0.002 | 1.6e-14 | 16.6% |

Three things follow, and all three are measurements rather than arguments.

* **Order from mesh halving cannot discriminate at all.** Every mutation
  self-converges at order ~1.85, indistinguishable from the correct run. A
  partitioned scheme that iterates the wrong transmission condition converges
  cleanly to the wrong fixed point. This is why order-only grading is never
  sufficient for a coupled cell.
* **The two-sided interface flux jump is the reference-free discriminator.**
  Its correct limit is known to be zero without any solution, which is exactly
  what mesh halving does not provide. It catches the two interface-mechanism
  mutations with the keys still sealed. It does *not* catch a wrong coefficient
  — that is a consistent coupling of the wrong problem, and only the true error
  finds it. No single quantity catches everything, and this is stated rather
  than papered over.
* **A scalar summary of the interface is blind to a reversed mapping.** The mean
  interface temperature and the net interface flux were both invariant to six
  digits under `MUT_MAP`, because reversal permutes the profile and leaves its
  sum alone. The full profile moves. This is why the graded quantity is the
  profile and not one number.

### The measurement that proves the tautology, rather than asserting it

The same mutation was run on an instance with the **same material on both
sides** — the shape D1, D2 and D4 have. Exporting the raw normal derivative and
re-applying the receiver's conductivity is then *exactly the identity map*.
Every reported quantity — interface temperature, net flux, flux jump, order,
true error, iteration count — agreed with the correct run in every digit
printed. The old instances cannot detect the bug at all, for the same reason
their verification cannot fail.

## 3. The redesigned coupled family

Eight instances, all carrying an exact manufactured solution. Every one is
verified by substitution symbolically to exact zero and numerically to
machine precision, in both subdomains and in every interface condition.

| id | codes | physics | dim | interface | contrast |
|---|---|---|---|---|---|
| D1 | FEniCSx + NGSolve | anisotropic diffusion, TENSOR jump | 2 | straight | K11 1:3, K22 2:5 |
| D2 | FEniCSx + deal.II | two-material conduction | 3 | plane | 1:6 |
| D3 | FEniCSx + Kratos | two-material conduction | 2 | straight | 1:4 |
| D4 | FEniCSx + deal.II | elasticity, SHEAR-MODULUS jump (VECTOR) | 2 | straight | mu 1:3 |
| D5 | FEniCSx + scikit-fem | four materials, NOTCHED domain, BENT interface | 2 | L-polyline | 1:5/2 and 1:2 |
| D6 | NGSolve + Kratos | two-material conduction, SEVERE contrast | 2 | straight | 1:1000 |
| D7 | FEniCSx + NGSolve | diffusion coupled to REACTION-diffusion | 2 | straight | 1:3 |
| D8 | FEniCSx + deal.II | TRANSIENT two-material heat | 2+t | straight | 1:4 |

D1, D2 and D4 replace tautological instances of the same id. D3's physics is
unchanged and its hidden field was redrawn. D5 to D8 are new.

**D5 is the arrangement that is not the 2D rectangle.** The global domain is the
unit square with a corner square removed, so it is non-convex; subdomain A is
therefore not a rectangle either; the interface is a bent polyline whose two
legs have different outward normals *and* different material contrasts; and four
material cells meet at a cross point. Every material line and the notch lie on
mesh lines at every prescribed level, so nothing is paid in geometric error.

**D4 is the vector interface.** The shear modulus jumps by a factor three while
`lambda` is shared — which the construction forces, not assumes — so both the
displacement vector and both components of the traction are binding conditions.
The shear traction at the interface is verified NONZERO, so the instance cannot
be passed by treating it as two scalar problems.

**D7 has different operators on the two sides**, which is what real multiphysics
looks like: pure diffusion on A, diffusion with a linear reaction on B. The
reaction term enters neither transmission condition, so the construction is
untouched while the two subdomains genuinely solve different equations.

**D8 is transient and still exact.** `u = T(t) U(eta(x), y)` with `T(0) = 0`, so
the initial datum is `u = 0` everywhere and discloses nothing. `T(t) = t e^{t/2}`
is deliberately not a low-degree polynomial: a quadratic `T` would make
Crank-Nicolson exact in time and the instance would silently stop testing the
time discretisation.

## 4. Three grades of evidence, never pooled

* **Grade 1 — exact manufactured solution.** Proves the answer is CORRECT.
* **Grade 2 — monolithic reference.** Proves the SPLIT did not change the
  answer. Weaker: a discretisation bug shared by both solves makes them agree
  while both are wrong.
* **Grade 3 — mesh-halving order alone.** Proves the scheme CONVERGES, not that
  it converges to the right thing. Measured above: every mutation converges at
  order ~1.85. Never sufficient alone for a coupled cell.

Every task carries its grade in `spec_public.json` and in its key, the grader
reports it, and a grade-3 cell must never be pooled into an aggregate with
grade-1 cells. If that makes a table look uneven, the table is telling the
truth. All eight coupled instances above are grade 1. The retained rarefied-gas
instance has no closed form and is grade 3, labelled.

## 5. What the agent additionally reports, and why it is not a leak

Coupled tasks now require, per level and per subdomain, the interface field and
the subdomain's OUTWARD normal flux (traction, for the vector instance) at fixed
interface probe points, plus a per-level partitioned-iteration residual history.
These are outputs, not inputs: they disclose nothing, and every one of them
passes the leak gate. The flux is what a partitioned scheme already computes,
so it costs the agent nothing it was not doing.

Because the flux is agent-reported it could in principle be fabricated as
`q_B = -q_A`. The grader therefore recomputes the interface flux from the
agent's own submitted FIELD values by one-sided quadratic extrapolation and
compares. Faking the flux then requires faking a consistent field.

## 6. Harness defects found while doing this, and fixed

**The probe grid was aliasing with the mesh, and biased every observed order
down.** `PROBE_M` was `{2: 32, 3: 16}` against mesh levels 8/16/32. At the
finest level the probe grid has exactly the mesh's own spacing and sits at the
midpoint of each quad's diagonal — the worst point of the P1 interpolation
error — so the finest error is systematically over-estimated. Measured on four
manufactured fields solved on the prescribed sequence:

| field | true L2 order | M = 32 | M = 44 |
|---|---|---|---|
| A | 1.971 | **1.705** | 1.970 |
| B | 1.982 | **1.705** | 1.982 |
| C | 1.979 | **1.714** | 1.980 |
| D | 1.983 | **1.710** | 1.983 |

With `tol = 0.4` that bias eats two thirds of the tolerance budget: a correct
second-order run is graded 1.71, and a slightly rougher field would be graded
CONFIDENTLY_WRONG. `PROBE_M` is now `{2: 44, 3: 21}`, sharing no factor with any
mesh level, and `assert_probe_grid_incommensurate` refuses a commensurate
choice. 44 also avoids putting a probe exactly on `x = 1/2` or `3/4`, which 45
does.

**`spec_public.json` was never written.** `scripts/blind_grade.py` reads it to
get `dim` and the code list, and fell back to `dim = 2` — so the 3D instance
would have been parsed with the wrong column count and graded on nonsense. The
builder now writes it.

**The two-code evidence gate was defeatable by writing a text file.**
`code_ran` was a case-insensitive substring scan for the code's name over any
`.log/.out/.txt/.json`, excluding four filenames. The task statement names both
codes, so the agent's own notes contain both as a matter of course. Combined
with a tautological coupled instance, the rational strategy for either arm was:
solve the whole problem monolithically in one code, write any file mentioning
the second, submit — order 2, all gates green, and the cell has measured
single-code PDE solving. It is now `blind_eval.evidence`, which requires solver
output carrying a number the solver computed, requires the two codes' evidence
to come from different files, and requires a partitioned-iteration residual
history that has at least three iterations, falls by at least a factor of ten,
ends at or below the prescribed tolerance and agrees with the iteration count
the agent reports. A monolithic solve has no such history at all. This raises
the cost of faking from "write one text file" to "implement a plausible
partitioned iteration and log it"; it is not proof of execution and is not
described as such, which is why its verdicts are three-valued.

**`run_blind.py` treated a MISSING keys directory as sealed.** Amendment 1 §4
described this as fixed. The fix existed in the OASiS repo, under test, and the
runner never imported it — it carried its own copy with
`if not keys.exists(): return True`. The runner now imports
`blind_eval.keyvault.is_sealed`, runs the full custody preflight before any
cell (seal verified by execution, no plaintext keys, builders unreadable,
commitment present, exposure sweep clean), re-sweeps after **every** cell
because a run creates scratch that did not exist at launch, and halts the
campaign if the keys become readable mid-run.

**The leak audit never reached campaign 3.** `audit_leaks.py` globbed `single/`
and `matrix/`; campaign-3 runs live under `campaign3_blind/runs/`. Its
`BAD_BOTH` regex was dead code — `for m in set(BAD_BOTH.findall(txt)) if False
else []` — compiled and never evaluated. And `leak_invalidate.py` did
`d["outcome_pre_leak_audit"] = d["outcome"]` on ledgers that have no `outcome`
key, which every campaign-3 ledger is, so the first tainted campaign-3 run would
have raised `KeyError` and left the rest marked clean. All three are fixed, and
the audit now also compares the agent's context against the sealed solution
symbolically rather than as a substring.

## 7. Custody: the builder is no longer a leak surface

Amendment 1 found that `build_problems.py`, `build_extra.py` and
`build_coupled.py` hold every hidden field as a Python literal two directories
above the agent's working directory. It sealed them with `chmod 000`, which
stops an agent that wanders and not one that decides to attack, and which also
means they cannot be version-controlled — so a "frozen, pre-registered" design
sat in a directory that could be silently edited.

`build_coupled_v2.py` holds **no solution**. The hidden fields are drawn from a
CSPRNG at build time and the draw seed is written only into the sealed key.
Reading the builder tells you the family, never the instance. That is what makes
it committable, which is what makes the design actually pre-registered rather
than merely declared to be. The campaign directory — design, runner, grader,
phase gate, builder, task texts, public specs — is now in the OASiS repo. Keys,
runs and the legacy plaintext builders stay out, and the legacy builders are
covered by a hash commitment so they cannot be silently edited either.

## 8. Still not enforced, stated honestly

* **Blinding is still not cryptographic.** A closed-form source term and a
  low-parameter manufactured field can be inverted by assuming an ansatz and
  matching coefficients. Randomising the coefficients raises the work; it does
  not change the kind. The defensible claim is unchanged and narrow: no
  instance hands over its solution, and none is one algebraic step away.
* **The evidence gate is not proof of execution.** See §6.
* **The seal is still not a sandbox.** The agent runs as the directory's owner.

## 9. Every instance was solved before it was allowed near a paid run

Verifying a manufactured solution's algebra says the PROBLEM is right. It does
not say the problem is GRADEABLE. Each instance was therefore solved on the mesh
sequence and probe grid the task prescribes, and both grading phases computed:

| id | contrast | phase 1 (mesh halving) | phase 2 (true error) | agree | flux jump | solver |
|---|---|---|---|---|---|---|
| D1 | K11 1:3, K22 2:5 | 1.830 | 1.854 | yes | 1.2e-14 | partitioned DN |
| D2 | 1:6 | — | — | — | — | not solved here (3D) |
| D3 | 1:4 | 1.840 | 1.862 | yes | 1.3e-14 | partitioned DN |
| D4 | mu 1:3 | 1.827 | 1.909 | yes | — | monolithic reference |
| D4 | mu 1:3 | 1.838 | 1.838 | yes | 5.2e-15 | partitioned DN (VECTOR) |
| D5 | 1:5/2 and 1:2 | 1.889 | 1.941 | yes | — | monolithic, notched domain |
| D6 | 1:1000 | 1.960 | 1.978 | yes | 6.0e-15 | partitioned DN |
| D7 | 1:3 | 1.817 | 1.847 | yes | 1.7e-14 | partitioned DN |
| D8 | 1:4 | — | — | — | — | not solved here (transient) |

Four of them were additionally re-solved **from their sealed keys**, so what was
measured is the instance that will actually be graded and not a rebuild from the
same family: D1 1.830/1.854, D3 1.839/1.869, D6 1.961/1.977, D7 1.854/1.875, all
agreeing between phases with a flux jump at roundoff. That mode prints orders
only and is how a shipped instance can be checked without anything being
disclosed.

The key-free and key-based orders agree everywhere they are both defined, which
is the cross-check that catches a wrong sealed solution. The two-sided flux jump
of a *correct* partitioned coupling is at roundoff, which is what makes an O(1)
jump a usable signal rather than a threshold argument.

D2 and D8 are not solved by the verification solver, which is 2D and steady.
They are verified symbolically to exact zero and numerically by finite
differences including the time derivative, and their construction is shared
exactly with D3, which is solved. That is stated rather than omitted.

## 10. The vector coupling path has never executed

Of the ten shipped coupling participants, **one** — FEBio — implements a vector
interface. `participant_fenics.py`, `participant_dealii.py`,
`participant_ngsolve.py`, `participant_kratos.py`, `participant_skfem.py`,
`participant_dune.py` and `participant_fourc.py` are conduction-only. Of 29
coupling fixtures, 2 contain any vector construct and 14 are the same split
conduction problem.

D4 names FEniCSx and deal.II, and neither has a vector participant. So the
generic coupling path would execute its first-ever vector interface for those
codes inside the paid, graded campaign, where a tool bug reads as agent failure
and is charged to the OASiS arm.

What is established here is narrower and is stated as such: **the arrangement is
solvable by exactly the scheme D4 describes.** A partitioned Dirichlet-Neumann
iteration exchanging a displacement vector and a two-component traction across
the shear-modulus jump converges in 23-24 iterations at every level, reproduces
the manufactured solution at order 1.838, and closes the traction jump to
5e-15. D4 is therefore a fair task. Whether OASiS's own participants can serve
it is a separate question, it is currently answered NO for seven of nine codes,
and it must be answered before D4 is run — that work belongs to the vector
coupling participants, not to the problem set.

## 11. One defect of this amendment's own making, and its guard

The coupled builder was made solution-free so that a pre-registered design could
be version-controlled: the hidden fields are drawn from a CSPRNG at build time
and only the draw seed reaches the sealed key. That property was then destroyed
by hard-coding the seed in a verification script, which was committed. A seed in
the repository re-derives every hidden field by re-running the builder,
whatever the builder itself holds.

Found while auditing this amendment's own work, before any run. The instances
were rebuilt from a fresh CSPRNG draw whose seed is in no file and was never
printed, the scripts and tests now use a clearly-labelled DEMONSTRATION seed
that is not the campaign's, and
`tests/test_blind_campaign_integrity.py::test_no_live_draw_seed_appears_in_any_tracked_file`
reads every live key's `draw_seed` and fails if it appears in any tracked file.
The check is a test rather than a review note because a control that depends on
somebody remembering is not a control — which is the same reasoning §4 of the
original document applies to the phase gate.

The seed that built the superseded instances remains in this branch's git
history. Those instances no longer exist, so it recovers nothing.

## 12. The campaign's own lifecycle, and what must happen before it runs

The problem set is BUILT, not yet sealed. In order:

1. `scripts/campaign_custody.py commit` — hashes the design, runner, grader,
   phase gate, builder and every task text into
   `data/blind_campaign_commitment.json`. Done.
2. `scripts/blind_keys.py commit` — hashes every key file into
   `data/blind_key_commitment.json`. Done.
3. `scripts/blind_keys.py encrypt` — **operator action, requires the
   passphrase.** The eight rebuilt coupled keys are currently plaintext on
   disk. `test_no_plaintext_keys_are_left_on_disk` fails until this is run, and
   that failure is the control working.
4. `scripts/blind_keys.py seal`.
5. `run_blind.py` — its preflight now refuses on any of: keys not sealed, seal
   not surviving an executed read, plaintext keys present, builder sources
   readable, missing commitment, a non-empty exposure sweep, or a coupled task
   whose intended path has no recorded throwaway run (§10).

---

# Amendment 3 — 2026-08-07. The vector path exists; three of its findings change D4

The vector coupling participants landed on `feature/vector-coupling`, which
closes the blocker §10 recorded. Verified by execution here rather than taken
from a report: `vector_pair_fenics_dealii` — **D4's exact pair** — was run in
this session and passed, covering both arrangements, vector exchange through the
registered `couple` tool, non-matching interface meshes, displacement continuity
and traction equilibrium on the interface interior, agreement with an un-split
monolithic solve, and `failures_count=0`.
`vector_traction_recovery_at_the_interface_ends` was also run here and passed.

`path_readiness.json` records **D4 verified**, from those runs. The other seven
stay false: their paths have no throwaway run recorded, and the preflight refuses
them.

## 1. The interface ENDS are not gradeable, and now are not graded

Where a Dirichlet-Neumann interface meets a constrained outer boundary, the
Neumann subproblem acquires a corner the monolithic problem does not have — a
Dirichlet-Neumann corner, which carries a flux (scalar) or stress (vector)
singularity. Measured over a 4x refinement: the displacement converges
(1.22e-02 -> 2.34e-03) while the exported traction at the interface END gets
**worse** (2.11x -> 2.51x the true value). Refinement does not fix it, because
it is a property of the split rather than a discretisation error.

Grading those points would have failed a **correct** submission — the same class
of false `CONFIDENTLY_WRONG` as the interface that lay on no mesh line, and for
the same reason: a defect of the instance charged to the agent. Every coupled
instance's interface probes now cover the interface **interior** only, with a
clearance of a quarter of the interface length at each end (two elements at the
coarsest prescribed level, eight at the finest). D5's two legs each get the same
treatment, since one end of each meets the outer boundary and the other meets
the corner where the legs join. The band is recorded in each `spec_public.json`
as `iface_graded_band`, and the grader refuses interface points outside it —
excluding the ends must not become a way for an agent to choose where it is
measured.

The clearance is a fixed fraction of the interface, not a multiple of `h`,
because the corner is a property of the continuum problem.

## 2. Relaxation from the worst component

A vector interface has one interface-stiffness ratio per component: the P-wave
modulus governs the normal direction and `mu` the shear. Measured on the
fixtures, `rho_x = 3.328` against `rho_y = 0.311` — a 10.7x spread — and the
theta chosen from the smaller ratio diverges in the stiff component
specifically. So `theta = 1 / (1 + max_c rho_c)`, which is now what
`blind_eval.femdd.vector_theta` computes and what the verification uses.

For D4 as specified the spread is 1.40x (`rho_normal = 0.653`,
`rho_shear = 0.467`), so the instance is not itself in the dangerous regime and
the worst-component choice coincides with the normal-component one — which is
what the earlier D4 verification happened to use. Stated because it is a
measurement, not a reassurance: a different material pair would not be so
forgiving.

The task now carries a NOTE that the two components' interface stiffness ratios
differ and a single scalar relaxation factor can diverge in the other component.
That discloses nothing: both ratios follow from the Lame parameters and
subdomain widths already printed in the task. It goes in the TASK, not the
served knowledge, so both arms get it — a correct agent should not lose a run to
a property of the problem it was not told about.

## 3. Interface corners belong to the outer boundary on both sides

Getting this wrong produces a run that converges, balances the interface to
1e-10, and is still 4.7% off in displacement and 28% off in traction — a
silently wrong answer that passes every internal check, which is precisely the
failure mode this campaign exists to detect rather than to inflict. It is a
statement about the stated boundary value problem, not a hint, so every coupled
task now says it explicitly:

> INTERFACE CORNERS: at the points where the interface meets the outer boundary,
> the outer boundary condition above applies on BOTH subdomains. Those points
> belong to the outer boundary, not to the interface, on either side.

## 4. What is still unproven, taken from the fixture evidence

* **Only four backends have a vector participant** — FEniCSx, scikit-fem,
  NGSolve, deal.II (plus FEBio, pre-existing). 4C, DUNE-fem, Kratos and SPARTA
  have none. That is absence of evidence, not demonstrated inability, and is
  recorded as such. D4 names FEniCSx and deal.II, both covered.
* **There is no 3-D coupled solver run.** The surface quadrature is verified
  arithmetically and by unit tests, but every vector participant is 2-D. D2 is
  a *scalar* 3-D instance, so the vector work does not bear on it either way;
  its path remains unproven and the preflight still refuses it.
* The seven non-D4 instances have no recorded throwaway run through their named
  path.
