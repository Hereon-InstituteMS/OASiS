# Contributing to OASiS

Thanks for the interest.  This MCP server gets more useful as more
people add verified knowledge — but a single wrong parameter key in the
catalog produces silently-broken input files for every downstream agent
session.  The rules below are designed to keep that from happening.

For the *why* behind these rules, see the [design paper](README.md#citation)
(Section 3.1 — generality, pitfall awareness, mandatory quality control,
vendor agnosticism).  This file is the *how*.

## Quick checklist for any PR touching the knowledge base

- [ ] Every new or changed parameter key / keyword string is **cited to a
      source-of-truth file:line** (4C `4C_mat_*.cpp`, dolfinx attribute,
      FEBio XML schema, etc.).  Cite in the commit message.
- [ ] `pytest tests/test_catalog_consistency.py -v -s` passes locally
      (with whichever backends you have installed; the rest skip
      gracefully).
- [ ] The PR description states *what changed and why* — what failure
      mode the entry catches, which source file was checked, which
      probe ran.
- [ ] At least one independent reviewer (human or sub-agent) has run
      through the change before merge.  See **Mandatory critic gate**
      below.

## What goes in the catalog

The catalog has three layers; each accepts a different kind of
contribution.

| Layer | Lives in | What goes here |
|---|---|---|
| **Physics catalog** | `src/backends/<solver>/generators/*.py` and `src/tools/deep_knowledge.py` | What the FE code *can do*: registered materials, element types, time integrators, section names.  Every named string must exist in the backend's source. |
| **Pitfall DB** | `src/tools/deep_knowledge.py` and per-backend pitfall lists | What goes *wrong*: silent input-parser mismatches, locking modes, version-specific footguns.  Each entry should be code-agnostic ("this pattern fails on solver X for reason Y") not problem-specific recipes. |
| **Operational guide** | YAML / Python templates in `src/backends/<solver>/generators/*.py` | *How to drive* the FE code: validated input templates the agent can fill in. |

**Generality rule.**  Every knowledge entry must benefit *all* simulations
on its target backend, not be fine-tuned for a specific example.  No
"Mg-10Gd compression test parameters" entries; those are problem
setups, not catalog knowledge.

## The verification gate

`tests/test_catalog_consistency.py` is run automatically on every PR
that touches `src/backends/**`, `tests/groundtruth/**`, or the test
itself (see `.github/workflows/knowledge-freshness.yml`).  It cross-
checks every catalog claim against the backend's source-of-truth.

Two probe families currently exist; see `tests/groundtruth/__init__.py`
for the extension recipe.

- **Source-grep** (used for 4C, eventually deal.II / FEBio): grep the
  C++/Fortran source for the canonical names the input parser
  accepts.  `tests/groundtruth/fourc.py` is the worked example.
- **Python introspection** (used for scikit-fem, eventually FEniCSx /
  NGSolve / Kratos / DUNE-fem): import the package, list public
  classes/functions, assert every catalog mention is real.
  `tests/groundtruth/skfem.py` is the worked example.

If you add or change a catalog entry, the test will refuse to merge
the PR until the entry references only names that exist in source.
If you add a new backend, add a probe module + a test class
(typically ~50 lines).

### Strict mode

By default the test surfaces "catalog *under-declares* a required
parameter" as a stderr warning.  Run with `OFA_STRICT_CATALOG=1` to
promote those warnings to hard failures.  Use it locally when
enriching a catalog entry — that way the test fails until you've
declared every required key.

## Mandatory critic gate

Catalog edits authored without independent review have repeatedly
shipped factually wrong keys (lowercase vs uppercase, wrong section
name, missing required parameters).  Every PR with catalog changes
must have at least one of:

1. **An independent human reviewer** on the PR.
2. **A sub-agent critic** invoked via Claude Code / Cursor / Windsurf
   with a "ruthlessly critical" system prompt, looking up the cited
   sources and confirming the entries match.  Quote the critic's
   verdict in the PR description.

A green CI run alone is *not* enough — the test catches structural
drift, not factual hallucination in descriptions, units, ranges.

## How to enrich an existing catalog entry

Worked example — adding a missing required parameter:

1. Run the consistency test to see what's flagged as under-declared:

   ```bash
   pytest tests/test_catalog_consistency.py::TestFourcMaterialParameters::test_required_keys_present_in_catalog -v -s
   ```

2. For each under-declared key, find its declaration in the backend
   source:

   ```bash
   curl -s https://raw.githubusercontent.com/4C-multiphysics/4C/main/src/mat/4C_mat_<name>.cpp \
     | grep -E 'parameters\.(get|get_or)<[^(]+>\("KEY_NAME"'
   ```

3. Note required (`get`) vs optional (`get_or`) and the type.  Read the
   surrounding context for default values, allowed value ranges, and any
   `FOUR_C_THROW` that rules out parameter combinations.

4. Add the entry to the appropriate generator's `materials` block with
   `description`, `range`, and (if non-obvious) a comment citing the
   source file:line.

5. Re-run the test — should pass.  Commit with a message that includes
   the source citation.

## How to add a new pitfall

A pitfall describes a specific failure mode the agent should avoid.

1. Reproduce the failure once (manually or via an existing E2E test).
2. Run the 5-question post-mortem from
   [E2E_POSTMORTEMS.md](E2E_POSTMORTEMS.md): what went wrong, which
   workarounds, which tools were useful, what online info was needed,
   what would you do differently.
3. Strip the entry to the code-agnostic kernel — "X happens when Y is
   set with Z" — not "for this specific problem with these specific
   parameters".
4. Add to the relevant generator's `pitfalls` list with a one-liner
   capturing the rule.  If a longer explanation is needed, add it as
   a multi-paragraph entry in `deep_knowledge.py`.

## How to add a new backend probe

The MCP currently covers 4C and scikit-fem in the verification test.
The remaining seven backends (deal.II, FEniCSx, NGSolve, Kratos,
DUNE-fem, FEBio) have stub placeholders in
`tests/groundtruth/_stubs.py`.

Recipe:

1. Identify the backend's *source of truth* for names the catalog
   promises:
   - For C++/Fortran-source backends (deal.II, FEBio): a specific set
     of source files that enumerate the input keywords.
   - For Python-importable backends (FEniCSx, NGSolve, Kratos,
     DUNE-fem): the top-level package attributes (classes, functions,
     constants).
2. Write `tests/groundtruth/<backend>.py` exposing one zero-arg probe
   per categorical section (e.g. `element_types()`,
   `solver_strategies()`).  Each probe returns `None` when the source
   is unreachable so the test skips rather than fails.
3. Add a `Test<Backend>...` class to
   `tests/test_catalog_consistency.py` that collects the catalog's
   claims and asserts subset-of-source.
4. If your backend is Python-importable, add it to the install line in
   `.github/workflows/knowledge-freshness.yml` so the test actually
   runs in CI.
5. Remove the corresponding stub from
   `tests/groundtruth/_stubs.py`.
6. Open a PR.  CI will run the new test against the live source.

## Commit message conventions

Follow the existing repo style: sentence-case title starting with a
verb, no trailing punctuation, body wrapped at ~72 cols.

For catalog changes, the body should cite each source file:line you
checked.  Example:

```
Fix MAT_Struct_PlasticGTN parameter keys

f0/fn/fc/kappa/k1-3 → F0/FN/FC/KAPPA/K1-3 (4C requires uppercase, see
src/mat/4C_mat_plasticgtn.cpp:45-54).  Added the five missing required
keys (DENS, HARDENING_FUNC, EF, MAXITER, TOL) per the same source.
```

## Code style

- Python: follow PEP 8, type hints on new public functions.
- Bash scripts: use `set -u` at minimum; wrap `cd` in subshells so
  cwd doesn't leak across commands.
- YAML workflows: validate locally with `python -c 'import yaml;
  yaml.safe_load(open("..."))'` before pushing.

## Where to ask questions

- Bugs in existing catalog entries: open an issue with a source
  citation showing the mismatch.
- Design questions about extending the MCP: open a discussion (or
  reference Section 3 of the paper).
- Anything else: feel free to open a draft PR — the maintainers prefer
  concrete code to abstract proposals.
