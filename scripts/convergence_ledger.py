#!/usr/bin/env python3
"""Is the knowledge converged yet? Round over round, as a number.

The freeze rule is "keep running verification rounds until a round stops
finding things". That is only actionable if each round records what it checked
and what it found, so the next one can be compared against it. Otherwise
"converged" becomes a feeling, and the feeling arrives exactly when everyone is
tired of looking — which is the failure this whole effort exists to avoid.

WHAT A ROUND IS
A round is a full pass over a backend by an agent that EXECUTES claims rather
than reading them, followed by an independent adversarial audit of that pass.
Round 1 was the extraction/verification pass plus the seven audits.

THE METRIC MUST BE COMPUTED HERE, NOT REPORTED BY THE AGENT
An audit of the deal.II pass established why. That pass reported 566 of 817
claims executed — 69.3%. Its own two ledger scripts printed 449/817 = 55.0% and
184, and neither yielded 566. Worse than the disagreement, the metric was a
category error: it summed EXECUTION counts against a denominator of CLAIM
STRINGS. None of the 817 claim ids mentioned "essentials" though 55 leaves were
credited; none were step citations though 79 were credited; several entries were
experiment counts. The one defensible figure was the one the script itself
printed for pitfalls: 43/133 = 32.3%.

So a self-reported percentage is not comparable between backends, and a freeze
criterion built on incomparable numbers measures nothing. `surface_covered` is
therefore defined as ONE thing that this script computes from the tree:

    pitfall claims with an execution fixture behind them / pitfall claims total

Pitfalls are the right unit because they are what the paper sells, they are
countable without judgement, and a fixture is unambiguous evidence of execution.
An agent's own richer accounting is welcome in its report; it does not enter the
criterion.

THE METRIC
For each backend, per round:

  checked      — claims actually executed this round (not claims that exist)
  falsified    — executed claims that did not hold
  rate         — falsified / checked

and separately, because they are different failure modes:

  audit_defects       — errors the ADVERSARIAL PASS found in that round's own
                        corrections. This is the number that matters most: it
                        says whether the people fixing things are getting them
                        right. Round 1 was non-zero on every single backend.
  unmatchable_signals — Signal clauses quoting strings the software cannot emit
  contamination_hits  — measured answers, machine paths, campaign references
                        reachable by an agent
  surface_covered     — executed / total claims. A 2% rate on 5% of the surface
                        is not convergence, it is a small sample.

FREEZE CRITERION — all four, per backend:
  * rate               <= 0.02   (a round barely finds anything)
  * audit_defects      == 0      (the corrections are right)
  * unmatchable_signals== 0      (no invented diagnostics)
  * contamination_hits == 0      (nothing readable as an answer)
  * surface_covered    >= 0.80   (and it looked at most of the surface)

The last one is the guard against declaring victory by checking less. Round 1
found the damage concentrated in the fraction nobody executed: FEBio's 25
invented error messages and 11 non-existent material names were all in the 73%
its pass never touched. A round that checks a smaller surface will find fewer
problems and mean nothing by it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "data" / "convergence"

THRESHOLDS = {
    "rate": 0.02,
    "audit_defects": 0,
    "unmatchable_signals": 0,
    "contamination_hits": 0,
    "surface_covered": 0.80,
}


def load_rounds() -> dict[int, dict]:
    rounds = {}
    if LEDGER.exists():
        for p in sorted(LEDGER.glob("round_*.json")):
            n = int(p.stem.split("_")[1])
            rounds[n] = json.loads(p.read_text())
    return rounds


def backend_verdict(entry: dict) -> tuple[bool, list[str]]:
    """Would this backend pass the freeze criterion on this round's numbers?"""
    fails = []
    checked = entry.get("checked") or 0
    total = entry.get("total_claims") or 0
    falsified = entry.get("falsified")
    if checked and falsified is not None:
        rate = falsified / checked
        if rate > THRESHOLDS["rate"]:
            fails.append(f"falsification rate {rate:.1%} > {THRESHOLDS['rate']:.0%}")
    else:
        fails.append("no falsification rate recorded")
    for key in ("audit_defects", "unmatchable_signals", "contamination_hits"):
        v = entry.get(key)
        if v is None:
            fails.append(f"{key} not measured")
        elif v > THRESHOLDS[key]:
            fails.append(f"{key}={v}")
    if total:
        cov = checked / total
        if cov < THRESHOLDS["surface_covered"]:
            fails.append(f"only {cov:.0%} of the surface executed "
                         f"(need {THRESHOLDS['surface_covered']:.0%})")
    else:
        fails.append("total claim count unknown, so coverage cannot be judged")
    return (not fails), fails


def main() -> int:
    rounds = load_rounds()
    if not rounds:
        print(f"No rounds recorded under {LEDGER}.")
        return 1

    latest = max(rounds)
    print(f"Rounds recorded: {', '.join(str(r) for r in sorted(rounds))}\n")

    hdr = (f"{'backend':<10} {'checked':>8} {'falsified':>10} {'rate':>7} "
           f"{'audit':>6} {'sig':>5} {'contam':>7} {'cover':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))

    ready = []
    for be, e in sorted(rounds[latest].get("backends", {}).items()):
        checked = e.get("checked") or 0
        fals = e.get("falsified")
        total = e.get("total_claims") or 0
        rate = f"{fals / checked:.1%}" if (checked and fals is not None) else "?"
        cov = f"{checked / total:.0%}" if total else "?"
        ok, why = backend_verdict(e)
        ready.append(ok)
        print(f"{be:<10} {checked:>8} {str(fals):>10} {rate:>7} "
              f"{str(e.get('audit_defects')):>6} "
              f"{str(e.get('unmatchable_signals')):>5} "
              f"{str(e.get('contamination_hits')):>7} {cov:>7}  "
              + ("READY" if ok else "; ".join(why)[:60]))

    print()
    if all(ready) and ready:
        print(f"Round {latest}: every backend meets the freeze criterion. "
              f"Run one more round to confirm it finds nothing, then freeze.")
        return 0
    print(f"Round {latest}: NOT converged — "
          f"{sum(1 for r in ready if not r)}/{len(ready)} backends fail the "
          f"criterion. Another round is required.")
    return 2


# ── computed coverage, so no backend can self-report an incomparable number ──
def measured_pitfall_coverage(repo: Path | None = None) -> dict[str, dict]:
    """Pitfall claims and their tier-2 fixtures, counted from the tree.

    Deliberately crude and uniform. An agent that walks its own knowledge dict
    can produce a defensible-looking percentage from any denominator it likes —
    one pass reported 69.3% from a metric that mixed execution counts with claim
    strings, while its own scripts printed 55.0% and a bare 184. Comparability
    beats sophistication when the number decides whether we freeze.

    THE DENOMINATOR MUST COVER EVERY SERVED SURFACE, NOT JUST src/backends/<be>.
    This walked only the per-backend directories, which quietly excluded three
    things an agent can actually be served:

      * `src/backends/_cross.py` — 35 cross-backend collation pitfalls, the ones
        that fire only on the delta between two codes (units, node ordering,
        Dirichlet strong-vs-penalty, restart incompatibility). Skipped because
        the name starts with an underscore.
      * `src/tools/deep_knowledge.py` — 100 entries, served through the same
        pitfalls surface as a supplement for physics a backend does not
        enumerate.
      * `src/tools/consolidated.py` — 1.

    Excluding them made the bar easier to clear on paper while an agent could
    still be handed those 136 unverified entries, which is precisely backwards:
    the freeze criterion exists to bound what a user can be told, so its
    denominator has to be everything a user can be told.

    COUPLING has the opposite failure and it is worse. It scored 0 pitfall
    claims against 10 fixtures — not because it is verified, but because its
    143,291 chars of knowledge carry no `Signal:` clause at all, so nothing here
    can see it. A capability with an unmeasurable denominator cannot pass or
    fail the 80% bar; it simply is not judged. That is reported explicitly below
    rather than being allowed to look like a clean sheet, because the flagship
    silently exempting itself from the freeze criterion is the last thing this
    project can afford.
    """
    import ast

    repo = repo or REPO
    backends_dir = repo / "src" / "backends"
    fixtures_dir = repo / "scripts" / "tier2_fixtures"

    SENTINEL_INDEX = 90        # >= this is a synthetic fixture; it credits no claim

    def _claim_text_map(backend: str):
        """{(physics, index) -> claim TEXT} straight from the registry.

        Resolving a fixture key to the TEXT it names is what makes the coverage
        numerator and denominator the same universe. Without it the metric
        divided a count of KEYS by a count of TEXTS — two unrelated cardinalities
        — which is why four backends reported above 100%, the exact symptom this
        function's own docstring calls proof that the metric is not measuring
        coverage.
        """
        try:
            sys.path.insert(0, str(repo / "src"))
            from core.registry import get_backend, load_all_backends
            load_all_backends()
            be = get_backend(backend)
        except Exception:
            return None
        if be is None:
            return None
        out = {}
        names = []
        try:
            names = [c.name for c in be.supported_physics()]
        except Exception:
            pass
        seen_phys = set()
        for phys in names:
            if phys in seen_phys:
                continue
            seen_phys.add(phys)
            try:
                k = be.get_knowledge(phys)
            except Exception:
                continue
            if isinstance(k, dict):
                for i, entry in enumerate(k.get("pitfalls") or []):
                    if isinstance(entry, str):
                        out[(phys, i)] = entry
        return out

    def _resolve(backend: str, cmap, phys: str, idx: int):
        """Claim text for one key, or None. Tries the physics name, then the
        backend's own alias resolution, so a fixture keyed on an alias is not
        silently counted as covering nothing."""
        if cmap is None:
            return None
        if (phys, idx) in cmap:
            return cmap[(phys, idx)]
        try:
            sys.path.insert(0, str(repo / "src"))
            from core.registry import get_backend
            be = get_backend(backend)
            k = be.get_knowledge(phys) if be is not None else None
            if isinstance(k, dict):
                ps = k.get("pitfalls") or []
                if 0 <= idx < len(ps) and isinstance(ps[idx], str):
                    return ps[idx]
        except Exception:
            return None
        return None

    def _claims_with_a_fixture(fx_dir: Path, backend: str = "") -> set[str]:
        """Distinct CLAIMS a backend's fixtures attest to, not fixture count.

        Counting fixture directories is wrong in both directions, and both
        errors showed up on the same day:

          * It UNDERSTATES. One fixture may legitimately cover several claims —
            DUNE JIT-compiles C++ per distinct form, so splitting claims that
            share a compiled module multiplies build time without adding
            evidence. DUNE reads 34% by directory count and 80.2% by claim
            attribution: 30 fixtures covering 89 of 111 claims.
          * It OVERSTATES. 4C reads 104% because 14 legacy fixtures carry keys
            matching no current claim. A coverage figure above 100% is proof on
            its face that the metric is not measuring coverage.

        So a fixture declares what it covers, and a `covers` entry naming a
        claim that does not exist counts for nothing — otherwise the metric
        could be raised by inventing keys. Fixtures with no `covers` key fall
        back to their own physics/pitfall_index pair, which is how the older
        ones are counted.

        Deliberately silent about fixture QUALITY: this answers "is there a
        fixture for this claim at all". Whether it passes is the runner's job,
        and whether its mutation kills it is the mutation harness's — which was
        itself scoring KILLED for fixtures that never ran until it was fixed.
        """
        found: set[str] = set()
        if not fx_dir.is_dir():
            return found
        cmap = _claim_text_map(backend) if backend else None
        for d in sorted(fx_dir.iterdir()):
            manifest = d / "fixture.json"
            if not manifest.is_file():
                continue
            try:
                spec = json.loads(manifest.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            covers = spec.get("covers")
            keys = []
            if isinstance(covers, list) and covers:
                keys = [str(c) for c in covers if isinstance(c, str)]
            else:
                physics = spec.get("physics")
                idx = spec.get("pitfall_index")
                if physics is not None and idx is not None:
                    keys = [f"{physics}::{idx}"]
            for raw in keys:
                # BOTH CONVENTIONS. `covers` entries are written `physics::index`
                # (deal.II, coupling) and `physics:index` (SPARTA); the fallback
                # used to emit the single-colon form regardless. Reading only one
                # spelling silently dropped every fixture that used the other.
                m = re.match(r"^\s*(.+?):{1,2}(\d+)\s*$", raw)
                if not m:
                    continue
                phys, i = m.group(1), int(m.group(2))
                # SENTINELS CREDIT NOTHING. An index >= 90 is the documented
                # range for synthetic fixtures — install diagnostics and the
                # like — which are worth keeping and are not evidence for any
                # claim. Counting them inflated the numerator with keys that
                # name no claim at all.
                if i >= SENTINEL_INDEX:
                    continue
                text = _resolve(backend, cmap, phys, i)
                # A key naming a claim that does not exist counts for nothing,
                # exactly as this function's contract says: otherwise coverage
                # could be raised by inventing keys.
                if text:
                    found.add(text)
        return found

    def _registry_claims(backend: str) -> set[str] | None:
        """Distinct pitfall TEXTS the backend actually serves, or None.

        Two corrections in one, both measured rather than assumed.

        THE REGISTRY, NOT THE SOURCE TREE. Reading Signal-bearing strings out of
        `src/backends/<name>/**.py` with an AST walk misses claims that arrive by
        merging, aliasing or importing across modules. It counted 252 for 4C
        where the registry serves 382 — a third missing — and that undercount is
        what made 4C look like 98% covered when it is 66%.

        DISTINCT TEXTS, NOT POSITIONS. SPARTA appends 10 universal pitfalls to
        every one of its 10 physics rows, so a positional count reports 165
        claims where 75 distinct things exist; on another branch the same shape
        gives 1343 positional against 53 distinct. Counting positions would let
        a backend inflate its denominator by supporting more physics without
        writing a single new claim, and would make its coverage unreachable for
        no reason. Measured across all backends, the inflation is 1303 of 2677
        positions — essentially all of it SPARTA, with 4C contributing 13 and
        every other backend exactly zero.

        Returns None when the backend cannot be loaded here, so the caller can
        fall back rather than report a confident zero.
        """
        try:
            sys.path.insert(0, str(repo / "src"))
            from core.registry import get_backend, load_all_backends
            load_all_backends()
            be = get_backend(backend)
        except Exception:
            return None
        if be is None:
            return None
        texts: set[str] = set()
        try:
            for cap in be.supported_physics():
                k = be.get_knowledge(cap.name)
                if isinstance(k, dict):
                    for entry in (k.get("pitfalls") or []):
                        if isinstance(entry, str):
                            texts.add(entry)
        except Exception:
            return None
        return texts or None

    def _signals_in(paths) -> set[str]:
        """Every distinct Signal-carrying string under these paths.

        FALLBACK ONLY — see _registry_claims for why this undercounts. Kept
        because a backend whose library is absent still has a knowledge surface
        worth reporting approximately, but a number from here is a floor.

        Deduplicated by value: a shared sub-dict attached by reference to many
        physics entries is one claim to verify, and a naive walk inflated a
        DUNE count 111 -> 127 exactly that way.
        """
        found: set[str] = set()
        for py in paths:
            try:
                tree = ast.parse(py.read_text(errors="ignore"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and "Signal:" in node.value):
                    found.add(node.value)
        return found

    out: dict[str, dict] = {}
    for be_dir in sorted(p for p in backends_dir.iterdir() if p.is_dir()):
        name = be_dir.name
        if name.startswith("_"):
            continue
        signals = _registry_claims(name)
        source = "registry"
        if signals is None:
            signals = _signals_in(be_dir.rglob("*.py"))
            source = "source-tree floor"
        fx = fixtures_dir / name
        n_fx = 0
        if fx.is_dir():
            n_fx = len([d for d in fx.iterdir() if (d / "fixture.json").is_file()])
        # Can this surface's keys be resolved to claim TEXTS at all? Only a
        # registered backend exposes an ordered pitfall list per physics. When
        # it cannot, attribution is UNMEASURABLE — not zero. Reporting zero
        # would be the same lie in the other direction: `coupling` keys its 18
        # fixtures on a flat 0..17 numbering under topic labels, so there is no
        # positional list to resolve against, and printing 0% would read as
        # "the flagship has no evidence" when what is true is "this metric
        # cannot see it".
        resolvable = _claim_text_map(name) is not None
        attributed = (_claims_with_a_fixture(fx, name) & (signals or set())
                      if resolvable else None)
        out[name] = {
            "pitfall_claims": len(signals),
            "fixtures": n_fx,
            # Fixture directories over claims. A FLOOR, not the figure: see
            # _claims_with_a_fixture for why it is wrong in both directions.
            "covered_crude": round(n_fx / len(signals), 4) if signals else None,
            "claims_attributed": (len(attributed) if attributed is not None
                                  else None),
            "denominator_source": source,
            "covered": (round(len(attributed) / len(signals), 4)
                        if (attributed is not None and signals) else None),
        }

    # The served surfaces that live outside src/backends/<name>/. Grouped under
    # their own rows so they are visible in the table rather than folded into a
    # backend's score, where they would dilute rather than be judged.
    extra_surfaces = {
        "cross_backend": [backends_dir / "_cross.py"],
        "deep_knowledge": [repo / "src" / "tools" / "deep_knowledge.py"],
    }
    for label, paths in extra_surfaces.items():
        sigs = _signals_in(p for p in paths if p.is_file())
        if not sigs:
            continue
        fx = fixtures_dir / label
        n_fx = len([d for d in fx.iterdir()
                    if (d / "fixture.json").is_file()]) if fx.is_dir() else 0
        out[label] = {
            "pitfall_claims": len(sigs),
            "fixtures": n_fx,
            # These surfaces have no per-physics ordered list either, so a
            # fixture key cannot be resolved to the claim it names. The crude
            # fixture-count ratio is kept as a floor and clearly labelled, but
            # `covered` stays None so the freeze criterion judges them as
            # UNMEASURABLE rather than on a number that is not coverage.
            "covered_crude": round(n_fx / len(sigs), 4),
            "claims_attributed": None,
            "covered": None,
        }

    # Fixture directories with no denominator at all. `coupling` is the one that
    # matters: it has fixtures but its knowledge carries no Signal: clause, so
    # the numerator is real and the denominator is missing. Reporting it as a
    # row with covered=None makes the gap visible; omitting it would let the
    # flagship look complete by being absent.
    if fixtures_dir.is_dir():
        for fx in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
            if fx.name in out:
                continue
            n_fx = len([d for d in fx.iterdir() if (d / "fixture.json").is_file()])
            if n_fx:
                out[fx.name] = {
                    "pitfall_claims": 0,
                    "fixtures": n_fx,
                    "covered": None,
                    "note": ("fixtures exist but the knowledge carries no "
                             "Signal: clause, so coverage is UNMEASURABLE — "
                             "not zero, and not complete"),
                }
    return out


def print_measured_coverage() -> None:
    rows = measured_pitfall_coverage()
    print(f"\n{'surface':<15} {'claims':>7} {'fixtures':>9} {'attributed':>11} "
          f"{'covered':>9}  bar")
    print("-" * 60)
    tp = tf = ta = 0
    unmeasurable = []
    for be, d in sorted(rows.items()):
        tp += d["pitfall_claims"]
        tf += d["fixtures"]
        ta += d.get("claims_attributed") or 0
        if d["covered"] is None:
            cov, bar = "UNMEASURABLE", ""
        else:
            cov = f"{d['covered']:.1%}"
            bar = "MEETS" if d["covered"] >= THRESHOLDS["surface_covered"] else ""
        if d["covered"] is None:
            unmeasurable.append(be)
        print(f"{be:<15} {d['pitfall_claims']:>7} {d['fixtures']:>9} "
              f"{(d.get('claims_attributed') if d.get('claims_attributed') is not None else '—'):>11} {cov:>9}  {bar}")
    print("-" * 60)
    print(f"{'TOTAL':<15} {tp:>7} {tf:>9} {ta:>11} "
          f"{(ta / tp if tp else 0):>8.1%}")
    print("\nsurface_covered for the freeze criterion is the `covered` column: "
          "DISTINCT CLAIMS with a fixture, over claims. Not fixture count over "
          "claims — that reads 34% for a backend measuring 80% (one fixture can "
          "cover several claims) and 104% for another (legacy fixtures keyed to "
          "claims that no longer exist). A ratio that can exceed 100% is not "
          "measuring coverage.")

    if unmeasurable:
        # Loud, because an unmeasurable row is the one failure mode this table
        # cannot express as a number, and a blank cell reads as "fine".
        print(f"\n!! {len(unmeasurable)} surface(s) CANNOT BE JUDGED: "
              f"{', '.join(unmeasurable)}")
        for be in unmeasurable:
            note = rows[be].get("note")
            if note:
                print(f"   {be}: {note}")
        print("   These have fixtures but no countable claims behind them. "
              "They are neither passing nor failing the 80% bar — they are "
              "exempt from it, which is worse. Give their knowledge the "
              "corpus format ([Category] ... Signal: ...) so it can be "
              "counted, retrieved by symptom, and held to the same bar as "
              "every other surface.")


if __name__ == "__main__":
    import sys as _sys
    # `--coverage` prints the computed, cross-backend-comparable figure. A first
    # version appended this dispatch AFTER `raise SystemExit(main())`, so it was
    # dead code that silently did nothing — the same shape as the
    # `--write-results` flag that was parsed and never read.
    if "--coverage" in _sys.argv:
        print_measured_coverage()
        raise SystemExit(0)
    raise SystemExit(main())
