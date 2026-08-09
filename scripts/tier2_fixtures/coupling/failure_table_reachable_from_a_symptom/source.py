"""The coupling failure table is only worth its size if a broken run can REACH
it, and until now nothing could.

THE CLAIM UNDER TEST. Every other family of knowledge in OASiS ships its
symptoms as `[Category] ... Signal: ...` entries, and `knowledge(signal=...)` is
how a post-execution critic gets from a symptom to the entry that explains it.
The coupling payload had a failure table whose left-hand column IS a set of
signals — and no pitfall list, no `Signal:` shape and no search. A coupling
failure could not be routed to the rows that name it: the reader had to already
know to ask for `topic='coupling'` and then read the whole payload.

WHERE THE QUERIES BELOW CAME FROM, because that is what makes them evidence.
They are NOT the entry author's phrasings. A hostile third party was given the
search and no sight of the entries, wrote a hundred queries from its own idea of
how a partitioned coupling goes wrong, and only then read the table to judge
each result. The first version of the router scored ~62% recall and ~52% top-1
against that set: it divided by the query length, so `nan` found the right entry
and `solver crashed with nan after the third coupling step` found nothing; it
weighted every token alike, so `interface` and `step` decided rankings; and its
entries spoke thermal while half the queries spoke mechanics, so every
force / load / traction / mapping phrasing missed. The queries kept here are
that party's, including the ones it phrased with typos, and they are a
REGRESSION SET: they are the evidence that those three faults are fixed, not
evidence that the router generalises to a NEXT party's vocabulary.

AND A SECOND BLIND PARTY WAS THEN RUN, because the paragraph above is only worth
something if it is acted on. It wrote ninety-five fresh queries without seeing
any of this, and measured 60% recall and 53% top-1 — that is, most of the gain
the first party's queries had shown was fitted to the first party's vocabulary,
exactly as this file predicted. It found two more structural faults, both fixed:
an unrecognised word was scored at the MAXIMUM weight in the coverage
denominator, so a longer query was punished after all ("handshake" returned four
rows, "handshake never completes" returned none); and the synonym table was
keyed on stems the stemmer never produces, so `wobble` could not reach the entry
`wobbles` reached. Its queries for the classes that then still missed are in the
set below too.

THE HONEST STATE, because two blind passes is still not a claim of generality.
The rows below reach a much wider vocabulary than they did, and there remain
whole failure families the TABLE has no row for at all — staggering lag,
subcycling, checkpoint and restart, mapping quality, a discontinuity in the
transferred VALUE rather than the flux — and a query about one of those gets a
confidently wrong neighbour rather than a refusal. That is a content gap, not a
matcher gap, and it is not fixed here.

WHAT THIS FIXTURE CHECKS, all through the REGISTERED `knowledge` tool:
  1. NO DRIFT. The markdown table and the searchable entries are generated from
     one list, every row appears in both, and each marker below identifies
     exactly one entry.
  2. ROUTING FROM AN OBSERVATION. Every query describes what somebody SAW, with
     no mechanism in it — no "theta", no "sign convention", no "units". A query
     that names the cause proves nothing: the entry contains the cause, so of
     course it matches, and the person with the problem does not know the cause
     yet.
  3. DISCRIMINATION. Queries about single-code problems, installation, licences
     and linear solvers must be REFUSED. A matcher that answers everything has
     routed nothing, and the first version answered a preconditioner question
     with five coupling entries.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

# A fragment that identifies exactly ONE entry. Uniqueness is asserted below, so
# a future edit that makes a marker ambiguous fails here rather than quietly
# weakening every query that uses it.
M_DIED = "script died"
M_BADJSON = "TRUNCATED"
M_SIZE = "different number of points"
M_TIMEOUT = "raise `timeout`"
M_NOCONV = "max_iter above"
M_OSC = "theta too large"
M_GROW = "stability limit"
M_UNBAL = "same sign"
M_UNCHECKED = "unguarded"
M_NOTCOUPLED = "nobody was given"
M_ONEWAY = "master/slave"
M_FROZEN = "initial guess"
M_NONFINITE = "solve diverged"
M_PLAUSIBLE = "wrongly located"
M_UNITS = "UNIT MISMATCH"
M_RATIO = "CONDUCTANCE RATIO"
M_STOCH = "SEED IS FIXED"

MARKERS = [M_DIED, M_BADJSON, M_SIZE, M_TIMEOUT, M_NOCONV, M_OSC, M_GROW,
           M_UNBAL, M_UNCHECKED, M_NOTCOUPLED, M_ONEWAY, M_FROZEN, M_NONFINITE,
           M_PLAUSIBLE, M_UNITS, M_RATIO, M_STOCH]

# (what somebody saw, any one of these entries is a correct answer)
QUERIES = [
    ("the whole thing exploded after a few timesteps and the displacements "
     "went to infinity", [M_GROW, M_NONFINITE]),
    ("my fsi with water and a thin flexible flap diverges no matter what i do",
     [M_GROW]),
    ("it ran fine with a heavy solid but goes unstable with a light one",
     [M_GROW]),
    ("solver crashed with nan after the third coupling step",
     [M_NONFINITE, M_DIED]),
    ("coupling iterations oscillate back and forth between two values", [M_OSC]),
    ("sub-iterations hit the maximum count every single step", [M_NOCONV]),
    ("the outer loop residual plateaus at 1e-3 and refuses to go lower",
     [M_STOCH, M_NOCONV]),
    ("convergence is painfully slow, takes 40 sub iterations per timestep",
     [M_NOCONV]),
    ("it converged but the answer is wrong", [M_PLAUSIBLE, M_UNITS]),
    ("both solvers converge on their own but the coupled result is nonsense",
     [M_PLAUSIBLE, M_UNITS, M_FROZEN]),
    ("the forces I send over don't add up to the same total on the other side",
     [M_UNBAL]),
    ("field arrives on the other side but scaled by some random factor",
     [M_UNITS]),
    ("displacements transferred fine but the tractions look mirrored",
     [M_OSC, M_UNBAL]),
    ("structure moves the opposite direction to the pressure load",
     [M_OSC, M_UNBAL]),
    ("my interface meshes don't line up exactly, there are small gaps",
     [M_PLAUSIBLE]),
    ("displacement is off by a factor of 1000", [M_UNITS]),
    ("the deflection came out a thousand times too large", [M_UNITS]),
    ("pressures are enormous, like 1e12", [M_GROW]),
    ("energy at the interface grows every step", [M_UNBAL, M_GROW]),
    ("total energy is not conserved in the coupled run", [M_UNBAL]),
    ("it just hangs, nothing happens, no error", [M_TIMEOUT]),
    ("one participant waits forever for the other", [M_TIMEOUT]),
    ("the run deadlocks at startup", [M_TIMEOUT, M_DIED]),
    ("process exits with no output and no error message", [M_DIED]),
    ("interface temperature oscillates in a checkerboard pattern between the "
     "two solvers", [M_OSC]),
    ("conjugate heat transfer diverges when the solid is much more conductive",
     [M_GROW, M_NOCONV]),
    ("the coupling converged in one iteration on the very first step which "
     "seems too good", [M_FROZEN]),
    ("my answer keeps changing every time I rerun it", [M_STOCH]),
    # written with the third party's typos, on purpose
    ("coupling iterations dont converge, residual oscilating", [M_OSC, M_NOCONV]),
    ("staggerd scheme giving wrong load transfer", [M_OSC, M_UNBAL]),
    ("load transfer is not conservative", [M_UNBAL, M_UNCHECKED]),
    ("interface flux imbalance", [M_UNBAL, M_UNCHECKED]),
    ("unstable", [M_GROW]),
    ("nan", [M_NONFINITE]),
    ("hangs", [M_TIMEOUT]),
    ("exports.json has different keys than the other participant expects",
     [M_BADJSON]),
    ("the second code receives all zeros", [M_FROZEN, M_NOTCOUPLED]),
    ("there is a visible gap between fluid and structure in paraview after "
     "coupling", [M_PLAUSIBLE]),
    ("the flux is wrong although the temperature is fine", [M_RATIO]),
    ("one side never reacts to what the other sends", [M_ONEWAY, M_NOTCOUPLED]),
    # the second blind party's, for the classes that missed before
    ("wobble", [M_OSC]),
    ("ping-pong between the participants but no progress", [M_OSC, M_NOCONV]),
    ("the log prints the same number for every iteration", [M_FROZEN]),
    ("the export did not change between iterations", [M_FROZEN]),
    ("only one of the two codes actually moved", [M_ONEWAY, M_NOTCOUPLED]),
    ("the coupled result is offset by a constant everywhere", [M_UNITS]),
    ("lift computed by the aero code is not the lift the structure feels",
     [M_UNBAL]),
    ("the run aborted with an MPI error partway through", [M_DIED]),
    ("the second program never starts", [M_DIED, M_TIMEOUT]),
    ("handshake never completes", [M_TIMEOUT, M_DIED]),
    ("the coupling residual goes down but the physics is wrong",
     [M_UNITS, M_PLAUSIBLE]),
    ("the contact pressure is spiky at the boundary between the two meshes",
     [M_PLAUSIBLE]),
]

# Nothing here is a coupling failure. Every one must come back empty-handed.
REFUSALS = [
    "how do I refine my mesh near a corner in gmsh",
    "what element type should I use for incompressible elasticity",
    "how do I install dolfinx with conda",
    "my direct solver runs out of memory on a 10 million dof problem",
    "how do I plot the von mises stress in paraview",
    "what is the correct poisson ratio for steel",
    "gmres is not converging, which preconditioner should I use",
    "wrong",
    "my simulation is wrong",
    "asdfqwerty zzz qqq",
    "added mass",
    "subcycling",
    # the second party's, on generic English that used to decide the ranking
    "how do I compute the condition number of my stiffness matrix",
    "the AMG preconditioner uses too much memory",
]


def ask(signal: str) -> str:
    """Ask the REGISTERED knowledge tool, the way an agent would."""
    return L.call_tool("knowledge", {"topic": "coupling", "signal": signal})


ROUTED = "# Coupling failure entries matching"
REFUSED = "No coupling failure entry matches"


def body() -> None:
    from tools.coupling_knowledge import (_FAILURE_ROWS, coupling_core,
                                          coupling_pitfall_entries)

    rows = _FAILURE_ROWS
    entries = coupling_pitfall_entries()
    core = coupling_core()
    print(f"failure_rows={len(rows)}")
    print(f"searchable_entries={len(entries)}")
    L.check(len(rows) == len(entries), "rows_and_entries_disagree",
            f"{len(rows)} rows rendered {len(entries)} searchable entries — the "
            f"two surfaces are supposed to be generated from one list")

    missing_table = [r["seen"] for r in rows if r["seen"] not in core]
    print(f"rows_missing_from_the_served_table={len(missing_table)}")
    L.check(not missing_table, "row_missing_from_the_served_payload",
            "; ".join(missing_table)[:300])
    print(f"every_row_is_in_both_surfaces="
          f"{not missing_table and len(rows) == len(entries)}")

    # A marker that matched two entries would make every query using it pass
    # for the wrong reason.
    ambiguous = [m for m in MARKERS
                 if sum(1 for e in entries if m in e) != 1]
    print(f"ambiguous_markers={len(ambiguous)}")
    L.check(not ambiguous, "marker_is_not_unique", str(ambiguous))
    print(f"markers_identify_one_entry_each={not ambiguous}")

    routed = 0
    for q, accept in QUERIES:
        out = ask(q)
        # BOTH conditions, and the first is not decoration: a miss used to
        # return the whole core payload, in which nearly every marker word
        # appears somewhere, so `marker in out` alone went green for a query of
        # pure nonsense. The mutation harness caught exactly that.
        hit = out.startswith(ROUTED) and any(m in out for m in accept)
        if hit:
            routed += 1
        else:
            L.check(False, "symptom_did_not_route",
                    f"{q!r} did not reach any of {accept}")
    print(f"queries_routed={routed}/{len(QUERIES)}")
    print(f"every_observation_reached_its_entry={routed == len(QUERIES)}")

    refused = 0
    for q in REFUSALS:
        out = ask(q)
        if REFUSED in out:
            refused += 1
        else:
            L.check(False, "unrelated_query_was_answered",
                    f"{q!r} came back with failure entries, so a match carries "
                    f"no information")
    print(f"unrelated_queries_refused={refused}/{len(REFUSALS)}")
    print(f"every_unrelated_query_is_refused={refused == len(REFUSALS)}")

    # The routed answer must be SHORT — the point is not to hand back the whole
    # payload with the answer inside it.
    sample = ask(QUERIES[0][0])
    print(f"routed_answer_is_shorter_than_the_full_payload="
          f"{len(sample) < len(core) // 4}")
    L.check(len(sample) < len(core) // 4, "routing_returns_the_whole_payload",
            f"a signal query returned {len(sample)} chars against a full "
            f"payload of {len(core)}; that is not routing, that is a dump")

    print("claims_checked=6")


L.main(body)
