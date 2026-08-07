#!/usr/bin/env python3
"""Audit the HOE-v2 campaign for prompts that hand the agent the answer.

The v2 prompt set is the one the paper's numbers currently rest on.  This
script classifies every task by *how* it is compromised, if it is, and counts
the affected cells, so the finding is reproducible rather than asserted.

Three distinct defects, which matter differently:

``SOLUTION_STATED``
    the prompt writes out ``u_exact`` and then asks for the error against it.
    Any convergence order reported is not evidence that the agent solved the
    problem -- it is evidence that it can subtract.

``ANSWER_STATED``
    the prompt prints the numeric values the agent is asked to report, and the
    grading band is "within x% of" those same values.  Pure pattern-match.

``BENCHMARK_MEMORISABLE``
    no leak in the prompt, but the target is a famous published number
    (Ghia, Schaefer-Turek, Leissa, Roshko).  A model may recall it rather than
    compute it.  This is a weaker and different problem from a prompt leak and
    is reported separately so the two are not conflated.

Run:  .venv/bin/python scripts/audit_hoe_v2_leak.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

V2_DOCS = Path("/home/alexander/Schreibtisch/open-fem-agent/"
               "papers/overleaf-paper/prompts")

CONDITIONS, SEEDS = 3, 3

# Patterns that constitute stating the solution.  Each is a phrase with no
# legitimate place in a task the agent is supposed to solve.
SOLUTION_MARKERS = (
    r"u_?exact", r"exact solution", r"manufactured exact",
    r"manufactured (displacement|solution|field)s?",
    r"the exact solution is", r"exact eigenvalues?",
)
SELF_GRADE_MARKERS = (
    r"error (against|vs\.?|versus) (the )?(exact|u_exact|analytic)",
    r"l2 error (against|of).{0,40}exact",
    r"relative l2 error of .* vs exact",
    r"compute the l2 (norm|error).{0,60}u_exact",
)

# Tier-by-tier verdicts, each justified by the quoted prompt text.
FINDINGS = {
    "A4": ("SOLUTION_STATED", "u_exact(x,y) = sin(pi x) sin(pi y) given; asks for "
                              "L2 error against it and the observed rate"),
    "C1": ("SOLUTION_STATED", "'The manufactured exact solution is: u_exact(x,y) "
                              "= sin(3 pi x) sin(2 pi y)'"),
    "C2": ("SOLUTION_STATED", "u1_exact/u2_exact/u3_exact given; Dirichlet BC "
                              "stated as 'u = u_exact on the entire boundary'"),
    "C3": ("SOLUTION_STATED", "u_exact(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 "
                              "gamma t) given for both refinement studies"),
    "C4": ("SOLUTION_STATED", "u1_exact, u2_exact and p_exact all given"),
    "C5": ("SOLUTION_STATED", "u_exact and v_exact given for both species"),
    "D1": ("SOLUTION_STATED", "u_exact(x,y) = sinh(2x) cos(y) given -- but the "
                              "task IS 'edit step-7 to use this solution', so "
                              "disclosure is intrinsic to the task as posed"),
    "E2": ("SOLUTION_STATED", "'the manufactured displacement field u1 = sin(pi x) "
                              "sin(pi y) sin(pi z)' + L2 error against u_exact"),
    "E3": ("SOLUTION_STATED", "'(exact solution u = sin(pi x) sin(pi y))' and "
                              "'RESULT err_A = <relative L2 error ... vs exact>'"),
    "E7": ("SOLUTION_STATED", "'The exact solution is the 1D boundary-layer "
                              "profile u(x,y) = (exp(x/eps)-1)/(exp(1/eps)-1)'"),
    "E8": ("ANSWER_STATED",   "prints lambda_1 = 5.78319 and lambda_2 = lambda_3 "
                              "= 14.68197, then asks the agent to report those "
                              "three numbers; gate is 'within 0.5% of exact'"),
    "A5": ("BENCHMARK_MEMORISABLE", "Leissa CCCC plate first eigenfrequency"),
    "B3": ("BENCHMARK_MEMORISABLE", "Strouhal number at Re=200 (Roshko/Williamson)"),
    "B4": ("BENCHMARK_MEMORISABLE", "Hertz contact -- gate is self-consistency, "
                                    "which blunts this"),
    "E5": ("BENCHMARK_MEMORISABLE", "Ghia et al. lid-driven cavity, named in prompt"),
    "E6": ("BENCHMARK_MEMORISABLE", "DFG 2D-1 Schaefer-Turek, named in prompt"),
    "A3": ("ANALYTICALLY_TRIVIAL", "IC sin(pi x) with homogeneous BCs: separation "
                                   "of variables gives u = sin(pi x) exp(-pi^2 t) "
                                   "in one line. Not a prompt leak, but the task "
                                   "does not discriminate solving from deriving"),
}

CLEAN = ["A1", "A2", "B1", "B2", "B5", "D2", "E1", "E4"]
ALL = (["A1", "A2", "A3", "A4", "A5"] + [f"B{i}" for i in range(1, 6)]
       + [f"C{i}" for i in range(1, 6)] + ["D1", "D2"]
       + [f"E{i}" for i in range(1, 9)])


def scan_docs():
    """Confirm the markers really are present in the generated documents."""
    hits = {}
    for doc in sorted(V2_DOCS.glob("PROMPTS_HOE_V2*.md")):
        text = doc.read_text(errors="ignore").lower()
        hits[doc.name] = {
            "solution_markers": sum(len(re.findall(p, text)) for p in SOLUTION_MARKERS),
            "self_grade_markers": sum(len(re.findall(p, text)) for p in SELF_GRADE_MARKERS),
            "cells": text.count("## cell "),
        }
    return hits


def main():
    by_class = {}
    for t in ALL:
        cls = FINDINGS.get(t, ("CLEAN", "no exact solution or answer value in prompt"))[0]
        by_class.setdefault(cls, []).append(t)

    compromised = by_class.get("SOLUTION_STATED", []) + by_class.get("ANSWER_STATED", [])
    print("HOE-v2 prompt-leak audit")
    print("=" * 72)
    for cls in ("SOLUTION_STATED", "ANSWER_STATED", "BENCHMARK_MEMORISABLE",
                "ANALYTICALLY_TRIVIAL", "CLEAN"):
        ts = sorted(by_class.get(cls, []))
        print(f"\n{cls}  ({len(ts)} tasks): {', '.join(ts) or '-'}")
        for t in ts:
            if t in FINDINGS:
                print(f"    {t}: {FINDINGS[t][1]}")

    n_cells = len(ALL) * CONDITIONS * SEEDS
    n_bad = len(compromised) * CONDITIONS * SEEDS
    print("\n" + "=" * 72)
    print(f"tasks compromised by a prompt leak : {len(compromised)}/{len(ALL)}"
          f"  ({', '.join(sorted(compromised))})")
    print(f"main-document cells affected       : {n_bad}/{n_cells}"
          f"  ({100 * n_bad / n_cells:.0f}%)")
    tierc = [t for t in compromised if t.startswith("C")]
    print(f"Tier C (the convergence-order tier): {len(tierc)}/5 compromised"
          f" -- {', '.join(sorted(tierc))}")
    print("\nEvery Tier-C task states its manufactured solution, so every Tier-C")
    print("convergence rate in the paper is a self-reported error against a")
    print("reference the agent was handed.")

    print("\nMarker counts in the generated documents (sanity check):")
    for name, h in scan_docs().items():
        print(f"  {name:34} cells={h['cells']:4}  "
              f"solution_markers={h['solution_markers']:4}  "
              f"self_grade={h['self_grade_markers']:3}")

    print("\nStructural note: the grading bands are appended to the SAME file as")
    print("the prompts (PROMPTS_HOE_V2.md). The agent not seeing them relies on")
    print("the operator pasting only the prompt block -- procedure, not structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
