#!/usr/bin/env python3
"""Screen the tier-2 corpus for assertions that do not measure what they name.

WHY
---
A fixture proves a knowledge claim by reproducing a failure and matching strings
in its output.  A mutation control proves the OUTPUT DEPENDS on the pathology.
Neither proves the assertion measures the quantity its NAME says.  Two
independent passes put that failure rate near 5% of the corpus, and reading 1300
fixtures by hand is not feasible, so the mechanical part is taken first.

`tests/test_expectations_assert_values.py` already takes the easiest slice: a
fixture whose EVERY expectation is a bare `key=` prefix.  This screen takes eight
more shapes, each visible statically.  Against the corpus at f1d4fbf5 it flags
113 expectations in 62 of 1308 fixtures (4.7%).

  BAKED_QUOTE  an expectation with NO `=` -- the corpus's form for a quoted
               diagnostic, i.e. words the TOOL emitted -- that is in fact a
               constant the fixture writes itself.  34 hits / 26 fixtures.
               `dealii/grid_in_malformed_format` expects "GridIn" and writes the
               word in both arms.  Hand-read: 24 of 55 pre-tuning hits confirmed;
               the 31 misses were two rules, both since fixed -- a `grep`
               PATTERN is not a write, and a literal inside an f-string
               replacement field is code, not output.  Post-fix precision on the
               same corpus: 24 confirmed of 34.

  BAKED_ALL    every expectation of the fixture is such a constant, so the whole
               verdict is the fixture agreeing with itself.  22 fixtures, 6
               confirmed.  The 13 misses all route their discrimination through
               `forbid_in_output` instead -- a gated success token plus a
               forbidden `FAIL:` does carry information, and the screen cannot
               see that from the expect list alone.

  SELFSAME     a printed boolean comparing a prescribed constant against a value
               built FROM that constant -- `G == G`.  1 hit, confirmed.

  EMPTYEXC     a boolean read out of a caught exception's text, co-asserted with
               the flag that makes that text empty.  2 hits, both confirmed.

  SUCCESS      a forbid tripwire whose needle is a substring of one of the
               fixture's own expectations, so it fires on the line that says
               nothing went wrong.  4 hits, all confirmed, all Kratos.

  ARGMAX       `argmax` over a boolean array with no `.any()` guard.  numpy
               answers 0 on all-False, so "the first index where the condition
               holds" reads as index 0 when it never holds.  1 hit, confirmed.

  SYNTHETIC    an asserted boolean computed from arange/eye with no solve in its
               dependency chain.  24 hits / 9 fixtures, 4 confirmed.

  CONSTCMP     an asserted EQUALITY between two named quantities that both
               resolve to literals the fixture typed.  6 hits.  Found after a
               random-sample read turned up `sparta/axisymmetric_...` comparing
               two module-level strings under a name that claims a fact about
               SPARTA's diagnostics.

WHAT IT IS NOT
--------------
A screen, not a verdict.  Every hit is a candidate for a person to read, and the
false-positive rate above was measured by reading them, not assumed.  A random
sample of 60 fixtures the screen does NOT flag was read the same way: 2 wrong and
3 suspect, i.e. a residual of 3.3% wrong / 8.3% wrong-or-suspect among the
unflagged.  Two blind spots are known and named there: discrimination that lives
entirely in `forbid_in_output`, and assertions that are substring greps over
generated source text rather than over a tool's output.

Usage:
    python scripts/screen_wrong_assertions.py            # summary
    python scripts/screen_wrong_assertions.py --detail   # every hit
    python scripts/screen_wrong_assertions.py --json OUT
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Overridable so the screen can be pointed at an older checkout of the corpus
# and the before/after counts compared without moving the working tree.
FIXTURES = Path(os.environ.get("T2_FIXTURES_DIR")
                or REPO / "scripts" / "tier2_fixtures")

# A call that turns inputs into outputs.  If one of these sits between a
# quantity and the value it is compared against, the comparison is not `G == G`.
SOLVERS = {
    "solve", "spsolve", "lsqr", "lstsq", "cg", "gmres", "bicgstab", "minres",
    "Inverse", "Solve", "CGSolver", "GMResSolver", "factorized", "splu",
    "eigsh", "eigs", "solve_linear", "newton", "NewtonSolver", "assemble",
    "Assemble", "interpolate", "project", "Set", "run", "solve_ivp",
    "LinearSolver", "SolveLinearSystem", "Run", "Execute", "check_output",
    "solve_problem", "apply", "condense", "Solve_", "linsolve",
}
OUTPUT_FUNCS = {"print", "write", "echo"}
_KEYVAL = re.compile(r"^[A-Za-z_][\w.\[\]]*=")
# An `except` handler's text lands in one of these shapes.
_EXC_TEXT = re.compile(r"\b(str|repr|format_exc|traceback)\b")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fixtures() -> list[tuple[str, Path, dict]]:
    out = []
    for fj in sorted(FIXTURES.glob("*/*/fixture.json")):
        try:
            spec = json.loads(fj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append((f"{fj.parent.parent.name}/{fj.parent.name}", fj.parent, spec))
    return out


def _const_chunks(node: ast.AST) -> list[str]:
    """Every statically-known string chunk that is actually WRITTEN.

    An f-string contributes its literal pieces only; the interpolated parts are
    unknown at read time, which is the whole point -- a value that is
    interpolated was measured, a value inside the literal was not.

    A literal inside a REPLACEMENT FIELD is code, not output:

        print(f"newton_printed_banner={'Newton iteration' in chatter}")

    writes `newton_printed_banner=True`, never the phrase.  Counting those made
    the screen report two ngsolve fixtures as baked when the phrase reaches the
    output only through the tool's own captured text.  So a FormattedValue's
    subtree is skipped entirely.
    """
    chunks: list[str] = []

    def walk(n: ast.AST) -> None:
        if isinstance(n, ast.JoinedStr):
            for part in n.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    chunks.append(part.value)
                # FormattedValue: the literals inside it are operands, not text
            return
        if isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                chunks.append(n.value)
            return
        for child in ast.iter_child_nodes(n):
            walk(child)

    walk(node)
    return chunks


class _PrintScan(ast.NodeVisitor):
    """Collect (chunk, guard) for every print-like call in a module.

    guard is 'always'   -- no enclosing conditional
             'allarms'  -- filled in afterwards for chunks seen in every arm
             'branch'   -- inside a conditional or a handler
    """

    def __init__(self) -> None:
        self.chunks: list[tuple[str, str]] = []
        self._depth = 0
        self.if_arms: list[tuple[set[str], set[str]]] = []

    def _is_output(self, node: ast.Call) -> bool:
        f = node.func
        if isinstance(f, ast.Name):
            return f.id in OUTPUT_FUNCS
        if isinstance(f, ast.Attribute):
            return f.attr in OUTPUT_FUNCS
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_output(node):
            guard = "branch" if self._depth else "always"
            for a in list(node.args) + [k.value for k in node.keywords]:
                for c in _const_chunks(a):
                    self.chunks.append((c, guard))
        self.generic_visit(node)

    def _guarded_body(self, body) -> None:
        self._depth += 1
        for st in body:
            self.visit(st)
        self._depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        if node.orelse:
            a = _PrintScan()
            for st in node.body:
                a.visit(st)
            b = _PrintScan()
            for st in node.orelse:
                b.visit(st)
            self.if_arms.append(({c for c, _ in a.chunks}, {c for c, _ in b.chunks}))
        self._guarded_body(node.body)
        self._guarded_body(node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        self._guarded_body(node.body)
        for h in node.handlers:
            self._guarded_body(h.body)
        self._guarded_body(node.orelse)
        for st in node.finalbody:      # finally always runs
            self.visit(st)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._guarded_body(node.body)
        self._guarded_body(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        # A loop over a non-empty literal always runs its body.
        literal = isinstance(node.iter, (ast.Tuple, ast.List, ast.Set)) and node.iter.elts
        if literal:
            for st in node.body:
                self.visit(st)
        else:
            self._guarded_body(node.body)
        self._guarded_body(node.orelse)


def _sh_chunks(text: str) -> list[tuple[str, str]]:
    """Same idea for cmd.sh: literal words an `echo`/`printf` writes.

    Depth is tracked over the shell's own block keywords.  Anything with a `$`
    in it is a chunk boundary -- the expansion is the measured part.

    TWO THINGS THAT ARE NOT WRITES, and cost the screen 19 false positives on
    the 4C set before they were excluded:

      * a `#` comment;
      * a PATTERN handed to grep/sed/awk.  `grep -m1 -F "Expected parameter
        'DENS'" 4c.log` puts that phrase in the script and prints it only if 4C
        put it in the log.  That is the opposite of baked -- it is the strongest
        shape in the corpus.  So a line that pipes through, or invokes, a
        matcher is not treated as a write of its own arguments.
    """
    chunks: list[tuple[str, str]] = []
    depth = 0
    open_kw = re.compile(r"^\s*(if|case|while|until|for)\b")
    close_kw = re.compile(r"^\s*(fi|esac|done)\b")
    matcher = re.compile(r"\b(grep|egrep|fgrep|rg|sed|awk|perl)\b")
    for line in text.splitlines():
        s = line.strip()
        if close_kw.match(s):
            depth = max(0, depth - 1)
        m = re.search(r"\b(echo|printf)\b(.*)$", s)
        if m and not s.startswith("#") and not matcher.search(s):
            payload = m.group(2)
            guard = "branch" if depth else "always"
            for piece in re.split(r"\$\{[^}]*\}|\$\(|\$[A-Za-z_][\w]*|`", payload):
                piece = piece.strip().strip('"').strip("'")
                if piece:
                    chunks.append((piece, guard))
        if open_kw.match(s):
            depth += 1
    return chunks


# --------------------------------------------------------------------------
# detector: BAKED — the expectation is a constant the fixture writes itself
# --------------------------------------------------------------------------
def _cpp_chunks(text: str) -> list[tuple[str, str]]:
    """String literals a C++ fixture writes to cout/cerr/printf.

    Guard depth follows brace nesting under `if`/`else`/`catch`, which is crude
    but enough to separate "written whatever happens" from "written in one arm".
    """
    chunks: list[tuple[str, str]] = []
    depth = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        if re.match(r"^\}?\s*(else\b|catch\b)", s) or re.match(r"^(if|for|while|switch)\b", s):
            depth += 1
        if re.match(r"^\}", s) and depth:
            depth -= 1
        if re.search(r"\b(std::cout|std::cerr|printf|fprintf|deallog)\b", s):
            for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', s):
                lit = lit.replace('\\n', '').replace('\\"', '"').strip()
                if lit:
                    chunks.append((lit, "branch" if depth else "always"))
    return chunks


def _all_chunks(py, sh, cpp):
    chunks: list[tuple[str, str]] = []
    arms: list[tuple[set[str], set[str]]] = []
    if py is not None:
        try:
            sc = _PrintScan()
            sc.visit(ast.parse(py))
            chunks += sc.chunks
            arms += sc.if_arms
        except SyntaxError:
            pass
    if sh is not None:
        chunks += _sh_chunks(sh)
    if cpp is not None:
        chunks += _cpp_chunks(cpp)
    return chunks, arms


def _bake_class(needle: str, chunks, arms) -> str | None:
    best = None
    for c, guard in chunks:
        if needle in c:
            if guard == "always":
                return "always"
            best = best or "branch"
    if best == "branch":
        for a, b in arms:
            if any(needle in c for c in a) and any(needle in c for c in b):
                return "allarms"
    return best


def detect_baked(spec: dict, py: str | None, sh: str | None,
                 cpp: str | None = None) -> list[dict]:
    """An expectation the fixture writes itself, so nothing was measured.

    Two shapes are separated, because they are not equally bad.

    BAKED_QUOTE  the expectation carries no `=`.  An expectation in that form
                 is a QUOTED DIAGNOSTIC -- the corpus uses it to pin words the
                 tool under test emitted.  If the fixture writes those words,
                 the expectation matches the fixture, not the tool.

    BAKED_ALL    every expectation of the fixture is a constant it writes, so
                 the whole verdict is the fixture agreeing with itself.  Same
                 shape as the bare-`key=` gate already in the test suite, one
                 step further along.
    """
    expect = [str(e).strip() for e in (spec.get("expect_in_output") or [])]
    if not expect:
        return []
    chunks, arms = _all_chunks(py, sh, cpp)
    if not chunks:
        return []
    baked: dict[str, str] = {}
    for e in expect:
        if len(e) < 6:
            continue
        cls = _bake_class(e, chunks, arms)
        if cls:
            baked[e] = cls
    hits = []
    for e, cls in baked.items():
        if "=" not in e:
            hits.append({"expectation": e, "how": f"quoted-diagnostic/{cls}",
                         "kind": "QUOTE"})
    if len(baked) == len([e for e in expect if len(e) >= 6]) and baked:
        hits.append({"expectation": "<all>", "how": "every-expectation-baked",
                     "kind": "ALL", "n": len(baked)})
    return hits


# --------------------------------------------------------------------------
# detector: SELFSAME — an assertion computed from an input, not an output
# --------------------------------------------------------------------------
def _roots(name: str, defs: dict[str, ast.AST], seen: set[str], depth: int = 0
           ) -> tuple[set[str], set[str]]:
    """(terminal names, function names called) that `name` depends on."""
    if depth > 8 or name in seen:
        return {name}, set()
    seen = seen | {name}
    expr = defs.get(name)
    if expr is None:
        return {name}, set()
    return _expr_roots(expr, defs, seen, depth + 1)


def _expr_roots(expr: ast.AST, defs: dict[str, ast.AST], seen: set[str],
                depth: int = 0) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    calls: set[str] = set()
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
        elif isinstance(sub, ast.Name):
            n, c = _roots(sub.id, defs, seen, depth)
            names |= n
            calls |= c
    return names, calls


def _collect_defs(fn: ast.AST) -> dict[str, ast.AST]:
    """name -> defining expression, following tuple unpacking elementwise.

    Unpacking has to be followed or the chain breaks at the first
    `a, b = f(), g()` and every root behind it reads as a terminal input.  That
    silently hid `x = np.arange(total)` behind `bad_u, bad_p = x[:cut], x[cut:]`
    in `skfem/ns_block_split_uses_basis_N` — the fixture this detector exists
    for.
    """
    defs: dict[str, ast.AST] = {}

    def bind(target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            defs.setdefault(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and \
                    len(value.elts) == len(target.elts):
                for t, v in zip(target.elts, value.elts):
                    bind(t, v)
            else:
                for t in target.elts:
                    bind(t, value)

    counts: dict[str, int] = {}

    def note(target: ast.AST) -> None:
        for s in ast.walk(target):
            if isinstance(s, ast.Name):
                counts[s.id] = counts.get(s.id, 0) + 1

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                bind(t, node.value)
                note(t)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
            note(node.target)
        elif isinstance(node, ast.AugAssign):
            note(node.target)
            note(node.target)          # an accumulator is never a constant
        elif isinstance(node, ast.For):
            bind(node.target, node.iter)
            note(node.target)
    _REBOUND[id(defs)] = {n for n, c in counts.items() if c > 1}
    return defs


def _printed_bools(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """(key, expression) for every `print(f"key={<expr>}")` in the module."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for arg in node.args:
            if not isinstance(arg, ast.JoinedStr):
                continue
            prev_lit = ""
            for part in arg.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    prev_lit = part.value
                elif isinstance(part, ast.FormattedValue):
                    m = re.search(r"([A-Za-z_][\w.\[\]]*)=\s*$", prev_lit)
                    if m:
                        out.append((m.group(1), part.value))
                    prev_lit = ""
    return out


_COMPARE_FUNCS = {"allclose", "array_equal", "isclose", "array_equiv",
                  "equal", "isequal"}


def detect_selfsame(spec: dict, py: str | None) -> list[dict]:
    if py is None:
        return []
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    expect = {str(e).strip() for e in (spec.get("expect_in_output") or [])}
    keys_asserted = {e.split("=", 1)[0] for e in expect if _KEYVAL.match(e)}
    defs = _collect_defs(tree)
    hits = []
    for key, expr in _printed_bools(tree):
        if key not in keys_asserted:
            continue
        # resolve one level of aliasing: printed a variable holding the test
        target = expr
        if isinstance(expr, ast.Name) and expr.id in defs:
            target = defs[expr.id]
        pairs = []
        for sub in ast.walk(target):
            if isinstance(sub, ast.Compare) and len(sub.comparators) == 1:
                pairs.append((sub.left, sub.comparators[0]))
            elif isinstance(sub, ast.Call):
                f = sub.func
                nm = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else "")
                if nm in _COMPARE_FUNCS and len(sub.args) >= 2:
                    pairs.append((sub.args[0], sub.args[1]))
        for left, right in pairs:
            ln, _ = _expr_roots(left, defs, set())
            # (a) the prescribed round-trip: one side is a prescribed constant
            #     and the other was BUILT from that same constant, so the
            #     comparison can only come out true.  This is the `G == G`
            #     shape that `condense(..., x=x, D=outflow)` produces.
            if isinstance(right, ast.Name) and right.id.isupper():
                assigned_from = _assigned_from(tree, right.id)
                if assigned_from & ln:
                    hits.append({"expectation": key,
                                 "how": "prescribed-roundtrip",
                                 "shared": sorted(assigned_from & ln)})
                    continue
            # (b) structural identity: the two sides are the same expression.
            if ast.dump(left) == ast.dump(right):
                hits.append({"expectation": key, "how": "identical-operands"})
    return hits


def _all_literal(expr: ast.AST, defs: dict[str, ast.AST], seen: set[str],
                 depth: int = 0) -> bool:
    """True if `expr` is decided before the fixture runs anything.

    Every Name in it has to resolve, transitively, to a literal the fixture
    typed.  `sparta/axisymmetric_weight_radius_and_bad_moves` prints
    `the_two_ordering_constraints_have_different_messages` from
    `QUOTED["a"] != QUOTED["b"]`, where QUOTED is a module-level dict of string
    constants -- the name asserts a fact about SPARTA's diagnostics, the code
    asserts that the author typed two different strings.
    """
    if depth > 6:
        return False
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Call):
            f = sub.func
            nm = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else "")
            if nm not in {"len", "str", "int", "float", "bool", "sorted", "set",
                          "tuple", "list", "lower", "upper", "strip"}:
                return False
        if isinstance(sub, ast.Name):
            if sub.id in seen:
                continue
            # A name that is written more than once, or accumulated into, is
            # not a constant even if its FIRST binding is a literal.  `ok =
            # True` followed by `ok = False` in a branch, and `n = 0` followed
            # by `n += 1`, are the two common shapes; counting them as literal
            # flagged 162 fixtures instead of the handful that are real.
            if sub.id in _REBOUND.get(id(defs), set()):
                return False
            d = defs.get(sub.id)
            if d is None:
                return False
            if not _all_literal(d, defs, seen | {sub.id}, depth + 1):
                return False
    return True


# Populated by _collect_defs, keyed by the identity of the defs dict it built.
_REBOUND: dict[int, set[str]] = {}


def detect_constcmp(spec: dict, py: str | None) -> list[dict]:
    """An asserted boolean whose answer is fixed by the fixture's own literals."""
    if py is None:
        return []
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    expect = {str(e).strip() for e in (spec.get("expect_in_output") or [])}
    keys_asserted = {e.split("=", 1)[0] for e in expect if _KEYVAL.match(e)}
    defs = _collect_defs(tree)
    hits = []
    for key, expr in _printed_bools(tree):
        if key not in keys_asserted:
            continue
        target = defs[expr.id] if isinstance(expr, ast.Name) and expr.id in defs \
            else expr
        # Only an EQUALITY between two named, fixture-supplied quantities.  A
        # threshold comparison against a literal (`PE > 1.0`) is also decided in
        # advance but is honest bookkeeping about the fixture's own setup, and
        # flagging those buried the handful that claim to be about the tool.
        cmps = [s for s in ast.walk(target)
                if isinstance(s, ast.Compare) and len(s.comparators) == 1
                and isinstance(s.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot))]
        for c in cmps:
            sides = (c.left, c.comparators[0])
            if any(isinstance(s, ast.Constant) for s in sides):
                continue
            if _all_literal(c, defs, set()):
                hits.append({"expectation": key, "how": "decided-by-literals"})
                break
    return hits


_SYNTH = {"arange", "eye", "identity", "indices", "meshgrid"}


def detect_synthetic(spec: dict, py: str | None) -> list[dict]:
    """An assertion about a field, computed from a synthetic index vector.

    `skfem/ns_block_split_uses_basis_N` asserts
    `naive_split_contaminates_pressure` on `x = np.arange(total)`: every entry
    of the mis-taken slice is below `bu.N` because `arange` put it there.  The
    line is an arithmetic identity on the index vector, not a statement about a
    velocity/pressure solution, so it holds for any mesh, any element and any
    physics.
    """
    if py is None or not any(s in py for s in _SYNTH):
        return []
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    expect = {str(e).strip() for e in (spec.get("expect_in_output") or [])}
    keys_asserted = {e.split("=", 1)[0] for e in expect if _KEYVAL.match(e)}
    defs = _collect_defs(tree)
    hits = []
    for key, expr in _printed_bools(tree):
        if key not in keys_asserted:
            continue
        names, calls = _expr_roots(expr, defs, set())
        if isinstance(expr, ast.Name) and expr.id in defs:
            n2, c2 = _expr_roots(defs[expr.id], defs, set())
            names |= n2
            calls |= c2
        if (calls & _SYNTH) and not (calls & SOLVERS):
            hits.append({"expectation": key, "how": "computed-from-synthetic",
                         "via": sorted(calls & _SYNTH)})
    return hits


def _assigned_from(tree: ast.AST, const_name: str) -> set[str]:
    """Names that were written FROM `const_name` (incl. via subscript stores)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            src = {s.id for s in ast.walk(node.value) if isinstance(s, ast.Name)}
            if const_name in src:
                for t in node.targets:
                    base = t
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name):
                        out.add(base.id)
    return out


# --------------------------------------------------------------------------
# detector: EMPTYEXC — a boolean read out of an exception string that is empty
# --------------------------------------------------------------------------
_EXCISH = re.compile(r"(exc|err|msg|rais|caught|stderr|trace|warn|abort|fail)",
                     re.I)
_STRINGY = {"lower", "upper", "find", "count", "search", "match", "startswith",
            "endswith", "split", "strip"}


def _direct_names(expr: ast.AST) -> set[str]:
    return {s.id for s in ast.walk(expr) if isinstance(s, ast.Name)}


def _is_truthiness_test(expr: ast.AST) -> bool:
    """`bool(s)`, `not s`, `s != ""`, `len(s) > 0` — a test for "is it empty"."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) \
            and expr.func.id == "bool":
        return True
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return True
    if isinstance(expr, ast.Compare) and len(expr.comparators) == 1:
        c = expr.comparators[0]
        if isinstance(c, ast.Constant) and c.value in ("", 0):
            return True
    return False


def _does_string_work(expr: ast.AST) -> bool:
    """The expression inspects the CONTENT of a string, not its emptiness."""
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Compare) and any(
                isinstance(o, (ast.In, ast.NotIn)) for o in sub.ops):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _STRINGY:
            return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id in {"any", "all"}:
            return True
    return False


def detect_emptyexc(spec: dict, py: str | None) -> list[dict]:
    """A boolean read out of an exception string that is empty by construction.

    `ngsolve/dg_advection_breaks_symmetry_cg_silent` asserts
    `cg_emitted_positive_definite_message=False`.  That flag is computed only
    from the text of the caught exception, and the co-asserted
    `cg_raised_on_unsymmetric_matrix=False` says no exception was caught -- so
    the text is "" and the message flag is False whatever the solver printed.
    The two expectations are one measurement, and neither looks at the solver's
    own output stream.
    """
    if py is None:
        return []
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    expect = {str(e).strip().lower() for e in (spec.get("expect_in_output") or [])}
    asserted_false = {e.split("=", 1)[0] for e in expect if e.endswith("=false")}
    if len(asserted_false) < 2:
        return []
    # An exception handler that turns the exception into text at all.
    has_exc_text = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if _EXC_TEXT.search(body) or (node.name and f"'{node.name}'" in body):
                has_exc_text = True
    if not has_exc_text:
        return []
    defs = _collect_defs(tree)
    printed = _printed_bools(tree)
    hits = []
    for key, expr in printed:
        if key.lower() not in asserted_false:
            continue
        e1 = defs[expr.id] if isinstance(expr, ast.Name) and expr.id in defs else expr
        if not _does_string_work(e1):
            continue
        # DIRECT names only, not the transitive roots: the exception text is
        # usually unpacked out of a tuple, and chasing the chain past that
        # replaces the name that carries the evidence (`exc_cg_a`) with the
        # container it came out of.
        n1 = _direct_names(e1)
        excish = {n for n in n1 if _EXCISH.search(n)}
        if not excish:
            continue
        for k2, e2 in printed:
            if k2 == key or k2.lower() not in asserted_false:
                continue
            t2 = defs[e2.id] if isinstance(e2, ast.Name) and e2.id in defs else e2
            if not _is_truthiness_test(t2):
                continue
            n2 = _direct_names(t2)
            if excish & n2:
                hits.append({"expectation": key,
                             "how": "empty-when-co-asserted",
                             "co_asserted": k2,
                             "via": sorted(excish & n2)})
                break
    return hits


# --------------------------------------------------------------------------
# detector: SUCCESS — a tripwire that fires on the fixture's own success line
# --------------------------------------------------------------------------
def detect_success(spec: dict) -> list[dict]:
    expect = [str(e).strip() for e in (spec.get("expect_in_output") or [])]
    forbid = [str(f).strip() for f in (spec.get("forbid_in_output") or [])]
    hits = []
    for f in forbid:
        if len(f) < 4:
            continue
        for e in expect:
            if f.lower() in e.lower() and f.lower() != e.lower():
                hits.append({"expectation": f, "how": "forbid-inside-expect",
                             "collides_with": e})
                break
    return hits


# --------------------------------------------------------------------------
# detector: ARGMAX — first-index-where on a condition that may never hold
# --------------------------------------------------------------------------
def detect_argmax(spec: dict, py: str | None) -> list[dict]:
    if py is None or "argmax" not in py:
        return []
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        nm = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else "")
        if nm != "argmax" or not node.args:
            continue
        arg = node.args[0]
        # boolean argument -> "first index where"; a plain array -> a real argmax
        is_bool = any(isinstance(s, (ast.Compare,)) for s in ast.walk(arg)) or \
            any(isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.Not)
                for s in ast.walk(arg))
        if not is_bool:
            continue
        # guarded if the enclosing expression is an IfExp testing .any()/.sum()
        src = ast.get_source_segment(py, node) or ""
        line = py.splitlines()[node.lineno - 1] if node.lineno <= len(
            py.splitlines()) else ""
        guarded = ".any()" in line or "if " in line and "any" in line
        if not guarded:
            hits.append({"expectation": f"argmax@line{node.lineno}",
                         "how": "unguarded-argmax", "src": src[:90]})
    return hits


# --------------------------------------------------------------------------
def screen() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = defaultdict(list)
    for fid, d, spec in fixtures():
        py = (d / "source.py").read_text() if (d / "source.py").is_file() else None
        sh = (d / "cmd.sh").read_text() if (d / "cmd.sh").is_file() else None
        cpp = (d / "source.cpp").read_text() if (d / "source.cpp").is_file() else None
        baked = detect_baked(spec, py, sh, cpp)
        for cat, hits in (
            ("BAKED_QUOTE", [h for h in baked if h.get("kind") == "QUOTE"]),
            ("BAKED_ALL", [h for h in baked if h.get("kind") == "ALL"]),
            ("SELFSAME", detect_selfsame(spec, py)),
            ("EMPTYEXC", detect_emptyexc(spec, py)),
            ("SUCCESS", detect_success(spec)),
            ("ARGMAX", detect_argmax(spec, py)),
            ("SYNTHETIC", detect_synthetic(spec, py)),
            ("CONSTCMP", detect_constcmp(spec, py)),
        ):
            for h in hits:
                found[cat].append({"fixture": fid, **h})
    return found


CATEGORIES = ("BAKED_QUOTE", "BAKED_ALL", "SELFSAME", "EMPTYEXC", "SUCCESS",
              "ARGMAX", "SYNTHETIC", "CONSTCMP")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--only", help="restrict to one category")
    a = ap.parse_args()
    found = screen()
    total_fx = len(fixtures())
    print(f"screened {total_fx} fixtures\n")
    flagged_fx: set[str] = set()
    for cat in CATEGORIES:
        hits = found.get(cat, [])
        fx = {h["fixture"] for h in hits}
        flagged_fx |= fx
        by_how = defaultdict(set)
        for h in hits:
            by_how[h["how"]].add(h["fixture"])
        extra = "  " + ", ".join(f"{k}={len(v)}" for k, v in
                                 sorted(by_how.items())) if len(by_how) > 1 else ""
        print(f"{cat:12s} {len(hits):5d} hits in {len(fx):4d} fixtures{extra}")
        if a.detail and (not a.only or a.only == cat):
            for h in sorted(hits, key=lambda x: x["fixture"]):
                print("   ", json.dumps(h))
    print(f"\nunion: {len(flagged_fx)} of {total_fx} fixtures flagged "
          f"({100.0 * len(flagged_fx) / total_fx:.1f}%)")
    if a.json:
        a.json.write_text(json.dumps(found, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
