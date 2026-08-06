"""Every unstarred "yes" in the sides table has a fixture that runs it.

THE CLAIM UNDER TEST is the sentence directly under the table:

  "Every UNSTARRED 'yes' means a real two-code coupling was run in that role on
   THIS install and CONVERGED; it is not copied from a tool docstring."

That is a claim about EVIDENCE, and until the coupling fixtures existed there
was none: eighteen table cells asserting eighteen runs, with nothing in the
repository that reruns any of them. A table is the cheapest thing in a knowledge
base to extend and the hardest to keep honest, because adding a row costs one
line and proving it costs a working participant script.

So this fixture reads the table AS SERVED, and for every unstarred "yes" it
finds the sibling coupling fixture that puts that backend in that role, by
parsing the arrangement calls rather than by a list kept in step by hand. Delete
a pair fixture, or add a row to the table, and this goes red.

What it deliberately does NOT do is assert that those fixtures passed. Reading a
recorded results snapshot would make this fixture green or red according to a
file rather than according to the code, and a stale snapshot is exactly the kind
of evidence the table already had too much of. The pair fixtures assert their own
physics; this one asserts that the table cannot claim a role no fixture runs.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent

# The table writes display names; the rest of the repo uses canonical ones.
CANON = {"fenicsx": "fenics", "4c": "fourc", "ngsolve": "ngsolve",
         "scikit-fem": "skfem", "dune-fem": "dune", "deal.ii": "dealii",
         "febio": "febio", "kratos": "kratos", "sparta": "sparta"}

ARRANGERS = ("heat_arrangement", "elastic_arrangement")


def table_rows() -> list[tuple[str, str, str]]:
    """(canonical backend, dirichlet cell, neumann cell) from the SERVED table."""
    from tools.coupling_knowledge import coupling_sides_table
    rows = []
    for line in coupling_sides_table().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() in ("backend", ""):
            continue
        if set(cells[0]) <= set("-: "):
            continue
        key = CANON.get(cells[0].lower())
        if key is None:
            raise AssertionError(
                f"the sides table has a row for {cells[0]!r} that this fixture "
                f"cannot map to a backend name — add it to CANON and give it a "
                f"fixture, or the row is unbacked")
        rows.append((key, cells[1], cells[2]))
    return rows


def runs_by_fixture() -> dict[tuple[str, str], list[str]]:
    """(backend, role) -> the fixtures that run that backend in that role.

    Parsed from the arrangement calls, so it cannot drift from what actually
    runs the way a hand-maintained list would.
    """
    found: dict[tuple[str, str], list[str]] = {}
    for src in sorted(FIXTURES.glob("*/source.py")):
        if src.parent == HERE:
            continue
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ARRANGERS or len(node.args) < 4:
                continue
            try:
                _, left, right, dirichlet = [ast.literal_eval(a)
                                             for a in node.args[:4]]
            except ValueError:
                continue
            roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
                     "right": "dirichlet" if dirichlet == "right" else "neumann"}
            for backend, position in ((left, "left"), (right, "right")):
                found.setdefault((backend, roles[position]),
                                 []).append(src.parent.name)
    return found


def body() -> None:
    rows = table_rows()
    runs = runs_by_fixture()
    print(f"sides_table_rows={len(rows)}")
    print(f"backend_role_pairs_with_a_fixture={len(runs)}")

    unstarred = 0
    starred = 0
    for backend, dirichlet_cell, neumann_cell in rows:
        for role, cell in (("dirichlet", dirichlet_cell),
                           ("neumann", neumann_cell)):
            if cell.startswith("yes*"):
                # A starred cell claims the ROLE is possible, explicitly NOT
                # that a converged run exists here. Nothing to back.
                starred += 1
                print(f"starred_cell={backend}/{role}")
                continue
            if not cell.startswith("yes"):
                print(f"negative_cell={backend}/{role}={cell}")
                continue
            unstarred += 1
            where = runs.get((backend, role), [])
            print(f"unstarred_cell={backend}/{role} "
                  f"fixtures={','.join(sorted(set(where))) or 'NONE'}")
            L.check(bool(where), f"unbacked_yes_{backend}_{role}",
                    f"the sides table gives {backend} an unstarred 'yes' in the "
                    f"{role} column, which the text under the table says means a "
                    f"real coupling was run in that role on this install — but "
                    f"no coupling fixture puts {backend} on the {role} side. "
                    f"Either add the fixture or star the cell.")
    print(f"unstarred_yes_cells={unstarred}")
    print(f"starred_yes_cells={starred}")
    L.check(unstarred > 0, "no_unstarred_yes_found",
            "the parser found no unstarred 'yes' at all, which means it is not "
            "reading the table it thinks it is")


L.main(body)
