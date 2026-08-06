"""4C and FEBio cannot do preCICE — checked against the binaries, not the docs.

THE CLAIM UNDER TEST, from `knowledge(topic='precice', solver=...)`:

  4C:    "CANNOT — 4C has no preCICE entry point ... Searching the installed 4C
          source tree for `precice` returns nothing, the built binary contains
          no preCICE symbols, and it links no preCICE library."
  FEBio: "CANNOT (as installed) — no preCICE in the binary, no Python API. The
          installed FEBio binary contains no preCICE symbols and FEBio ships no
          Python module, so there is nothing to call `precice` from."

Why a "cannot" needs a fixture more than a "can" does. A capability claim fails
loudly the first time somebody uses it. A refusal never gets exercised: agents
are steered away, nothing runs, and the statement sits there being true or false
without anything finding out. If somebody builds a preCICE-enabled 4C, the
knowledge quietly becomes wrong and every agent keeps being told to use the file
handshake instead. This fixture is what notices.

Each of the three probes the claim names is run for real — `ldd` for the linked
libraries, `nm -D` for the dynamic symbol table, `strings` for anything textual
the build left behind — plus a search of each source tree. Then, because the
knowledge does not merely say "no" but redirects to `couple`, the fixture checks
that both payloads still carry that redirect: a refusal without an alternative
is how an agent concludes the coupling is impossible.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def probe(tool: str, args: list[str], binary: Path) -> int:
    """Count case-insensitive `precice` hits in one probe's output."""
    if not shutil.which(tool):
        raise L.Absent(f"{tool} is not on PATH; the claim names it as evidence")
    r = subprocess.run([tool] + args + [str(binary)], capture_output=True,
                       text=True, errors="ignore", timeout=300)
    return sum(1 for line in (r.stdout + r.stderr).splitlines()
               if "precice" in line.lower())


def source_tree_hits(root: Path) -> int:
    n = 0
    for p in root.rglob("*precice*"):
        if ".git" in p.parts:
            continue
        n += 1
    return n


def check_binary(label: str, binary: Path, tree: Path) -> None:
    binary = binary.resolve()
    print(f"{label}_binary={binary}")
    L.check(binary.is_file(), f"{label}_binary_missing", str(binary))
    if not binary.is_file():
        return
    for tool, args in (("ldd", []), ("nm", ["-D"]), ("strings", ["-a"])):
        hits = probe(tool, args, binary)
        name = "nm_D" if tool == "nm" else tool
        print(f"{label}_{name}_precice_hits={hits}")
        L.check(hits == 0, f"{label}_{name}_found_precice",
                f"{hits} hit(s) — the served knowledge says this binary "
                f"contains no preCICE symbols and links no preCICE library. If "
                f"a preCICE-enabled {label} was built here, the knowledge is "
                f"now wrong and must be rewritten.")
    print(f"{label}_source_tree={tree}")
    if not tree.is_dir():
        L.check(False, f"{label}_source_tree_missing", str(tree))
        return
    hits = source_tree_hits(tree)
    print(f"{label}_source_tree_precice_files={hits}")
    L.check(hits == 0, f"{label}_source_tree_found_precice",
            f"{hits} path(s) matching *precice* under {tree}")


def body() -> None:
    L.require_available("fourc", "febio")
    from backends.fourc.backend import _find_fourc_binary
    from backends.febio.backend import _find_febio_binary

    fourc = _find_fourc_binary()
    febio = _find_febio_binary()
    if not fourc:
        raise L.Absent("no 4C binary resolved")
    if not febio:
        raise L.Absent("no FEBio binary resolved")

    # The source tree is the binary's build root, walked upwards until a
    # directory that is not part of the build sits at the top. Resolving it
    # from the binary keeps the claim about THIS install rather than about a
    # path somebody typed once.
    check_binary("fourc", Path(fourc), _tree_of(Path(fourc)))
    check_binary("febio", Path(febio), _tree_of(Path(febio)))

    # A refusal that does not say what to do instead is how an agent concludes
    # the coupling is impossible. Both payloads must still redirect to `couple`.
    from tools.coupling_knowledge import precice_knowledge
    for name, verdict in (("fourc", "CANNOT"), ("febio", "CANNOT")):
        text = precice_knowledge(name)
        print(f"{name}_precice_payload_chars={len(text)}")
        L.check(verdict in text, f"{name}_verdict_softened",
                f"the payload no longer states {verdict!r}")
        L.check("`couple`" in text or "couple(" in text or
                "topic='coupling'" in text,
                f"{name}_no_redirect_to_couple",
                "the payload refuses preCICE without pointing at the driver "
                "that does work for this backend")
    print("binaries_probed=2")
    print("probes_per_binary=4")


def _tree_of(binary: Path) -> Path:
    """The checkout the binary was built in: walk up from the build dir until a
    directory that has no parent build marker."""
    p = binary.resolve().parent
    for _ in range(6):
        if (p / ".git").exists() or (p / "CMakeLists.txt").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return binary.resolve().parent


L.main(body)
