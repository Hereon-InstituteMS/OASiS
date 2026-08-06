"""Shared helpers for the FEBio Tier-2 fixtures.

This is a MODULE, not a fixture: `scripts/run_tier2_fixtures.py` only
walks directories, so a bare .py file next to the fixture directories is
never mistaken for one.

Every FEBio fixture in this tree follows the same shape, and the shape is
the point:

  * it builds the WRONG variant of a deck and runs the real binary,
  * it builds the RIGHT variant of the same deck and runs that too,
  * it asserts the wrong one shows the pitfall's Signal text and the
    right one does not.

The positive control is what makes the observed message ATTRIBUTABLE to
the trigger. A message can be present in the binary and still belong to
a different trigger — that happened here: a pitfall claimed the LU
linear solver reports "Linear solver failed to find solution. Aborting
run.", which IS a string in the binary, while an LU failure actually
prints "Fatal error in factorization of stiffness matrix." Only running
the wrong variant and reading what comes out settles it.

ABSENCE BEHAVIOUR. A fixture that cannot run the solver must not report
a pass. `binary()` therefore never returns None and never prints a
`<key>=skipped...` line: it prints a line beginning `FAIL:` and exits 1.
Both halves matter — the exit status is what the runner sees, and
`FAIL:` is in every fixture's forbid_in_output so the row is red even if
the exit status were ignored.

It also REFUSES a FEBIO_BINARY that is set but not executable rather
than silently falling through to a binary found elsewhere. Falling
through is what made `_find_febio_binary()` in the backend unusable for
absence testing: pointing FEBIO_BINARY at a non-existent path measured
nothing, because the search continued and found the real binary anyway.

NOTE ON STDIN. `febio4` with no input file drops into an interactive
prompt and hangs on an open stdin, so every invocation here passes
stdin=DEVNULL.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
#  Binary resolution
# ─────────────────────────────────────────────────────────────────────

_CANDIDATES = (
    "FEBio/bin/febio4",
    "FEBioStudio/bin/febio4",
)
_ABS_CANDIDATES = (
    "/opt/febio/bin/febio4",
    "/usr/local/bin/febio4",
)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    """Print a failure line the runner's forbid list catches, and exit 1."""
    print(f"FAIL: {msg}")
    sys.stdout.flush()
    sys.exit(1)


def binary() -> Path:
    """Locate febio4, or fail the fixture.

    Never returns None; never reports a skip. See the module docstring
    for why absence has to be loud.
    """
    env = os.environ.get("FEBIO_BINARY")
    if env is not None:
        p = Path(env)
        if not (p.is_file() and os.access(p, os.X_OK)):
            die(f"FEBIO_BINARY={env!r} is set but is not an executable "
                f"file. Refusing to search elsewhere: silently falling "
                f"through to another binary would make an absence test "
                f"measure nothing.")
        return p
    home = Path(os.environ.get("HOME") or "/nonexistent-home")
    for rel in _CANDIDATES:
        c = home / rel
        if c.is_file() and os.access(c, os.X_OK):
            return c
    for a in _ABS_CANDIDATES:
        c = Path(a)
        if c.is_file() and os.access(c, os.X_OK):
            return c
    w = shutil.which("febio4") or shutil.which("febio3") or shutil.which("febio")
    if w:
        return Path(w)
    die("FEBio binary not found. This fixture RUNS the solver and "
        "cannot be satisfied without it. Build FEBio (cmake "
        "-DUSE_MKL=OFF ...) and symlink it to ~/FEBio/bin/febio4, or "
        "set FEBIO_BINARY.")


# ─────────────────────────────────────────────────────────────────────
#  Running a deck
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    rc: int
    out: str                       # stdout + stderr
    log: str = ""                  # contents of <deck>.log if written
    work: Path = field(default_factory=Path)
    files: dict = field(default_factory=dict)   # name -> text of extra files

    # ── observables, named after what the catalogue quotes ──
    @property
    def text(self) -> str:
        return self.out + "\n" + self.log

    @property
    def read_success(self) -> bool:
        return "SUCCESS!" in self.out

    @property
    def read_failed(self) -> bool:
        return "FAILED!" in self.out

    @property
    def normal_termination(self) -> bool:
        return "N O R M A L   T E R M I N A T I O N" in self.text

    @property
    def error_termination(self) -> bool:
        return "E R R O R   T E R M I N A T I O N" in self.text

    @property
    def segfault(self) -> bool:
        return self.rc < 0 or self.rc == 139

    @property
    def steps_completed(self) -> int:
        m = re.search(r"Number of time steps completed\s*\.*\s*:\s*(\d+)",
                      self.text)
        return int(m.group(1)) if m else -1

    @property
    def flat(self) -> str:
        """stdout+stderr+log with the ERROR/WARNING box unwrapped.

        The box breaks a long diagnostic after 71 columns and puts a `*`
        at each end of every line, so a message that fits on one line in
        the source does not appear as one string in the output. Anything
        matching a WHOLE quoted message must go through this.
        """
        t = re.sub(r"\s*\*?\s*\n\s*\*?\s*", " ", self.text)
        return re.sub(r"\s+", " ", t)

    def has(self, needle: str) -> bool:
        """Substring search over stdout+stderr+log, whitespace-collapsed.

        FEBio's ERROR box wraps at 71 columns, so a long diagnostic is
        split across lines inside the star frame. Collapsing runs of
        whitespace (and dropping the leading star of a wrapped line)
        lets a whole sentence be matched. Short messages are unaffected.
        """
        flat = re.sub(r"\s*\*?\s*\n\s*\*?\s*", " ", self.text)
        flat = re.sub(r"\s+", " ", flat)
        return needle in flat or needle in self.text


def run(deck: str, *, extra: dict | None = None, args=(),
        timeout: int = 240, collect=()) -> Run:
    """Write `deck` to in.feb in a scratch dir, run febio4, collect output.

    extra   — {filename: text} written next to the deck first.
    collect — filenames to read back after the run (logfile CSVs …).
    """
    b = binary()
    work = Path(tempfile.mkdtemp(prefix="febio_t2_"))
    try:
        (work / "in.feb").write_text(deck)
        for name, text in (extra or {}).items():
            (work / name).write_text(text)
        cmd = [str(b), "-i", "in.feb", "-nosplash", *args]
        try:
            p = subprocess.run(cmd, cwd=str(work), capture_output=True,
                               text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired as e:
            rc, out = 124, f"(timeout after {timeout}s) {e}"
        logp = work / "in.log"
        log = logp.read_text(errors="ignore") if logp.is_file() else ""
        files = {}
        for name in collect:
            f = work / name
            files[name] = f.read_text(errors="ignore") if f.is_file() else None
        return Run(rc=rc, out=out, log=log, work=work, files=files)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_raw(deck: str, *, extra=None, args=(), timeout: int = 240,
            collect=()) -> Run:
    """Like run(), but keeps the scratch directory (caller must clean)."""
    b = binary()
    work = Path(tempfile.mkdtemp(prefix="febio_t2_"))
    (work / "in.feb").write_text(deck)
    for name, text in (extra or {}).items():
        (work / name).write_text(text)
    cmd = [str(b), "-i", "in.feb", "-nosplash", *args]
    try:
        p = subprocess.run(cmd, cwd=str(work), capture_output=True,
                           text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = 124, f"(timeout after {timeout}s) {e}"
    logp = work / "in.log"
    log = logp.read_text(errors="ignore") if logp.is_file() else ""
    files = {}
    for name in collect:
        f = work / name
        files[name] = f.read_text(errors="ignore") if f.is_file() else None
    return Run(rc=rc, out=out, log=log, work=work, files=files)


# ─────────────────────────────────────────────────────────────────────
#  Meshes
# ─────────────────────────────────────────────────────────────────────

def hex8_box(n=1, lx=1.0, ly=1.0, lz=1.0, part="Part1"):
    """Structured n x n x n hex8 mesh of a box, FEBio 4.0 <Mesh> body.

    Returns (mesh_xml, info) where info carries the node-set names and
    the node coordinates so a fixture can check results per node.
    """
    nodes = []
    coords = {}
    nid = 0
    idx = {}
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nid += 1
                x, y, z = lx * i / n, ly * j / n, lz * k / n
                nodes.append(f'      <node id="{nid}">{x:.10g},{y:.10g},'
                             f'{z:.10g}</node>')
                coords[nid] = (x, y, z)
                idx[(i, j, k)] = nid
    elems = []
    eid = 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                eid += 1
                c = [idx[(i, j, k)], idx[(i + 1, j, k)],
                     idx[(i + 1, j + 1, k)], idx[(i, j + 1, k)],
                     idx[(i, j, k + 1)], idx[(i + 1, j, k + 1)],
                     idx[(i + 1, j + 1, k + 1)], idx[(i, j + 1, k + 1)]]
                elems.append(f'      <elem id="{eid}">'
                             + ",".join(str(v) for v in c) + "</elem>")

    def sel(pred):
        return sorted(v for kk, v in idx.items() if pred(*kk))

    sets = {
        "all_nodes": sorted(coords),
        "bottom": sel(lambda i, j, k: k == 0),
        "top": sel(lambda i, j, k: k == n),
        "x0": sel(lambda i, j, k: i == 0),
        "y0": sel(lambda i, j, k: j == 0),
        "z0": sel(lambda i, j, k: k == 0),
        "xn": sel(lambda i, j, k: i == n),
        "yn": sel(lambda i, j, k: j == n),
    }
    setxml = "\n".join(
        f'    <NodeSet name="{name}">'
        + ",".join(str(v) for v in ids) + "</NodeSet>"
        for name, ids in sets.items())
    mesh = (
        "  <Mesh>\n"
        '    <Nodes name="AllNodes">\n' + "\n".join(nodes) + "\n"
        "    </Nodes>\n"
        f'    <Elements type="hex8" name="{part}">\n' + "\n".join(elems) + "\n"
        "    </Elements>\n" + setxml + "\n"
        "  </Mesh>")
    return mesh, {"coords": coords, "sets": sets, "idx": idx,
                  "n_nodes": nid, "n_elems": eid, "part": part}


# ─────────────────────────────────────────────────────────────────────
#  Decks
# ─────────────────────────────────────────────────────────────────────

HEADER = '<?xml version="1.0" encoding="ISO-8859-1"?>\n'

LOADCURVE = """  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>"""


def solid_deck(*, mesh=None, material=None, control=None, boundary=None,
               loaddata=LOADCURVE, output=None, module='<Module type="solid"/>',
               globals_=None, extra_sections="", n=1, part="Part1",
               domains=None):
    """A complete, runnable FEBio 4.0 solid deck with every section
    overridable. Defaults give a one-element uniaxial compression that
    reaches normal termination on FEBio 4.12 / USE_MKL=OFF."""
    if mesh is None:
        mesh, _ = hex8_box(n, part=part)
    if material is None:
        material = (
            "  <Material>\n"
            '    <material id="1" name="Material1" type="isotropic elastic">\n'
            "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n"
            "    </material>\n"
            "  </Material>")
    if control is None:
        control = (
            "  <Control>\n"
            "    <analysis>STATIC</analysis>\n"
            "    <time_steps>2</time_steps>\n"
            "    <step_size>0.5</step_size>\n"
            '    <solver type="solid">\n'
            "      <symmetric_stiffness>symmetric</symmetric_stiffness>\n"
            "    </solver>\n"
            "  </Control>")
    if domains is None:
        domains = ("  <MeshDomains>\n"
                   f'    <SolidDomain name="{part}" mat="Material1"/>\n'
                   "  </MeshDomains>")
    if boundary is None:
        boundary = (
            "  <Boundary>\n"
            '    <bc name="fix" type="zero displacement" node_set="bottom">\n'
            "      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n"
            "    </bc>\n"
            '    <bc name="push" type="prescribed displacement" '
            'node_set="top">\n'
            "      <dof>z</dof><value lc=\"1\">-0.05</value>"
            "<relative>0</relative>\n"
            "    </bc>\n"
            "  </Boundary>")
    parts = [HEADER, '<febio_spec version="4.0">\n', module + "\n"]
    if globals_:
        parts.append(globals_ + "\n")
    for sec in (control, material, mesh, domains, boundary):
        if sec:
            parts.append(sec + "\n")
    if extra_sections:
        parts.append(extra_sections + "\n")
    if loaddata:
        parts.append(loaddata + "\n")
    if output:
        parts.append(output + "\n")
    parts.append("</febio_spec>\n")
    return "".join(parts)


def template(name: str, **params) -> str:
    """The deck the shipped generator emits for `name`.

    Basing a fixture on the shipped template rather than a hand-written
    copy means the fixture also fails if the template regresses — which
    is what you want from a claim about how this backend's own decks
    behave. The exotic modules (thermo-fluid, biphasic, fluid, fluid-FSI,
    multiphasic, polar fluid) are only reachable this way without
    re-authoring several hundred lines of correct XML per fixture.
    """
    repo = Path(__file__).resolve().parents[3]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from backends.febio.generators import GENERATORS  # noqa: E402
    gen = GENERATORS.get(name)
    if gen is None:
        die(f"no shipped template named {name!r}; available: "
            f"{sorted(GENERATORS)}")
    return gen(dict(params))


def swap(deck: str, old: str, new: str, *, count: int = 0) -> str:
    """Replace `old` with `new`, failing loudly if `old` is not there.

    A silent no-op replacement is how a fixture ends up running the
    RIGHT deck twice and reporting a pass for a trigger it never
    pulled.
    """
    if old not in deck:
        die(f"deck does not contain {old!r} — the mutation this fixture "
            f"depends on did not apply, so nothing was actually "
            f"triggered. The template or the deck builder changed.")
    return deck.replace(old, new) if count == 0 else deck.replace(
        old, new, count)


def drop(deck: str, fragment: str) -> str:
    """Delete `fragment`, failing loudly if it is not there."""
    return swap(deck, fragment, "")


def logfile(*items):
    """<Output><logfile> block. items are (kind, data, filename) triples."""
    rows = "\n".join(
        f'      <{kind} data="{data}" delim="," file="{fn}"/>'
        for kind, data, fn in items)
    return ("  <Output>\n    <logfile>\n" + rows
            + "\n    </logfile>\n  </Output>")


def parse_log_csv(text: str):
    """Parse an FEBio logfile CSV into [(time, {id: [floats]}), ...].

    FEBio writes blocks headed `*Step  = k` / `*Time  = t` / `*Data =`
    then `id,v1,v2,...` rows.
    """
    if not text:
        return []
    blocks = []
    time = None
    rows: dict = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("*Step"):
            if rows:
                blocks.append((time, rows))
                rows = {}
            continue
        if s.startswith("*Time"):
            try:
                time = float(s.split("=", 1)[1])
            except (IndexError, ValueError):
                time = None
            continue
        if s.startswith("*") or not s:
            continue
        bits = s.split(",")
        try:
            rows[int(bits[0])] = [float(x) for x in bits[1:]]
        except ValueError:
            continue
    if rows:
        blocks.append((time, rows))
    return blocks


def report(ok: bool, key: str, good: str, bad: str, detail: str = "") -> int:
    """Print the fixture's verdict token and return an exit status."""
    if detail:
        print(detail)
    print(f"{key}={good if ok else bad}")
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────
#  The two recurring shapes
# ─────────────────────────────────────────────────────────────────────

def parse_error(key: str, *, wrong: str, right: str, message: str,
                also=(), timeout: int = 240,
                right_must_run: bool = True) -> int:
    """The wrong deck must be REJECTED BY THE READER with `message`;
    the right deck must read and run without it.

    `also` holds further strings the wrong run must print (e.g. the
    second half of a two-part diagnostic).

    The right-hand run is the whole point: it establishes that the
    message is attributable to the one thing that was changed. Without
    it, any deck that fails for any reason satisfies the expectation.
    """
    w = run(wrong, timeout=timeout)
    r = run(right, timeout=timeout)
    needles = (message, *also)
    w_ok = w.read_failed and w.rc != 0 and all(w.has(x) for x in needles)
    r_ok = ((r.read_success and r.rc == 0 and r.normal_termination)
            if right_must_run else True)
    r_clean = not any(r.has(x) for x in needles)
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          + " ".join(f"msg[{i}]={int(w.has(x))}"
                     for i, x in enumerate(needles)))
    print(f"right: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} "
          f"steps={r.steps_completed} msg_absent={int(r_clean)}")
    if not w_ok:
        print("NOTE wrong-variant output follows:")
        print(w.text[:1500])
    if not (r_ok and r_clean):
        print("NOTE right-variant (positive control) output follows:")
        print(r.text[:1500])
    return report(w_ok and r_ok and r_clean, key, "reproduced",
                  "not_reproduced")


def init_error(key: str, *, wrong: str, right: str, message: str,
               also=(), timeout: int = 240) -> int:
    """The wrong deck must READ SUCCESSFULLY and only then fail, with
    `message`; the right deck must run clean.

    This shape is its own pitfall class: a wrapper that stops at the
    reader line believes the deck is fine.
    """
    w = run(wrong, timeout=timeout)
    r = run(right, timeout=timeout)
    needles = (message, *also)
    w_ok = (w.read_success and not w.read_failed and w.rc != 0
            and all(w.has(x) for x in needles))
    r_ok = r.read_success and r.rc == 0 and r.normal_termination
    r_clean = not any(r.has(x) for x in needles)
    print(f"wrong: rc={w.rc} read_success={int(w.read_success)} "
          f"read_failed={int(w.read_failed)} "
          + " ".join(f"msg[{i}]={int(w.has(x))}"
                     for i, x in enumerate(needles)))
    print(f"right: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} "
          f"steps={r.steps_completed} msg_absent={int(r_clean)}")
    if not w_ok:
        print("NOTE wrong-variant output follows:")
        print(w.text[:1500])
    if not (r_ok and r_clean):
        print("NOTE right-variant (positive control) output follows:")
        print(r.text[:1500])
    return report(w_ok and r_ok and r_clean, key, "reproduced",
                  "not_reproduced")
