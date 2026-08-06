"""Tier-2: an unregistered <Module type=...> SEGFAULTS FEBio.

Verifies febio::heat#0. FEBioXML/FEBioModuleSection.cpp reads the
type attribute and hands it straight to
FEModelBuilder::SetActiveModule(szt) with no existence check, so
a module name FEBio does not know dereferences a null module.

This is the worst failure shape a wrapper can meet: no ERROR box,
no "FAILED!", no .log file, stdout truncated mid-line at
"Reading file <name>.feb ...". The ONLY observable is the process
signal.

The fixture runs four decks that differ from a known-good solid
deck by the module string alone:

  * "heat"          — the module OASiS's heat template used, and
                      which does not exist in FEBio 4.12
  * "biphasic-FSI"  — likewise; the real 4.12 spelling is a
                      MATERIAL inside the fluid-FSI module
  * "Solid"/"SOLID" — pure case errors on a REAL module name

plus the positive control "solid", which must read cleanly. The
positive control matters: it proves the four crashes are caused
by the module string and not by anything else in the deck.

Verified 2026-08-03 on FEBio 4.12.0.86045466d: all four bad
strings exit 139 (SIGSEGV) with no diagnostic; "solid" reads
SUCCESS.

MUTATION CONTROL. T2_MUTATE=1 replaces the four unregistered module
strings with the REGISTERED name "solid" — the pathology removed. None
of the four runs then dies undiagnosed, the crash count drops to 0 and
'unknown_module_segfault_count=4' is no longer printed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATE = os.environ.get("T2_MUTATE") == "1"

DECK = '''<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="__MODULE__"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" name="M1" type="isotropic elastic">
      <density>1.0</density><E>1000.0</E><v>0.3</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="N">
      <node id="1">0,0,0</node><node id="2">1,0,0</node>
      <node id="3">1,1,0</node><node id="4">0,1,0</node>
      <node id="5">0,0,1</node><node id="6">1,0,1</node>
      <node id="7">1,1,1</node><node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" name="P1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="bot">1,2,3,4</NodeSet>
  </Mesh>
  <MeshDomains><SolidDomain name="P1" mat="M1"/></MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="bot">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
  </Boundary>
</febio_spec>
'''

BAD_MODULES = ("heat", "biphasic-FSI", "Solid", "SOLID")


def find_febio() -> Path | None:
    env = os.environ.get("FEBIO_BINARY")
    if env and Path(env).is_file():
        return Path(env)
    for c in (Path.home() / "FEBio" / "bin" / "febio4",
              Path.home() / "FEBioStudio" / "bin" / "febio4",
              Path("/opt/febio/bin/febio4"),
              Path("/usr/local/bin/febio4")):
        if c.is_file():
            return c
    w = shutil.which("febio4") or shutil.which("febio")
    return Path(w) if w else None


def run_module(binary: Path, module: str) -> tuple[int, str, bool]:
    work = Path(tempfile.mkdtemp(prefix="febio_mod_"))
    try:
        (work / "in.feb").write_text(DECK.replace("__MODULE__", module))
        p = subprocess.run([str(binary), "-i", "in.feb", "-nosplash"],
                           cwd=str(work), capture_output=True,
                           text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out, (work / "in.log").is_file()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    binary = find_febio()
    if binary is None:
        # A missing binary is a FAILURE, not a skip. This fixture
        # exists to verify solver behaviour; with no solver it
        # verifies nothing, and returning 0 would let the tier-2
        # floor certify an unrun check. "FAIL:" is in this
        # fixture's forbid_in_output, so the runner records a
        # failure even if the exit status were ignored.
        print("FAIL: FEBio binary not found. This fixture RUNS the "
              "solver and cannot be satisfied without it. Set "
              "FEBIO_BINARY or symlink ~/FEBio/bin/febio4.")
        return 1

    modules = ("solid",) * len(BAD_MODULES) if MUTATE else BAD_MODULES
    if MUTATE:
        print("mutation=every_module_string_is_the_registered_solid")
    crashed = 0
    for m in modules:
        rc, out, has_log = run_module(binary, m)
        # A crash is a negative returncode from subprocess (signal)
        # or 139 when a shell re-encodes it.
        sigsegv = (rc < 0) or (rc == 139)
        diagnosed = ("FAILED!" in out) or ("ERROR" in out)
        print(f"module={m!r} rc={rc} sigsegv={int(sigsegv)} "
              f"diagnosed={int(diagnosed)} log_written={int(has_log)}")
        if sigsegv and not diagnosed:
            crashed += 1
        else:
            print(f"NOTE: {m!r} no longer crashes silently — the "
                  f"FEBio build may have gained a module-name check")

    rc_ok, out_ok, _ = run_module(binary, "solid")
    control_reads = "SUCCESS!" in out_ok
    print(f"positive_control_solid_reads={int(control_reads)}")
    print(f"unknown_module_segfault_count={crashed}")

    if not control_reads:
        print("FAIL: positive control with Module type=\"solid\" did "
              "not read cleanly — the fixture deck itself is broken, "
              "so the crashes prove nothing")
        return 1
    if crashed != len(modules):
        print(f"FAIL: expected all {len(BAD_MODULES)} unregistered "
              f"module strings to die undiagnosed, got {crashed}. "
              f"If FEBio added validation this is good news — "
              f"update febio::heat#0 and this fixture together")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
