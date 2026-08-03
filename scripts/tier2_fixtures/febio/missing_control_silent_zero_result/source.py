"""Tier-2: a deck with no <Control> writes output and solves nothing.

Verifies febio::linear_elasticity#5 — the most dangerous FEBio
outcome for an automated wrapper. Removing the <Control> section
from an otherwise valid deck leaves a run that

  * READS successfully ("Reading file in.feb ...SUCCESS!"),
  * prints NO error message of any kind,
  * writes a full .xplt AND every requested logfile CSV,
  * yet completes zero time steps and ends with
    "E R R O R   T E R M I N A T I O N" and exit status 1.

Any check of the form "did the solver produce output files?"
passes. Any check of the form "did stderr mention an error?"
passes. Only the termination banner, the completed-step count,
and the fact that the CSV holds a single all-zero *Step = 0 block
give it away.

The fixture asserts all four observables at once, and contrasts
them against the same deck WITH <Control> so the difference is
attributable to that section alone.

Verified 2026-08-03 on FEBio 4.12.0.86045466d.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONTROL = '''  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
    </solver>
  </Control>
'''

REST = '''  <Material>
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
    <NodeSet name="top">5,6,7,8</NodeSet>
  </Mesh>
  <MeshDomains><SolidDomain name="P1" mat="M1"/></MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="bot">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="pull" type="prescribed displacement" node_set="top">
      <dof>z</dof><value lc="1">0.01</value><relative>0</relative>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio"><var type="displacement"/></plotfile>
    <logfile>
      <node_data data="ux;uy;uz" delim="," file="nd.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''

HEAD = ('<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<febio_spec version="4.0">\n'
        '  <Module type="solid"/>\n')


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


def run(binary: Path, deck: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix="febio_ctrl_"))
    try:
        (work / "in.feb").write_text(deck)
        p = subprocess.run([str(binary), "-i", "in.feb", "-nosplash"],
                           cwd=str(work), capture_output=True,
                           text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(
            r"Number of time steps completed\s*\.*\s*:\s*(\d+)", out)
        csv = work / "nd.csv"
        blocks = 0
        if csv.is_file():
            blocks = csv.read_text().count("*Step")
        # An ERROR box is a line of stars around the word ERROR.
        has_error_box = bool(re.search(r"\*\s+ERROR\s+\*", out))
        return {
            "rc": p.returncode,
            "read_success": "SUCCESS!" in out,
            "normal": "N O R M A L   T E R M I N A T I O N" in out,
            "errterm": "E R R O R   T E R M I N A T I O N" in out,
            "steps": int(m.group(1)) if m else -1,
            "xplt_bytes": ((work / "in.xplt").stat().st_size
                           if (work / "in.xplt").is_file() else 0),
            "csv_blocks": blocks,
            "error_box": has_error_box,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    binary = find_febio()
    if binary is None:
        print("SKIP: FEBio binary not found (set FEBIO_BINARY)")
        print("missing_control_silent=skipped_no_binary")
        return 0

    good = run(binary, HEAD + CONTROL + REST)
    bad = run(binary, HEAD + REST)
    for label, r in (("with_control", good), ("no_control", bad)):
        print(f"{label}: rc={r['rc']} read_success={int(r['read_success'])} "
              f"normal={int(r['normal'])} errterm={int(r['errterm'])} "
              f"steps={r['steps']} xplt_bytes={r['xplt_bytes']} "
              f"csv_blocks={r['csv_blocks']} "
              f"error_box={int(r['error_box'])}")

    ok = True
    if not (good["rc"] == 0 and good["normal"] and good["steps"] == 1):
        print("FAIL: positive control did not solve — the fixture "
              "deck is broken, so the negative case proves nothing")
        ok = False
    # The pitfall, observable by observable.
    if not bad["read_success"]:
        print("FAIL: no-Control deck did not report a clean read; "
              "the silent-failure claim no longer holds")
        ok = False
    if bad["error_box"]:
        print("FAIL: no-Control deck printed an ERROR box — FEBio "
              "gained a diagnostic, update linear_elasticity#5")
        ok = False
    if not bad["errterm"] or bad["rc"] == 0:
        print("FAIL: no-Control deck did not end in E R R O R "
              "T E R M I N A T I O N / non-zero exit")
        ok = False
    if bad["steps"] != 0:
        print(f"FAIL: no-Control deck reported {bad['steps']} "
              f"completed steps, expected 0")
        ok = False
    if bad["xplt_bytes"] <= 0 or bad["csv_blocks"] != 1:
        print(f"FAIL: no-Control deck did not leave the misleading "
              f"output behind (xplt={bad['xplt_bytes']} bytes, "
              f"csv_blocks={bad['csv_blocks']}, expected >0 and "
              f"exactly 1 step-0 block)")
        ok = False
    if not ok:
        return 1
    print("missing_control_silent=reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
