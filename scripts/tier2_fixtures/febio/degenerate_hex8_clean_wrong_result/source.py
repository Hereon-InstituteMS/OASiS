"""Tier-2: a degenerate hex8 runs clean and returns wrong stress.

Verifies febio::linear_elasticity#7. Repeating a node id inside a
hex8 connectivity list — the classic off-by-one in a generated
mesh — is not rejected by FEBio. The run reaches
"N O R M A L   T E R M I N A T I O N" with exit 0, writes a
complete .xplt, and produces a materially different stress state.
The only hint is a WARNING box ("1 isolated vertex removed."),
which is not an error and is trivially missed.

The fixture runs the same one-element uniaxial deck twice — once
with the correct connectivity 1,2,3,4,5,6,7,8 and once with
1,2,3,4,5,6,7,7 — and asserts:

  * both runs terminate normally with exit 0 (the failure really
    is silent),
  * the degenerate run breaks the x/y symmetry that the correct
    one has (sx == sy for a cube pulled along z),
  * the two stress states differ by more than 10%.

Compare-against-a-good-run is what makes this a correctness
check rather than a smoke test: it fails both if FEBio starts
rejecting degenerate elements (good news, update the pitfall)
and if the wrong answer silently becomes the right one.

Verified 2026-08-03 on FEBio 4.12.0.86045466d: correct run
sx = sy = 1.64474220205, sz = 11.2026953216; degenerate run
sx = 2.05167782746, sy = 1.48697290222, sz = 11.2776298464.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DECK = '''<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
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
      <elem id="1">__CONN__</elem>
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
    <logfile>
      <element_data data="sx;sy;sz" delim="," file="ed.csv"/>
    </logfile>
  </Output>
</febio_spec>
'''


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


def run(binary: Path, conn: str) -> tuple[int, bool, list[float], str]:
    work = Path(tempfile.mkdtemp(prefix="febio_degen_"))
    try:
        (work / "in.feb").write_text(DECK.replace("__CONN__", conn))
        p = subprocess.run([str(binary), "-i", "in.feb", "-nosplash"],
                           cwd=str(work), capture_output=True,
                           text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        normal = "N O R M A L   T E R M I N A T I O N" in out
        stress: list[float] = []
        csv = work / "ed.csv"
        if csv.is_file():
            blocks: list[list[str]] = []
            cur: list[str] | None = None
            for line in csv.read_text().splitlines():
                if line.startswith("*Step"):
                    cur = []
                    blocks.append(cur)
                elif line.startswith("*"):
                    continue
                elif cur is not None and line.strip():
                    cur.append(line)
            if len(blocks) >= 2 and blocks[-1]:
                stress = [float(v)
                          for v in blocks[-1][0].split(",")[1:]]
        return p.returncode, normal, stress, out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    binary = find_febio()
    if binary is None:
        print("SKIP: FEBio binary not found (set FEBIO_BINARY)")
        print("degenerate_hex8=skipped_no_binary")
        return 0

    rc_ok, norm_ok, s_ok, _ = run(binary, "1,2,3,4,5,6,7,8")
    rc_bad, norm_bad, s_bad, out_bad = run(binary, "1,2,3,4,5,6,7,7")
    print(f"correct:    rc={rc_ok} normal={int(norm_ok)} stress={s_ok}")
    print(f"degenerate: rc={rc_bad} normal={int(norm_bad)} "
          f"stress={s_bad}")
    print(f"degenerate_warning_isolated_vertex="
          f"{int('isolated vertex' in out_bad)}")

    ok = True
    if not (rc_ok == 0 and norm_ok and len(s_ok) == 3):
        print("FAIL: positive control did not solve; the fixture "
              "deck is broken")
        ok = False
    if not (rc_bad == 0 and norm_bad and len(s_bad) == 3):
        print("FAIL: the degenerate element no longer runs to normal "
              "termination — FEBio gained a connectivity check, "
              "which is good news; update linear_elasticity#7")
        ok = False
    if ok:
        sym_ok = abs(s_ok[0] - s_ok[1])
        sym_bad = abs(s_bad[0] - s_bad[1])
        rel = max(abs(a - b) / max(abs(a), 1e-30)
                  for a, b in zip(s_ok, s_bad))
        print(f"xy_symmetry_break_correct={sym_ok:.3e}")
        print(f"xy_symmetry_break_degenerate={sym_bad:.3e}")
        print(f"max_relative_stress_difference={rel:.4f}")
        if sym_ok > 1e-9:
            print("FAIL: the correct run is not x/y symmetric; the "
                  "reference is not trustworthy")
            ok = False
        if sym_bad < 1e-3:
            print("FAIL: the degenerate run did not break x/y "
                  "symmetry — the wrong-result claim no longer holds")
            ok = False
        if rel < 0.10:
            print(f"FAIL: degenerate and correct stresses agree to "
                  f"{rel:.4f} relative; the pitfall claims a "
                  f"materially wrong answer")
            ok = False
    if not ok:
        return 1
    print("degenerate_hex8=clean_run_wrong_stress")
    return 0


if __name__ == "__main__":
    sys.exit(main())
