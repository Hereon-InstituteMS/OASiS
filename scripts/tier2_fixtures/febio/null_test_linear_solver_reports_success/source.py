"""Tier-2: <linear_solver type="test"/> solves nothing and says it did.

Verifies febio::biphasic#5, the one case found on FEBio 4.12 that
defeats the whole standard acceptance rule. `test` is a
registered FELINEARSOLVER_ID factory, so it passes every
validation FEBio does, and NumCore/TestSolver.cpp:

    bool TestSolver::BackSolve(double* x, double* b)
    {
        if (m_pA->Rows() == 0) return true;
        for (int i = 0; i < m_n; ++i) x[i] = 0.0;
        ...
        return true;
    }

zeroes the solution vector and reports success. FEBio takes zero
Newton increments, declares convergence, completes every
requested time step and ends with
"N O R M A L   T E R M I N A T I O N" and exit 0 — the exact
combination (banner + exit 0 + non-zero completed-step count)
that the silent-failure checklist otherwise relies on.

The fixture runs one confined-compression biphasic step twice,
identical but for the linear solver, and asserts:

  * both runs pass every standard acceptance check (exit 0,
    normal-termination banner, completed steps > 0) — i.e. the
    checks genuinely cannot separate them;
  * the two stresses nevertheless disagree.

It fails if FEBio ever starts rejecting the null solver (good
news, update the pitfall) and if the two answers ever agree.

Verified 2026-08-03 on FEBio 4.12.0.86045466d: reference
(skyline, symmetric) sz = -2.00691787274, test-solver
sz = -2.00200400802, both with exit 0 and the normal banner.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DECK = '''<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="biphasic"/>
  <Control>
    <analysis>TRANSIENT</analysis>
    <time_steps>2</time_steps>
    <step_size>0.1</step_size>
    <solver type="biphasic">
      <symmetric_stiffness>__SYM__</symmetric_stiffness>
      __LS__
    </solver>
  </Control>
  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>
  <Material>
    <material id="1" name="M1" type="biphasic">
      <phi0>0.2</phi0>
      <solid type="neo-Hookean">
        <density>1.0</density><E>1000</E><v>0.0</v>
      </solid>
      <permeability type="perm-const-iso"><perm>1e-3</perm></permeability>
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
    <bc name="press" type="prescribed displacement" node_set="top">
      <dof>z</dof><value lc="1">-0.01</value><relative>0</relative>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="top"/>
  </Boundary>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <logfile><element_data data="sz" delim="," file="ed.csv"/></logfile>
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


def run(binary: Path, sym: str, ls: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix="febio_null_"))
    try:
        deck = DECK.replace("__SYM__", sym).replace("__LS__", ls)
        (work / "in.feb").write_text(deck)
        p = subprocess.run([str(binary), "-i", "in.feb", "-nosplash"],
                           cwd=str(work), capture_output=True,
                           text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(
            r"Number of time steps completed\s*\.*\s*:\s*(\d+)", out)
        sz = None
        csv = work / "ed.csv"
        if csv.is_file():
            rows = [l for l in csv.read_text().splitlines()
                    if l and not l.startswith("*")]
            if rows:
                sz = float(rows[-1].split(",")[1])
        return {
            "rc": p.returncode,
            "normal": "N O R M A L   T E R M I N A T I O N" in out,
            "steps": int(m.group(1)) if m else -1,
            "sz": sz,
        }
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

    # Reference: symmetrised tangent on the default skyline solver.
    # (A non-symmetric reference would need bicgstab, which is
    # unpreconditioned and not robust — see biphasic#4.)
    ref = run(binary, "symmetric", "")
    nul = run(binary, "non-symmetric",
              '<linear_solver type="test"></linear_solver>')
    for label, r in (("reference", ref), ("test_solver", nul)):
        print(f"{label}: rc={r['rc']} normal={int(r['normal'])} "
              f"steps={r['steps']} sz={r['sz']}")

    ok = True
    if not (ref["rc"] == 0 and ref["normal"] and ref["steps"] == 2
            and ref["sz"] is not None):
        print("FAIL: reference run did not solve; the fixture deck "
              "is broken and the comparison proves nothing")
        ok = False
    if not (nul["rc"] == 0 and nul["normal"] and nul["steps"] == 2):
        print("FAIL: <linear_solver type=\"test\"/> no longer "
              "reports a clean, complete run — FEBio may have "
              "removed or gated the null solver, which is good "
              "news; update biphasic#5 and this fixture together")
        ok = False
    if ok:
        rel = abs(nul["sz"] - ref["sz"]) / max(abs(ref["sz"]), 1e-30)
        print(f"relative_sz_difference={rel:.6f}")
        print(f"both_pass_standard_acceptance_checks="
              f"{int(ref['normal'] and nul['normal'] and ref['rc'] == 0 and nul['rc'] == 0 and nul['steps'] > 0)}")
        if rel < 1e-6:
            print("FAIL: the null solver produced the reference "
                  "answer; either the deck has no interior degrees "
                  "of freedom left to get wrong, or TestSolver "
                  "changed")
            ok = False
    if not ok:
        return 1
    print("null_test_solver=clean_run_not_solved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
