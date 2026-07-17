# ⚠️ TEMPORARY — Mac verification handoff (DELETE before merging this branch)

This file is a scratch instruction sheet for a macOS agent (Fable‑5) to
independently re‑verify the work on `fix/issue-44-45` on real macOS hardware.
It is **not** part of the product and must be removed before the PR is merged.

Branch: `fix/issue-44-45` (5 commits ahead of `main`):

```
e4cd92d  Fix #44 (DUNE FindPython3 picks wrong Python) and #45 (critic block truncated)
c01b46e  Address audit findings on the #44/#45 fixes
1e1b57b  Second audit: fix venv-symlink prefix (#40 redux) + schema regression tests
2372893  V&V verification gate: attestation-driven verdict on every run/coupling result
d0ae323  V&V audit round 2: reject NaN/Inf output (fabrication slip), harden scan
```

Everything below was verified on Linux. **Your job is to prove it on macOS,
where the original #44/#39 bugs actually live (Xcode Python, deal.II.app SDK).**
Be adversarial. Report anything that is not green with the exact command + output.

---

## 0. Setup

```bash
cd <repo>            # the open-fem-agent checkout
git checkout fix/issue-44-45 && git log --oneline -5   # confirm the 5 commits above
# Use the project venv that runs the server:
PY=.venv/bin/python  # adjust if your venv path differs
$PY -c "import mcp, meshio, numpy; print('deps ok', meshio.__version__)"
```

Which backends are live on your Mac? Run and record:

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0,'src')
from core.registry import load_all_backends, get_backend, available_backends
load_all_backends()
for name in ('fenics','dealii','ngsolve','skfem','dune','fourc','kratos','febio'):
    b=get_backend(name)
    if not b: print(f"{name:8} MISSING"); continue
    st,msg=b.check_availability(); print(f"{name:8} {st.value:12} {msg[:70]}")
EOF
```

---

## 1. Issue #44 — DUNE JIT builds against the WRONG Python (macOS Xcode Python)

**The bug (macOS‑specific):** dune‑fem JIT‑compiles C++ at run time; its CMake
`find_package(Python3)` grabbed Xcode's bundled Python 3.9 (or any earlier
interpreter on PATH) instead of the conda env's Python, so the compiled module
could not `import dune.fem`. The fix makes `_dune_subprocess_env` mirror
activation (CONDA_PREFIX/VIRTUAL_ENV) and hand CMake `Python3_ROOT_DIR`.

**Check A — the env is correct for a conda dune install:**
```bash
$PY - <<'EOF'
import sys; sys.path.insert(0,'src')
from backends.dune.backend import _find_dune_python, _dune_subprocess_env, _interpreter_prefix
py=_find_dune_python()
print("resolved dune python:", py)
if py:
    e=_dune_subprocess_env(py)
    print("sys.prefix          :", _interpreter_prefix(py))
    print("CONDA_PREFIX        :", e.get("CONDA_PREFIX"))
    print("Python3_ROOT_DIR    :", e.get("Python3_ROOT_DIR"))
    print("CMAKE_PREFIX_PATH[0] :", e.get("CMAKE_PREFIX_PATH","").split(":")[0])
    print("Python3_EXECUTABLE in env (must be False):", "Python3_EXECUTABLE" in e)
    print("bin on PATH[0]      :", e.get("PATH","").split(":")[0])
EOF
```
Expect: CONDA_PREFIX / Python3_ROOT_DIR / CMAKE_PREFIX_PATH all point at the
dune conda env; `Python3_EXECUTABLE in env` is **False**.

**Check B — a REAL DUNE solve** (if dune‑fem is installed on the Mac): drive a
tiny Poisson through `run_simulation('dune', ...)` and confirm it completes and
`import dune.fem` succeeds inside the JIT build (no "wrong Python" / "cannot
import dune.fem"). If dune‑fem is not installed, say so — do not fake it.

**Check C — the macOS smoking gun (decoy CMake):** put an OLDER python earlier on
PATH (e.g. `/usr/bin/python3` or Xcode's) and confirm a bare
`find_package(Python3)` run with `_dune_subprocess_env(<conda-python>)` still
selects the conda interpreter, not the Xcode one. (On Linux we proved this with
system 3.16 + a pip cmake 4.x; do the equivalent with your Mac's cmake.)

---

## 2. Issue #40 redux (GAP 3) — venv symlink prefix

`_dune_subprocess_env` used `Path(python).resolve()`, which follows a venv's
`bin/python` symlink out to the **base** interpreter, so `<venv>/lib/cmake/dune-*`
went unfound (the #40 failure returns for venv installs). Fixed by asking the
interpreter its own `sys.prefix`.

```bash
$PY -m pytest tests/test_python_env_consistency.py -q
# Expect: all pass, including test_symlinked_venv_prefix_is_the_venv
```
macOS note: `/tmp` and `/var` are symlinked on macOS (`/tmp -> /private/tmp`).
The fix resolves the *bin directory* (not the interpreter symlink), which should
be macOS‑correct — but this is exactly the kind of thing that breaks on macOS, so
**check the venv test passes on the Mac specifically**, and try a real
`python3 -m venv` with a package installed only in the venv.

---

## 3. Issue #45 — mandatory‑critic block dropped by client truncation

The safety block now leads the server `instructions` string so prefix truncation
can't drop it.
```bash
$PY -m pytest tests/test_mcp_instructions.py -q      # expect all pass
$PY -c "import sys;sys.path.insert(0,'src');import server;print(server.mcp.instructions[:400])"
# 'MANDATORY CRITIC — READ THIS FIRST' must appear BEFORE '## Available Backends'
```

---

## 4. V&V verification gate (the big one) — attestation, no fabrications

Design, faithful to the paper §3: OASiS **verifies** (numerical checks +
**attestation**: every reported number is bound to run evidence) but does **not
validate** (physical reality stays the engineer's job); the critic is **optional**
pre‑execution review and is NOT what makes a result verified. Every run/coupling
result now carries `trustworthy_result` + a `verification` verdict; an unverified
result is **flagged, never refused** (no thrown errors).

```bash
$PY -m pytest tests/test_verification_gate.py -q     # expect 14 passed
```

**Real end‑to‑end checks (do these on the Mac):**
- A normal `run_simulation` (fenics/ngsolve/skfem — whatever is live) that
  produces output → `status=completed`, `trustworthy_result=true`, `verification`
  starts `VERIFIED`.
- A run that **exits 0 but writes no output file** → `status=completed_unverified`,
  `trustworthy_result=false`, even with `critic_approved=true`.
- A run that writes an **all‑NaN field** → `trustworthy_result=false`,
  `verification` starts `NOT VERIFIED`, `validation` lists non‑finite values.
  (This is the fabrication slip fixed in d0ae323 — confirm meshio on macOS reads
  the `.vtu` and the scan flags NaN. Watch for any meshio reader that raises
  `SystemExit`; the scan is guarded with `except BaseException`, verify it holds.)

---

## 5. deal.II — verify SUPER carefully (uses `run_with_generator`, macOS SDK trap)

deal.II is a **compiled** backend: the generator writes `main.cpp`, then
`run()` does `cmake .` → `make` → `./fem_solve`. My V&V change added the
finiteness scan + verdict to `run_with_generator`; the compile/run flow is
unchanged. On Linux a real `poisson_2d` solve returned `status=completed`,
`trustworthy_result=true`, `output_files=['result.vtu']`, VERIFIED.

**macOS trap:** deal.II.app ships with an Xcode SDK header conflict (`isinf`,
ambiguous `abs`, `MacOSX.sdk`). `backend.run()` already prints a hint to
`export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`. Confirm:

```bash
$PY - <<'EOF'
import sys, asyncio, json; sys.path.insert(0,'src')
from core.registry import load_all_backends; load_all_backends()
from backends.dealii.generators import get_template
from mcp.server.fastmcp import FastMCP
from tools.consolidated import register_consolidated_tools
cpp=get_template('poisson_2d')({})
gen="open('main.cpp','w').write(r'''"+cpp+"''')\n"
m=FastMCP('t'); cap={}; o=m.tool
def c(*a,**k):
    d=o(*a,**k)
    def w(f):
        r=d(f); cap[f.__name__]=f; return r
    return w
m.tool=c; register_consolidated_tools(m)
out=asyncio.new_event_loop().run_until_complete(cap['run_with_generator']('dealii',gen,critic_approved=False))
d=json.loads(out)
print("status:",d.get('status'),"| trustworthy:",d.get('trustworthy_result'),
      "| files:",d.get('output_files'))
print("verification:", (d.get('verification') or '')[:90])
print("ERROR:", (d.get('error') or '')[:400])
EOF
```
Expect on a healthy Mac: `status=completed, trustworthy=True, files=['result.vtu']`.
If compilation fails with the SDK conflict, confirm the **hint appears in the
error** and that re‑running with `SDKROOT` set fixes it (this is a deal.II.app
packaging issue, not an OASiS bug — but the hint must be present and correct).

Also sanity‑check the deal.II NaN path is caught the same way: any `result.vtu`
full of NaN must come back `NOT VERIFIED` (same scan as §4).

---

## 6. Beyond — regression & "at least as good"

```bash
$PY -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -3
```
On Linux: **32 failed, 529 passed, 32 skipped, 3 errors** — and all 32 failures
are pre‑existing/environmental (skfem/numpy `np.long`, Kratos/FEBio not
installed, backend snapshots). **On the Mac the pass set may differ by
environment; what matters is that NO failure is caused by these 5 commits.**
To prove that, diff against the branch point:
```bash
git worktree add /tmp/oasis-base e4cd92d^   # the commit before this work
# run the same pytest in /tmp/oasis-base, diff the FAILED sets; anything only-in-HEAD is on us
git worktree remove /tmp/oasis-base
```

Report back, per section: PASS/FAIL, the command, and the actual output for
anything not green. Focus fire on #44 (Xcode Python) and §5 (deal.II SDK) —
those are the macOS‑only failure modes we cannot see from Linux.
