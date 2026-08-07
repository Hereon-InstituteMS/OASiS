"""Tier-2 for fenics thermal_structural#2: after a form-compile failure the FFCx
JIT cache is left holding a stale generated .c file, and re-running the SAME
script hides the real error behind a JIT timeout.

The fixture drives the exact cycle in one process, on a private cache directory
passed through jit_options so nothing else can interfere. Attempt 1 compiles the
thermo-elastic bilinear form with the thermal term wrongly left inside it and
gets the true error, ArityMismatch. Attempt 2 compiles the identical form again
against the same cache: instead of the ArityMismatch it pauses for the JIT
timeout and raises TimeoutError("JIT compilation timed out, probably due to a
failed previous compile. Try cleaning cache (e.g. remove
<cache>/libffcx_forms_<hash>.c) or increase timeout option.") chained onto
FileExistsError: [Errno 17] File exists: '<that same path>'. Attempt 3 deletes
the named file and the real ArityMismatch is back.

The JIT timeout is set to 3 s (default 10 s) so the fixture stays quick; the
message is the same, it just arrives sooner.

Mutation control: T2_MUTATE=1 removes the stale .c file before attempt 2, which
is the documented cure, so attempt 2 reports the real error.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def attempt(form, jit_options):
    t0 = time.time()
    try:
        dolfinx.fem.form(form, jit_options=jit_options)
        return "compiled", "", "", time.time() - t0
    except BaseException as exc:  # noqa: BLE001
        ctx = exc.__context__
        ctx_text = f"{type(ctx).__name__}: {ctx}" if ctx is not None else ""
        return type(exc).__name__, str(exc), ctx_text, time.time() - t0


def main() -> int:
    cache = tempfile.mkdtemp(prefix="t2_ffcx_cache_")
    jit_options = {"cache_dir": cache, "timeout": 3}
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    # The thermal term does not belong in the bilinear form (see #1); any form
    # that fails to compile does the same job here.
    bad = ufl.inner(2.0 * eps(u) + 1.0 * ufl.tr(eps(u)) * ufl.Identity(d)
                    - 6.3e6 * 50.0 * ufl.Identity(d), eps(v)) * ufl.dx

    kind1, msg1, _, _ = attempt(bad, jit_options)
    left = sorted(os.listdir(cache))
    print(f"attempt1_error={kind1}")
    print(f"attempt1_message={msg1.splitlines()[0] if msg1 else ''}")
    print(f"cache_after_failure={left}")
    stale = [f for f in left if f.endswith(".c")]
    print(f"failed_compile_left_a_c_file={bool(stale)}")

    if MUTATE:
        for f in stale:
            os.remove(os.path.join(cache, f))
        print(f"removed_stale_c_file_before_second_attempt=True")

    kind2, msg2, ctx2, secs = attempt(bad, jit_options)
    print(f"attempt2_error={kind2} after_seconds={secs:.1f}")
    print(f"attempt2_message={msg2}")
    print(f"attempt2_chained_from={ctx2}")
    hidden = (kind2 == "TimeoutError"
              and "failed previous compile" in msg2
              and ctx2.startswith("FileExistsError"))
    print(f"second_run_hides_the_real_error={hidden}")

    for f in os.listdir(cache):
        os.remove(os.path.join(cache, f))
    kind3, msg3, _, _ = attempt(bad, jit_options)
    print(f"attempt3_error={kind3}")
    restored = kind3 == "ArityMismatch" and kind1 == "ArityMismatch"
    print(f"deleting_the_named_file_restores_the_real_error={restored}")
    shutil.rmtree(cache, ignore_errors=True)

    if hidden and restored and bool(stale):
        print("VERDICT=stale_ffcx_cache_masks_the_real_form_error")
        return 0
    print("VERDICT=cache_did_not_mask_anything")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
