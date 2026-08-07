"""Tier-2: the recorded Kratos environment claim, re-measured.

Pitfall (kratos.curved_mms #0) states that Kratos on this host
works ONLY under the system /usr/bin/python3 (Python 3.8) and
that the pip wheel inside a venv is BROKEN because its shared
libraries require GLIBC 2.32.

That was true of the 10.4.2 wheel. It is NOT a property of the
host. This fixture runs under a venv interpreter, on the same
glibc, and imports Kratos successfully.

Mutation control: the pathology is the stale operational advice "use
/usr/bin/python3", and the way to remove it from this fixture is to OBEY it.
T2_MUTATE=1 re-execs this same file under /usr/bin/python3, the interpreter
the claim prescribes. Kratos loads there too, so every probe still runs for
real and reports a correct measurement -- it simply stops being able to
falsify anything: python_major_minor=3.12, running_in_venv=True and
interpreter_is_system_python3=False all disappear and env_claim_mismatches
rises from 0 to 2.
"""
from __future__ import annotations

import os
import sys
import sysconfig

os.environ.setdefault("OMP_NUM_THREADS", "2")

MUTATE = os.environ.get("T2_MUTATE") == "1"
_SYSTEM_PYTHON = "/usr/bin/python3"
if MUTATE and os.environ.get("T2_MUTATE_REEXEC") != "1":
    # Re-exec before importing Kratos, so the child does the import itself and
    # the measurement really belongs to the prescribed interpreter. The guard
    # variable is what stops the child re-execing again.
    if not os.path.exists(_SYSTEM_PYTHON):
        print(f"mutation_unavailable={_SYSTEM_PYTHON}_absent", flush=True)
    else:
        print("mutation=reexec_under_the_system_python3_the_claim_prescribes",
              flush=True)
        os.environ["T2_MUTATE_REEXEC"] = "1"
        os.execv(_SYSTEM_PYTHON,
                 [_SYSTEM_PYTHON, os.path.abspath(__file__)])

import KratosMultiphysics as KM


def main() -> int:
    bad = 0
    print(f"kratos_imports=True")
    print(f"interpreter_is_system_python3={sys.executable == '/usr/bin/python3'}")
    print(f"python_major_minor={sys.version_info[0]}.{sys.version_info[1]}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"running_in_venv={in_venv}")

    if sys.version_info[:2] == (3, 8):
        print("FAIL: running under Python 3.8, so this fixture cannot "
              "falsify the 'only under python3.8' claim", file=sys.stderr)
        bad += 1
    if not in_venv:
        print("FAIL: not running in a venv, so this fixture cannot falsify "
              "the 'venv wheel is broken' claim", file=sys.stderr)
        bad += 1

    # The claim blames GLIBC 2.32. Report what the interpreter was built
    # against, so the record says which libc this ran on.
    libc = sysconfig.get_config_var("HOST_GNU_TYPE") or "unknown"
    print(f"host_gnu_type={libc}")

    # A real Kratos call, not just the import: if the shared libraries were
    # unusable this would not return.
    model = KM.Model()
    mp = model.CreateModelPart("env")
    mp.SetBufferSize(1)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    print(f"kratos_call_works={mp.NumberOfNodes() == 1}")
    if mp.NumberOfNodes() != 1:
        print("FAIL: Kratos imported but does not function", file=sys.stderr)
        bad += 1

    print(f"env_claim_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
