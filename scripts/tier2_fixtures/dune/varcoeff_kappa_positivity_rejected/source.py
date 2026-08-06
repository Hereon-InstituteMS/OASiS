"""Tier-2: the variable-coefficient MMS family rejects an indefinite
diffusion field before emitting any code (poisson_mms#3).

The claim is that kappa must stay positive on the whole cube, that the
exact minimum of the affine field is k0 + min(0,kx) + min(0,ky) +
min(0,kz) at a corner, and that validate_parameters rejects kmin <= 0
BEFORE any code is generated. All three are checkable without running
DUNE at all, which is the point: the guard is the cheap detector for a
condition whose runtime symptom (a stalling linear solve) is expensive.

The fixture still imports dune-fem and generates the script, so it
cannot pass on a machine without the backend.

Verified against the shipped generator and dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Fails loudly without dune-fem, the same as every other fixture here.
import dune.fem                                                 # noqa: E402,F401

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src"))

from backends.dune.generators import poisson_mms3d as gen       # noqa: E402


def kappa_min_by_sampling(p, n=12):
    """Brute-force minimum of the affine kappa over the cube."""
    merged = {**gen._DEFAULTS, **p}
    L = merged["L"]
    k0, kx, ky, kz = (merged["k0"], merged["kx"], merged["ky"],
                      merged["kz"])
    best = None
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                X, Y, Z = (i / n * L, j / n * L, k / n * L)
                val = k0 + kx * X + ky * Y + kz * Z
                best = val if best is None else min(best, val)
    return best


def main() -> int:
    fail: list[str] = []

    # ── the formula matches a brute-force minimum ──────────────────
    cases = [
        {},
        {"kx": -0.5},
        {"kx": -0.4, "ky": -0.4, "kz": -0.4},
        {"k0": 1.0, "kx": -1.5},
    ]
    for i, p in enumerate(cases):
        formula = gen.kappa_min({**gen._DEFAULTS, **p})
        sampled = kappa_min_by_sampling(p)
        agree = abs(formula - sampled) < 1e-9
        print(f"case{i}_formula={formula:.6f} sampled={sampled:.6f} "
              f"agree={agree}")
        if not agree:
            fail.append(f"case {p}: the corner formula gives {formula} "
                        f"but the sampled minimum is {sampled}")

    # ── the guard fires on an indefinite field ─────────────────────
    bad = {"k0": 1.0, "kx": -1.5}
    kmin = gen.kappa_min({**gen._DEFAULTS, **bad})
    problems = gen.validate_parameters(bad)
    joined = " | ".join(problems)
    print(f"indefinite_kappa_min={kmin:.6f}")
    print(f"indefinite_rejected={len(problems) > 0}")
    print(f"indefinite_message={joined[:220]}")
    print(f"message_names_ellipticity={'ellipticity' in joined}")
    if kmin > 0:
        fail.append(f"the test case has kappa_min {kmin} > 0, so it "
                    f"does not exercise the guard")
    if not problems:
        fail.append("validate_parameters accepted an indefinite kappa; "
                    "the claim is that it rejects kmin <= 0")
    if "ellipticity" not in joined:
        fail.append(f"the rejection message no longer explains the "
                    f"consequence: {joined[:220]}")

    # ── and it fires BEFORE any code is emitted ────────────────────
    try:
        script = gen.GENERATORS["poisson_mms_3d_varcoeff"](bad)
        print(f"generator_emitted_code_anyway={len(script)}")
        fail.append("the generator produced a script for an indefinite "
                    "kappa; the claim is that the rejection happens "
                    "before any code is emitted")
    except Exception as exc:                                 # noqa: BLE001
        msg = " ".join(str(exc).split())
        print(f"generator_refused={type(exc).__name__}")
        print(f"generator_refusal_message={msg[:200]}")
        if "ellipticity" not in msg and "kappa" not in msg:
            fail.append(f"the generator's refusal does not mention "
                        f"kappa or ellipticity: {msg[:200]}")

    # ── the control: a valid field is accepted and DOES emit code ──
    good = {"k0": 1.0, "kx": 0.5, "ky": 0.25, "kz": 0.1}
    kmin_ok = gen.kappa_min({**gen._DEFAULTS, **good})
    problems_ok = gen.validate_parameters(good)
    print(f"valid_kappa_min={kmin_ok:.6f}")
    print(f"valid_accepted={problems_ok == []}")
    if problems_ok:
        fail.append(f"a positive kappa was rejected: {problems_ok}")
    script_ok = gen.GENERATORS["poisson_mms_3d_varcoeff"](good)
    print(f"valid_script_length={len(script_ok) > 0}")
    print(f"valid_script_is_runnable="
          f"{'DUNE_TEMPLATE_COMPLETE' in script_ok}")
    if "DUNE_TEMPLATE_COMPLETE" not in script_ok:
        fail.append("the accepted script does not print the terminal "
                    "sentinel, so it is not a complete template")

    if not fail:
        print("dune_kappa_positivity_guard_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
