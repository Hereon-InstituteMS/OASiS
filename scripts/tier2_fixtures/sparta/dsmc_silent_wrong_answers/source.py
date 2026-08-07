"""Tier-2: the four SPARTA failures that produce a plausible answer, not an error.

DSMC's characteristic failure is not a crash. It is a run that completes, exits
0, prints no ERROR and no WARNING, and reports a number that is wrong. This
fixture executes four such cases against the installed binary and asserts that
each one really does stay silent while the physics goes away, and that the
corrected form of the same deck really does recover it.

  A  global nrho / fnum omitted. Both default to 1.0 (update.cpp), so the gas is
     a molecule per cubic metre and never collides. Ncoll is 0 on every stats
     row while Np is large.
  B  mixture fractions that sum to less than 1.0. mixture.cpp's init_fraction
     errors only when the sum EXCEEDS 1.0, then forces the cumulative array's
     last entry to 1.0 — so the whole deficit lands on the last species. 25/25
     is silently sampled as 25/75.
  C  react file whose species are not all declared. Every reaction is dropped
     and the summary says 'Gas reactions = 0'.
  D  compute lambda/grid on a grid with empty cells. Empty cells get the
     sentinel BIG = 1.0e20 (compute_lambda_grid.cpp), which poisons any spatial
     average taken over the field.

Statistical assertions. B is the only probe whose observable is stochastic: it
counts how many particles of each species were sampled. The bound used here is
'the last species takes more than 65% of the particles', which was chosen after
running the broken deck at six seeds (12345, 777, 90210, 424242, 31337, 5150)
and observing a last-species share of 0.737 to 0.769 — so the threshold sits
about eight standard deviations below the observed minimum, and the correct
deck's share (about 0.50) is equally far above it on the other side. The deck
also pins its own seed so a single run is reproducible; the bound is what makes
the verdict independent of that choice. A, C and D are exact: 0 collisions, 0
reactions, and the literal value 1e+20.

Missing binary. If no SPARTA executable is found this fixture prints
FIXTURE_ABORT and exits 1, and FIXTURE_ABORT is in forbid_in_output, so a host
without SPARTA records a FAILURE rather than a pass. A fixture that goes green
when nothing ran is worse than no fixture, because the suite counts it as
evidence.

Mutation control. T2_MUTATE=1 replaces each of the four BROKEN decks with its
corrected twin — the pathologies are removed while every assertion stays in
place. The four '*_is_silently_wrong' keys then go False and the fixture fails.
Re-verify: python source.py (passes, rc=0), then T2_MUTATE=1 python source.py
(fails, rc=1).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATE = os.environ.get("T2_MUTATE") == "1"

# Data files each deck references, resolved out of the SPARTA distribution.
NEEDED = ("ar.species", "ar.vss", "air.species", "air.vss", "air.tce")


def find_binary() -> str | None:
    env = os.environ.get("SPARTA_BINARY")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    if env:
        return None          # explicit override that does not exist: do not fall back
    for name in ("spa_serial", "spa_mpi", "sparta"):
        p = shutil.which(name)
        if p:
            return p
    for cand in (Path.home() / "sparta" / "src" / "spa_serial",
                 Path.home() / "Schreibtisch" / "sparta" / "src" / "spa_serial"):
        if cand.is_file():
            return str(cand)
    return None


def find_data(binary: str) -> Path | None:
    """The distribution's data/ dir, located relative to the binary."""
    roots = []
    env = os.environ.get("SPARTA_ROOT")
    if env:
        roots.append(Path(env))
    roots.append(Path(binary).resolve().parent.parent)
    for r in roots:
        d = r / "data"
        if d.is_dir() and all((d / n).is_file() for n in NEEDED):
            return d
    return None


BINARY = find_binary()
if BINARY is None:
    print("FIXTURE_ABORT=no_binary "
          "(set SPARTA_BINARY or build spa_serial; this fixture cannot be "
          "evaluated without one and refuses to report a pass)")
    sys.exit(1)

DATA = find_data(BINARY)
if DATA is None:
    print("FIXTURE_ABORT=no_data "
          "(SPARTA distribution data/ dir with ar.species, ar.vss, "
          "air.species, air.vss, air.tce not found next to the binary)")
    sys.exit(1)


def run(deck: str) -> tuple[int, str]:
    d = Path(tempfile.mkdtemp(prefix="spa_t2_"))   # honours TMPDIR; keep it ext4
    try:
        for n in NEEDED:
            shutil.copy(DATA / n, d / n)
        (d / "in.deck").write_text(deck)
        p = subprocess.run([BINARY, "-in", "in.deck"], cwd=str(d),
                           capture_output=True, text=True, timeout=600)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def stats_rows(txt: str, header_re: str) -> list[list[str]]:
    """Rows of the stats table whose header matches header_re."""
    lines = txt.splitlines()
    out, grabbing = [], False
    for ln in lines:
        if re.search(header_re, ln):
            grabbing = True
            continue
        if grabbing:
            if re.match(r"^\s*[-0-9][-0-9.eE+ \t]*$", ln) and ln.split():
                out.append(ln.split())
            else:
                grabbing = False
    return out


def clean(rc: int, txt: str) -> bool:
    """The run completed and said nothing was wrong."""
    return rc == 0 and "ERROR" not in txt and "WARNING" not in txt


# ─────────────────────────── A: nrho / fnum omitted ───────────────────────────
_A = """seed 12345
dimension 3
global gridcut 1.0e-5
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 10 10 10
species ar.species Ar
mixture air Ar vstream 0.0 0.0 0.0 temp 273.15
{scale}collide vss air ar.vss
create_particles air n 10000
stats 250
stats_style step np nattempt ncoll
timestep 7.00E-9
run 500
"""
A_BROKEN = _A.format(scale="")                                    # defaults 1.0 / 1.0
A_FIXED = _A.format(scale="global nrho 7.07043E22 fnum 7.07043E6\n")

rc, txt = run(A_FIXED if MUTATE else A_BROKEN)
rows = stats_rows(txt, r"^Step\s+Np\s+Natt\s+Ncoll")
a_ncoll = [int(r[3]) for r in rows] if rows else []
a_np = [int(r[1]) for r in rows] if rows else []
a_silent = clean(rc, txt) and bool(a_ncoll) and max(a_ncoll) == 0 and min(a_np) > 0
print(f"A_ncoll_per_row={a_ncoll} np_per_row={a_np} rc={rc}")
print(f"A_missing_nrho_fnum_is_silently_wrong={a_silent}")

rc, txt = run(A_FIXED)
rows = stats_rows(txt, r"^Step\s+Np\s+Natt\s+Ncoll")
a_ref = max((int(r[3]) for r in rows), default=0)
a_control = clean(rc, txt) and a_ref > 100
print(f"A_reference_run_collides={a_control} (max Ncoll per row {a_ref})")

# ───────────────────── B: mixture fractions that under-sum ────────────────────
_B = """seed 12345
dimension 3
global gridcut 1.0e-5
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 10 10 10
species air.species N2 O2
mixture m N2 frac {f}
mixture m O2 frac {f}
mixture m vstream 0 0 0 temp 273.15
global nrho 7.07043E22 fnum 7.07043E7
create_particles m n 0
compute cN count N2
compute cO count O2
stats 100
stats_style step np c_cN c_cO
timestep 7.0e-9
run 100
"""
B_BROKEN = _B.format(f="0.25")     # sums to 0.5 -> deficit falls on the LAST species
B_FIXED = _B.format(f="0.5")       # sums to 1.0


def last_species_share(deck: str) -> tuple[bool, float]:
    rc, txt = run(deck)
    rows = stats_rows(txt, r"^Step\s+Np\s+c_cN\s+c_cO")
    if not rows:
        return False, -1.0
    np_, _n2, o2 = int(rows[-1][1]), int(rows[-1][2]), int(rows[-1][3])
    return clean(rc, txt), (o2 / np_ if np_ else -1.0)

b_clean, b_share = last_species_share(B_FIXED if MUTATE else B_BROKEN)
# Bound, not a pinned value: 0.737-0.769 measured over six seeds for the broken
# deck, about 0.50 for the correct one. 0.65 separates them by roughly eight
# standard deviations on either side, so the verdict does not turn on the seed.
b_silent = b_clean and b_share > 0.65
print(f"B_last_species_share={b_share:.4f} (broken deck runs 0.737-0.769 "
      f"across seeds; correct deck about 0.50; threshold 0.65)")
print(f"B_mixture_fraction_deficit_is_silently_wrong={b_silent}")

b_ok_clean, b_ok_share = last_species_share(B_FIXED)
b_control = b_ok_clean and b_ok_share < 0.60
print(f"B_reference_run_is_balanced={b_control} (share {b_ok_share:.4f})")

# ──────────────────── C: react file whose species are undeclared ──────────────
_C = """seed 12345
dimension 3
global gridcut 1.0e-5
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 10 10 10
species air.species {species}
mixture air {species} temp 20000.0
global nrho 7.07043E22 fnum 7.07043E6
collide vss air air.vss
react tce air.tce
create_particles air n 0
timestep 7.0e-9
run 20
"""
C_BROKEN = _C.format(species="N2 O2")            # products N, O, NO undeclared
C_FIXED = _C.format(species="N2 O2 NO N O")


def gas_reactions(deck: str) -> tuple[bool, int]:
    rc, txt = run(deck)
    m = re.search(r"^Gas reactions\s*=\s*(\d+)", txt, re.M)
    return clean(rc, txt), (int(m.group(1)) if m else -1)

c_clean, c_n = gas_reactions(C_FIXED if MUTATE else C_BROKEN)
c_silent = c_clean and c_n == 0
print(f"C_gas_reactions={c_n} rc_clean={c_clean}")
print(f"C_undeclared_products_drop_all_chemistry_silently={c_silent}")

c_ok_clean, c_ok_n = gas_reactions(C_FIXED)
c_control = c_ok_clean and c_ok_n > 1000
print(f"C_reference_run_has_chemistry={c_control} ({c_ok_n} gas reactions)")

# ───────────────── D: lambda/grid sentinel on a grid with empty cells ─────────
_D = """seed 12345
dimension 3
global gridcut 1.0e-5
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid {n} {n} {n}
species ar.species Ar
mixture air Ar vstream 0 0 0 temp 273.15
global nrho 7.07043E22 fnum 7.07043E6
collide vss air ar.vss
create_particles air n 0
compute gn grid all species nrho
compute gt thermal/grid all all temp
compute lam lambda/grid c_gn[*] c_gt[1] lambda tau
compute lmax reduce max c_lam[1]
compute lmin reduce min c_lam[1]
stats 100
stats_style step np c_lmin c_lmax
timestep 7.00E-9
run 100
"""
D_BROKEN = _D.format(n=20)     # 8000 cells, 10000 particles -> empty cells appear
D_FIXED = _D.format(n=5)       # 125 cells, 10000 particles -> none empty


def lambda_extremes(deck: str) -> tuple[bool, float, float]:
    rc, txt = run(deck)
    rows = stats_rows(txt, r"^Step\s+Np\s+c_lmin\s+c_lmax")
    if not rows:
        return False, -1.0, -1.0
    return clean(rc, txt), float(rows[-1][2]), float(rows[-1][3])

d_clean, d_min, d_max = lambda_extremes(D_FIXED if MUTATE else D_BROKEN)
# BIG is a literal 1.0e20 in compute_lambda_grid.cpp: an exact comparison, not a
# tolerance, and the healthy minimum in the same run is about 2e-6 m.
d_silent = d_clean and d_max == 1e20 and 0.0 < d_min < 1e-3
print(f"D_lambda_min={d_min:.6g} lambda_max={d_max:.6g} rc_clean={d_clean}")
print(f"D_empty_cells_return_the_sentinel_silently={d_silent}")

d_ok_clean, d_ok_min, d_ok_max = lambda_extremes(D_FIXED)
d_control = d_ok_clean and d_ok_max < 1e-3
print(f"D_reference_run_has_no_sentinel={d_control} (max lambda {d_ok_max:.6g})")

# ─────────────────────────────────── verdict ──────────────────────────────────
if MUTATE:
    print("mutation=broken_decks_replaced_by_their_corrected_twins")

ok = (a_silent and a_control and b_silent and b_control
      and c_silent and c_control and d_silent and d_control)
print(f"all_four_silent_failures_reproduce={ok}")
if not ok:
    print("FAIL: at least one probe did not behave as the catalog claims")
sys.exit(0 if ok else 1)
