#!/bin/bash
# Tier-2 for fourc::fsi#7 — the per-field NUMDOF rule is real (fluid = dim+1,
# structure = dim, ALE = dim) but 4C only checks it in ONE direction and the
# diagnostic is not the claimed one.
#
# Claimed: "a structural Dirichlet with NUMDOF=3 on a 2D problem (or NUMDOF=2 on
#           a 3D problem) aborts at setup with 'invalid NUMDOF'."
# Observed: the check is Core::FE::Dbc::read_dirichlet_condition in
#           core/fem/src/discretization/4C_fem_discretization_utils_dbc.cpp
#           line 292, and its source reads `if (num_dbc_dofs < numdf)`.  Too FEW
#           entries abort; too MANY are accepted without a word.  On the 3D
#           upstream deck fsi_fp_mono_fs_ga_ga.4C.yaml:
#
#     fluid point Dirichlet 4 -> 3   "3 DOFs given but 4 expected in Point
#                                     Dirichlet boundary condition"     ABORT
#     ALE volume Dirichlet  3 -> 2   "2 DOFs given but 3 expected in Volume
#                                     Dirichlet boundary condition"     ABORT
#     structure point Dirichlet 3 -> 4                     exit 0, OK (6)
#
#           The message never says "invalid NUMDOF", never names the field, and
#           an over-declared NUMDOF slips through silently — which is the case an
#           agent copying a 3D block into a 2D deck will actually hit.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^DESIGN VOL ALE DIRICH CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_vol_ale_dirich"; exit 3; }

# The pathology: give a field's Dirichlet the wrong DOF count.
MISDECLARE_NUMDOF=yes

cp "$BASE" "$TMP/correct.yaml"
python3 - "$BASE" "$TMP" "$MISDECLARE_NUMDOF" <<'PY'
import sys
src, tmp, do = sys.argv[1:4]
t = open(src).read()
fluid4 = """  - E: 1
    NUMDOF: 4
    ONOFF: [0, 0, 0, 0]
    VAL: [0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]"""
fluid3 = """  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]"""
struct3 = """  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [1, 0, 0]
    FUNCT: [1, 0, 0]"""
struct4 = """  - E: 2
    NUMDOF: 4
    ONOFF: [1, 1, 1, 0]
    VAL: [1, 0, 0, 0]
    FUNCT: [1, 0, 0, 0]"""
ale3 = """DESIGN VOL ALE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]"""
ale2 = """DESIGN VOL ALE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]"""
for a, b in ((fluid4, fluid3), (struct3, struct4), (ale3, ale2)):
    assert a in t, "upstream Dirichlet block changed"
n = (lambda a, b: b) if do == "yes" else (lambda a, b: a)
open(tmp + "/fluid_short.yaml", "w").write(t.replace(fluid4, n(fluid4, fluid3), 1))
open(tmp + "/struct_long.yaml", "w").write(t.replace(struct3, n(struct3, struct4), 1))
open(tmp + "/ale_short.yaml", "w").write(t.replace(ale3, n(ale3, ale2), 1))
PY

probe CORRECT     "$TMP/correct.yaml"
probe FLUIDSHORT  "$TMP/fluid_short.yaml"
probe ALESHORT    "$TMP/ale_short.yaml"
probe STRUCTLONG  "$TMP/struct_long.yaml"

grep -m1 -F "OK (6)" "$TMP/CORRECT.log"
grep -m1 -F "processor 0 finished normally" "$TMP/CORRECT.log"

# Too few entries: hard abort, and this is the wording.
grep -m1 -F "3 DOFs given but 4 expected in Point Dirichlet boundary condition"  "$TMP/FLUIDSHORT.log"
grep -m1 -F "2 DOFs given but 3 expected in Volume Dirichlet boundary condition" "$TMP/ALESHORT.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/FLUIDSHORT.log"

# Too many entries: accepted in silence, same pinned results.
grep -m1 -F "OK (6)" "$TMP/STRUCTLONG.log"
echo "STRUCTLONG_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/STRUCTLONG.log")"
echo "STRUCTLONG_NUMDOF_WARNINGS=$(grep -ciE 'numdof|DOFs given but' "$TMP/STRUCTLONG.log")"

# The claimed diagnostic exists in none of the arms.
echo "CLAIMED_INVALID_NUMDOF=$(cat "$TMP"/FLUIDSHORT.log "$TMP"/ALESHORT.log "$TMP"/STRUCTLONG.log \
      | grep -ci 'invalid NUMDOF')"
echo "ABORT_MESSAGE_NAMES_THE_FIELD=$(grep -c 'DOFs given but.*\(fluid\|structure\|ale\)' "$TMP/FLUIDSHORT.log")"
exit 0
