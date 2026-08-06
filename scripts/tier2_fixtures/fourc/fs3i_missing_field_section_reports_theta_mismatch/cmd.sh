#!/bin/bash
# Tier-2 for fourc::fs3i#0 — FS3I really does police its five fields, but not
# with the message the entry quoted, and not by naming the missing field.
#
# Claimed:  "missing any field section aborts at setup with 'FS3I field N not
#            found' from 4C_fs3i_factory.cpp; the failure message names the
#            missing field."
# Observed, on upstream fs3i_part_1wc_infperm.4C.yaml (Gas_Fluid_Structure_
# Interaction: fluid + structure + ALE + scatra1 + scatra2):
#   NOSCATRADYN : delete SCALAR TRANSPORT DYNAMIC and the abort is FS3I's own
#                 cross-field consistency check, fs3i/4C_fs3i.cpp line 188:
#                 "Parameter(s) theta for one-step-theta time-integration scheme
#                  defined in one or more of the individual fields do(es) not
#                  match for partitioned FS3I computation."
#                 It names THETA — the parameter that silently reverted to its
#                 default when the section went away — not the missing section.
#   NOALE       : delete ALE DYNAMIC and you get the generic ALE adapter
#                 message, "No linear solver defined for ALE problems...", from
#                 adapter/4C_adapter_ale.cpp; the message itself says nothing
#                 about FS3I (the word only appears once in the whole log, in a
#                 stack-frame symbol).
#
# No 4C_fs3i_factory.cpp exists and 'FS3I field' appears nowhere.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q '^SCALAR TRANSPORT DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_scatra_section_changed"; exit 3; }
grep -q '^ALE DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_ale_section_changed"; exit 3; }
# The deck's Teko preconditioner XMLs are resolved relative to the CWD.
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The two pathologies.
DROP_SCATRA_SECTION=yes
DROP_ALE_SECTION=yes

cp "$BASE" "$TMP/full.yaml"
python3 - "$BASE" "$TMP/noscatra.yaml" "$DROP_SCATRA_SECTION" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
if sys.argv[3] == "yes":
    t2 = re.sub(r'SCALAR TRANSPORT DYNAMIC:\n(  \S.*\n)+', '', t, count=1)
    assert 'SCALAR TRANSPORT DYNAMIC:\n' not in t2, "scatra section not removed"
    t = t2
open(sys.argv[2], "w").write(t)
PY
python3 - "$BASE" "$TMP/noale.yaml" "$DROP_ALE_SECTION" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
if sys.argv[3] == "yes":
    t2 = re.sub(r'ALE DYNAMIC:\n(  \S.*\n)+', '', t, count=1)
    assert 'ALE DYNAMIC:' not in t2, "ALE section not removed"
    t = t2
open(sys.argv[2], "w").write(t)
PY
echo "NOSCATRA_DECK_HAS_SECTION=$(grep -c '^SCALAR TRANSPORT DYNAMIC:$' "$TMP/noscatra.yaml")"
echo "NOALE_DECK_HAS_SECTION=$(grep -c '^ALE DYNAMIC:' "$TMP/noale.yaml")"

probe FULL     "$TMP/full.yaml"
probe NOSCATRA "$TMP/noscatra.yaml"
probe NOALE    "$TMP/noale.yaml"

grep -m1 -F "OK (3)" "$TMP/FULL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "Parameter(s) theta for one-step-theta time-integration scheme defined in one or more of the individual fields do(es) not match for partitioned FS3I computation." "$TMP/NOSCATRA.log"
grep -m1 -F "4C_fs3i.cpp" "$TMP/NOSCATRA.log"
grep -m1 -F "No linear solver defined for ALE problems. Please set LINEAR_SOLVER in ALE DYNAMIC to a valid number!" "$TMP/NOALE.log"
grep -m1 -F "4C_adapter_ale.cpp" "$TMP/NOALE.log"

# The quoted diagnostic and file do not exist.
echo "CLAIMED_FS3I_FIELD_TEXT=$(grep -ciE 'FS3I field [0-9N] not found' "$TMP/NOSCATRA.log")$(grep -ciE 'FS3I field [0-9N] not found' "$TMP/NOALE.log")"
echo "CLAIMED_FS3I_FACTORY_FILE=$(grep -c '4C_fs3i_factory' "$TMP/NOSCATRA.log")"
# Neither message names the section that was removed.
echo "NOSCATRA_NAMES_ITS_SECTION=$(grep -c 'SCALAR TRANSPORT DYNAMIC' "$TMP/NOSCATRA.log")"
echo "NOALE_MESSAGE_MENTIONS_FS3I=$(grep -c 'No linear solver defined for ALE problems.*FS3I' "$TMP/NOALE.log")"
exit 0
