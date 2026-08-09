#!/bin/bash
# Tier-2 for fourc::particles#6 — the rule is right, the default is not.
#
# Claimed:  "omitting PDBODYID gives all PD particles the default body ID -1;
#           force assembly is applied across body boundaries that should be
#           separate, producing non-physical coupling between bodies."
# Observed: the consequence is exactly right and the number is wrong.  PDBODYID
#           is an optional per-particle state and unset states are zero-filled,
#           so the default is 0, not -1.  Bonds form only between particles whose
#           rounded PDBODYID agree, so any two bodies that both take the default
#           are welded together.
#
# Upstream particle_sph_3d_pdbody_colliding_slow.4C.yaml is two 515-particle
# blocks 0.2 apart with an interaction horizon of 0.3, tagged PDBODYID 1 and 2.
# Four arms:
#   BOTHIDS    1 and 2, as upstream          41126 bonds, all 9 result tests pass
#   OMITBOTH   PDBODYID stripped everywhere  41129 bonds, 6 of 9 tests fail
#   ZEROANDOMIT body 1 -> 0, body 2 stripped 41129 bonds — identical to OMITBOTH,
#              which is only possible if the default is 0.  Were it -1 the two
#              bodies would still differ and the count would stay at 41126.
#   ZEROANDTWO body 1 -> 0, body 2 stays 2   41126 bonds, all 9 tests pass —
#              the control: what matters is that the ids DIFFER, not their values.
# The three welding bonds are enough to wreck the collision, and 4C prints no
# warning about bodies, ids or defaults anywhere.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_3d_pdbody_colliding_slow.4C.yaml) || exit 3

python3 - "$BASE" "$TMP" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
d = sys.argv[2]
one, two = ' PDBODYID 1"', ' PDBODYID 2"'
assert t.count(one) == 515 and t.count(two) == 515, "upstream two-body deck changed"
open(os.path.join(d, "bothids.yaml"),    "w").write(t)
open(os.path.join(d, "omitboth.yaml"),   "w").write(t.replace(one, '"').replace(two, '"'))
open(os.path.join(d, "zeroandomit.yaml"),"w").write(t.replace(one, ' PDBODYID 0"').replace(two, '"'))
open(os.path.join(d, "zeroandtwo.yaml"), "w").write(t.replace(one, ' PDBODYID 0"'))
print("PARTICLES_PER_BODY=515")
PY

probe BOTHIDS     "$TMP/bothids.yaml"
probe OMITBOTH    "$TMP/omitboth.yaml"
probe ZEROANDOMIT "$TMP/zeroandomit.yaml"
probe ZEROANDTWO  "$TMP/zeroandtwo.yaml"

bonds() { grep -m1 -oE 'peridynamic bonds on this proc: [0-9]+' "$TMP/$1.log" | grep -oE '[0-9]+$'; }

grep -m1 -F "processor 0 finished normally" "$TMP/BOTHIDS.log"
echo "BOTHIDS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BOTHIDS.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 41126" "$TMP/BOTHIDS.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 41129" "$TMP/OMITBOTH.log"

B=$(bonds BOTHIDS)
echo "BONDS_BOTHIDS=$B"
echo "BONDS_OMITBOTH=$(bonds OMITBOTH)"
echo "BONDS_ZEROANDOMIT=$(bonds ZEROANDOMIT)"
echo "BONDS_ZEROANDTWO=$(bonds ZEROANDTWO)"
echo "CROSS_BODY_BONDS_WHEN_OMITTED=$(( $(bonds OMITBOTH) - B ))"
echo "OMITBOTH_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OMITBOTH.log")"
echo "ZEROANDTWO_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ZEROANDTWO.log")"
echo "OMITBOTH_BODY_WARNINGS=$(grep -ciE 'body id|pdbodyid|default body' "$TMP/OMITBOTH.log")"

# An explicit 0 next to an omitted tag behaves exactly like two omitted tags,
# so the unset value IS 0.
if [ "$(bonds ZEROANDOMIT)" = "$(bonds OMITBOTH)" ]; then
  echo "DEFAULT_PDBODYID_IS_ZERO=yes"; else echo "DEFAULT_PDBODYID_IS_ZERO=no"; fi
if [ "$(bonds ZEROANDOMIT)" = "$B" ]; then
  echo "DEFAULT_PDBODYID_IS_MINUS_ONE=yes"; else echo "DEFAULT_PDBODYID_IS_MINUS_ONE=no"; fi
exit 0
