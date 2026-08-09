#!/bin/bash
# Tier-2 for fourc::fsi#17 — the rule is real, the quoted Signal and file are
# wrong, and the recommended fix is not a fix.
#
# Claimed: "with iter_monolithicstructuresplit (structure=slave), a structural
#           Dirichlet on a node that also belongs to the FSI coupling interface
#           aborts with 'slave node carries Dirichlet' from
#           4C_fsi_monolithic_structuresplit.cpp.  Fix: switch to
#           iter_monolithicfluidsplit (structure=master)."
# Observed: upstream fsi_fp_mono_ss_ga_ga.4C.yaml already declares a Dirichlet on
#           DNODE 2 (structure nodes 17..20, which are the structure side of the
#           FSI interface) with every ONOFF entry at 0.  Turning those three
#           flags on aborts before time step 1 — but with a fourteen-line boxed
#           banner headed
#             "DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE"
#           containing "The slave side of the interface is not allowed to carry
#           Dirichlet boundary conditions." and naming the split explicitly
#           ("MASTER  = FLUID", "SLAVE   = STRUCTURE"), from
#           fsi/src/monolithic/model_evaluator/4C_fsi_monolithicstructuresplit.cpp
#           line 131.  The claimed sentence appears nowhere and the claimed file
#           name (with the extra underscore) does not exist.
#
#           The FIX arm is the interesting one.  Switching COUPALGO to
#           iter_monolithicfluidsplit, as the entry advises, aborts too — the
#           same banner from 4C_fsi_monolithicfluidsplit.cpp line 135, because
#           this deck's fluid interface nodes 13..16 sit inside a domain-wide
#           DESIGN VOL DIRICH condition with ONOFF [0,1,1,0].  The CONTROL arm
#           shows that is not caused by the structural edit: the UNMODIFIED deck
#           with only COUPALGO swapped fails identically.  The constraint is
#           "whichever field is the slave must have a Dirichlet-free interface",
#           and swapping the split only moves which side has to be clean.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_ss_ga_ga.4C.yaml) || exit 3
grep -q 'COUPALGO: "iter_monolithicstructuresplit"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_is_not_structuresplit"; exit 3; }

# The pathology: activate the Dirichlet that sits on the slave interface.
SLAVE_INTERFACE_ONOFF="[1, 1, 1]"

cp "$BASE" "$TMP/clean.yaml"
python3 - "$BASE" "$TMP" "$SLAVE_INTERFACE_ONOFF" <<'PY'
import sys
src, tmp, onoff = sys.argv[1:4]
t = open(src).read()
off = "  - E: 2\n    NUMDOF: 3\n    ONOFF: [0, 0, 0]"
assert off in t, "upstream DNODE 2 Dirichlet block changed"
on = t.replace(off, "  - E: 2\n    NUMDOF: 3\n    ONOFF: " + onoff, 1)
open(tmp + "/slavedirich.yaml", "w").write(on)
swap = ('  COUPALGO: "iter_monolithicstructuresplit"',
        '  COUPALGO: "iter_monolithicfluidsplit"')
assert swap[0] in on
open(tmp + "/claimedfix.yaml", "w").write(on.replace(*swap, 1))
open(tmp + "/fixcontrol.yaml", "w").write(t.replace(*swap, 1))
PY
echo "SLAVEDIRICH_INTERFACE_FLAGS_ON=$(grep -A2 '  - E: 2' "$TMP/slavedirich.yaml" | grep -c 'ONOFF: \[1, 1, 1\]')"
echo "CLAIMEDFIX_USES_FLUIDSPLIT=$(grep -c 'COUPALGO: "iter_monolithicfluidsplit"' "$TMP/claimedfix.yaml")"

probe CLEAN       "$TMP/clean.yaml"
probe SLAVEDIRICH "$TMP/slavedirich.yaml"
probe CLAIMEDFIX  "$TMP/claimedfix.yaml"
probe FIXCONTROL  "$TMP/fixcontrol.yaml"

# Control: the untouched structuresplit deck runs.
grep -m1 -F "processor 0 finished normally" "$TMP/CLEAN.log"
grep -m1 -F "OK (12)" "$TMP/CLEAN.log"

# The real banner, and where it is raised.
grep -m1 -F "DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE" "$TMP/SLAVEDIRICH.log"
grep -m1 -F "The slave side of the interface is not allowed to carry Dirichlet boundary conditions." "$TMP/SLAVEDIRICH.log"
grep -m1 -F "SLAVE   = STRUCTURE" "$TMP/SLAVEDIRICH.log"
grep -m1 -F "4C_fsi_monolithicstructuresplit.cpp" "$TMP/SLAVEDIRICH.log"
echo "SLAVEDIRICH_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/SLAVEDIRICH.log")"
echo "SLAVEDIRICH_CLAIMED_SENTENCE=$(grep -ci 'slave node carries Dirichlet' "$TMP/SLAVEDIRICH.log")"
echo "SLAVEDIRICH_CLAIMED_FILENAME=$(grep -ci '4C_fsi_monolithic_structuresplit.cpp' "$TMP/SLAVEDIRICH.log")"

# The recommended fix aborts as well, from the mirror-image file.
grep -m1 -F "4C_fsi_monolithicfluidsplit.cpp" "$TMP/CLAIMEDFIX.log"
grep -m1 -F "SLAVE   = FLUID" "$TMP/CLAIMEDFIX.log"
echo "CLAIMEDFIX_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/CLAIMEDFIX.log")"
# ...and it aborts for a reason that has nothing to do with the structural edit.
echo "FIXCONTROL_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/FIXCONTROL.log")"
echo "FIXCONTROL_SAME_BANNER=$(grep -c 'DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE' "$TMP/FIXCONTROL.log")"
exit 0
