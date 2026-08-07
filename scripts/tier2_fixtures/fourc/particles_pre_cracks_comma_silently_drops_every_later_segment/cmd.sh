#!/bin/bash
# Tier-2 for fourc::particles#5 — the separator rule is real, the consequence is
# the opposite of the one claimed.
#
# PROVENANCE WARNING.  PRE_CRACKS and the deck
# particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml are NOT in upstream 4C main;
# they are local branch work on bond-based peridynamics in the checkout this ran
# against.  This fixture therefore certifies branch behaviour and aborts loudly
# (FIXTURE_ABORT=no_upstream_decks) wherever the deck is absent.
#
# Claimed:  "mis-formatted PRE_CRACKS (e.g. comma separator, or missing
#           semicolons between segments) parses as ONE crack with concatenated
#           endpoints — bonds across all spurious segments break instead of just
#           the intended ones."
# Observed: nothing spurious breaks.  The parser splits on ';' and then reads
#           exactly four numbers from each piece, so a comma-separated or
#           run-together list collapses to the FIRST crack and everything after
#           it is thrown away.  You get FEWER broken bonds than you asked for,
#           not more, and no endpoints are concatenated.
#
# Counting bonds on the same 10x10 plate:
#   no crack                                       1058 bonds
#   one segment                                     974 bonds  ( 84 broken)
#   two segments, ';' separated                     890 bonds  (168 broken)
#   the same two, ',' separated                     974 bonds  ( 84 broken)
#   the same two, no separator at all               974 bonds  ( 84 broken)
# The two malformed spellings land on the one-segment count exactly, and 4C
# prints "Number of pre-crack segments: 1" and then a completely clean run —
# exit 0, every result test passing — because the deck's reference values were
# built for one crack.  A silently halved crack set that still says PASS is a
# far worse failure mode than the claimed over-breaking, which cannot happen.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml) || exit 3
grep -q '  PRE_CRACKS: "-5.0 0.0 0.0 0.0"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

SEG1="-5.0 0.0 0.0 0.0"
SEG2="0.0 2.0 5.0 2.0"
mk() { python3 -c "
import sys
t = open(sys.argv[1]).read()
open(sys.argv[2], 'w').write(t.replace('  PRE_CRACKS: \"-5.0 0.0 0.0 0.0\"',
                                       '  PRE_CRACKS: \"%s\"' % sys.argv[3]))" "$BASE" "$1" "$2"; }

mk "$TMP/nocrack.yaml" ""
mk "$TMP/oneseg.yaml"  "$SEG1"
mk "$TMP/semi.yaml"    "$SEG1 ; $SEG2"
mk "$TMP/comma.yaml"   "$SEG1 , $SEG2"
mk "$TMP/nosep.yaml"   "$SEG1 $SEG2"

probe NOCRACK "$TMP/nocrack.yaml"
probe ONESEG  "$TMP/oneseg.yaml"
probe SEMI    "$TMP/semi.yaml"
probe COMMA   "$TMP/comma.yaml"
probe NOSEP   "$TMP/nosep.yaml"

bonds() { grep -m1 -oE 'peridynamic bonds on this proc: [0-9]+' "$TMP/$1.log" | grep -oE '[0-9]+$'; }
segs()  { grep -m1 -oE 'Number of pre-crack segments: [0-9]+' "$TMP/$1.log" | grep -oE '[0-9]+$'; }

grep -m1 -F "processor 0 finished normally" "$TMP/ONESEG.log"
echo "ONESEG_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ONESEG.log")"

# 4C reports how many segments it understood.  The semicolon form gets both.
grep -m1 -F "Number of pre-crack segments: 2" "$TMP/SEMI.log"
# The comma form gets one, and says so in a line nobody reads.
grep -m1 -F "Number of pre-crack segments: 1" "$TMP/COMMA.log"
echo "SEGMENTS_SEMI=$(segs SEMI)"
echo "SEGMENTS_COMMA=$(segs COMMA)"
echo "SEGMENTS_NOSEP=$(segs NOSEP)"

N0=$(bonds NOCRACK)
echo "BONDS_NOCRACK=$N0"
echo "BONDS_ONESEG=$(bonds ONESEG)"
echo "BONDS_SEMI=$(bonds SEMI)"
echo "BONDS_COMMA=$(bonds COMMA)"
echo "BONDS_NOSEP=$(bonds NOSEP)"
echo "BROKEN_SEMI=$(( N0 - $(bonds SEMI) ))"
echo "BROKEN_COMMA=$(( N0 - $(bonds COMMA) ))"
echo "EXTRA_BONDS_BROKEN_BY_COMMA_FORM=$(( $(bonds ONESEG) - $(bonds COMMA) ))"
[ "$(bonds COMMA)" = "$(bonds ONESEG)" ] && echo "COMMA_IS_EXACTLY_THE_FIRST_SEGMENT=yes" \
                                         || echo "COMMA_IS_EXACTLY_THE_FIRST_SEGMENT=no"

# And the malformed deck looks perfectly healthy.
echo "COMMA_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/COMMA.log")"
echo "COMMA_PRE_CRACK_WARNINGS=$(grep -ciE 'ignored|discard|malformed|invalid.*crack' "$TMP/COMMA.log")"
exit 0
