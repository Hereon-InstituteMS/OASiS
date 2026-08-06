#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#0 -- 4C has no "DNS" mode to select, and
# the y+ check the entry recommends is only produced if you ask for it.
#
# TURBULENCE_APPROACH accepts exactly two values, CLASSICAL_LES and
# DNS_OR_RESVMM_LES.  The default is DNS_OR_RESVMM_LES: 4C does not distinguish
# a DNS from a residual-based-VMM LES at all, and performs no resolution check of
# any kind, so nothing in the code will tell you a "DNS" is under-resolved.
# The y+ diagnostic does exist -- but only in <output>.flow_statistics, and only
# when DUMPING_PERIOD > 0.  The upstream channel deck ships DUMPING_PERIOD: 0 and
# therefore writes a statistics file with a header and zero rows.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q "  DUMPING_PERIOD: 0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^FLUID DYNAMIC/TURBULENCE MODEL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/base.yaml"
sed 's|^FLUID DYNAMIC/TURBULENCE MODEL:|FLUID DYNAMIC/TURBULENCE MODEL:\n  TURBULENCE_APPROACH: "DNS"|' "$BASE" > "$TMP/dns.yaml"
sed 's/  DUMPING_PERIOD: 0/  DUMPING_PERIOD: 1/' "$BASE" > "$TMP/dump.yaml"

probe BASE "$TMP/base.yaml"
probe DNS  "$TMP/dns.yaml"
probe DUMP "$TMP/dump.yaml"

# "DNS" is not a value; 4C prints the two that are.
grep -m1 -F "Could not match this input" "$TMP/DNS.log"
grep -m1 -F "possible values: CLASSICAL_LES|DNS_OR_RESVMM_LES" "$TMP/DNS.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
# The resolution check the entry asks for is not something 4C does...
echo "RESOLUTION_CHECK_MENTIONS=$(grep -ciE 'kolmogorov|under-resolved|resolution check' "$TMP/BASE.log")"
# ...and the y+ column only appears once DUMPING_PERIOD is turned on.
echo "BASE_STATS_ROWS=$(grep -c '^ *-\?[0-9]' "$TMP/o_BASE.flow_statistics")"
echo "DUMP_STATS_ROWS=$(grep -c '^ *-\?[0-9]' "$TMP/o_DUMP.flow_statistics")"
echo "DUMP_HAS_YPLUS_COLUMN=$(grep -c 'y+' "$TMP/o_DUMP.flow_statistics")"
grep -m1 -F "(u_tau)^2 = tau_W/rho" "$TMP/o_DUMP.flow_statistics"
exit 0
