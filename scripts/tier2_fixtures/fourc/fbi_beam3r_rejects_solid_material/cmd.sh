#!/bin/bash
# Tier-2 for fourc::fbi#5 — beams do reject solid materials, but the diagnostic
# is the element's, not a shared beam-material base class, and it is far more
# helpful than the entry claimed.
#
# Claimed:  "using MAT_ElastHyper for beams raises 'beam element requires beam
#            material' from 4C_mat_beam_base.cpp".
# Observed: replacing MAT 2 (MAT_BeamReissnerElastHyper) of upstream
#           fbi_mortar_solidcoupling.4C.yaml with MAT_ElastHyper + a
#           ELAST_CoupNeoHooke sub-material aborts in
#           beam3/src/4C_beam3_reissner_input.cpp line 40 with
#             "The material parameter definition 'm_elasthyper' is not supported
#              by Beam3r element! Choose MAT_BeamReissnerElastHyper,
#              MAT_BeamReissnerElastHyper_ByModes or MAT_BeamReissnerElastPlastic!"
#           — it names the offending material AND enumerates the three legal
#           ones.  The claimed phrase and file do not exist in 4C.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '    MAT_BeamReissnerElastHyper:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_beam_material_changed"; exit 3; }
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology: the material block MAT 2 is given.
SOLID_MATERIAL='    MAT_ElastHyper:\n      NUMMAT: 1\n      MATIDS: [3]\n      DENS: 0.1\n  - MAT: 3\n    ELAST_CoupNeoHooke:\n      YOUNG: 1e+08\n      NUE: 0.0\n'

cp "$BASE" "$TMP/beammat.yaml"
python3 - "$BASE" "$TMP/solidmat.yaml" "$SOLID_MATERIAL" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
new = sys.argv[3].encode().decode("unicode_escape")
m = re.search(r'^    MAT_BeamReissnerElastHyper:\n(      \S.*\n)+', t, re.M)
assert m, "upstream deck no longer carries a MAT_BeamReissnerElastHyper block"
open(sys.argv[2], "w").write(t[:m.start()] + new + t[m.end():])
PY

probe BEAMMAT  "$TMP/beammat.yaml"
probe SOLIDMAT "$TMP/solidmat.yaml"

grep -m1 -F "OK (6)" "$TMP/BEAMMAT.log"
grep -m1 -F "The material parameter definition 'm_elasthyper' is not supported by Beam3r element!" "$TMP/SOLIDMAT.log"
grep -m1 -F "Choose MAT_BeamReissnerElastHyper, MAT_BeamReissnerElastHyper_ByModes or MAT_BeamReissnerElastPlastic!" "$TMP/SOLIDMAT.log"
grep -m1 -F "4C_beam3_reissner_input.cpp" "$TMP/SOLIDMAT.log"

# The claimed wording and file are absent.
echo "CLAIMED_REQUIRES_BEAM_MATERIAL_TEXT=$(grep -ci 'requires beam material' "$TMP/SOLIDMAT.log")"
echo "CLAIMED_MAT_BEAM_BASE_FILE=$(grep -c '4C_mat_beam_base' "$TMP/SOLIDMAT.log")"
# It fails while reading the element, before any field is set up.
echo "SOLIDMAT_FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/SOLIDMAT.log")"
exit 0
