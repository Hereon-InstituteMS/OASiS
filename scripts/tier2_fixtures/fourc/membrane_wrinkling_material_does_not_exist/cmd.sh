#!/bin/bash
# Tier-2 for fourc::membrane#2 — and a FALSIFICATION of it.
#
# The entry said: "For wrinkling: enable wrinkling model in material definition
# (e.g. MAT_MembraneWrinkling)."  No such material exists in this build.  The
# only membrane materials are MAT_Membrane_ElastHyper and
# MAT_Membrane_ActiveStrain, neither of which relaxes compressive stress, so a
# tension-field / wrinkling model is simply not available and a compressive
# membrane state has to be detected by the user.
#
# Upstream deck membrane_cyl_new_struc, three ways:
#   BASE         MAT_Membrane_ElastHyper as shipped -> exit 0
#   WRINKLING    MAT_MembraneWrinkling              -> unknown, parse error
#   PLAIN_HYPER  MAT_ElastHyper (the general 3D
#                wrapper, same sub-materials)       -> parses, then the ELEMENT
#                                                      rejects it by name
# The third arm is the useful one: swapping the membrane-specific wrapper for
# the ordinary hyperelastic one is the mistake an agent actually makes, and 4C
# catches it with a message that says exactly what is wrong.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream membrane_cyl_new_struc.4C.yaml) || exit 3
grep -q "MAT_Membrane_ElastHyper:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.4C.yaml"
sed 's/MAT_Membrane_ElastHyper:/MAT_MembraneWrinkling:/' "$BASE" > "$TMP/wrink.4C.yaml"
sed 's/MAT_Membrane_ElastHyper:/MAT_ElastHyper:/'        "$BASE" > "$TMP/plain.4C.yaml"

probe BASE        "$TMP/base.4C.yaml"
probe WRINKLING   "$TMP/wrink.4C.yaml"
probe PLAIN_HYPER "$TMP/plain.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
# No wrinkling material: the name is not in the material catalogue at all.
grep -m1 -F "Could not match this input" "$TMP/WRINKLING.log"
grep -m1 -F "MAT_MembraneWrinkling" "$TMP/WRINKLING.log"
# The general hyperelastic wrapper parses but the element refuses it.
grep -m1 -F "The material does not support the evaluation of membranes" "$TMP/PLAIN_HYPER.log"
echo "WRINKLING_STEPS=$(grep -c '^Finalised step' "$TMP/WRINKLING.log")"
echo "PLAIN_HYPER_STEPS=$(grep -c '^Finalised step' "$TMP/PLAIN_HYPER.log")"
exit 0
