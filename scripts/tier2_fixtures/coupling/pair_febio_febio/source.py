"""FEBio <-> FEBio, both roles — and NOT for heat.

THE CLAIM UNDER TEST is unusual because it contains a refusal. The sides table
row reads: "FEBio | yes | yes | Python wrapper + XML | FEBio-to-FEBio, both
roles — ELASTICITY, not heat: FEBio 4 has no heat module". The payload expands
it: FEBioHeat "was removed upstream and survives only as a plugin", so "a
conduction participant is impossible here", and the shipped script solves "the
exact linear analogue instead: a uniaxial-strain elastic bar, where displacement
plays the role of temperature and the P-wave modulus the role of conductivity".

That analogue has its own closed form, so the claim is checkable on exactly the
same terms as the conduction pairs: the interface displacement, the interface
traction with the sign each side's own outward normal implies, and conservation
across the interface. The P-wave modulus M = E(1-nu)/((1+nu)(1-2nu)) is what
plays the conductance role, and it is computed here from E and nu rather than
copied, so a change to either constant in the shipped script moves the expected
answer with it instead of silently invalidating the fixture.

FEBio is also the backend whose failure mode is the least readable: "AN UNKNOWN
`<Module type>` SEGFAULTS — it does not produce an error message", which "looks
exactly like a corrupted deck". A wrapper that writes a deck FEBio rejects
produces no exports.json, and the driver reports that as a participant failure —
so this fixture reaching a converged, correct answer is also evidence that the
shipped deck names a module FEBio 4 actually registers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def body() -> None:
    L.require_available("febio")
    p = L.ElasticProblem()
    print(f"p_wave_modulus_left={p.modulus(p.el):.6f}")
    print(f"p_wave_modulus_right={p.modulus(p.er):.6f}")
    L.elastic_arrangement("febio_L_D_febio_R_N", "febio", "febio", "left")
    L.elastic_arrangement("febio_L_N_febio_R_D", "febio", "febio", "right")
    print("pairs_run=2")


L.main(body)
