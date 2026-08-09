"""THE PARTITIONED TWO-WAY TSI, AGAINST 4C's NATIVE Thermo_Structure_Interaction.

THE CLAIM UNDER TEST: the answer OASiS's cross-code partitioned coupling
converges to is the answer a production multiphysics code gets when it solves
the same coupled problem monolithically in one piece.

WHY THIS AND NOT ONLY THE IN-HOUSE MONOLITHIC REFERENCE. The monolithic
reference in _lib/tsi_monolithic.py shares an author and an understanding of the
physics with the participants; it can catch a partitioning mistake and cannot
catch a modelling one. 4C can. It is a different code, written by other people,
with trilinear hexahedra instead of triangles, its own time integrators, its own
material implementation and a monolithic Newton solve of the coupled system.

WHAT THIS FIXTURE ESTABLISHES, in order:

  1. 4C's own reverse direction is the classical thermoelastic one — checked
     BLACK BOX, inside 4C alone, with no reference to anything here: in uniaxial
     strain the two-way solve at CAPA must equal the ONE-WAY solve at
     CAPA*(1+delta). If it does, then 4C's `- N^T ctemp : (B_L d') N T` term is
     beta * T * d/dt tr(eps) with the same beta, and 4C is a reference for the
     reverse direction and not only for the forward one.
  2. 4C agrees with the in-house monolithic solve, one-way and two-way.
  3. THE PARTITIONED CROSS-CODE COUPLING agrees with 4C.
  4. Switching the reverse direction off moves the partitioned answer, and 4C's
     one-way run moves by the same amount. Both codes' reverse directions are
     alive and the same size.

THE ONE MODELLING DIFFERENCE IS MEASURED, NOT ASSUMED. 4C's reverse term uses
the CURRENT temperature where the classical linear theory uses T_ref, so the two
models differ by the factor T/T_ref. The problem here is run with a temperature
excursion of 0.3 K on a reference of 293 K, which bounds that factor at ~1e-3;
the fixture prints the bound, and prints the one-way agreement — where the
models are identical — beside the two-way one, so the reader can see the
difference the linearisation costs rather than take it on trust.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                          # noqa: E402
import tsilib as T                                               # noqa: E402
import tsi_fourc as F                                            # noqa: E402
import numpy as np                                               # noqa: E402

NX4 = 320               # 4C's own mesh along the bar
REFN = (1600, 2)        # the in-house reference, refined well past both


def _line(coords, vals):
    """Collapse a 2-D field on a uniaxial problem onto its x-profile."""
    co = np.asarray(coords, float)
    sel = np.isclose(co[:, 1], co[:, 1].min())
    order = np.argsort(co[sel, 0])
    return co[sel, 0][order], np.asarray(vals, float)[sel][order]


def _rel(got, want):
    return float(np.linalg.norm(got - want)) / max(float(np.linalg.norm(want)),
                                                   1e-30)


def body() -> None:
    L.require_available("fourc", "fenics", "skfem")
    p = F.FOURC_NATIVE
    q = F.FOURC_COUPLED
    root = Path(tempfile.mkdtemp(prefix="t2tsi_4c_"))
    print(f"fourc_delta={p.delta:.6f}")
    print(f"fourc_linearisation_bound={F.linearisation_bound(p):.3e}")
    print(f"fourc_coupled_linearisation_bound={F.linearisation_bound(q):.3e}")

    # ── 0. every key the deck writes must be in 4C's own accepted grammar.
    # A mis-cased key and an invented one produce the SAME "Failed to match
    # specification" message, so `4C -p` is the only thing that can tell them
    # apart; this asks it, by name, before anything is run.
    bad = F.audit_deck_keys(p) + F.audit_deck_keys(q)
    L.check(not bad, "fourc_deck_has_keys_outside_the_grammar", "; ".join(bad)[:400])
    print(f"fourc_deck_keys_all_in_grammar={bool(not bad)}")

    # ── 1. is 4C's reverse direction the classical one? Asked inside 4C alone.
    f2 = F.run(root / "fourc_two_way", p, nx=NX4, two_way=True)
    f1 = F.run(root / "fourc_one_way", p, nx=NX4, two_way=False)
    fe = F.run(root / "fourc_one_way_capa_eff", p, nx=NX4, two_way=False,
               capa_scale=1.0 + p.delta)
    print(f"fourc_transverse_spread_T={f2['transverse_spread_T']:.3e}")
    L.check(f2["transverse_spread_T"] < 1e-9, "fourc_solution_not_uniaxial",
            f"4C's temperature varies across the section by "
            f"{f2['transverse_spread_T']:.3e} K, so it is not the 1-D problem "
            f"the comparison assumes")
    scale = float(np.max(np.abs(f2["T"] - p.t_ref))) or 1.0
    ident = float(np.max(np.abs(f2["T"] - fe["T"]))) / scale
    size = float(np.max(np.abs(f2["T"] - f1["T"]))) / scale
    print(f"fourc_effective_capacity_identity_dev={ident:.3e}")
    print(f"fourc_reverse_direction_size={size:.3e}")
    print(f"fourc_identity_over_effect={size / max(ident, 1e-30):.1f}")
    L.check(ident < 1e-3 and size > 100 * ident,
            "fourc_reverse_term_is_not_the_classical_one",
            f"4C's two-way solve at CAPA differs from its one-way solve at "
            f"CAPA*(1+delta) by {ident:.3e}, against a reverse-direction effect "
            f"of {size:.3e}")
    print(f"fourc_reverse_term_is_classical={bool(ident < 1e-3 and size > 100 * ident)}")

    # ── 2. 4C against the in-house monolithic solve, both directions
    import tsi_monolithic as M
    ref2 = M.solve_monolithic(p, *REFN, coupling=1.0)
    ref1 = M.solve_monolithic(p, *REFN, coupling=0.0)
    x2, th2 = _line(ref2["coordinates"], ref2["theta"])
    x1, th1 = _line(ref1["coordinates"], ref1["theta"])
    _, ux2 = _line(ref2["coordinates"], ref2["ux"])
    e2 = _rel(f2["T"] - p.t_ref, np.interp(f2["x"], x2, th2))
    e1 = _rel(f1["T"] - p.t_ref, np.interp(f1["x"], x1, th1))
    eu = _rel(f2["ux"], np.interp(f2["x"], x2, ux2))
    print(f"fourc_vs_monolithic_twoway_theta_relL2={e2:.3e}")
    print(f"fourc_vs_monolithic_oneway_theta_relL2={e1:.3e}")
    print(f"fourc_vs_monolithic_twoway_ux_relL2={eu:.3e}")
    L.check(e2 < 1e-3, "fourc_twoway_disagrees_with_monolithic", f"{e2:.3e}")
    L.check(e1 < 2e-4, "fourc_oneway_disagrees_with_monolithic", f"{e1:.3e}")
    L.check(eu < 1e-3, "fourc_ux_disagrees_with_monolithic", f"{eu:.3e}")
    print(f"fourc_matches_monolithic={bool(e2 < 1e-3 and e1 < 2e-4 and eu < 1e-3)}")

    # ── 3. THE PARTITIONED CROSS-CODE COUPLING against 4C, on the offset
    # problem (see tsi_fourc.FOURC_COUPLED for why the coupled comparison
    # cannot use the same temperatures as the identity check).
    g2 = F.run(root / "fourc_coupled_two_way", q, nx=NX4, two_way=True)
    g1 = F.run(root / "fourc_coupled_one_way", q, nx=NX4, two_way=False)
    gsize = (float(np.max(np.abs(g2["T"] - g1["T"])))
             / max(float(np.max(np.abs(g2["T"] - q.t_ref))), 1e-30))
    print(f"fourc_coupled_reverse_direction_size={gsize:.3e}")
    two = T.run_tsi("cpl_2way", "fenics", "skfem", p=q)
    if not T.assert_run_clean("cpl_2way", two):
        return
    xc, thc = _line(two["theta_coords"], two["theta_field"])
    c2 = _rel(thc, np.interp(xc, g2["x"], g2["T"] - q.t_ref))
    print(f"coupled_vs_fourc_twoway_theta_relL2={c2:.3e}")
    L.check(c2 < 2e-3, "coupled_disagrees_with_native_fourc", f"{c2:.3e}")
    print(f"coupled_matches_native_fourc={bool(c2 < 2e-3)}")

    # ── 4. both reverse directions alive, and the same size
    one = T.run_tsi("cpl_1way", "fenics", "skfem", p=q, thermal_reads=False)
    T.assert_run_clean("cpl_1way", one, expect_one_way=True)
    _, tho = _line(one["theta_coords"], one["theta_field"])
    csize = float(np.max(np.abs(thc - tho))) / max(float(np.max(np.abs(thc))), 1e-30)
    print(f"coupled_reverse_direction_size={csize:.3e}")
    d = abs(csize - gsize) / max(gsize, 1e-30)
    print(f"reverse_direction_size_vs_fourc_rel={d:.3e}")
    L.check(csize > 1e-3, "coupled_reverse_direction_is_inert",
            f"switching mechanical->thermal off moved the coupled answer by only "
            f"{csize:.3e}")
    L.check(d < 0.05, "reverse_direction_size_disagrees_with_fourc",
            f"OASiS measures the reverse direction at {csize:.3e} of the answer "
            f"and 4C at {gsize:.3e} — {d:.1%} apart")
    print(f"both_codes_agree_on_reverse_direction={bool(csize > 1e-3 and d < 0.05)}")
    print("pairs_run=1")


L.main(body)
