"""SPARTA has no native flux boundary condition — and a flux still reaches it.

THE CLAIM UNDER TEST is the most-edited sentence in the coupling knowledge, and
it has been wrong in both directions. The current text says:

  "There is NO native flux boundary condition: of the nine `surf_collide` styles
   (`adiabatic`, `cll`, `diffuse`, `impulsive`, `piston`, `specular`, `td`,
   `transparent`, `vanish`) the four that take a thermal datum — `diffuse`,
   `cll`, `td`, `impulsive` — all take a TEMPERATURE, and none accepts a
   prescribed heat flux.

   A flux CAN still be imposed INDIRECTLY ... `fix surf/temp` converts a
   per-surface energy flux into a per-surface temperature through the gray-body
   Stefan-Boltzmann law `q = sigma*emisurf*T^4`, i.e.
   `T = (q/(sigma*emisurf))^(1/4)` ... SPARTA's own doc says 'SPARTA does not
   check that the specified compute/fix calculates an energy flux'."

Both halves are falsifiable against the installed SPARTA source, and this
fixture checks both. The negative half is the one that matters for planning a
coupling — it is why the sides table says "not natively" in SPARTA's Neumann
column — and a negative claim is exactly the kind that rots silently, because a
new upstream style would not break anything that currently runs.

The counts are DERIVED from the source tree, not typed in: the nine styles come
from the `SurfCollideStyle(...)` registrations, and the four thermal ones from
which of them parse a surface temperature. If SPARTA gains a tenth style, or a
flux-accepting one, this fixture fails and the knowledge has to be rewritten —
which is the whole point of pinning a "cannot".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def sparta_src() -> Path:
    """The SPARTA source tree that goes with the binary this install uses."""
    from backends.sparta.backend import _find_sparta_binary
    binary = _find_sparta_binary()
    if not binary:
        raise L.Absent("no SPARTA binary resolved on this install")
    src = Path(binary).resolve().parent
    if not (src / "surf_collide.cpp").is_file():
        raise L.Absent(f"SPARTA source not next to the binary at {src}")
    print(f"sparta_src={src}")
    return src


def body() -> None:
    L.require_available("sparta")
    src = sparta_src()

    # ── the styles, enumerated from the registration macro ─────────────────
    styles = sorted(
        m.group(1)
        for h in src.glob("surf_collide_*.h")
        for m in [re.search(r"SurfCollideStyle\(\s*([A-Za-z0-9_/]+)\s*,", h.read_text())]
        if m)
    print(f"surf_collide_styles={len(styles)}")
    print(f"surf_collide_names={','.join(styles)}")
    documented = ["adiabatic", "cll", "diffuse", "impulsive", "piston",
                  "specular", "td", "transparent", "vanish"]
    L.check(styles == sorted(documented), "style_list_drifted",
            f"the knowledge names {sorted(documented)}, the source registers "
            f"{styles}")

    # ── which of them take a thermal datum, and what KIND of datum ─────────
    # `parse_tsurf` is SPARTA's own surface-TEMPERATURE parser (it is what
    # accepts both a constant and an `s_<custom>` per-surf vector).
    thermal = sorted(p.stem[len("surf_collide_"):] for p in
                     src.glob("surf_collide_*.cpp")
                     if "parse_tsurf" in p.read_text())
    print(f"thermal_styles={len(thermal)}")
    print(f"thermal_names={','.join(thermal)}")
    L.check(thermal == sorted(["diffuse", "cll", "td", "impulsive"]),
            "thermal_style_list_drifted",
            f"the knowledge names diffuse, cll, td, impulsive; the source has "
            f"{thermal}")

    # ── and NONE of them accepts a flux. This is the "not natively" ────────
    flux_hits = []
    for p in sorted(list(src.glob("surf_collide_*.cpp"))
                    + list(src.glob("surf_collide_*.h"))):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"flux", line, re.I):
                flux_hits.append(f"{p.name}:{i}")
    print(f"surf_collide_flux_mentions={len(flux_hits)}")
    L.check(not flux_hits, "a_surf_collide_style_mentions_flux",
            f"the knowledge says none accepts a prescribed heat flux, but "
            f"{flux_hits[:6]} mention one — recheck the claim")

    # ── the INDIRECT route the knowledge says exists. A "cannot" that hides
    #    a workaround steers an agent away from a coupling that is possible,
    #    which is how this sentence was wrong the first time.
    fst = src / "fix_surf_temp.cpp"
    if not fst.is_file():
        L.check(False, "fix_surf_temp_missing",
                "the knowledge offers `fix surf/temp` as the indirect route; "
                "the source has no fix_surf_temp.cpp")
        return
    body_text = fst.read_text()
    # T = (q / (sigma * emisurf))^(1/4) — a reciprocal emissivity-times-
    # Stefan-Boltzmann prefactor, then a fourth root.
    has_prefactor = re.search(r"prefactor\s*=\s*1\.0\s*/\s*\(\s*emi\s*\*\s*SB_",
                              body_text) is not None
    has_fourth_root = re.search(r"pow\(\s*prefactor\s*\*\s*qw\s*,\s*0\.25\s*\)",
                                body_text) is not None
    print(f"fix_surf_temp_sb_prefactor={has_prefactor}")
    print(f"fix_surf_temp_fourth_root={has_fourth_root}")
    L.check(has_prefactor and has_fourth_root, "sb_conversion_not_found",
            "the knowledge states the conversion is "
            "T = (q/(sigma*emisurf))^(1/4); fix_surf_temp.cpp does not show it")

    # …and the caveat the knowledge quotes, from SPARTA's own doc: the flux is
    # taken from ANY per-surf compute or fix, unchecked. That is what makes an
    # IMPORTED flux reach it, and also what makes it not a Neumann condition.
    doc = src.parent / "doc" / "fix_surf_temp.txt"
    if not doc.is_file():
        L.check(False, "fix_surf_temp_doc_missing", str(doc))
        return
    doc_text = " ".join(doc.read_text(errors="ignore").split())
    quoted = "SPARTA does not check that the specified compute/fix calculates an energy flux"
    print(f"doc_disclaimer_present={quoted in doc_text}")
    L.check(quoted in doc_text, "doc_disclaimer_not_found",
            "the knowledge quotes this sentence from SPARTA's doc verbatim; it "
            "is not in doc/fix_surf_temp.txt on this install")

    # ── the served payload must still SAY the negative, in both places ─────
    from tools.coupling_knowledge import coupling_knowledge, coupling_sides_table
    payload = coupling_knowledge("sparta")
    table = coupling_sides_table()
    L.check("NO native flux boundary condition" in payload,
            "payload_dropped_the_negative",
            "the SPARTA payload no longer states that no native flux BC exists")
    sparta_row = [r for r in table.splitlines() if r.startswith("| SPARTA")]
    print(f"sparta_table_rows={len(sparta_row)}")
    L.check(len(sparta_row) == 1 and "not natively" in sparta_row[0],
            "sides_table_lost_the_neumann_caveat",
            f"expected SPARTA's Neumann column to read 'not natively', row is "
            f"{sparta_row!r}")
    print("claims_checked=7")


L.main(body)
