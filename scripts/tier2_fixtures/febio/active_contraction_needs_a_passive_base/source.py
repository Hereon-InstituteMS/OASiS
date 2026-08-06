"""Tier-2: an active-contraction material standing alone destroys the
element — but not the way the claim said.

Verifies febio::active_contraction#0 and CORRECTS its Signal. The material
must sit inside a `solid mixture` next to a passive elastic solid; alone,
nothing resists the contraction.

Executed on the shipped template with the mixture replaced by the bare
active material: the deck READS successfully (so this is not a schema
failure, as the claim rightly says), and then dies. What it prints is:

  * WARNING `No force acting on the system.`,
  * ERROR `Negative jacobian detected.` — SINGULAR, and with no count,
  * `------- failed to converge at time`,
  * `E R R O R  T E R M I N A T I O N`, exit 1.

The claim quotes `8 negative jacobians detected.` (plural, with the count
of inverted integration points) and `Number of time steps completed
.... : 0`. Neither is what happens: the plural string never appears, and
the run gets through some steps before dying, so the completed count is
not zero. The failure IS an element inversion at an active material with
no passive base — the substance stands — but a wrapper grepping for the
quoted plural string would never match.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MIXTURE = ('    <material id="1" name="Material1" type="solid mixture">\n'
           "      <density>1.0</density>\n"
           '      <mat_axis type="vector">\n'
           "        <a>0,0,1</a>\n"
           "        <d>1,0,0</d>\n"
           "      </mat_axis>\n"
           '      <solid type="neo-Hookean">\n'
           "        <density>1.0</density>\n"
           "        <E>50.0</E>\n"
           "        <v>0.45</v>\n"
           "      </solid>\n"
           '      <solid type="prescribed uniaxial active contraction">\n'
           '        <T0 lc="1">100.0</T0>\n'
           "      </solid>\n"
           "    </material>")
ALONE = ('    <material id="1" name="Material1" '
         'type="prescribed uniaxial active contraction">\n'
         '      <T0 lc="1">100.0</T0>\n'
         "    </material>")


def main() -> int:
    base = L.template("active_contraction_3d_fiber")
    w = L.run(L.swap(base, MIXTURE, ALONE), timeout=900)
    r = L.run(base, timeout=900)

    singular = w.has("Negative jacobian detected.")
    plural = "negative jacobians detected." in w.text
    print(f"standalone_active: rc={w.rc} "
          f"read_success={int(w.read_success)} "
          f"read_failed={int(w.read_failed)} "
          f"no_force_warning={int(w.has('No force acting on the system.'))} "
          f"negative_jacobian_SINGULAR={int(singular)} "
          f"negative_jacobians_PLURAL={int(plural)} "
          f"failed_to_converge="
          f"{int('------- failed to converge at time' in w.text)} "
          f"error_termination={int(w.error_termination)} "
          f"steps_completed={w.steps_completed}")
    print(f"with_passive_base: rc={r.rc} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    good = (w.read_success and not w.read_failed and w.rc != 0
            and singular and not plural
            and "------- failed to converge at time" in w.text
            and w.error_termination
            and r.rc == 0 and r.normal_termination)
    if not good:
        print(w.text[:1200])
    return L.report(good, "active_needs_passive",
                    "reproduced_with_correction", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
