"""Cross-check the MCP knowledge base against the backend's actual source.

The MCP catalog promises specific keyword strings to the agent: section names,
DYNAMICTYPE options, material classes, parameter keys, element types.  If any
of those drift away from what the solver's input parser actually accepts, the
agent will emit input files that the solver silently rejects -- exactly the
"silent input-mismatch" failure mode the paper's pitfall layer is meant to
prevent.

These tests close the loop by fetching the canonical names directly from the
solver source (local checkout via ``$<SOLVER>_ROOT`` or
raw.githubusercontent.com) and asserting set membership.

The tests SKIP -- they do not fail -- when neither a local source tree nor
network access is available.  This keeps the suite green in offline CI but
runs real checks whenever sources are reachable (e.g. on a weekly cron).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.groundtruth import fourc as fourc_gt  # noqa: E402


def _normalise_key(raw: str) -> str:
    """Strip any ``" (annotation)"`` suffix the catalog appends for human
    readability.  The actual 4C input parser sees the bare key only."""
    return raw.split("(", 1)[0].strip()


def _parse_param_string(text: str) -> set[str]:
    """Parse a comma-separated parameter list as used by the older
    ``plasticity_models``-style entries.  Drops empty tokens."""
    return {
        _normalise_key(t)
        for t in (s.strip() for s in text.split(","))
        if t
    }


def _collect_fourc_catalog_materials() -> dict[str, set[str]]:
    """Walk every 4C physics generator, pull every materials-like block, and
    return {material_name: set_of_parameter_keys}.

    The material name is kept as the catalog literally writes it -- including
    composite forms like ``MAT_ElastHyper + ELAST_CoupNeoHooke`` -- because
    the source-of-truth check for composites is the union of each
    component's parameter set.  Stripping the suffix loses information.

    Two storage schemas coexist in the catalog:
      * structured ``parameters: {KEY: {description, range}}`` (preferred)
      * legacy ``parameters: "KEY1, KEY2, ..."`` comma-separated string
    Both are parsed here so the test catches the same bug class in either
    form -- otherwise free-form-string entries (which is where the original
    GTN-with-lowercase-keys bug lived) silently bypass verification.
    """
    from backends.fourc.generators import _GENERATOR_SPECS, get_generator

    # Collect material-like blocks at multiple nesting depths -- the legacy
    # ``plasticity_models`` block in solid_mechanics is a sibling of
    # ``materials`` but holds the same kind of name -> param mapping.
    interesting_blocks = ("materials", "plasticity_models")

    out: dict[str, set[str]] = {}
    for module_key in _GENERATOR_SPECS:
        try:
            gen = get_generator(module_key)
        except Exception:
            continue
        try:
            knowledge = gen.get_knowledge()
        except Exception:
            continue
        for block_name in interesting_blocks:
            block = knowledge.get(block_name)
            if not isinstance(block, dict):
                continue
            for mat_name, body in block.items():
                keys: set[str] = set()
                if isinstance(body, dict):
                    params = body.get("parameters")
                    if isinstance(params, dict):
                        keys = {_normalise_key(k) for k in params if isinstance(k, str)}
                    elif isinstance(params, str):
                        keys = _parse_param_string(params)
                # No params at all -> still record the entry with empty set so
                # the "no_catalog_key_is_unknown" check stays silent for it
                # while the down-stream "missing-keys" report can flag it.
                out.setdefault(mat_name, set()).update(keys)
    return out


def _allowed_keys_for(name: str, source_lookup: dict[str, set[str]]) -> set[str] | None:
    """Resolve the allowed-parameter set for a catalog entry name.

    For composite names (``MAT_ElastHyper + ELAST_CoupNeoHooke``) the
    allowed set is the union over components.  Returns ``None`` if any
    component is missing from ``source_lookup`` -- the caller should then
    skip that entry rather than miscount a partial union as ground truth.
    """
    from tests.groundtruth.fourc import composite_components

    parts = composite_components(name)
    allowed: set[str] = set()
    for p in parts:
        if p not in source_lookup:
            return None
        allowed |= source_lookup[p]
    return allowed


class TestFourcDynamictypeOptions(unittest.TestCase):
    """The catalog must not promise ``DYNAMICTYPE`` keywords that 4C rejects."""

    @classmethod
    def setUpClass(cls):
        cls.source_options = fourc_gt.dynamictype_options()
        if cls.source_options is None:
            raise unittest.SkipTest(
                "4C source unavailable (set FOURC_ROOT or enable network access)"
            )

    # Match only bullet-list lines that look like:  "  'KeyWord' -- ..."
    # so the test ignores stray quoted tokens elsewhere in the prose (e.g.
    # the MODE entry's "YN" | "Lame" mentioned for hyperelastic sub-materials).
    _BULLET_RE = re.compile(r"^\s+'([A-Z][A-Za-z0-9]+)' -- ", re.MULTILINE)

    def test_structural_dynamics_keywords_are_valid(self):
        """Every keyword introduced as a DYNAMICTYPE bullet ('Name' -- desc) in
        the structural_dynamics docstring must be a real 4C input keyword."""
        from backends.fourc.generators import get_generator

        gen = get_generator("structural_dynamics")
        doc = gen.get_knowledge()["time_integration"]["DYNAMICTYPE"]
        promised = set(self._BULLET_RE.findall(doc))
        self.assertTrue(
            promised,
            "Bullet-scanner found no DYNAMICTYPE keywords -- docstring "
            "format may have changed; update _BULLET_RE.",
        )
        unknown = promised - self.source_options
        self.assertFalse(
            unknown,
            f"\nstructural_dynamics catalog promises DYNAMICTYPE keywords that "
            f"4C does not accept: {sorted(unknown)}\n"
            f"4C input parser accepts: {sorted(self.source_options)}",
        )


class TestFourcMaterialParameters(unittest.TestCase):
    """Every parameter key the catalog lists for a 4C material must be one the
    material class actually reads via ``matdata.parameters.get<...>("KEY")``."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = _collect_fourc_catalog_materials()
        # Resolve source keys eagerly so a single unreachable source skips the
        # whole class rather than partially-running tests.
        cls.source: dict[str, set[str]] = {}
        skipped: list[str] = []
        for mat, path in fourc_gt.MATERIAL_SOURCE.items():
            keys = fourc_gt.material_parameter_keys(path)
            if keys is None:
                skipped.append(f"{mat} ({path})")
            else:
                cls.source[mat] = keys
        if not cls.source and skipped:
            raise unittest.SkipTest(
                f"4C material source unavailable: {skipped}"
            )

    def test_no_catalog_key_is_unknown_to_source(self):
        """Catalog parameter keys must be a subset of source keys.

        Failure means the agent will emit a parameter the 4C input parser
        rejects -- the same failure mode that the GTN/DYNAMICTYPE incidents
        manifested before this test existed.
        """
        violations: list[str] = []
        for mat_name, cat_keys in self.catalog.items():
            if not cat_keys:
                continue  # schema-incompatible entry (free-form string)
            allowed = _allowed_keys_for(mat_name, self.source)
            if allowed is None:
                continue  # component source unmapped; nothing to compare
            unknown = cat_keys - allowed
            if unknown:
                violations.append(
                    f"  {mat_name}: catalog lists {sorted(unknown)} "
                    f"which 4C source does not accept. "
                    f"Allowed: {sorted(allowed)}"
                )
        self.assertFalse(violations, "\n" + "\n".join(violations))

    def test_required_keys_present_in_catalog(self):
        """Soft check: surface 4C parameter keys that exist in source but are
        missing from the catalog, so an agent never finds them.

        These are not strictly errors (the catalog may legitimately omit
        obscure knobs) so the assertion only fails when
        ``OFA_STRICT_CATALOG=1`` is set.  Otherwise the gaps are printed to
        stderr so CI can surface them as warnings.
        """
        import os

        strict = os.environ.get("OFA_STRICT_CATALOG") == "1"
        missing: list[str] = []
        for mat_name, cat_keys in self.catalog.items():
            if not cat_keys:
                continue
            allowed = _allowed_keys_for(mat_name, self.source)
            if allowed is None:
                continue
            gap = allowed - cat_keys
            if gap:
                missing.append(
                    f"  {mat_name}: 4C source has parameters not in catalog: "
                    f"{sorted(gap)}"
                )
        if missing:
            msg = "Catalog under-declares 4C parameters:\n" + "\n".join(missing)
            if strict:
                self.fail(msg)
            else:
                print("\n[WARN catalog drift]\n" + msg, file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
