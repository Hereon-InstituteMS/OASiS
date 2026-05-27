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

from tests.groundtruth import dealii as dealii_gt  # noqa: E402
from tests.groundtruth import fourc as fourc_gt  # noqa: E402
from tests.groundtruth import skfem as skfem_gt  # noqa: E402


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
                continue  # catalog entry declares no parameters at all
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
            allowed = _allowed_keys_for(mat_name, self.source)
            if allowed is None:
                continue  # no source mapping for this entry; cannot compare
            # Note: we intentionally do NOT skip entries with empty
            # cat_keys here -- a catalog entry that promises a material
            # but lists no parameters is itself an under-declaration the
            # agent needs to know about.
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


# ── scikit-fem ───────────────────────────────────────────────────────────────


_SKFEM_ELEMENT_RE = re.compile(r"\bElement[A-Z]\w*\b")
_SKFEM_MESH_RE = re.compile(r"\bMesh[A-Z]\w*\b")


def _collect_skfem_class_mentions() -> dict[str, set[str]]:
    """Walk every string value reachable under
    ``backends.skfem.generators.KNOWLEDGE`` and pull out all ``Element*``
    and ``Mesh*`` identifiers referenced by name.  These are the names the
    agent sees when it asks the MCP for guidance, so each one must be a
    real class on the installed ``skfem`` module.

    Returns ``{"Element": set, "Mesh": set}``.
    """
    from backends.skfem.generators import KNOWLEDGE

    def walk(obj, into: set[str], regex: re.Pattern[str]) -> None:
        if isinstance(obj, str):
            into.update(regex.findall(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, into, regex)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v, into, regex)

    elements: set[str] = set()
    meshes: set[str] = set()
    walk(KNOWLEDGE, elements, _SKFEM_ELEMENT_RE)
    walk(KNOWLEDGE, meshes, _SKFEM_MESH_RE)
    return {"Element": elements, "Mesh": meshes}


class TestSkfemElementMeshClasses(unittest.TestCase):
    """Every ``Element*`` / ``Mesh*`` name the scikit-fem catalog mentions
    must be an actual class on the installed ``skfem`` package.

    This closes the same catalog-vs-source loop that
    ``TestFourcMaterialParameters`` does for 4C, but via Python
    introspection rather than C++ source-grep -- demonstrating the second
    probe family described in ``tests/groundtruth/__init__.py``.
    """

    @classmethod
    def setUpClass(cls):
        cls.source_elements = skfem_gt.element_classes()
        cls.source_meshes = skfem_gt.mesh_classes()
        if cls.source_elements is None or cls.source_meshes is None:
            raise unittest.SkipTest(
                "scikit-fem not installed (pip install scikit-fem)"
            )
        cls.mentions = _collect_skfem_class_mentions()

    def test_no_unknown_element_class_in_knowledge(self):
        promised = self.mentions["Element"]
        unknown = promised - self.source_elements
        self.assertFalse(
            unknown,
            f"\nscikit-fem catalog references Element* classes that do "
            f"not exist on `skfem`: {sorted(unknown)}\n"
            f"Typo or recent rename.  Available: "
            f"{sorted(self.source_elements)[:10]}... "
            f"({len(self.source_elements)} total)",
        )

    def test_no_unknown_mesh_class_in_knowledge(self):
        promised = self.mentions["Mesh"]
        unknown = promised - self.source_meshes
        self.assertFalse(
            unknown,
            f"\nscikit-fem catalog references Mesh* classes that do "
            f"not exist on `skfem`: {sorted(unknown)}\n"
            f"Available: {sorted(self.source_meshes)}",
        )


# ── deal.II ──────────────────────────────────────────────────────────────────


_DEALII_FE_RE = re.compile(r"\bFE_[A-Z][A-Za-z0-9_]*\b")


def _collect_dealii_fe_mentions() -> set[str]:
    """Return every ``FE_*`` class name referenced anywhere in the
    deal.II generator package.

    Scans the raw Python source of each generator module rather than
    walking the runtime knowledge dicts -- the catalog embeds template
    code as multi-line string literals in those modules and the agent
    will execute the templates, so every ``FE_*`` mention in those
    files (KNOWLEDGE prose AND template code) must resolve to a real
    deal.II class.  Reading the .py source covers both at once.
    """
    out: set[str] = set()
    generators_dir = Path(__file__).parent.parent / "src" / "backends" / "dealii" / "generators"
    if generators_dir.is_dir():
        for py in generators_dir.glob("*.py"):
            out.update(
                _DEALII_FE_RE.findall(
                    py.read_text(encoding="utf-8", errors="replace")
                )
            )
    return out


class TestDealiiFiniteElementClasses(unittest.TestCase):
    """Every ``FE_*`` name the deal.II catalog mentions (in either
    knowledge prose or generated Python template code) must be a real
    concrete class under ``include/deal.II/fe/``."""

    @classmethod
    def setUpClass(cls):
        cls.source = dealii_gt.fe_class_names()
        if cls.source is None:
            raise unittest.SkipTest(
                "deal.II source unavailable "
                "(set DEALII_ROOT or enable network access)"
            )
        cls.mentions = _collect_dealii_fe_mentions()

    def test_no_unknown_fe_class_in_catalog(self):
        unknown = self.mentions - self.source
        # Tolerate a small denylist of "marker" or "abstract" base
        # references that show up in prose but never get instantiated
        # in template code.
        unknown -= {"FE_Base", "FE_Data"}
        self.assertFalse(
            unknown,
            f"\ndeal.II catalog references FE_* classes that do not "
            f"exist in deal.II source: {sorted(unknown)}\n"
            f"Source has {len(self.source)} concrete FE_* classes. "
            f"First 15: {sorted(self.source)[:15]}",
        )

    def test_under_declared_fe_classes(self):
        """Soft check: ``FE_*`` classes that exist in deal.II source but
        are never mentioned by the catalog.  Each unmentioned class is
        a feature the agent will never propose -- not strictly wrong,
        but a gap worth surfacing.

        Mirrors the 4C ``test_required_keys_present_in_catalog`` --
        prints to stderr by default, fails the test only when
        ``OFA_STRICT_CATALOG=1`` is set so the under-declaration list
        does not block PRs.
        """
        import os

        strict = os.environ.get("OFA_STRICT_CATALOG") == "1"
        gap = self.source - self.mentions
        if gap:
            msg = (
                f"deal.II catalog under-declares FE_* classes "
                f"({len(gap)} unmentioned of {len(self.source)} in source): "
                f"{sorted(gap)}"
            )
            if strict:
                self.fail(msg)
            print(f"\n[WARN catalog drift]\n{msg}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
