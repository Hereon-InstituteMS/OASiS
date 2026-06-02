"""Tier-2: FEBio catalog self-consistency.

FEBio is not installed in this sandbox (no conda/pip/apt
package; gated download from febio.org). So we cannot
runtime-verify the FEBio binary's input parser. But the
catalog has internal self-consistency requirements that
can be checked statically:

  (1) Every physics declared in supported_physics() has
      a matching _TEMPLATES[<physics>_<variant>] entry —
      otherwise generate_input() raises ValueError.

  (2) Every physics declared in supported_physics() has
      a _FEBIO_KNOWLEDGE[<physics>] entry — otherwise
      get_knowledge() returns {}.

  (3) Every shipped template emits well-formed XML with
      <febio_spec version="4.0"> root.

  (4) Every shipped template uses <Module type="X"> with
      X matching the physics-required module type
      (solid for elasticity/hyperelasticity, biphasic
      for biphasic, heat for heat).

Catalog falsifications #29 + #30 found via this scanner:
  * supported_physics declared 4 physics; _TEMPLATES had
    1 → 3 ValueError-paths exposed.
  * 2 of 4 declared physics had no knowledge entries.

Fix landed alongside this fixture (3 template stubs +
2 knowledge entries).
"""
from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PHYSICS_TO_MODULE = {
    "linear_elasticity": "solid",
    "hyperelasticity": "solid",
    "biphasic": "biphasic",
    "heat": "heat",
    # ── Audit pass 4 (2026-06-02): catch-up after FEBio
    #    refactor passes 181 + 182 added 12 new physics
    #    (active_contraction, damage, fiber_reinforced,
    #    fluid, fluid_fsi, biphasic_fsi, growth_remodeling,
    #    multiphasic, plasticity, polar_fluid, rigid_body,
    #    viscoelasticity). Mapping verified against
    #    `<Module type="X"/>` in each generators/*.py.
    "active_contraction": "solid",
    "damage": "solid",
    "fiber_reinforced": "solid",
    "fluid": "fluid",
    "fluid_fsi": "fluid-FSI",
    "biphasic_fsi": "biphasic-FSI",
    "growth_remodeling": "solid",
    "multiphasic": "multiphasic",
    "plasticity": "solid",
    "polar_fluid": "polar fluid",
    "rigid_body": "solid",
    "viscoelasticity": "solid",
}


def _load_febio_backend():
    import types
    cb = types.ModuleType("core.backend")

    class _PhysicsCapability:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _BackendStatus:
        AVAILABLE = "AVAILABLE"
        NOT_INSTALLED = "NOT_INSTALLED"

    class _InputFormat:
        XML = "XML"

    class _SolverBackend:
        pass

    class _JobHandle:
        pass

    cb.PhysicsCapability = _PhysicsCapability
    cb.BackendStatus = _BackendStatus
    cb.InputFormat = _InputFormat
    cb.SolverBackend = _SolverBackend
    cb.JobHandle = _JobHandle
    sys.modules["core.backend"] = cb
    sys.modules["core"] = types.ModuleType("core")
    sys.modules["core"].backend = cb
    cr = types.ModuleType("core.registry")
    cr.register_backend = lambda *a, **k: None
    sys.modules["core.registry"] = cr
    sys.modules["core"].registry = cr

    # Load febio backend as a real PACKAGE so the relative
    # import `from .generators import ...` inside backend.py
    # resolves. spec_from_file_location with a bare module
    # name does NOT set up __package__ correctly for
    # relative imports; loading via submodule_search_locations
    # makes febio_backend a package whose __path__ contains
    # the febio backend dir. Audit pass 4 fix (2026-06-02).
    febio_dir = (Path(__file__).resolve().parents[4]
                 / "src" / "backends" / "febio")
    backend_path = febio_dir / "backend.py"
    spec = importlib.util.spec_from_file_location(
        "febio_backend.backend", backend_path,
        submodule_search_locations=[str(febio_dir)],
    )
    # Pre-register the package so the relative import sees it.
    pkg = types.ModuleType("febio_backend")
    pkg.__path__ = [str(febio_dir)]
    sys.modules["febio_backend"] = pkg
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "febio_backend"
    sys.modules["febio_backend.backend"] = mod
    # The relative `from .generators import ...` needs the
    # generators subpackage to be importable via the parent.
    gen_path = febio_dir / "generators"
    gen_spec = importlib.util.spec_from_file_location(
        "febio_backend.generators",
        gen_path / "__init__.py",
        submodule_search_locations=[str(gen_path)],
    )
    gen_mod = importlib.util.module_from_spec(gen_spec)
    gen_mod.__package__ = "febio_backend.generators"
    sys.modules["febio_backend.generators"] = gen_mod
    gen_spec.loader.exec_module(gen_mod)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = _load_febio_backend()
    backend = mod.FebioBackend()
    physics_list = backend.supported_physics()
    templates = mod._TEMPLATES
    knowledge = mod._FEBIO_KNOWLEDGE

    missing_templates = []
    missing_knowledge = []
    for cap in physics_list:
        if cap.name not in knowledge:
            missing_knowledge.append(cap.name)
        for variant in cap.template_variants:
            key = f"{cap.name}_{variant}"
            if key not in templates:
                missing_templates.append(key)
    print(f"physics_count={len(physics_list)}")
    print(f"templates_count={len(templates)}")
    print(f"knowledge_count={len(knowledge)}")
    print(f"missing_templates={missing_templates}")
    print(f"missing_knowledge={missing_knowledge}")

    xml_errors = []
    module_mismatches = []
    # Iterate physics keys in length-descending order so
    # 'fluid_fsi' and 'biphasic_fsi' match before their
    # 'fluid' / 'biphasic' prefixes (audit pass 4 fix).
    _physics_keys_by_len = sorted(
        PHYSICS_TO_MODULE.keys(), key=len, reverse=True)
    for key, gen in templates.items():
        physics_name = None
        for p in _physics_keys_by_len:
            if key.startswith(p + "_"):
                physics_name = p
                break
        if physics_name is None:
            xml_errors.append(
                f"{key}: cannot map template key to known "
                f"physics")
            continue
        try:
            xml_str = gen({})
        except Exception as e:
            xml_errors.append(
                f"{key}: gen({{}}) raised "
                f"{type(e).__name__}")
            continue
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            xml_errors.append(
                f"{key}: XML parse error {e!s}")
            continue
        if root.tag != "febio_spec":
            xml_errors.append(
                f"{key}: root is <{root.tag}>")
            continue
        if root.attrib.get("version") != "4.0":
            xml_errors.append(
                f"{key}: version="
                f"{root.attrib.get('version')}")
            continue
        mod_el = root.find("Module")
        if mod_el is None:
            module_mismatches.append(f"{key}: no Module")
            continue
        expected = PHYSICS_TO_MODULE[physics_name]
        actual = mod_el.attrib.get("type")
        if actual != expected:
            module_mismatches.append(
                f"{key}: Module type='{actual}' "
                f"expected='{expected}'")
    print(f"template_xml_errors={xml_errors}")
    print(f"template_module_mismatches={module_mismatches}")

    ok = (not missing_templates
          and not missing_knowledge
          and not xml_errors
          and not module_mismatches)
    if ok:
        return 0
    print("FAIL: febio catalog self-consistency regression",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
