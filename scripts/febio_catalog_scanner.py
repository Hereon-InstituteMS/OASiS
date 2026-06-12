"""Layer A — FEBio catalog self-consistency scanner.

FEBio is an XML-input solver and not installed in this
sandbox (no conda/pip/apt package, gated download). So
'runtime introspection of the library' is not available
the way it is for kratos/fenics/skfem/ngsolve. The most
valuable check is INTERNAL CATALOG SELF-CONSISTENCY:

  * Every physics declared in supported_physics() must
    have a corresponding _TEMPLATES['<physics>_<variant>']
    entry — otherwise generate_input() raises ValueError
    unconditionally for that physics × variant.
  * Every physics declared in supported_physics() must
    have a _FEBIO_KNOWLEDGE['<physics>'] entry — otherwise
    get_knowledge() returns an empty dict.
  * Every emitted XML template must parse as
    well-formed XML and use <febio_spec version="4.0">
    as the root.
  * Every emitted template must contain a <Module type="X">
    where X matches the physics-specific Module type the
    catalog claims.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_febio_backend():
    """Import the febio backend module by file path so we
    don't need the rest of the MCP package to import.
    Stub out core.backend + core.registry."""
    import sys
    import types
    # Stub core.backend
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
    # Stub core.registry
    cr = types.ModuleType("core.registry")
    cr.register_backend = lambda *a, **k: None
    sys.modules["core.registry"] = cr
    sys.modules["core"].registry = cr
    # Load febio backend
    spec = importlib.util.spec_from_file_location(
        "febio_backend",
        Path(__file__).resolve().parent.parent / "src"
        / "backends" / "febio" / "backend.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["febio_backend"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_febio_backend()
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

    # Verify each shipped template is well-formed XML AND
    # uses febio_spec v4.0 + a Module type matching the
    # physics name (or a known synonym).
    physics_to_module = {
        "linear_elasticity": "solid",
        "hyperelasticity": "solid",
        "biphasic": "biphasic",
        "heat": "heat",
    }
    template_xml_errors = []
    template_module_mismatches = []
    for key, gen in templates.items():
        physics = key.rsplit("_", 1)[0] if "_" in key else key
        # Some variants are like 'linear_elasticity_3d_cube';
        # physics is everything before the last '_<variant>'.
        # Fall back: walk known physics names.
        for p in physics_to_module:
            if key.startswith(p + "_"):
                physics = p
                break
        try:
            xml_str = gen({})
        except Exception as e:
            template_xml_errors.append(
                f"{key}: gen({{}}) raised {type(e).__name__}: {e!s}")
            continue
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            template_xml_errors.append(
                f"{key}: XML parse error {e!s}")
            continue
        if root.tag != "febio_spec":
            template_xml_errors.append(
                f"{key}: root is <{root.tag}>, "
                f"expected <febio_spec>")
            continue
        if root.attrib.get("version") != "4.0":
            template_xml_errors.append(
                f"{key}: version={root.attrib.get('version')}, "
                f"expected 4.0")
        # Module type matches physics
        mod_el = root.find("Module")
        if mod_el is None:
            template_module_mismatches.append(
                f"{key}: no <Module> element")
            continue
        mod_type = mod_el.attrib.get("type")
        expected_module = physics_to_module.get(physics)
        if expected_module and mod_type != expected_module:
            template_module_mismatches.append(
                f"{key}: Module type='{mod_type}', "
                f"expected '{expected_module}'")
    print(f"template_xml_errors={template_xml_errors}")
    print(f"template_module_mismatches="
          f"{template_module_mismatches}")

    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "febio_catalog_scan.json"
    out.write_text(json.dumps({
        "physics_count": len(physics_list),
        "templates_count": len(templates),
        "knowledge_count": len(knowledge),
        "missing_templates": missing_templates,
        "missing_knowledge": missing_knowledge,
        "template_xml_errors": template_xml_errors,
        "template_module_mismatches":
            template_module_mismatches,
    }, indent=2))
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
