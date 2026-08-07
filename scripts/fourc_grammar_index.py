#!/usr/bin/env python3
"""Searchable index of the 4C input grammar, built from `4C -p`.

Top-level keys of the dump:
  metadata, legacy_element_specs, legacy_particle_specs, cell_types,
  sections, legacy_string_sections

`sections` is an all_of of 478 named specs. A section NAME may itself contain
slashes (`IO/RUNTIME VTK OUTPUT/STRUCTURE`) — that is part of the literal
section header, not a nesting operator, so section names are never split.

Inside a section, keys nest through group / list / one_of / selection specs.
This module flattens those to `A/B/C` paths RELATIVE to the section.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DUMP = HERE / "fourc_params.yaml"
CACHE = HERE / "fourc_params.index.pkl"


class Grammar:
    def __init__(self, doc):
        self.doc = doc
        self.refs = {}
        self._collect_refs(doc)
        self.sections: dict[str, dict] = {}   # section -> {relpath: spec}
        self.section_spec: dict[str, dict] = {}
        self._build()

    # -- ref table --------------------------------------------------------
    def _collect_refs(self, node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.isdigit() and isinstance(v, dict) \
                        and ("type" in v or "name" in v):
                    self.refs[k] = v
                self._collect_refs(v)
        elif isinstance(node, list):
            for it in node:
                self._collect_refs(it)

    def _resolve(self, spec):
        seen = 0
        while isinstance(spec, dict) and "$ref" in spec and len(spec) == 1 and seen < 8:
            spec = self.refs.get(str(spec["$ref"]), {})
            seen += 1
        return spec

    # -- flatten ----------------------------------------------------------
    def _walk(self, spec, out, prefix, depth=0):
        if depth > 40:
            return
        spec = self._resolve(spec)
        if not isinstance(spec, dict):
            return
        t = spec.get("type")
        name = spec.get("name")
        if name:
            path = f"{prefix}/{name}" if prefix else name
        else:
            path = prefix
        if t in ("all_of", "one_of", "any_of"):
            for s in spec.get("specs", []) or []:
                self._walk(s, out, prefix, depth + 1)
            return
        if t == "group":
            out[path] = spec
            for s in spec.get("specs", []) or []:
                self._walk(s, out, path, depth + 1)
            return
        if t == "list":
            out[path] = spec
            inner = spec.get("spec")
            if isinstance(inner, dict):
                self._walk(inner, out, path, depth + 1)
            return
        if t == "selection":
            out[path] = spec
            for ch in spec.get("choices", []) or []:
                if isinstance(ch, dict) and isinstance(ch.get("spec"), dict):
                    self._walk(ch["spec"], out, path, depth + 1)
            return
        if name:
            out[path] = spec
            return
        for s in spec.get("specs", []) or []:
            self._walk(s, out, prefix, depth + 1)

    def _build(self):
        for s in self.doc["sections"]["specs"]:
            nm = s["name"]
            self.section_spec[nm] = s
            flat: dict = {}
            # flatten the section's CONTENT with an empty prefix
            body = dict(s)
            body.pop("name", None)
            self._walk(body, flat, "")
            flat.pop("", None)
            self.sections[nm] = flat

    # -- queries ----------------------------------------------------------
    def section_names(self):
        return sorted(self.sections)

    def has_section(self, name):
        return name in self.sections

    def paths_of(self, section):
        return sorted(self.sections.get(section, {}))

    def keys_of(self, section):
        return sorted({p.split("/")[-1] for p in self.sections.get(section, {})})

    def spec(self, section, relpath):
        return self.sections.get(section, {}).get(relpath)

    def all_keys(self):
        out = set()
        for sec, flat in self.sections.items():
            for p in flat:
                out.add(p.split("/")[-1])
        return out

    def groups_named(self, pattern):
        """Every group path whose last component matches (materials,
        conditions, boundary-condition sub-groups...)."""
        res = []
        for sec, flat in self.sections.items():
            for p, sp in flat.items():
                if sp.get("type") == "group" and pattern.upper() in p.split("/")[-1].upper():
                    res.append((sec, p))
        return sorted(res)

    def material_names(self):
        flat = self.sections.get("MATERIALS", {})
        return sorted({p for p, sp in flat.items()
                       if sp.get("type") == "group" and "/" not in p})

    def cell_types(self):
        return sorted(self.doc.get("cell_types", {}))

    def legacy_elements(self):
        return sorted(self.doc.get("legacy_element_specs", {}))

    def legacy_string_sections(self):
        return list(self.doc.get("legacy_string_sections", []) or [])

    def element_cells(self, elem):
        return [c.get("cell_type") for c in
                self.doc.get("legacy_element_specs", {}).get(elem, [])]

    def element_spec(self, elem, cell):
        for c in self.doc.get("legacy_element_specs", {}).get(elem, []):
            if c.get("cell_type") == cell:
                return c.get("spec")
        return None

    def enum_choices(self, section, relpath):
        sp = self.spec(section, relpath)
        if not sp:
            return None
        ch = sp.get("choices")
        if ch is None:
            return None
        return [c.get("name") if isinstance(c, dict) else c for c in ch]


_G = None


def get():
    global _G
    if _G is not None:
        return _G
    if CACHE.exists() and CACHE.stat().st_mtime > DUMP.stat().st_mtime:
        with open(CACHE, "rb") as f:
            _G = pickle.load(f)
        return _G
    with open(DUMP) as f:
        doc = yaml.safe_load(f)
    _G = Grammar(doc)
    with open(CACHE, "wb") as f:
        pickle.dump(_G, f)
    return _G


def _fmt(p, sp):
    bits = [f"{p}  [{sp.get('type')}]"]
    if sp.get("required"):
        bits.append("REQ")
    if "default" in sp:
        bits.append("def=" + json.dumps(sp.get("default"), default=str))
    if sp.get("type") == "enum":
        bits.append("{" + ",".join(
            c.get("name") if isinstance(c, dict) else str(c)
            for c in (sp.get("choices") or [])) + "}")
    if sp.get("type") == "vector":
        bits.append(f"size={sp.get('size')}")
    return " ".join(bits)


def main():
    g = get()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    if cmd == "stats":
        paths = sum(len(v) for v in g.sections.values())
        leaves = sum(1 for v in g.sections.values() for p, sp in v.items()
                     if sp.get("type") not in ("group", "list", "selection"))
        keys = len(g.all_keys())
        groups = sum(1 for v in g.sections.values() for p, sp in v.items()
                     if sp.get("type") == "group")
        print(f"sections           {len(g.sections)}")
        print(f"paths              {paths}")
        print(f"leaf specs         {leaves}")
        print(f"distinct key names {keys}")
        print(f"groups             {groups}")
        print(f"materials          {len(g.material_names())}")
        print(f"cell types         {len(g.cell_types())}")
        print(f"legacy elements    {len(g.legacy_elements())}")
        print(f"legacy str sect.   {len(g.legacy_string_sections())}")
        print(f"version            {g.doc['metadata']}")
    elif cmd == "sections":
        for s in g.section_names():
            if arg.upper() in s.upper():
                print(s)
    elif cmd == "keys":
        if not g.has_section(arg):
            print(f"NO SUCH SECTION: {arg!r}")
            near = [s for s in g.section_names() if arg.upper()[:12] in s.upper()]
            for n in near[:20]:
                print("  near:", n)
            return 1
        for p in g.paths_of(arg):
            print(_fmt(p, g.sections[arg][p]))
    elif cmd == "find":
        n = arg.upper()
        for sec in g.section_names():
            for p, sp in sorted(g.sections[sec].items()):
                if n in p.upper():
                    print(f"{sec} :: {_fmt(p, sp)}")
    elif cmd == "mat":
        for m in g.material_names():
            if arg.upper() in m.upper():
                print(m)
    elif cmd == "matkeys":
        for p in sorted(g.sections["MATERIALS"]):
            if p == arg or p.startswith(arg + "/"):
                print(_fmt(p, g.sections["MATERIALS"][p]))
    elif cmd == "elements":
        for e in g.legacy_elements():
            if arg.upper() in e.upper():
                print(e, g.element_cells(e))
    elif cmd == "elemspec":
        e, c = arg.split(":")
        print(json.dumps(g.element_spec(e, c), indent=1, default=str))
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
