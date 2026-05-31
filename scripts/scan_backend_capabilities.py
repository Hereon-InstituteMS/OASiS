#!/usr/bin/env python3
"""
Layer A — per-backend source-capability scanner.

For each backend, walk the installed Python package (and where
applicable the on-disk source tree) to enumerate every capability
the backend ACTUALLY exposes: Applications, Elements, Conditions,
Constitutive laws, Variables, Mesh generators, Element families,
... — whatever the backend's introspection API surfaces.

Emit a JSON snapshot per backend under `scripts/scan_results/`,
plus a top-level summary.  The snapshots are the input for the
catalog-vs-scan consistency test (tests/test_catalog_vs_scan.py,
landing in a follow-up PR) which surfaces gaps in both directions:

  * **drift**: in catalog but not in source — the MCP advertises a
    capability the backend does not actually ship.
  * **coverage gap**: in source but not in catalog — the MCP misses
    a capability the backend does ship.

Usage:
    python scripts/scan_backend_capabilities.py [--backend NAME]

This is the first foundation of the multi-week "scan every backend's
physics / modules / capabilities and encode the gaps into the MCP"
pipeline.  As of this PR the Kratos, 4C, scikit-fem and FEniCSx
scanners are wired in; NGSolve / deal.II / DUNE-fem follow in later
PRs.

Important: run via `./.venv/bin/python` (or `source .venv/bin/activate`
first) so the in-process scanners can import `KratosMultiphysics` /
`skfem`.  The FEniCSx scanner internally dispatches to its own
conda env's python via `FENICS_PYTHON`, so it works regardless of
the outer interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_RESULTS = REPO_ROOT / "scripts" / "scan_results"


def _redact_path(p: Path | str) -> str:
    """Replace the user's home directory with `~` and strip the repo
    root so committed snapshots do not embed machine-specific
    absolute paths (which leak usernames into the repo and produce
    noisy diffs across developers / CI).
    """
    s = str(p)
    home = str(Path.home())
    if s.startswith(home):
        s = "~" + s[len(home):]
    repo = str(REPO_ROOT)
    repo_home = "~" + repo[len(home):] if repo.startswith(home) else repo
    if s.startswith(repo_home):
        s = "<repo>" + s[len(repo_home):]
    return s


# ── data model ─────────────────────────────────────────────────────────


@dataclass
class BackendCapabilities:
    """What a backend exposes at the source level.

    Each field is best-effort populated by the per-backend scanner —
    when a backend does not expose a category through its Python
    introspection surface (e.g. Kratos elements need a C++ registry
    walk we cannot do from pure Python), the field stays empty and
    the consistency test treats that as "no information" rather than
    "no capability".
    """

    backend: str
    version: str = ""
    applications: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    constitutive_laws: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    mesh_generators: list[str] = field(default_factory=list)
    element_families: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    modelers: list[str] = field(default_factory=list)
    other: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ── Kratos scanner ─────────────────────────────────────────────────────


def scan_kratos() -> BackendCapabilities:
    """Enumerate what KratosMultiphysics exposes through Python.

    Kratos splits its capability surface across many places:

      * `import KratosMultiphysics.<X>Application` — each application
        module is itself a pybind11 binding registering Elements /
        Conditions / ConstitutiveLaws into the global C++ Registry.
        We discover installed applications by walking the package
        directory and attempting an import; success means the
        application is wired into the runtime.

      * `KM.KratosGlobals.Kernel.GetAllVariableNames()` — every
        registered nodal/element variable, across all currently-
        imported applications.

      * `KM.Registry` — a python-facing tree of registered Processes,
        Modelers, Stages, OutputProcesses.  Walked via
        `Registry.HasItem(path)` + `Registry.NumberOfItems(path)`.

      * Constitutive laws — `KM.KratosGlobals.HasConstitutiveLaw(name)`
        only answers point queries, never enumerates.  As a
        best-effort proxy we filter `dir(<application_module>)` for
        names ending in `Law` per imported application, and emit the
        result as `App::LawName` strings.  A future improvement
        would correlate the proxy list against the C++ Registry once
        Kratos exposes it through Python.

    Element / Condition class names are similarly approximated by
    filtering `dir(<application_module>)` for `*Element` / `*Condition`
    suffixes — the C++-registered class list is not exposed through
    Python, so this dir-walk is the closest stand-in available.
    """
    import re

    cap = BackendCapabilities(backend="kratos")

    try:
        import KratosMultiphysics as KM
    except ImportError as e:
        cap.notes.append(f"KratosMultiphysics not importable: {e}")
        return cap

    # Kratos's Kernel.Version() returns a string like
    # `10.4."2"--0-Release-x86_64` with embedded literal quote characters
    # around the patch component.  Strip those so downstream consumers
    # do JSON serialisation / semver parsing on a clean version string.
    cap.version = str(KM.KratosGlobals.Kernel.Version()).replace('"', "")

    # ── 1. installed applications.  Two layouts are supported:
    #   (a) `KratosMultiphysics/<X>Application/__init__.py` — pip
    #       wheels typically ship apps as proper sub-packages.
    #   (b) `KratosMultiphysics/<X>Application*.so` —  some
    #       installations ship apps as top-level extension modules
    #       without a Python wrapper.
    # Try both; either match counts as a candidate.  Validate by
    # actually importing — only successful imports go into
    # `cap.applications`; failures are recorded separately.
    kpkg = Path(KM.__file__).parent
    candidate_apps: set[str] = set()
    for p in kpkg.iterdir():
        if p.is_dir() and p.name.endswith("Application") and (p / "__init__.py").exists():
            candidate_apps.add(p.name)
        # Match e.g. StructuralMechanicsApplication.cpython-312-x86_64-linux-gnu.so
        m = re.match(r"([A-Z][A-Za-z0-9]*Application)(?:\.[^.]+)*\.so$", p.name)
        if m:
            candidate_apps.add(m.group(1))
    # No separate sorted alias — iterate the set deterministically below.
    apps_seen: list[str] = []
    apps_failed: dict[str, str] = {}
    for app in sorted(candidate_apps):
        try:
            __import__(f"KratosMultiphysics.{app}")
            apps_seen.append(app)
        except Exception as e:
            apps_failed[app] = f"{type(e).__name__}: {e!s:.120}"
    cap.applications = apps_seen
    if apps_failed:
        cap.other["applications_failed_to_import"] = [
            f"{k}: {v}" for k, v in apps_failed.items()
        ]

    # ── 2. variables registered globally after all imported applications.
    # `GetAllVariableNames()` returns a single whitespace-indented string
    # with one variable name per line (not a list).  We split + strip
    # and drop empty / typedef-name lines.
    try:
        raw = KM.KratosGlobals.Kernel.GetAllVariableNames()
        if isinstance(raw, str):
            var_names = [ln.strip() for ln in raw.splitlines()]
        else:
            var_names = list(raw)
        var_names = [v for v in var_names if v]
        cap.variables = sorted(set(var_names))
    except Exception as e:
        cap.notes.append(f"GetAllVariableNames failed: {e!s:.120}")

    # ── 3. Registry tree — Processes / Modelers / Stages / OutputProcesses
    for cat_attr, cat_path in [
        ("processes", "Processes"),
        ("modelers", "Modelers"),
    ]:
        try:
            if KM.Registry.HasItem(cat_path):
                n = KM.Registry.NumberOfItems(cat_path)
                # Record the count in `notes` rather than injecting a
                # placeholder string into the typed list field.  The
                # downstream consistency test compares list contents
                # against the MCP catalog — appending a synthetic
                # `"<N items ...>"` entry would look like a real
                # capability name and would always show as drift.
                cap.notes.append(
                    f"Registry[{cat_path!r}] has {n} items "
                    f"(deep enumeration is a follow-up; the Python "
                    f"Registry binding does not expose child iteration)"
                )
        except Exception as e:
            cap.notes.append(f"Registry[{cat_path}] walk failed: {e!s:.80}")

    # ── 4. element / condition / constitutive-law names per application
    # Python introspection only: filter `dir(<app>)` for typical naming.
    elements: list[str] = []
    conditions: list[str] = []
    claws: list[str] = []
    for app in apps_seen:
        try:
            mod = sys.modules[f"KratosMultiphysics.{app}"]
        except KeyError:
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            if name.endswith("Element") and not name.endswith("BoundaryElement"):
                elements.append(f"{app}::{name}")
            elif name.endswith("Condition"):
                conditions.append(f"{app}::{name}")
            elif name.endswith("Law"):
                claws.append(f"{app}::{name}")
    cap.elements = sorted(elements)
    cap.conditions = sorted(conditions)
    cap.constitutive_laws = sorted(claws)

    # ── 5. for completeness — record the package install path
    cap.notes.append(f"package_dir={_redact_path(kpkg)}")

    return cap


# ── 4C scanner ─────────────────────────────────────────────────────────


def scan_fourc() -> BackendCapabilities:
    """Walk the 4C source tree to enumerate physics modules,
    registered materials, and input parser definitions.

    Unlike Kratos (pip-installed, introspect Python), 4C is a C++
    project we have on disk.  We grep the canonical registry files
    rather than parse C++ AST:

      * `src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp`
        contains every material declaration via
        `group("MAT_<name>", {...})` calls — one `MAT_*` per
        material the input parser will accept.

      * `src/core/legacy_enum_definitions/4C_legacy_enum_definitions_materials.cpp`
        carries the matching `Core::Materials::m_<name>` C++ enum
        symbols; the two must stay in sync (a mismatch is itself a
        4C-internal bug worth surfacing in the scan).

      * `src/<module>/` directories under `src/` correspond to
        physics modules.  We skip generic-infrastructure
        directories (`core`, `config`, `deal_ii`, ...) and report
        the rest as the "modules" capability under `other`.

      * `src/inpar/*.cpp` defines input keywords and
        `ConditionDefinition` entries.  We extract the section-name
        string literals each file registers.

    The 4C source root is taken from `FOURC_ROOT` env var or the
    repo-relative `~/Schreibtisch/4C-src/4C` location (matching
    `sweep_layer3.py` discovery).
    """
    import os
    import re

    cap = BackendCapabilities(backend="fourc")

    candidates = [
        os.environ.get("FOURC_ROOT", ""),
        str(Path.home() / "Schreibtisch/4C-src/4C"),
        str(Path.home() / "4C"),
    ]
    root: Path | None = None
    for c in candidates:
        if c and (Path(c) / "src").is_dir():
            root = Path(c)
            break
    if root is None:
        cap.notes.append("4C source root not found; tried FOURC_ROOT + "
                         "~/Schreibtisch/4C-src/4C + ~/4C")
        return cap
    cap.notes.append(f"source_root={_redact_path(root)}")

    # ── 1. modules: every src/<dir> that looks like a physics module
    src = root / "src"
    skip = {
        "core", "config", "module_registry", "global_data",
        "global_legacy_module", "deal_ii", "legacy",
    }
    modules: list[str] = []
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skip:
            continue
        if not (child / "CMakeLists.txt").exists():
            continue
        modules.append(child.name)
    cap.other["modules"] = modules

    # ── 2. materials: every MAT_<name> in the valid-materials registry
    mats_file = src / "global_legacy_module" / "4C_global_legacy_module_validmaterials.cpp"
    if mats_file.is_file():
        txt = mats_file.read_text(encoding="utf-8", errors="replace")
        materials = sorted(set(re.findall(r'group\("(MAT_[A-Za-z_0-9]+)"', txt)))
        cap.constitutive_laws = materials
        cap.notes.append(f"materials_source={mats_file.relative_to(root)}")
    else:
        cap.notes.append(f"materials_file_missing: {mats_file}")

    # ── 3. material enum symbols (the C++ side of the same table)
    enum_file = src / "core" / "legacy_enum_definitions" / "4C_legacy_enum_definitions_materials.cpp"
    if enum_file.is_file():
        txt = enum_file.read_text(encoding="utf-8", errors="replace")
        enum_syms = sorted(set(re.findall(r'\bm_([A-Za-z_0-9]+)\b', txt)))
        cap.other["material_enum_symbols"] = enum_syms

    # ── 4. ConditionDefinition string literals.  In 4C the type is
    # written as `Core::Conditions::ConditionDefinition`, and the call
    # site is one of:
    #     ConditionDefinition tbc_turb_inflow("DESIGN ...", ...)
    #     ConditionDefinition(\n   "DESIGN ...", ...)
    #     ... = ConditionDefinition("DESIGN ...", ...)
    # So the regex allows an optional variable-name identifier and
    # arbitrary whitespace (incl. newlines) between the keyword and
    # the string literal.  Walk the entire 4C src tree, not just
    # src/inpar/, because beaminteraction and others register their
    # own conditions outside inpar.
    cond_re = re.compile(
        r'ConditionDefinition\s*(?:[A-Za-z_]\w*\s*)?\(\s*"([^"]+)"',
        re.DOTALL,
    )
    sections: set[str] = set()
    condition_files: list[str] = []
    for p in sorted(src.rglob("*.cpp")):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "ConditionDefinition" not in txt:
            continue
        hits = cond_re.findall(txt)
        if hits:
            sections.update(hits)
            condition_files.append(str(p.relative_to(src)))
    cap.conditions = sorted(sections)
    cap.other["condition_source_files"] = condition_files

    # ── 5. element-registration directories under src/*_ele/
    elem_modules: list[str] = []
    for p in sorted(src.glob("*_ele")):
        if p.is_dir() and (p / "CMakeLists.txt").exists():
            elem_modules.append(p.name)
    cap.other["element_modules"] = elem_modules

    # 4C has no global variable registry to enumerate the way Kratos
    # does — variables in 4C are field names inside each physics
    # module's element implementation.  Per-module field extraction
    # is a deeper-scan follow-up.
    return cap


# ── scikit-fem scanner ─────────────────────────────────────────────────


def scan_skfem() -> BackendCapabilities:
    """scikit-fem ships as a pure-Python package — enumerate every
    Element* and Mesh* class on the top-level module, plus the
    bundled `skfem.models.*` form submodules and the canonical
    refinement / IO helpers under `skfem.io`.
    """
    cap = BackendCapabilities(backend="skfem")
    try:
        import skfem
    except ImportError as e:
        cap.notes.append(f"scikit-fem not importable: {e}")
        return cap

    cap.version = getattr(skfem, "__version__", "")

    # Element classes
    elements = sorted(
        n for n in dir(skfem)
        if n.startswith("Element") and n[7:] and n[7] != "_"
    )
    cap.elements = elements

    # Mesh classes
    meshes = sorted(
        n for n in dir(skfem)
        if n.startswith("Mesh") and n[4:] and n[4] != "_"
    )
    cap.mesh_generators = meshes

    # Bundled form modules
    try:
        import skfem.models as M
        model_modules = sorted(
            n for n in dir(M) if not n.startswith("_") and n in (
                "elasticity", "general", "helmholtz", "poisson"
            )
        )
        cap.element_families = model_modules
    except ImportError:
        pass

    cap.notes.append(f"package_dir={_redact_path(Path(skfem.__file__).parent)}")
    return cap


# ── FEniCSx (dolfinx) scanner ──────────────────────────────────────────


def scan_fenics() -> BackendCapabilities:
    """FEniCSx lives in its own conda env (matching the user's setup
    where the .venv runs the MCP but dolfinx is in ofa-fenicsx).
    We dispatch a small introspection script to the env's python
    via subprocess and parse the JSON it prints.

    Three artefacts captured:
      * `basix.ElementFamily` — every continuous, discontinuous,
        bubble, Nedelec, RT, BDM, etc. element family the
        installed basix provides.
      * `basix.CellType` — every cell topology dolfinx can mesh
        (point, interval, triangle, tetrahedron, ..., pyramid).
      * `dolfinx.mesh.create_*` — every built-in mesh-generator
        function name.

    The env's python is discovered through `FENICS_PYTHON` (matches
    `tools/developer.py` convention) with a fallback to the conda
    location used by `sweep_layer3.py`.
    """
    import json as _json
    import os
    import subprocess

    cap = BackendCapabilities(backend="fenics")

    fenics_py = (
        os.environ.get("FENICS_PYTHON", "")
        or str(Path.home() / "miniconda3/envs/ofa-fenicsx/bin/python")
    )
    if not Path(fenics_py).is_file():
        cap.notes.append(
            f"FEniCSx python not found at {fenics_py}; set FENICS_PYTHON"
        )
        return cap

    probe = """
import json, sys
try:
    import basix
    import dolfinx
    import dolfinx.mesh
    out = {
        'dolfinx_version': dolfinx.__version__,
        'basix_version': basix.__version__,
        'element_families': [f.name for f in basix.ElementFamily],
        'cell_types': [c.name for c in basix.CellType],
        'mesh_creators': sorted(
            n for n in dir(dolfinx.mesh) if n.startswith('create_')
        ),
    }
except Exception as e:
    out = {'error': f'{type(e).__name__}: {e}'}
sys.stdout.write(json.dumps(out))
"""
    r = subprocess.run(
        [fenics_py, "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        cap.notes.append(
            f"FEniCSx probe exited {r.returncode}: {r.stderr[:200]}"
        )
        return cap
    try:
        data = _json.loads(r.stdout)
    except _json.JSONDecodeError as e:
        cap.notes.append(f"FEniCSx probe output not JSON: {e}")
        return cap
    if "error" in data:
        cap.notes.append(f"FEniCSx probe raised: {data['error']}")
        return cap

    cap.version = data.get("dolfinx_version", "")
    cap.element_families = data.get("element_families", [])
    cap.mesh_generators = data.get("mesh_creators", [])
    cap.other["cell_types"] = data.get("cell_types", [])
    cap.other["basix_version"] = data.get("basix_version", "")
    return cap


# ── NGSolve scanner ────────────────────────────────────────────────────


def scan_ngsolve() -> BackendCapabilities:
    """Enumerate what NGSolve exposes through its Python surface.

    NGSolve is pure-Python (over a pybind11 C++ core) and lives in the
    same ``.venv`` as the MCP, so the scanner imports it directly —
    no subprocess dispatch needed.

    Capability buckets populated:
      * ``element_families`` — finite-element-space classes from
        ``ngsolve.comp`` (H1, HCurl, HDiv, L2, VectorH1, ... plus the
        many specialised facet / surface / div-div spaces). Filter
        rule: ends in ``FESpace`` or matches one of the well-known
        short names that NGSolve registered as aliases (H1, HCurl, ...).
      * ``elements`` — finite-element shape classes from ``ngsolve.fem``
        ending in ``FE`` (H1FE, L2FE, HCurlFE, HDivFE, ...). These
        are the element-level objects the spaces are built from.
      * ``mesh_generators`` — Netgen geometry / mesher entry points
        (``netgen.occ.OCCGeometry``, ``netgen.csg.CSGeometry``,
        ``netgen.geom2d.SplineGeometry``, ``Make2DProblem``, ...).
      * ``processes`` — solver classes (``CGSolver``, ``GMRESSolver``,
        ``QMRSolver``, ``ArnoldiSolver``) — labelled "processes" to
        match the per-backend field naming.
      * ``other['preconditioners']`` — preconditioner classes
        (``MultiGridPreconditioner``, ``H1AMG``, ``HCurlAMG``,
        ``BDDCPreconditioner``, ...).
      * ``other['forms']`` — bilinear/linear-form classes.
    """
    cap = BackendCapabilities(backend="ngsolve")
    try:
        import ngsolve
    except ImportError as e:
        cap.notes.append(f"ngsolve not importable: {e}")
        return cap

    cap.version = getattr(ngsolve, "__version__", "")

    # Finite-element-space families on ngsolve.comp + the short
    # aliases on the top-level module.
    space_aliases = {
        "H1", "HCurl", "HDiv", "L2", "VectorH1", "VectorL2",
        "SurfaceL2", "HCurlDiv", "HDivDiv", "FacetSurface",
        "NormalFacetSurface", "HDivSurface", "VectorSurfaceL2",
        "TangentialSurfaceL2",
    }
    families = set()
    for name in dir(ngsolve):
        if name in space_aliases or (
            name.endswith("FESpace") and not name.startswith("_")
        ):
            families.add(name)
    try:
        import ngsolve.comp as _comp
        for name in dir(_comp):
            if name.endswith("FESpace") and not name.startswith("_"):
                families.add(name)
    except ImportError:
        pass
    cap.element_families = sorted(families)

    # Element-level classes on ngsolve.fem (shape functions).
    try:
        import ngsolve.fem as _fem
        cap.elements = sorted(
            n for n in dir(_fem)
            if n.endswith("FE") and not n.startswith("_")
        )
    except ImportError:
        pass

    # Solvers (called "processes" in the BackendCapabilities schema —
    # NGSolve does not have a Kratos-style Process registry, so the
    # solvers are the closest analogue).
    solver_classes = {
        "CGSolver", "GMRESSolver", "QMRSolver", "ArnoldiSolver",
    }
    cap.processes = sorted(
        n for n in dir(ngsolve) if n in solver_classes
    )

    # Preconditioners — split across top-level and ngsolve.comp.
    precond_names = set()
    for name in dir(ngsolve):
        if (name.endswith("Preconditioner") or name.endswith("AMG")) \
                and not name.startswith("_"):
            precond_names.add(name)
    try:
        import ngsolve.comp as _comp
        for name in dir(_comp):
            if (name.endswith("Preconditioner") or name.endswith("AMG")) \
                    and not name.startswith("_"):
                precond_names.add(name)
    except ImportError:
        pass
    if precond_names:
        cap.other["preconditioners"] = sorted(precond_names)

    # Bilinear / linear form classes.
    form_classes = sorted(
        n for n in dir(ngsolve)
        if (n.endswith("BilinearForm") or n.endswith("LinearForm")
            or n.endswith("BFI") or n.endswith("LFI")
            or n in ("SymbolicEnergy",))
        and not n.startswith("_")
    )
    if form_classes:
        cap.other["forms"] = form_classes

    # Netgen geometry / mesher.
    netgen_kinds: list[str] = []
    for modname, attrs in (
        ("netgen.occ", ("OCCGeometry", "WorkPlane", "Box", "Cylinder",
                        "Sphere", "Pnt", "Glue", "OffsetCurve")),
        ("netgen.csg", ("CSGeometry", "OrthoBrick", "Cylinder", "Plane",
                        "Sphere")),
        ("netgen.geom2d", ("SplineGeometry", "MakeCircle",
                           "MakeRectangle", "unit_square")),
    ):
        try:
            _m = __import__(modname, fromlist=["__name__"])
            for a in attrs:
                if hasattr(_m, a):
                    netgen_kinds.append(f"{modname}.{a}")
        except ImportError:
            continue
    if netgen_kinds:
        cap.mesh_generators = sorted(netgen_kinds)

    # On-disk demos — captures the discoverable tutorial surface
    # without needing a separate harvest pass. The tutorial walk
    # (extracting one-line descriptions per demo file) lands in
    # the follow-up encode pass.
    try:
        pkg_dir = Path(ngsolve.__file__).resolve().parent
        demos_dir = pkg_dir / "demos"
        if demos_dir.is_dir():
            demos = sorted(p.name for p in demos_dir.iterdir() if p.is_dir())
            if demos:
                cap.other["demo_groups"] = demos
            cap.notes.append(
                f"demos_dir={_redact_path(demos_dir)}")
    except Exception as e:  # pragma: no cover — defensive
        cap.notes.append(f"demos walk failed: {type(e).__name__}: {e}")

    cap.notes.append(
        f"package_dir={_redact_path(Path(ngsolve.__file__).parent)}")
    return cap


# ── deal.II scanner ────────────────────────────────────────────────────


def scan_dealii() -> BackendCapabilities:
    """Enumerate what deal.II exposes — via header walk.

    deal.II is C++ with no maintained Python bindings, so the
    capability surface is harvested by walking the installed
    ``include/deal.II/`` directory:

      * ``include/deal.II/fe/fe_*.h`` — every finite-element class.
        Each header declares ``class FE_<Name>`` matching the
        filename suffix.
      * ``include/deal.II/grid/grid_generator.h`` — every mesh
        generator (functions in the ``GridGenerator`` namespace).
      * ``include/deal.II/lac/solver_*.h`` — every Krylov solver.
      * ``include/deal.II/lac/precondition*.h`` — every preconditioner
        class.

    Install root is discovered through ``DEAL_II_DIR`` / ``DEALII_ROOT``
    with a fallback to the conda env used by the user's setup
    (``~/miniconda3/envs/ofa-dealii``). The version is read from the
    CMake ``deal.IIConfigVersion.cmake`` file.
    """
    import os
    import re

    cap = BackendCapabilities(backend="dealii")

    root_candidates = [
        os.environ.get("DEAL_II_DIR", ""),
        os.environ.get("DEALII_ROOT", ""),
        str(Path.home() / "miniconda3/envs/ofa-dealii"),
    ]
    install_root = next(
        (Path(r) for r in root_candidates if r and Path(r).is_dir()),
        None,
    )
    if install_root is None:
        cap.notes.append(
            "deal.II install root not found; set DEAL_II_DIR or "
            "install into ~/miniconda3/envs/ofa-dealii")
        return cap

    inc = install_root / "include" / "deal.II"
    if not inc.is_dir():
        cap.notes.append(
            f"include/deal.II not under {_redact_path(install_root)}")
        return cap

    # Version
    ver_file = (install_root / "lib" / "cmake" / "deal.II"
                / "deal.IIConfigVersion.cmake")
    if ver_file.is_file():
        m = re.search(
            r'set\(PACKAGE_VERSION\s+"([0-9.]+)"\)',
            ver_file.read_text())
        if m:
            cap.version = m.group(1)

    # Element families: walk fe_*.h
    fe_dir = inc / "fe"
    if fe_dir.is_dir():
        skip_substrings = (
            "_values", "_tools", "_update", "_collection",
            "_dgvector", "_face_q",  # face elements handled as fe_face.h
        )
        elements: list[str] = []
        for header in sorted(fe_dir.glob("fe_*.h")):
            name = header.stem
            if name.endswith(".templates") or any(s in name for s in skip_substrings):
                continue
            # Read the header and pick the first `class FE_<Name>`
            # declaration so we get the actual class name (which
            # can differ from the filename — e.g. fe_q_dg0.h →
            # FE_Q_DG0).
            try:
                text = header.read_text(errors="replace")
            except OSError:
                continue
            # Match either underscore form (FE_Q, FE_Nedelec) or
            # the camel-case wrapper FESystem — both are real
            # deal.II classes. The trailing `(?=[\s{:])` requires the
            # match to be followed by whitespace + `{`, inheritance
            # `:`, or end-of-line continuation — this filters out
            # forward declarations like `class FE_Enriched;` which
            # ARE NOT real declarations and which (in fe_system.h)
            # used to win the search and hide the real FESystem
            # declaration on the next line. We use findall so
            # multiple declarations per header (FE_Enriched fwd-decl
            # + FESystem real-decl, both in fe_system.h) are picked
            # up correctly.
            matches = re.findall(
                r"^class\s+(FE_[A-Za-z0-9_]+|FE[A-Z][A-Za-z0-9_]*)"
                r"(?=[\s:])",
                text, re.MULTILINE,
            )
            if matches:
                elements.extend(matches)
            else:
                # Fallback: derive from filename when the header
                # has no parseable `class FE_*` declaration (e.g.
                # template-only headers). Note: `for-else` (the
                # previous form) was wrong here — it fires after
                # EVERY iteration, polluting the list with the
                # title-cased fallback even when matches existed,
                # and the loop variable `name` was shadowed by the
                # match so the fallback produced garbage names
                # like `FE_Ystem` (from "FESystem"[3:].title()).
                elements.append("FE_" + name[3:].title().replace("_", ""))
        # De-dup while preserving order.
        seen = set()
        cap.elements = [x for x in elements if not (x in seen or seen.add(x))]

    # Mesh generators: GridGenerator namespace in grid_generator.h
    gg = inc / "grid" / "grid_generator.h"
    if gg.is_file():
        try:
            text = gg.read_text(errors="replace")
        except OSError:
            text = ""
        # Each generator is a function declaration in the
        # GridGenerator namespace. Functions take a Triangulation
        # by reference as their first argument, so we anchor on
        # that to avoid catching every void function in the header.
        gen_names = sorted(set(re.findall(
            r"^\s+void\s+([a-z][A-Za-z0-9_]*)\s*\([^;]*Triangulation",
            text, re.MULTILINE,
        )))
        cap.mesh_generators = gen_names

    # Solvers + preconditioners
    lac = inc / "lac"
    if lac.is_dir():
        solver_names: list[str] = []
        for header in sorted(lac.glob("solver_*.h")):
            name = header.stem
            if name.endswith(".templates"):
                continue
            try:
                text = header.read_text(errors="replace")
            except OSError:
                continue
            m = re.search(r"^class\s+(Solver[A-Z][A-Za-z0-9_]*)",
                          text, re.MULTILINE)
            if m:
                solver_names.append(m.group(1))
        if solver_names:
            cap.other["solvers"] = sorted(set(solver_names))

        precond_names: set[str] = set()
        for header in sorted(lac.glob("precondition*.h")):
            try:
                text = header.read_text(errors="replace")
            except OSError:
                continue
            for m in re.finditer(
                r"^class\s+(Precondition[A-Z][A-Za-z0-9_]*)",
                text, re.MULTILINE,
            ):
                precond_names.add(m.group(1))
        if precond_names:
            cap.other["preconditioners"] = sorted(precond_names)

    # Constitutive laws are usually user-written in deal.II — there
    # is no central "linear-elastic law" registry to enumerate.
    # Note this so the catalog-vs-scan diff treats the empty list
    # as "no information", not "no capability".
    cap.notes.append(
        "constitutive_laws: deal.II ships no central material/law "
        "registry; user-written. Catalog diff should treat empty "
        "list as 'no information'.")
    cap.notes.append(
        "tutorials: 97 step-* tutorials live in the upstream source "
        "tree, not in this conda install. Tutorial harvest pending.")
    cap.notes.append(
        f"install_root={_redact_path(install_root)}")
    return cap


# ── DUNE-fem scanner ───────────────────────────────────────────────────


def scan_dune() -> BackendCapabilities:
    """Enumerate what DUNE-fem exposes — try Python import first,
    fall back to source walk.

    DUNE-fem is a JIT-compiled UFL-on-C++ stack with a heavy
    pybind11 surface. The MPI / UCX setup can break the import path
    at runtime (task #31 documents the FieldVector incompatibility);
    the scanner therefore tries the Python introspection path first,
    and falls back to scanning the on-disk source / build tree for
    the same files (``_spaces.py``, ``_grids.py``, ``_schemes.py``,
    ``_functions.py``) so we get a useful snapshot even when the
    runtime is broken.

    Capability buckets:
      * ``element_families`` — space factory functions from
        ``dune.fem.space`` (lagrange, dgonb, raviartThomas, ...).
      * ``mesh_generators`` — grid factories from ``dune.grid`` +
        ``dune.alugrid``.
      * ``processes`` — scheme factories from ``dune.fem.scheme``.
      * ``other['function_factories']`` — function constructors
        from ``dune.fem.function``.
      * ``other['models']`` — model factories from ``dune.fem.model``.
      * ``other['storage_backends']`` — discrete-function storage
        backends supported (numpy / fem / istl / petsc).
    """
    import re

    cap = BackendCapabilities(backend="dune")

    # ── attempt 1: Python introspection ────────────────────────
    introspection_ok = False
    try:
        import dune.fem  # noqa: F401
        import dune.fem.space as _space
        import dune.fem.scheme as _scheme
        import dune.fem.function as _func
        try:
            import dune.fem.model as _model
        except ImportError:
            _model = None
        try:
            import dune.grid as _grid
        except ImportError:
            _grid = None
        try:
            import dune.alugrid as _alu
        except ImportError:
            _alu = None
    except Exception as e:
        cap.notes.append(
            f"dune python import failed: {type(e).__name__}: "
            f"{str(e)[:200]}; falling back to source walk")
    else:
        introspection_ok = True
        cap.version = getattr(__import__("dune.fem", fromlist=["fem"]),
                              "__version__", "")
        # Space factories
        cap.element_families = sorted(
            n for n in dir(_space)
            if not n.startswith("_") and callable(getattr(_space, n, None))
        )
        # Scheme factories
        cap.processes = sorted(
            n for n in dir(_scheme)
            if not n.startswith("_") and callable(getattr(_scheme, n, None))
        )
        # Function factories
        func_factories = sorted(
            n for n in dir(_func)
            if not n.startswith("_") and callable(getattr(_func, n, None))
        )
        if func_factories:
            cap.other["function_factories"] = func_factories
        if _model is not None:
            cap.other["models"] = sorted(
                n for n in dir(_model)
                if not n.startswith("_") and callable(getattr(_model, n, None))
            )
        # Grid generators across dune.grid and dune.alugrid
        mesh_kinds: list[str] = []
        for prefix, mod in (("dune.grid", _grid), ("dune.alugrid", _alu)):
            if mod is None:
                continue
            for n in dir(mod):
                if n.startswith("_") or not callable(getattr(mod, n, None)):
                    continue
                # Heuristic: factory names end in "Grid"
                if "Grid" in n or n.endswith("grid"):
                    mesh_kinds.append(f"{prefix}.{n}")
        cap.mesh_generators = sorted(mesh_kinds)

    # ── attempt 2: source walk (always runs, augments introspection
    #              and is the only path when import fails) ─────
    src_roots = [
        Path.home() / "Schreibtisch/dune-src/dune-fem"
                       "/build-cmake/python/dune",
        Path.home() / "Schreibtisch/dune-src/dune-fem/python/dune",
    ]
    src_root = next((p for p in src_roots if p.is_dir()), None)
    if src_root is None:
        if not introspection_ok:
            cap.notes.append(
                "dune source tree not found at "
                "~/Schreibtisch/dune-src/dune-fem; scanner produced "
                "no data")
        return cap

    # Storage backends: parse _spaces.py for the `storage=` choices.
    spaces_file = src_root / "fem" / "space" / "_spaces.py"
    if spaces_file.is_file():
        try:
            text = spaces_file.read_text(errors="replace")
        except OSError:
            text = ""
        # Look for the storage validation set (it has been a single
        # source of truth across versions).
        m = re.search(
            r'storage[\w_]*\s*=\s*[({\[]([^)}\]]+)[)}\]]',
            text)
        backends: set[str] = set()
        if m:
            for tok in re.findall(r'"([a-z]+)"', m.group(1)):
                if tok in ("numpy", "fem", "istl", "petsc"):
                    backends.add(tok)
        if backends:
            cap.other["storage_backends"] = sorted(backends)

    # If the import failed, derive the factory lists from the source
    # files so the snapshot is still informative.
    if not introspection_ok:
        for kind, rel in (
            ("element_families", "fem/space/_spaces.py"),
            ("processes",        "fem/scheme/_schemes.py"),
        ):
            f = src_root / rel
            if not f.is_file():
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            funcs = sorted(set(
                re.findall(r"^def\s+([a-zA-Z][A-Za-z0-9_]*)\s*\(",
                           text, re.MULTILINE)
            ))
            if kind == "element_families":
                cap.element_families = funcs
            else:
                cap.processes = funcs

        # Grid generators live in the sibling dune-grid and
        # dune-alugrid repos, NOT under dune-fem — so the
        # source walk above misses them. Walk up the path until
        # we land in the dune-src root that has both siblings,
        # rather than guess at parents[N].
        mesh_kinds: list[str] = []
        dune_src_root: Path | None = None
        for parent in src_root.parents:
            if (parent / "dune-grid").is_dir() \
                    or (parent / "dune-alugrid").is_dir():
                dune_src_root = parent
                break
        if dune_src_root is None:
            dune_src_root = src_root.parent
        for sibling, prefix in (
            ("dune-grid/python/dune/grid/_grids.py", "dune.grid"),
            ("dune-alugrid/python/dune/alugrid/_grids.py", "dune.alugrid"),
        ):
            f = dune_src_root / sibling
            if not f.is_file():
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for fn in sorted(set(
                re.findall(r"^def\s+([a-zA-Z][A-Za-z0-9_]*)\s*\(",
                           text, re.MULTILINE)
            )):
                if "Grid" in fn or fn.endswith("Grid") or fn.endswith("grid"):
                    mesh_kinds.append(f"{prefix}.{fn}")
        if mesh_kinds:
            cap.mesh_generators = sorted(mesh_kinds)

    cap.notes.append(
        f"source_root={_redact_path(src_root)}; "
        f"introspection_ok={introspection_ok}")
    return cap


# ── dispatch ───────────────────────────────────────────────────────────


SCANNERS = {
    "kratos": scan_kratos,
    "fourc": scan_fourc,
    "skfem": scan_skfem,
    "fenics": scan_fenics,
    "ngsolve": scan_ngsolve,
    "dealii": scan_dealii,
    "dune": scan_dune,
}


def run(backends: list[str]) -> dict[str, BackendCapabilities]:
    SCAN_RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict[str, BackendCapabilities] = {}
    for name in backends:
        scanner = SCANNERS.get(name)
        if scanner is None:
            print(f"  {name:8s}  (no scanner yet — pending PR)")
            continue
        print(f"  {name:8s}  scanning…", end="", flush=True)
        cap = scanner()
        out[name] = cap
        (SCAN_RESULTS / f"{name}.json").write_text(
            json.dumps(asdict(cap), indent=2, default=str)
        )
        print(f" applications={len(cap.applications)} "
              f"elements={len(cap.elements)} "
              f"conditions={len(cap.conditions)} "
              f"laws={len(cap.constitutive_laws)} "
              f"variables={len(cap.variables)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="",
                    help="restrict to one backend name; default scans all known")
    args = ap.parse_args()

    targets = [args.backend] if args.backend else sorted(SCANNERS.keys())
    print(f"Scanning {len(targets)} backend(s): {', '.join(targets)}")
    results = run(targets)

    # Top-level summary — keep ALL capability buckets so the
    # summary actually reflects what was found. Backends populate
    # different buckets (NGSolve/skfem use elements + element_families,
    # 4C uses conditions + constitutive_laws, Kratos uses applications
    # + variables, deal.II uses elements + mesh_generators + other),
    # so a 5-column print misrepresents the scan.
    summary = {
        name: {
            "version": cap.version,
            "n_applications": len(cap.applications),
            "n_elements": len(cap.elements),
            "n_element_families": len(cap.element_families),
            "n_conditions": len(cap.conditions),
            "n_constitutive_laws": len(cap.constitutive_laws),
            "n_variables": len(cap.variables),
            "n_mesh_generators": len(cap.mesh_generators),
            "n_processes": len(cap.processes),
            "n_modelers": len(cap.modelers),
            "other_counts": {k: len(v) for k, v in cap.other.items()
                             if isinstance(v, list)},
            "notes": cap.notes,
        }
        for name, cap in results.items()
    }
    (SCAN_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults under {SCAN_RESULTS}/")


if __name__ == "__main__":
    main()
