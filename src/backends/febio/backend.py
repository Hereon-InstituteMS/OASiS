"""
FEBio solver backend.

FEBio is an open-source FEM code for biomechanics. Uses XML input files (.feb).
Specialized for soft tissue mechanics, biphasic/multiphasic problems, and
biological applications.

FEBio website: https://febio.org
GitHub: https://github.com/febiosoftware/FEBio
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.backend import (
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend

logger = logging.getLogger("open-fem-agent.febio")


def _find_febio_binary() -> Optional[Path]:
    """Locate the FEBio binary."""
    env_path = os.environ.get("FEBIO_BINARY")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # Common locations
    candidates = [
        Path.home() / "FEBio" / "bin" / "febio4",
        Path.home() / "FEBioStudio" / "bin" / "febio4",
        Path("/opt/febio/bin/febio4"),
        Path("/usr/local/bin/febio4"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    p = shutil.which("febio4") or shutil.which("febio3") or shutil.which("febio")
    return Path(p) if p else None


class FebioBackend(SolverBackend):

    def name(self) -> str:
        return "febio"

    def display_name(self) -> str:
        return "FEBio"

    def check_availability(self) -> tuple[BackendStatus, str]:
        binary = _find_febio_binary()
        if not binary:
            return BackendStatus.NOT_INSTALLED, (
                "FEBio binary not found. Install from https://febio.org/downloads/ "
                "or set FEBIO_BINARY env var."
            )
        return BackendStatus.AVAILABLE, f"FEBio at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.XML

    def get_version(self) -> Optional[str]:
        binary = _find_febio_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or r.stderr.strip()
        except Exception:
            return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (small strain solid mechanics)",
                spatial_dims=[3],
                element_types=["hex8", "tet4", "tet10"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Nonlinear hyperelasticity (Neo-Hookean, Mooney-Rivlin)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="biphasic",
                description="Biphasic poroelasticity (solid + fluid phases)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_confined"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat conduction (steady-state)",
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_bar"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return _FEBIO_KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        generator = _TEMPLATES.get(key)
        if not generator:
            raise ValueError(f"No FEBio template for {key}. "
                             f"Available: {list(_TEMPLATES.keys())}")
        return generator(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "<febio_spec" not in content:
            errors.append("Missing <febio_spec> root element")
        if "<Material>" not in content and "<Material " not in content:
            errors.append("Missing Material section")
        if "<Geometry>" not in content and "<Mesh>" not in content:
            errors.append("Missing Geometry/Mesh section")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_febio_binary()
        if not binary:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="febio",
                work_dir=work_dir,
                status="failed",
                error="FEBio binary not found",
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        input_file = work_dir / "input.feb"
        input_file.write_text(input_content)

        cmd = [str(binary), "-i", str(input_file)]
        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="febio", work_dir=work_dir, status="running")

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                job.error = stderr.decode(errors="replace")[-2000:]
            (work_dir / "stdout.log").write_text(stdout.decode(errors="replace"))
            (work_dir / "stderr.log").write_text(stderr.decode(errors="replace"))
        except asyncio.TimeoutError:
            job.status = "failed"
            job.elapsed = timeout
            job.error = f"Timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.xplt", "*.vtk", "*.vtu", "*.log"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted(results)


# ─── Templates ───────────────────────────────────────────────────────────


def _elasticity_3d_cube(params: dict) -> str:
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Globals>
    <Constants>
      <T>0</T>
      <R>0</R>
      <Fc>0</Fc>
    </Constants>
  </Globals>
  <Material>
    <material id="1" type="isotropic elastic">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="load_top">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof>
      <y_dof>1</y_dof>
      <z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate>
      <extend>CONSTANT</extend>
      <points>
        <pt>0,0</pt>
        <pt>1,1</pt>
      </points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/>
      <var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


def _hyperelasticity_3d_cube(params: dict) -> str:
    """3D unit cube with neo-Hookean material under prescribed
    displacement. Catalog stub — FEBio v4.0 XML."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Control>
    <analysis>STATIC</analysis>
    <time_steps>10</time_steps>
    <step_size>0.1</step_size>
    <solver type="solid">
      <symmetric_stiffness>symmetric</symmetric_stiffness>
      <equation_scheme>staggered</equation_scheme>
    </solver>
  </Control>
  <Material>
    <material id="1" type="neo-Hookean">
      <density>1.0</density>
      <E>{E}</E>
      <v>{nu}</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="load_top">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.3</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
  <Output>
    <plotfile type="febio">
      <var type="displacement"/><var type="stress"/>
    </plotfile>
  </Output>
</febio_spec>
'''


def _biphasic_3d_confined(params: dict) -> str:
    """Confined-compression biphasic test — solid skeleton with
    interstitial fluid, isotropic permeability. Catalog stub."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.0)
    perm = params.get("permeability", 1.0e-3)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="biphasic"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>10</time_steps>
    <step_size>1.0</step_size>
    <solver type="biphasic">
      <symmetric_stiffness>non-symmetric</symmetric_stiffness>
    </solver>
  </Control>
  <Material>
    <material id="1" type="biphasic">
      <phi0>0.2</phi0>
      <solid type="neo-Hookean">
        <density>1.0</density>
        <E>{E}</E>
        <v>{nu}</v>
      </solid>
      <permeability type="perm-const-iso">
        <perm>{perm}</perm>
      </permeability>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="fix_bottom">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="load_top">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="fix" type="zero displacement" node_set="fix_bottom">
      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>
    </bc>
    <bc name="load" type="prescribed displacement" node_set="load_top">
      <dof>z</dof>
      <value lc="1">-0.1</value>
    </bc>
    <bc name="drain" type="zero fluid pressure" node_set="load_top"/>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
</febio_spec>
'''


def _heat_3d_bar(params: dict) -> str:
    """Steady-state heat conduction in a 1×1×1 bar with
    Dirichlet temperatures on opposite faces. Catalog stub."""
    k = params.get("conductivity", 1.0)
    return f'''\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="heat"/>
  <Control>
    <analysis>STEADY-STATE</analysis>
    <time_steps>1</time_steps>
    <step_size>1.0</step_size>
  </Control>
  <Material>
    <material id="1" type="isotropic Fourier">
      <density>1.0</density>
      <capacity>1.0</capacity>
      <conductivity>{k}</conductivity>
    </material>
  </Material>
  <Mesh>
    <Nodes name="Object1">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">1,1,0</node>
      <node id="4">0,1,0</node>
      <node id="5">0,0,1</node>
      <node id="6">1,0,1</node>
      <node id="7">1,1,1</node>
      <node id="8">0,1,1</node>
    </Nodes>
    <Elements type="hex8" mat="1" name="Part1">
      <elem id="1">1,2,3,4,5,6,7,8</elem>
    </Elements>
    <NodeSet name="cold_face">
      <n id="1"/><n id="2"/><n id="3"/><n id="4"/>
    </NodeSet>
    <NodeSet name="hot_face">
      <n id="5"/><n id="6"/><n id="7"/><n id="8"/>
    </NodeSet>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="Part1" mat="1"/>
  </MeshDomains>
  <Boundary>
    <bc name="cold" type="prescribed temperature" node_set="cold_face">
      <value lc="1">0.0</value>
    </bc>
    <bc name="hot" type="prescribed temperature" node_set="hot_face">
      <value lc="1">100.0</value>
    </bc>
  </Boundary>
  <LoadData>
    <load_controller id="1" type="loadcurve">
      <interpolate>LINEAR</interpolate><extend>CONSTANT</extend>
      <points><pt>0,0</pt><pt>1,1</pt></points>
    </load_controller>
  </LoadData>
</febio_spec>
'''


_TEMPLATES = {
    "linear_elasticity_3d_cube": _elasticity_3d_cube,
    "hyperelasticity_3d_cube": _hyperelasticity_3d_cube,
    "biphasic_3d_confined": _biphasic_3d_confined,
    "heat_3d_bar": _heat_3d_bar,
}


_FEBIO_KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity with FEBio — isotropic elastic material",
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": "Newton-Raphson with direct linear solver",
        "materials": {
            "isotropic elastic": {"E": "Young's modulus (Pa)", "v": "Poisson's ratio"},
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "material constant 1", "c2": "material constant 2"},
        },
        "pitfalls": [
            "FEBio uses lowercase 'v' for Poisson's ratio (not 'nu')",
            "Element connectivity is 1-indexed",
            "MeshDomains section required in v4.0 (links domain to material)",
            "LoadData section with load_controller needed for prescribed BCs",
        ],
    },
    "hyperelasticity": {
        "description": "Nonlinear hyperelasticity with FEBio — Neo-Hookean, Mooney-Rivlin",
        "materials": {
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "1st Mooney-Rivlin constant", "c2": "2nd constant",
                              "k": "bulk modulus"},
        },
        "pitfalls": [
            "Use 'STATIC' analysis for quasi-static loading",
            "Large deformations require proper step size control",
            "Convergence issues: reduce step size or use line search",
        ],
    },
    "biphasic": {
        "description": "Biphasic poroelasticity — solid skeleton + interstitial fluid (FEBio Module type='biphasic')",
        "input_format": "FEBio XML v4.0",
        "solver": "Non-symmetric Newton-Raphson (biphasic solver)",
        "materials": {
            "biphasic": {
                "phi0": "Solid volume fraction at reference",
                "solid": "Nested solid material (e.g. neo-Hookean)",
                "permeability": "Nested permeability model "
                                "(perm-const-iso, perm-Holmes-Mow, perm-exp-iso, etc.)",
            },
        },
        "pitfalls": [
            "[Syntax] Module type MUST be 'biphasic' (NOT 'solid'). "
            "Wrong Module type causes the biphasic material to be "
            "rejected at input-parse time. "
            "Signal: stderr contains 'unknown material type' or "
            "'invalid module' from FEBio.",
            "[Syntax] biphasic material requires NESTED <solid> and "
            "<permeability> elements with their own type attribute. "
            "Flat parameter lists (E, v directly inside biphasic) "
            "are not valid. "
            "Signal: input parse fails with 'invalid material parameter'.",
            "[Numerical] Pore-pressure boundary conditions use "
            "'zero fluid pressure' or 'prescribed fluid pressure' "
            "BC types — separate from displacement BCs. "
            "Signal: silent stagnation of pressure field if drainage "
            "BC is missing from the loaded surface.",
        ],
    },
    "heat": {
        "description": "Steady-state heat conduction (FEBio Module type='heat')",
        "input_format": "FEBio XML v4.0",
        "solver": "Linear or non-linear thermal solver",
        "materials": {
            "isotropic Fourier": {
                "density": "Mass density",
                "capacity": "Specific heat capacity",
                "conductivity": "Thermal conductivity",
            },
        },
        "pitfalls": [
            "[Syntax] Module type MUST be 'heat' (NOT 'solid'). "
            "Signal: stderr contains 'invalid module type' from "
            "FEBio when the wrong Module type is used.",
            "[Syntax] Temperature BCs use 'prescribed temperature' "
            "type (NOT 'prescribed displacement'). "
            "Signal: FEBio reports 'invalid BC type' at parse time.",
            "[Syntax] The thermal conductivity material in "
            "FEBio 4.0 is the <isotropic Fourier> type — NOT "
            "<thermal>, <conductive>, or <Fourier>. The exact "
            "attribute spelling 'isotropic Fourier' (lowercase "
            "noun, capitalized proper noun, space-separated) is "
            "required. "
            "Signal: emitting <material type='Fourier'> or "
            "<material type='isotropic_Fourier'> (underscore) "
            "raises an FEBio 'unknown material' parse error "
            "naming the supplied type string verbatim.",
        ],
    },
}


def register():
    register_backend(
        FebioBackend(),
        aliases=["febio", "FEBio", "febio4"],
    )
