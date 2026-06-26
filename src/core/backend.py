"""
Abstract base class for FEM solver backends.

Each backend (4C, FEniCS, deal.ii, ...) implements this interface,
providing solver-independent access to:
  - Installation check
  - Physics knowledge
  - Input generation (YAML, Python script, C++ code, ...)
  - Simulation execution
  - Result retrieval
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class BackendStatus(Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    MISCONFIGURED = "misconfigured"


class InputFormat(Enum):
    YAML = "yaml"           # 4C
    PYTHON = "python"       # FEniCS/dolfinx
    CPP = "cpp"             # deal.ii
    XML = "xml"             # FEBio
    JSON = "json"           # generic


def sorted_by_step(paths: list) -> list:
    """Sort result files by NUMERIC step index, not lexicographically.

    Fixes a silent-wrong bug: plain sorted() puts 'field_10.vtu' before
    'field_2.vtu', so callers taking [-1] as "final timestep" can read an EARLIER
    step (step 9 while step 10 exists) and report it as the converged result.
    Sort by the LAST integer in each filename stem, then name.
    """
    import re
    def _key(p):
        nums = re.findall(r"\d+", Path(p).stem)
        return (int(nums[-1]) if nums else -1, str(p))
    return sorted(paths, key=_key)


def detect_template_language(content: str, default: str) -> str:
    """Sniff a template's first-line markers and return the markdown
    fence language tag that matches the *content*, not the backend's
    declared ``input_format``.

    Catches the Kratos case: ``KratosBackend.input_format() == JSON``
    (its native config-file format), but its catalog generators emit
    Python scripts that use the ``KratosMultiphysics`` library
    directly.  Tagging those as ``json`` confuses any client that
    strips fences by language.

    Falls back to ``default`` (typically ``backend.input_format().value``)
    when no marker matches.
    """
    head = content.lstrip()[:200]
    if not head:
        return default
    python_markers = (
        '"""', "'''",
        "#!/",
        "from ", "import ",
        "def ", "class ",
        "# -*-",
    )
    if head.startswith(python_markers):
        return "python"
    if head.startswith(("<?xml", "<")):
        return "xml"
    if head.startswith(("{", "[")):
        return "json"
    if head.startswith(("//", "#include", "template<", "namespace ")):
        return "cpp"
    return default


@dataclass
class PhysicsCapability:
    """A physics problem this backend can solve."""
    name: str                          # e.g. "poisson", "linear_elasticity"
    description: str                   # human-readable
    spatial_dims: list[int]            # [2], [3], or [2, 3]
    element_types: list[str]           # e.g. ["QUAD4", "HEX8", "TRI3"]
    template_variants: list[str] = field(default_factory=list)


@dataclass
class JobHandle:
    """Tracks a running or completed simulation."""
    job_id: str
    backend_name: str
    work_dir: Path
    status: str = "pending"            # pending, running, completed, failed
    pid: Optional[int] = None
    return_code: Optional[int] = None
    elapsed: Optional[float] = None
    error: Optional[str] = None


class SolverBackend(ABC):
    """Abstract interface for a FEM solver backend."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'fourc', 'fenics', 'dealii'."""

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. '4C Multiphysics', 'FEniCSx (dolfinx)'."""

    @abstractmethod
    def check_availability(self) -> tuple[BackendStatus, str]:
        """Check if the solver is installed and usable.

        Returns (status, message) where message explains the status.
        """

    @abstractmethod
    def input_format(self) -> InputFormat:
        """What kind of input this backend expects."""

    @abstractmethod
    def supported_physics(self) -> list[PhysicsCapability]:
        """List of physics problems this backend can solve."""

    @abstractmethod
    def get_knowledge(self, physics: str) -> dict:
        """Return domain knowledge for a physics module.

        Keys: description, materials, solver, pitfalls, unit_systems, etc.
        """

    @abstractmethod
    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        """Generate solver-specific input (YAML / Python / C++ / XML).

        Args:
            physics: Physics module name (e.g. "poisson")
            variant: Template variant (e.g. "2d", "3d")
            params: User-specified parameters to override defaults

        Returns:
            Input file content as string.
        """

    @abstractmethod
    def validate_input(self, content: str) -> list[str]:
        """Validate input content. Returns list of errors (empty = valid)."""

    @abstractmethod
    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        """Execute the simulation.

        Args:
            input_content: Generated input (YAML/Python/C++)
            work_dir: Directory for input/output files
            np: Number of MPI processes
            timeout: Maximum runtime in seconds

        Returns:
            JobHandle with status and output location.
        """

    @abstractmethod
    def get_result_files(self, job: JobHandle) -> list[Path]:
        """Return paths to VTU/XDMF output files from a completed job."""

    def get_version(self) -> Optional[str]:
        """Return solver version string, if detectable."""
        return None


def get_python_executable() -> str:
    """Get the Python executable that's running this MCP server.

    This is critical: when the MCP server runs inside a venv, we must use
    sys.executable (the venv Python) rather than shutil.which('python3')
    (which may find the system Python that lacks our packages).
    """
    import sys
    return sys.executable


def get_env_with_source_root(env_var: str) -> dict:
    """Get an environment dict that includes a source build from *_ROOT.

    If the env var (e.g. KRATOS_ROOT, NGSOLVE_ROOT) points to a directory
    with a build/, that build path is prepended to PYTHONPATH so the
    source-built version takes priority over pip-installed packages.

    This enables the develop workflow: modify source → build → use.

    Returns a copy of os.environ with PYTHONPATH adjusted.
    """
    import os
    env = os.environ.copy()
    root = os.environ.get(env_var, "")
    if not root or not Path(root).is_dir():
        return env

    # Check for Python source builds — cmake install/ takes priority, then build/
    for build_dir in ["install", "install/lib", "build/lib", "build",
                      "build/Release/lib", "build/Release", "lib"]:
        candidate = Path(root) / build_dir
        if candidate.is_dir():
            # Check if it contains Python packages
            has_python = any(candidate.rglob("*.py"))
            if has_python:
                pp = env.get("PYTHONPATH", "")
                build_path = str(candidate)
                if build_path not in pp:
                    env["PYTHONPATH"] = f"{build_path}:{pp}" if pp else build_path
                break

    # Also add the root itself (for editable installs / src layouts)
    pp = env.get("PYTHONPATH", "")
    if root not in pp:
        env["PYTHONPATH"] = f"{root}:{pp}" if pp else root

    # Add shared library paths (for compiled C++ extensions)
    for libs_dir in ["install/libs", "install/lib", "build/libs", "build/lib", "lib"]:
        candidate = Path(root) / libs_dir
        if candidate.is_dir() and any(candidate.glob("*.so")):
            ld = env.get("LD_LIBRARY_PATH", "")
            libs_path = str(candidate)
            if libs_path not in ld:
                env["LD_LIBRARY_PATH"] = f"{libs_path}:{ld}" if ld else libs_path
            break

    return env
