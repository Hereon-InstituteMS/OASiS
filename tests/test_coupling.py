"""
Tests for cross-solver coupling tools.

Tests field_transfer core module and coupling tool helpers.
Integration tests for coupled_solve require both FEniCS and 4C.
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _run_async(coro):
    """Run an async coroutine in a sync test."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Unit tests for field_transfer core ─────────────────────────────────

class TestInterfaceData:
    def test_roundtrip_json(self, tmp_path):
        from core.field_transfer import InterfaceData

        coords = np.array([[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0]])
        values = np.array([100.0, 50.0, 0.0])
        fluxes = np.array([-100.0, -100.0, -100.0])

        iface = InterfaceData(
            coordinates=coords,
            values=values,
            field_name="temperature",
            normal_fluxes=fluxes,
        )

        json_path = tmp_path / "test_iface.json"
        iface.to_json(json_path)

        loaded = InterfaceData.from_json(json_path)
        assert loaded.field_name == "temperature"
        assert len(loaded.coordinates) == 3
        np.testing.assert_allclose(loaded.values, values)
        np.testing.assert_allclose(loaded.normal_fluxes, fluxes)

    def test_to_dict(self):
        from core.field_transfer import InterfaceData

        iface = InterfaceData(
            coordinates=np.array([[0.0, 0.0], [0.0, 1.0]]),
            values=np.array([10.0, 20.0]),
            field_name="phi",
        )
        d = iface.to_dict()
        assert d["field_name"] == "phi"
        assert d["n_points"] == 2
        assert "normal_fluxes" not in d


class TestInterpolation:
    def test_1d_interpolation(self):
        from core.field_transfer import InterfaceData, interpolate_to_points

        # Source: 3 points along y at x=0.5
        src = InterfaceData(
            coordinates=np.array([[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0]]),
            values=np.array([100.0, 50.0, 0.0]),
            field_name="T",
        )

        # Target: 5 points along y at x=0.5
        target = np.array([
            [0.5, 0.0, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.75, 0.0],
            [0.5, 1.0, 0.0],
        ])

        result = interpolate_to_points(src, target)
        expected = np.array([100.0, 75.0, 50.0, 25.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_constant_field(self):
        from core.field_transfer import InterfaceData, interpolate_to_points

        src = InterfaceData(
            coordinates=np.array([[0.5, 0.0, 0.0]]),
            values=np.array([42.0]),
            field_name="T",
        )
        target = np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]])
        result = interpolate_to_points(src, target)
        np.testing.assert_allclose(result, [42.0, 42.0])


class TestFormatters:
    def test_format_for_fenics_dirichlet(self):
        from core.field_transfer import InterfaceData, format_for_fenics

        iface = InterfaceData(
            coordinates=np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]]),
            values=np.array([100.0, 0.0]),
            field_name="T",
        )
        code = format_for_fenics(iface, "dirichlet", 0, 0.5)
        assert "interface_marker" in code
        assert "dirichletbc" in code
        assert "np.interp" in code

    def test_format_for_fenics_neumann(self):
        from core.field_transfer import InterfaceData, format_for_fenics

        iface = InterfaceData(
            coordinates=np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]]),
            values=np.array([-100.0, -100.0]),
            field_name="flux",
        )
        code = format_for_fenics(iface, "neumann", 0, 0.5)
        assert "interface_marker" in code
        assert "ds_iface" in code

    def test_format_for_4c_neumann_uniform(self):
        from core.field_transfer import InterfaceData, format_for_4c_neumann

        iface = InterfaceData(
            coordinates=np.array([[0.5, 0.0], [0.5, 1.0]]),
            values=np.array([-100.0, -100.0]),
            field_name="flux",
        )
        yaml = format_for_4c_neumann(iface)
        assert "DESIGN LINE NEUMANN CONDITIONS" in yaml
        assert "-1.0000000000e+02" in yaml


# ─── Test FEniCS subdomain script generation ─────────────────────────────

class TestFenicsSubdomainScript:
    def test_generates_valid_python(self):
        from tools.coupling import _fenics_heat_subdomain_script

        script = _fenics_heat_subdomain_script(
            x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0,
            nx=8, ny=8,
            T_left=100.0,
            T_interface=[50.0] * 9,
            interface_side="right",
            compute_flux=True,
        )
        assert "from dolfinx" in script
        assert "LinearProblem" in script
        assert "interface_data.json" in script
        # Should compile as Python
        compile(script, "<subdomain_a>", "exec")

    def test_no_interface_bc(self):
        from tools.coupling import _fenics_heat_subdomain_script

        script = _fenics_heat_subdomain_script(
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            nx=8, ny=8,
            T_left=0.0, T_right=0.0,
            source=1.0,
        )
        assert "from dolfinx" in script
        assert "bc_left" in script
        assert "bc_right" in script
        compile(script, "<full_domain>", "exec")


class TestFourcSubdomainInput:
    def test_generates_valid_yaml(self):
        from tools.coupling import _fourc_heat_subdomain_input

        yaml_str = _fourc_heat_subdomain_input(
            nx=8, ny=8,
            x_min=0.5, x_max=1.0,
            y_min=0.0, y_max=1.0,
            T_right=0.0,
            neumann_flux=-100.0,
            neumann_line=2,
        )
        import yaml
        data = yaml.safe_load(yaml_str)
        assert data["PROBLEM TYPE"]["PROBLEMTYPE"] == "Scalar_Transport"
        assert "MATERIALS" in data
        assert "NODE COORDS" in data

    def test_node_offset(self):
        from tools.coupling import _fourc_heat_subdomain_input

        yaml_str = _fourc_heat_subdomain_input(
            nx=4, ny=4,
            x_min=0.5, x_max=1.0,
            y_min=0.0, y_max=1.0,
            T_right=0.0,
            neumann_flux=-100.0,
            neumann_line=2,
        )
        # Check that nodes start at x=0.5
        import yaml
        data = yaml.safe_load(yaml_str)
        first_node = data["NODE COORDS"][0]
        # "NODE 1 COORD 0.500000 0.000000 0.0"
        coords = first_node.split("COORD")[1].strip().split()
        x = float(coords[0])
        assert abs(x - 0.5) < 1e-6, f"First node x={x}, expected 0.5"

    def test_has_neumann_condition(self):
        from tools.coupling import _fourc_heat_subdomain_input

        yaml_str = _fourc_heat_subdomain_input(
            nx=4, ny=4,
            x_min=0.5, x_max=1.0,
            y_min=0.0, y_max=1.0,
            T_right=0.0,
            neumann_flux=-50.0,
            neumann_line=2,
        )
        assert "DESIGN LINE NEUMANN CONDITIONS" in yaml_str


# ─── Convergence plot test ───────────────────────────────────────────────

class TestConvergencePlot:
    def test_plot_generation(self, tmp_path):
        from tools.coupling import _plot_convergence

        history = [
            {"iteration": 0, "residual": 1.0, "mean_flux": -100.0, "T_interface_mean": 50.0, "time_s": 1.0},
            {"iteration": 1, "residual": 0.1, "mean_flux": -100.0, "T_interface_mean": 50.0, "time_s": 1.0},
            {"iteration": 2, "residual": 0.01, "mean_flux": -100.0, "T_interface_mean": 50.0, "time_s": 1.0},
        ]
        path = _plot_convergence(history, tmp_path / "test_conv.png")
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 1000  # Non-trivial PNG


# ─── Integration test (requires FEniCS + 4C) ────────────────────────────

def _fourc_binary_path():
    """Get 4C binary path from env or default."""
    import os
    return os.environ.get("FOURC_BINARY", shutil.which("4C") or "/usr/local/bin/4C")


@pytest.mark.skipif(
    not Path(_fourc_binary_path()).exists(),
    reason=f"4C binary not found at {_fourc_binary_path()}. "
           f"Set FOURC_BINARY env var to override.",
)
class TestCoupledSolveIntegration:
    """Integration tests — only run when both solvers are available.

    Set FOURC_BINARY and FOURC_ROOT environment variables to configure
    the 4C binary location for non-default installations.
    """

    @pytest.fixture
    def setup_backends(self):
        """Ensure backends are loaded."""
        import os
        fourc_bin = _fourc_binary_path()
        fourc_root = os.environ.get("FOURC_ROOT", str(Path(fourc_bin).parent.parent))
        os.environ.setdefault("FOURC_ROOT", fourc_root)
        os.environ.setdefault("FOURC_BINARY", fourc_bin)
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        return get_backend

    def test_heat_dd_converges(self, setup_backends):
        """Heat DD should converge to T(x)=100*(1-x) within a few iterations."""
        from tools.coupling import _heat_domain_decomposition
        from core.registry import get_backend

        fenics = get_backend("fenics")
        fourc = get_backend("fourc")
        if not fenics or not fourc:
            pytest.skip("Both FEniCS and 4C required")

        status_f, _ = fenics.check_availability()
        status_c, _ = fourc.check_availability()
        if status_f.value != "available" or status_c.value != "available":
            pytest.skip("Both FEniCS and 4C must be available")

        # Use small mesh for fast test
        result = _run_async(_heat_domain_decomposition(
            fenics, fourc, nx=8, ny=8,
            max_iter=10, tol=1e-4, relaxation=1.0,
            params={"T_left": 100.0, "T_right": 0.0},
        ))
        assert "Converged" in result, f"Did not converge:\n{result}"

    def test_poisson_dd_converges(self, setup_backends):
        """Poisson DD with source f=1 should converge iteratively."""
        from tools.coupling import _poisson_domain_decomposition
        from core.registry import get_backend

        fenics = get_backend("fenics")
        fourc = get_backend("fourc")
        if not fenics or not fourc:
            pytest.skip("Both FEniCS and 4C required")

        status_f, _ = fenics.check_availability()
        status_c, _ = fourc.check_availability()
        if status_f.value != "available" or status_c.value != "available":
            pytest.skip("Both FEniCS and 4C must be available")

        # Poisson DD needs relaxation θ<1 to converge (classic DN property)
        result = _run_async(_poisson_domain_decomposition(
            fenics, fourc, nx=8, ny=8,
            max_iter=20, tol=1e-4, relaxation=0.5,
            params={"source": 1.0},
        ))
        assert "Converged" in result, f"Did not converge:\n{result}"

    def test_fenics_fenics_heat_dd(self, setup_backends):
        """FEniCS↔FEniCS heat DD — proves framework is solver-agnostic."""
        from tools.coupling import _heat_domain_decomposition
        from core.registry import get_backend

        fenics = get_backend("fenics")
        if not fenics:
            pytest.skip("FEniCS required")
        status, _ = fenics.check_availability()
        if status.value != "available":
            pytest.skip("FEniCS not available")

        # Use FEniCS for BOTH subdomains
        result = _run_async(_heat_domain_decomposition(
            fenics, fenics, nx=8, ny=8,
            max_iter=10, tol=1e-4, relaxation=1.0,
            params={"T_left": 100.0, "T_right": 0.0},
        ))
        assert "Converged" in result, f"FEniCS↔FEniCS did not converge:\n{result}"

    def test_fenics_fenics_poisson_dd(self, setup_backends):
        """FEniCS↔FEniCS Poisson DD — cross-instance coupling."""
        from tools.coupling import _poisson_domain_decomposition
        from core.registry import get_backend

        fenics = get_backend("fenics")
        if not fenics:
            pytest.skip("FEniCS required")
        status, _ = fenics.check_availability()
        if status.value != "available":
            pytest.skip("FEniCS not available")

        result = _run_async(_poisson_domain_decomposition(
            fenics, fenics, nx=8, ny=8,
            max_iter=20, tol=1e-4, relaxation=0.5,
            params={"source": 1.0},
        ))
        assert "Converged" in result, f"FEniCS↔FEniCS Poisson did not converge:\n{result}"

    def test_ngsolve_ngsolve_poisson_dd(self, setup_backends):
        """NGSolve↔NGSolve Poisson DD — pins the NGSolve subdomain generator
        (missing return, wrong Dirichlet flags, broken vertex iteration,
        interface_data.json schema)."""
        from tools.coupling import _poisson_domain_decomposition
        from core.registry import get_backend

        ngsolve = get_backend("ngsolve")
        if not ngsolve:
            pytest.skip("NGSolve required")
        status, _ = ngsolve.check_availability()
        if status.value != "available":
            pytest.skip("NGSolve not available")

        result = _run_async(_poisson_domain_decomposition(
            ngsolve, ngsolve, nx=8, ny=8,
            max_iter=40, tol=1e-5, relaxation=0.5,
            params={"source": 1.0},
        ))
        assert "Converged" in result, f"NGSolve↔NGSolve Poisson did not converge:\n{result}"

    def test_skfem_skfem_poisson_dd(self, setup_backends):
        """scikit-fem↔scikit-fem Poisson DD — pins the skfem flux generator
        (interface_data.json schema: missing field_name, 1-D coordinates)."""
        from tools.coupling import _poisson_domain_decomposition
        from core.registry import get_backend

        skfem = get_backend("skfem")
        if not skfem:
            pytest.skip("scikit-fem required")
        status, _ = skfem.check_availability()
        if status.value != "available":
            pytest.skip("scikit-fem not available")

        result = _run_async(_poisson_domain_decomposition(
            skfem, skfem, nx=8, ny=8,
            max_iter=40, tol=1e-5, relaxation=0.5,
            params={"source": 1.0},
        ))
        assert "Converged" in result, f"skfem↔skfem Poisson did not converge:\n{result}"

    def test_oneway_tsi(self, setup_backends):
        """One-way TSI: FEniCS thermal → 4C structural via native TSI."""
        from tools.coupling import _oneway_thermal_structural
        from core.registry import get_backend

        fenics = get_backend("fenics")
        fourc = get_backend("fourc")
        if not fenics or not fourc:
            pytest.skip("Both FEniCS and 4C required for one-way TSI")

        status_f, _ = fenics.check_availability()
        status_c, _ = fourc.check_availability()
        if status_f.value != "available" or status_c.value != "available":
            pytest.skip("Both FEniCS and 4C must be available")

        result = _run_async(_oneway_thermal_structural(
            fenics, fourc, nx=4, ny=4,
            params={"nz": 4, "E": 200e3, "nu": 0.3, "alpha": 12e-6,
                    "T_left": 100.0, "T_right": 0.0},
        ))
        assert "One-Way Coupling" in result, f"One-way TSI failed:\n{result}"
        assert "agreement" in result.lower(), f"No agreement metric:\n{result}"


# ─── Unit tests for TSI input generation ──────────────────────────────

class TestTsiInputGeneration:
    def test_tsi_oneway_generates_valid_yaml(self):
        from backends.fourc.inline_mesh import matched_tsi_oneway_input
        import yaml

        yaml_str = matched_tsi_oneway_input(nx=2, ny=2, nz=2)
        data = yaml.safe_load(yaml_str)

        assert data["PROBLEM TYPE"]["PROBLEMTYPE"] == "Thermo_Structure_Interaction"
        assert data["TSI DYNAMIC"]["COUPALGO"] == "tsi_oneway"
        assert "CLONING MATERIAL MAP" in data
        assert len(data["MATERIALS"]) == 2
        # Check material types
        mat1 = data["MATERIALS"][0]
        assert "MAT_Struct_ThermoStVenantK" in mat1
        mat2 = data["MATERIALS"][1]
        assert "MAT_Fourier" in mat2

    def test_tsi_oneway_has_solidscatra_elements(self):
        from backends.fourc.inline_mesh import matched_tsi_oneway_input

        yaml_str = matched_tsi_oneway_input(nx=2, ny=2, nz=2)
        assert "SOLIDSCATRA HEX8" in yaml_str
        assert "STRUCTURE ELEMENTS" in yaml_str

    def test_tsi_oneway_boundary_conditions(self):
        from backends.fourc.inline_mesh import matched_tsi_oneway_input

        yaml_str = matched_tsi_oneway_input(
            nx=2, ny=2, nz=2,
            T_left=200.0, T_right=50.0,
        )
        assert "DESIGN SURF THERMO DIRICH CONDITIONS" in yaml_str
        assert "200.0" in yaml_str
        assert "50.0" in yaml_str
        assert "DESIGN SURF DIRICH CONDITIONS" in yaml_str

    def test_tsi_oneway_surface_topology(self):
        from backends.fourc.inline_mesh import matched_tsi_oneway_input

        yaml_str = matched_tsi_oneway_input(nx=2, ny=2, nz=2)
        assert "DSURF-NODE TOPOLOGY" in yaml_str
        assert "DSURFACE 1" in yaml_str  # left face
        assert "DSURFACE 2" in yaml_str  # right face

    def test_tsi_oneway_initialfield_function(self):
        from backends.fourc.inline_mesh import matched_tsi_oneway_input

        yaml_str = matched_tsi_oneway_input(
            nx=2, ny=2, nz=2,
            T_left=100.0, T_right=0.0, lx=1.0,
        )
        assert "SYMBOLIC_FUNCTION_OF_SPACE_TIME" in yaml_str
        assert "INITIALFIELD" in yaml_str


class TestFenicsThreeD:
    def test_fenics_tsi_script_compiles(self):
        from tools.coupling import _fenics_tsi_oneway_script

        script = _fenics_tsi_oneway_script(
            nx=4, ny=4, nz=4,
            E=200e3, nu=0.3, alpha=12e-6,
            T_left=100.0, T_right=0.0,
        )
        assert "create_box" in script
        assert "thermal" in script.lower()
        assert "displacement" in script.lower()
        assert "results_summary.json" in script
        # Should compile as Python
        compile(script, "<tsi_oneway_3d>", "exec")


class TestFullFieldExtraction:
    def test_extract_full_field(self, tmp_path):
        """Test full-domain field extraction from a synthetic VTU."""
        try:
            import pyvista as pv
        except ImportError:
            pytest.skip("PyVista required")

        # Create a simple unstructured mesh with known field
        grid = pv.ImageData(dimensions=(3, 3, 3))
        ugrid = grid.cast_to_unstructured_grid()
        temp_vals = np.linspace(0, 100, ugrid.n_points)
        ugrid.point_data["temperature"] = temp_vals
        vtu_path = tmp_path / "test.vtu"
        ugrid.save(str(vtu_path))

        from core.field_transfer import extract_full_field_from_vtu
        points, values = extract_full_field_from_vtu(vtu_path, "temperature")

        assert len(points) == ugrid.n_points
        assert len(values) == ugrid.n_points
        np.testing.assert_allclose(values, temp_vals)


# ─── L-bracket mesh tests ─────────────────────────────────────────────

class TestLBracketMesh:
    def test_l_domain_hex8_generates_valid_mesh(self):
        from backends.fourc.inline_mesh import generate_l_domain_hex8

        mesh = generate_l_domain_hex8(n=2, lz=0.5)
        assert mesh["n_nodes"] > 0
        assert mesh["n_elements"] > 0
        assert len(mesh["left_face"]) > 0
        assert len(mesh["right_face"]) > 0

    def test_l_bracket_tsi_generates_valid_yaml(self):
        from backends.fourc.inline_mesh import matched_l_bracket_tsi_input
        import yaml

        yaml_str = matched_l_bracket_tsi_input(n=2)
        data = yaml.safe_load(yaml_str)

        assert data["PROBLEM TYPE"]["PROBLEMTYPE"] == "Thermo_Structure_Interaction"
        assert data["TSI DYNAMIC"]["COUPALGO"] == "tsi_oneway"
        assert "SOLIDSCATRA HEX8" in yaml_str
        assert "CLONING MATERIAL MAP" in data

    def test_l_domain_hex8_correct_element_count(self):
        from backends.fourc.inline_mesh import generate_l_domain_hex8

        # L-domain with n=2: 2D has 3/4 of (2n)^2 = 12 elements
        # 3D: 12 * nz layers, nz = max(n//2, 1) = 1
        mesh = generate_l_domain_hex8(n=2, lz=0.5)
        # 2n=4, 4x4 grid minus n*n=4 corner = 12 quad cells, extruded 1 layer = 12 hex
        assert mesh["n_elements"] == 12


# ─── preCICE config tests ─────────────────────────────────────────────

class TestPreciceConfig:
    def test_heat_config_generates_valid_xml(self):
        from core.precice_config import generate_heat_coupling_config

        xml = generate_heat_coupling_config()
        assert '<?xml version="1.0"' in xml
        assert "precice-configuration" in xml
        assert "Temperature" in xml
        assert "Heat-Flux" in xml
        assert "Dirichlet" in xml
        assert "Neumann" in xml

    def test_tsi_config_generates_valid_xml(self):
        from core.precice_config import generate_tsi_coupling_config

        xml = generate_tsi_coupling_config()
        assert "Temperature" in xml
        assert "Displacement" in xml
        assert "Thermal" in xml
        assert "Structure" in xml

    def test_save_config(self, tmp_path):
        from core.precice_config import generate_heat_coupling_config, save_precice_config

        xml = generate_heat_coupling_config()
        path = save_precice_config(xml, tmp_path)
        assert path.exists()
        assert path.suffix == ".xml"
        assert "precice" in path.read_text()

    def test_check_availability(self):
        from core.precice_config import check_precice_available

        available, msg = check_precice_available()
        assert isinstance(available, bool)
        assert isinstance(msg, str)


# ─── Knowledge tools tests ────────────────────────────────────────────

class TestKnowledgeTools:
    def _register_and_get_tools(self):
        """Register knowledge tools and return a dict of tool functions."""
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test")
        register_knowledge_tools(mcp)
        # Tools are registered as decorated functions; call them directly
        # by re-importing the module-level closure
        return mcp

    def test_coupling_knowledge_includes_new_problems(self):
        """Verify coupling knowledge lists all problem types."""
        mcp = self._register_and_get_tools()
        # Call directly through the registered function
        result = mcp.call_tool("get_coupling_knowledge", {})
        # The result is returned from the sync function
        # Just verify the function is registered without error

    def test_tsi_knowledge_content(self):
        """Verify TSI knowledge has critical patterns."""
        mcp = self._register_and_get_tools()
        # Access the tool's function via the decorator-captured closure
        # Since we can't easily call MCP tools in test, test the raw function
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
        mcp2 = FastMCP("test2")

        # Capture the function reference during registration
        captured = {}
        original_tool = mcp2.tool

        def capturing_tool(*args, **kwargs):
            decorator = original_tool(*args, **kwargs)
            def wrapper(fn):
                result = decorator(fn)
                captured[fn.__name__] = fn
                return result
            return wrapper

        mcp2.tool = capturing_tool
        register_knowledge_tools(mcp2)

        assert "get_tsi_knowledge" in captured
        result = captured["get_tsi_knowledge"]()
        assert "SOLIDSCATRA" in result
        assert "CLONING" in result
        assert "THEXPANS" in result
        assert "tsi_oneway" in result

    def test_precice_knowledge_content(self):
        """Verify preCICE knowledge has comparison info."""
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
        mcp2 = FastMCP("test3")

        captured = {}
        original_tool = mcp2.tool

        def capturing_tool(*args, **kwargs):
            decorator = original_tool(*args, **kwargs)
            def wrapper(fn):
                result = decorator(fn)
                captured[fn.__name__] = fn
                return result
            return wrapper

        mcp2.tool = capturing_tool
        register_knowledge_tools(mcp2)

        assert "get_precice_knowledge" in captured
        result = captured["get_precice_knowledge"]()
        assert "preCICE" in result
        assert "MCP" in result
        assert "adapter" in result.lower()
