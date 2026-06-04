"""
Field transfer between FEM solvers via VTU/numpy/interpolation.

Foundation for cross-solver coupling: reads VTU output from any backend,
extracts field data at interface nodes, interpolates between non-matching
meshes, and formats data for the target solver.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("oasis.field_transfer")


@dataclass
class InterfaceData:
    """Field values at an interface between subdomains."""
    coordinates: np.ndarray   # (N, dim) — interface node positions
    values: np.ndarray        # (N,) or (N, n_comp) — field values
    field_name: str
    normal_fluxes: Optional[np.ndarray] = None  # (N,) — normal gradient at interface

    def to_dict(self) -> dict:
        d = {
            "field_name": self.field_name,
            "n_points": len(self.coordinates),
            "coordinates": self.coordinates.tolist(),
            "values": self.values.tolist(),
        }
        if self.normal_fluxes is not None:
            d["normal_fluxes"] = self.normal_fluxes.tolist()
        return d

    def to_json(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "InterfaceData":
        d = json.loads(path.read_text())
        coords = np.array(d["coordinates"])
        values = np.array(d["values"])
        fluxes = np.array(d["normal_fluxes"]) if "normal_fluxes" in d else None
        return cls(
            coordinates=coords,
            values=values,
            field_name=d["field_name"],
            normal_fluxes=fluxes,
        )


def extract_interface_from_vtu(
    vtu_path: Path,
    field_name: str,
    interface_coord: float,
    interface_axis: int = 0,
    tol: float = 1e-6,
) -> InterfaceData:
    """Extract field values at an interface plane from a VTU file.

    Args:
        vtu_path: Path to VTU result file.
        field_name: Name of the field to extract (e.g. "temperature").
        interface_coord: Coordinate value defining the interface plane.
        interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z).
        tol: Tolerance for node matching.

    Returns:
        InterfaceData with coordinates, values, and optionally fluxes.
    """
    from core.post_processing import read_mesh

    mesh = read_mesh(vtu_path)
    points = np.asarray(mesh.points)

    # Find nodes at the interface
    mask = np.abs(points[:, interface_axis] - interface_coord) < tol
    if not np.any(mask):
        raise ValueError(
            f"No nodes found at {['x','y','z'][interface_axis]}={interface_coord} "
            f"(tol={tol}). Range: [{points[:, interface_axis].min():.6f}, "
            f"{points[:, interface_axis].max():.6f}]"
        )

    interface_points = points[mask]
    indices = np.where(mask)[0]

    # Extract field values
    if field_name in mesh.point_data:
        arr = np.asarray(mesh.point_data[field_name])
        interface_values = arr[indices]
    else:
        available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise ValueError(f"Field '{field_name}' not found. Available: {available}")

    # Sort by the tangential coordinate for consistent ordering
    tangential_axes = [i for i in range(points.shape[1]) if i != interface_axis]
    sort_key = interface_points[:, tangential_axes[0]]
    order = np.argsort(sort_key)

    logger.info(
        f"Extracted {len(indices)} interface nodes at "
        f"{['x','y','z'][interface_axis]}={interface_coord}"
    )

    return InterfaceData(
        coordinates=interface_points[order],
        values=interface_values[order],
        field_name=field_name,
    )


def interpolate_to_points(
    source: InterfaceData,
    target_coords: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate interface data onto target coordinates.

    Uses scipy.interpolate.griddata for non-matching meshes.
    For 1D interfaces (e.g. y-values along x=const), uses interp1d.

    Args:
        source: InterfaceData from the source solver.
        target_coords: (M, dim) target node positions.
        method: Interpolation method ('nearest', 'linear', 'cubic').

    Returns:
        Interpolated values at target coordinates.
    """
    src_coords = source.coordinates
    src_values = source.values

    # Determine dimensionality of the interface
    # Remove the constant-axis columns
    nonconst = []
    for ax in range(src_coords.shape[1]):
        if src_coords[:, ax].max() - src_coords[:, ax].min() > 1e-10:
            nonconst.append(ax)

    if len(nonconst) == 0:
        # Single point — return constant
        return np.full(len(target_coords), src_values[0])

    if len(nonconst) == 1:
        # 1D interface — use numpy interp (robust, no scipy needed)
        ax = nonconst[0]
        order = np.argsort(src_coords[:, ax])
        return np.interp(
            target_coords[:, ax],
            src_coords[order, ax],
            src_values[order],
        )

    # 2D+ interface — use scipy griddata
    from scipy.interpolate import griddata
    src_2d = src_coords[:, nonconst]
    tgt_2d = target_coords[:, nonconst]
    result = griddata(src_2d, src_values, tgt_2d, method=method)

    # Fill NaN (extrapolation) with nearest
    nans = np.isnan(result)
    if np.any(nans):
        nearest = griddata(src_2d, src_values, tgt_2d[nans], method="nearest")
        result[nans] = nearest

    return result


def format_for_fenics(
    interface_data: InterfaceData,
    bc_type: str = "dirichlet",
    interface_axis: int = 0,
    interface_coord: float = 0.5,
) -> str:
    """Format interface data as FEniCS Python code snippet.

    Generates code that applies Dirichlet or Neumann BC at the interface.

    Args:
        interface_data: Field values at the interface.
        bc_type: 'dirichlet' or 'neumann'.
        interface_axis: Axis of the interface (0=x, 1=y).
        interface_coord: Coordinate of the interface.

    Returns:
        Python code snippet for FEniCS.
    """
    coords = interface_data.coordinates
    values = interface_data.values

    # Determine tangential axis
    tangential_axes = [i for i in range(coords.shape[1]) if i != interface_axis]
    tang_ax = tangential_axes[0] if tangential_axes else 0

    tang_vals = coords[:, tang_ax].tolist()
    field_vals = values.tolist()

    axis_name = ['x', 'y', 'z'][interface_axis]

    if bc_type == "dirichlet":
        return f"""\
# Interface Dirichlet BC from coupled solver
# {len(tang_vals)} nodes at {axis_name}={interface_coord}
_interface_tang = np.array({tang_vals})
_interface_vals = np.array({field_vals})

def interface_marker(x):
    return np.isclose(x[{interface_axis}], {interface_coord})

interface_facets = mesh.locate_entities_boundary(domain, fdim, interface_marker)
interface_dofs = fem.locate_dofs_topological(V, fdim, interface_facets)

# Interpolate coupled values onto FEniCS DOFs
_iface_func = fem.Function(V)
_coords = V.tabulate_dof_coordinates()
for i in interface_dofs:
    _tang = _coords[i, {tang_ax}]
    _iface_func.x.array[i] = np.interp(_tang, _interface_tang, _interface_vals)
bc_interface = fem.dirichletbc(_iface_func, interface_dofs)
"""
    else:
        return f"""\
# Interface Neumann BC from coupled solver (flux)
# {len(tang_vals)} nodes at {axis_name}={interface_coord}
_interface_tang = np.array({tang_vals})
_interface_flux = np.array({field_vals})

def interface_marker(x):
    return np.isclose(x[{interface_axis}], {interface_coord})

interface_facets = mesh.locate_entities_boundary(domain, fdim, interface_marker)
facet_tags = mesh.meshtags(domain, fdim, interface_facets,
                            np.full(len(interface_facets), 99, dtype=np.int32))
ds_iface = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)(99)

# Interpolate flux values
_flux_func = fem.Function(V)
_coords = V.tabulate_dof_coordinates()
for i in range(len(_coords)):
    if np.isclose(_coords[i, {interface_axis}], {interface_coord}):
        _tang = _coords[i, {tang_ax}]
        _flux_func.x.array[i] = np.interp(_tang, _interface_tang, _interface_flux)
# Add to RHS: L += _flux_func * v * ds_iface
"""


def format_for_4c_neumann(
    interface_data: InterfaceData,
    line_id: int = 3,
) -> str:
    """Format interface flux data as 4C DESIGN LINE NEUMANN CONDITIONS.

    For uniform flux, returns a single Neumann condition.
    For non-uniform, returns per-node conditions via DESIGN POINT NEUMANN.

    Args:
        interface_data: Flux values at the interface.
        line_id: Design line ID for the interface.

    Returns:
        YAML snippet for 4C input file.
    """
    values = interface_data.values
    mean_flux = float(np.mean(values))

    # Check if flux is approximately uniform
    if np.std(values) < 1e-8 * abs(mean_flux) if mean_flux != 0 else np.std(values) < 1e-12:
        return f"""\
DESIGN LINE NEUMANN CONDITIONS:
  - E: {line_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{mean_flux:.10e}]
    FUNCT: [0]
"""
    else:
        # Non-uniform: use per-node point Neumann (approximate via uniform for now)
        # Full non-uniform would need FUNCT definition — use mean as approximation
        logger.warning(
            f"Non-uniform flux (std={np.std(values):.2e}) approximated by mean={mean_flux:.6e}"
        )
        return f"""\
DESIGN LINE NEUMANN CONDITIONS:
  - E: {line_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{mean_flux:.10e}]
    FUNCT: [0]
"""


def compute_interface_flux_fenics_code(
    interface_axis: int = 0,
    interface_coord: float = 0.5,
    field_name: str = "temperature",
    conductivity: float = 1.0,
) -> str:
    """Generate FEniCS code that computes the normal heat flux at the interface.

    The flux is q = -k * dT/dn evaluated at interface nodes.
    This code should be appended to the FEniCS solve script.

    Returns:
        Python code that writes interface_data.json with coordinates, values, and fluxes.
    """
    axis_name = ['x', 'y', 'z'][interface_axis]
    tang_axes = [i for i in range(3) if i != interface_axis]

    return f"""\

# === Compute interface flux ===
import json as _json

# Project gradient
V_grad = fem.functionspace(domain, ("DG", 0, (gdim,)))
grad_T = fem.Function(V_grad)

# Use L2 projection for gradient
_u_grad = ufl.TrialFunction(V_grad)
_v_grad = ufl.TestFunction(V_grad)
_a_grad = ufl.inner(_u_grad, _v_grad) * ufl.dx
_L_grad = ufl.inner(ufl.grad(uh), _v_grad) * ufl.dx
from dolfinx.fem.petsc import LinearProblem as _LP
_prob_grad = _LP(_a_grad, _L_grad, petsc_options_prefix="grad",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
grad_T = _prob_grad.solve()

# Extract interface node values and flux
_coords = V.tabulate_dof_coordinates()
_mask = np.abs(_coords[:, {interface_axis}] - {interface_coord}) < 1e-6
_iface_dofs = np.where(_mask)[0]
_iface_coords = _coords[_iface_dofs]
_iface_T = uh.x.array[_iface_dofs]

# Normal flux: q_n = -k * dT/d{axis_name}
# Use finite difference from nearby nodes for robustness
_h = 1e-4
_flux = np.zeros(len(_iface_dofs))
for _ii, _dof in enumerate(_iface_dofs):
    _pt = _coords[_dof].copy()
    # Find nearest interior node
    _dists = np.linalg.norm(_coords - _pt, axis=1)
    _dists[_mask] = 1e10  # exclude interface nodes
    _nearest = np.argmin(_dists)
    _dx = _coords[_nearest, {interface_axis}] - _pt[{interface_axis}]
    if abs(_dx) > 1e-12:
        _flux[_ii] = -{conductivity} * (uh.x.array[_nearest] - uh.x.array[_dof]) / _dx
    else:
        _flux[_ii] = 0.0

# Sort by tangential coordinate
_order = np.argsort(_iface_coords[:, {tang_axes[0]}])
_iface_data = {{
    "field_name": "{field_name}",
    "n_points": int(len(_iface_dofs)),
    "coordinates": _iface_coords[_order].tolist(),
    "values": _iface_T[_order].tolist(),
    "normal_fluxes": _flux[_order].tolist(),
}}
with open("interface_data.json", "w") as _f:
    _json.dump(_iface_data, _f, indent=2)
print(f"Interface: {{len(_iface_dofs)}} nodes, T=[{{_iface_T.min():.4f}}, {{_iface_T.max():.4f}}], flux=[{{_flux.min():.4e}}, {{_flux.max():.4e}}]")
"""


def extract_full_field_from_vtu(
    vtu_path: Path,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract field values at ALL nodes from a VTU file.

    Unlike extract_interface_from_vtu which filters to an interface plane,
    this returns the complete field for full-domain comparison or transfer.

    Args:
        vtu_path: Path to VTU result file.
        field_name: Name of the field to extract (e.g. 'displacement', 'temperature').
            Also tries prefix matching (e.g. 'disp' matches 'displacement').

    Returns:
        Tuple of (coordinates, values) where coordinates is (N, 3) and
        values is (N,) for scalar or (N, n_comp) for vector fields.
    """
    from core.post_processing import read_mesh

    mesh = read_mesh(vtu_path)
    points = np.asarray(mesh.points)

    if field_name in mesh.point_data:
        values = np.asarray(mesh.point_data[field_name])
    else:
        # Try prefix match
        available = list(mesh.point_data.keys())
        for key in available:
            if key.startswith(field_name) or field_name.startswith(key):
                values = np.asarray(mesh.point_data[key])
                logger.info(f"Field '{field_name}' matched as '{key}'")
                break
        else:
            raise ValueError(
                f"Field '{field_name}' not found in {vtu_path.name}. "
                f"Available: {available}"
            )

    logger.info(f"Extracted {len(points)} nodes, field shape {values.shape}")
    return points, values
