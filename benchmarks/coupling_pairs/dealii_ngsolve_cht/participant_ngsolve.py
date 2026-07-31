"""NGSolve participant (Neumann side of the CHT coupling).

Coupling-driver contract: run in work_dir, read imports.json (partner's
InterfaceData), write exports.json (own InterfaceData).

Steady heat conduction on the slab [x_min,x_max] x [y_min,y_max] with
constant conductivity k:
  * Neumann on y = y_min (coupling interface): incoming normal heat flux
    g(x) taken from the partner's exported `normal_fluxes` (the partner
    exports q = -k1 dT/dy, i.e. the heat flux in +y, which IS the flux
    entering this slab through its bottom boundary).
  * Dirichlet T = T_dirichlet on y = y_max (outer boundary).
  * Insulated (natural) lateral sides.
Exports the interface TEMPERATURE sampled at n_samples uniform x-positions.

On iteration 0 (empty imports.json) the interface flux falls back to
`flux_init` from params.json.

All problem numbers come from params.json — nothing is hardcoded.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (H1, BND, BilinearForm, CoefficientFunction, GridFunction,
                     LinearForm, Mesh, TaskManager, dx, ds, grad, x)


def flux_coefficient(params: dict, imports: dict) -> CoefficientFunction:
    """Least-squares polynomial fit of the imported flux samples -> CF(x)."""
    partner = params.get("partner", "dealii")
    if partner in imports and imports[partner].get("n_points", 0) > 0:
        d = imports[partner]
        xs = np.array([float(c[0]) for c in d["coordinates"]])
        qs = np.array([float(v) for v in d.get("normal_fluxes", [])])
        if qs.size == xs.size and xs.size > 0:
            deg = int(min(params.get("poly_degree", 3), xs.size - 1))
            coeffs = np.polyfit(xs, qs, deg)[::-1]  # lowest order first
        else:
            coeffs = np.array([float(params.get("flux_init", 0.0))])
    else:
        coeffs = np.array([float(params.get("flux_init", 0.0))])
    cf = CoefficientFunction(0.0)
    for i, c in enumerate(coeffs):
        cf = cf + float(c) * x ** i
    return cf


def main() -> None:
    params = json.loads(Path("params.json").read_text())
    imports = {}
    if Path("imports.json").exists():
        try:
            imports = json.loads(Path("imports.json").read_text())
        except json.JSONDecodeError:
            imports = {}

    k = float(params["k"])
    x_min, x_max = float(params["x_min"]), float(params["x_max"])
    y_if, y_out = float(params["y_min"]), float(params["y_max"])
    T_out = float(params["T_dirichlet"])

    g = flux_coefficient(params, imports)

    geo = SplineGeometry()
    # AddRectangle edge order: bottom, right, top, left
    geo.AddRectangle((x_min, y_if), (x_max, y_out),
                     bcs=("interface", "right", "outer", "left"))
    mesh = Mesh(geo.GenerateMesh(maxh=float(params["maxh"])))

    fes = H1(mesh, order=int(params["order"]), dirichlet="outer")
    u, v = fes.TnT()
    with TaskManager():
        a = BilinearForm(fes)
        a += k * grad(u) * grad(v) * dx
        a.Assemble()
        f = LinearForm(fes)
        f += g * v * ds("interface")
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.Set(CoefficientFunction(T_out), BND,
                definedon=mesh.Boundaries("outer"))
        res = f.vec.CreateVector()
        res.data = f.vec - a.mat * gfu.vec
        gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                      inverse="sparsecholesky") * res

    # Sample interface temperature at uniform x (tiny inward offset so the
    # point-location always lands inside the mesh).
    n_samples = int(params.get("n_samples", 15))
    eps = 1e-9 * (y_out - y_if)
    xs = np.linspace(x_min, x_max, n_samples)
    temps = [float(gfu(mesh(float(px), y_if + eps))) for px in xs]

    export = {
        "field_name": "temperature",
        "n_points": n_samples,
        "coordinates": [[float(px), y_if] for px in xs],
        "values": temps,
    }
    Path("exports.json").write_text(json.dumps(export, indent=2))


if __name__ == "__main__":
    main()
