"""deal.II participant wrapper (Dirichlet side of the CHT coupling).

Coupling-driver contract: run in work_dir, read imports.json (partner's
InterfaceData), write exports.json (own InterfaceData).

This wrapper is pure glue — the PDE solve is done by the compiled deal.II
executable (heat_slab_dealii.cc). The wrapper:
  1. reads params.json (problem numbers + path to the compiled solver),
  2. converts the partner's exported interface TEMPERATURE into the solver's
     plain-text input file (Dirichlet data on the interface),
  3. runs the deal.II solver,
  4. converts its output (x, T, q at the interface) into exports.json, with
     the interface normal heat flux q = -k dT/dy in `normal_fluxes`.

On iteration 0 (empty imports.json) the interface temperature falls back to
the initial guess `T_interface_init` from params.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    params = json.loads(Path("params.json").read_text())
    partner = params.get("partner", "ngsolve")

    imports = {}
    if Path("imports.json").exists():
        try:
            imports = json.loads(Path("imports.json").read_text())
        except json.JSONDecodeError:
            imports = {}

    # Interface temperature samples (x, T) from the partner, else initial guess.
    samples = []
    if partner in imports and imports[partner].get("n_points", 0) > 0:
        d = imports[partner]
        xs = [float(c[0]) for c in d["coordinates"]]
        Ts = [float(v) for v in d["values"]]
        samples = sorted(zip(xs, Ts))
    if not samples:
        t0 = float(params["T_interface_init"])
        samples = [(float(params["x_min"]), t0), (float(params["x_max"]), t0)]

    header = ("{k} {x_min} {x_max} {y_min} {y_max} {T_dirichlet} "
              "{nx} {ny} {degree}").format(**params)
    lines = [header, str(len(samples))]
    lines += [f"{x:.16g} {T:.16g}" for x, T in samples]
    Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

    env = dict(os.environ)
    extra_ld = params.get("ld_library_path")
    if extra_ld:
        env["LD_LIBRARY_PATH"] = (
            str(extra_ld) + os.pathsep + env.get("LD_LIBRARY_PATH", ""))

    r = subprocess.run([params["exe"], "dealii_input.txt", "dealii_output.txt"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not Path("dealii_output.txt").exists():
        sys.stderr.write("deal.II solver failed (rc=%s)\n%s\n%s\n"
                         % (r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
        sys.exit(1)

    y_if = float(params["y_max"])
    coords, temps, fluxes = [], [], []
    for line in Path("dealii_output.txt").read_text().splitlines():
        tok = line.split()
        if len(tok) != 3:
            continue
        coords.append([float(tok[0]), y_if])
        temps.append(float(tok[1]))
        fluxes.append(float(tok[2]))

    export = {
        "field_name": "temperature",
        "n_points": len(coords),
        "coordinates": coords,
        "values": temps,
        "normal_fluxes": fluxes,
    }
    Path("exports.json").write_text(json.dumps(export, indent=2))


if __name__ == "__main__":
    main()
