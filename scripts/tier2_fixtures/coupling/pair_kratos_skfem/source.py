"""Kratos on the DIRICHLET side of a real cross-code coupling — the starred
cell in the sides table, closed.

THE CLAIM UNDER TEST. The sides table said Kratos "yes*" in both columns and
explained the star as "proven in a SEPARATE Kratos install, not in OASiS's
interpreter here", which is a claim nothing on this machine could re-run. The
Neumann half is re-runnable — `tests/test_coupling_pair_fourc_kratos.py` drives
the real 4C binary against real Kratos ConvectionDiffusion and checks an
analytic series-resistance answer. The DIRICHLET half had nothing.

This runs the SHIPPED Kratos participant — the exact text
`knowledge(topic='coupling', solver='kratos')` serves — as the Dirichlet side of
the placeholder conduction problem, against a scikit-fem Neumann side, through
the REGISTERED `couple` tool, and checks the interface temperature, the
interface flux, their signs, conservation and the un-split monolithic re-solve.

THE INTERPRETER IS THE WHOLE STORY, and it is the point the payload makes. The
`command` field is an argv list, so a participant may run under ANY interpreter;
Kratos does not have to be importable where OASiS itself runs. This fixture
resolves a Kratos-capable Python by TRYING candidates and importing
ConvectionDiffusionApplication in each, and prints which one won — because
`discover(query='list')` probes Kratos in OASiS's own interpreter and so can
report it unavailable on a machine where the coupling works, and an agent that
believes `discover` there will not attempt this at all.

The two sides use different interpreters AND different interface
discretisations, which is what a partitioned coupling is for.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

# Which cells of the served sides table this fixture establishes. Read by
# sides_table_backed_by_runs: this run does not go through the arrangement
# helpers (Kratos needs its own interpreter and its own edit block), so the
# ast-parsed route cannot see it.
SIDES_COVERED = [("kratos", "dirichlet"), ("skfem", "neumann")]

# The applications the conduction participant needs. A core-only Kratos has no
# thermal element at all, which is the payload's first trap.
NEEDED = "import KratosMultiphysics, KratosMultiphysics.ConvectionDiffusionApplication, numpy"

# Kratos' own mesh on its subdomain. Deliberately not the partner's.
KRATOS_NX, KRATOS_NY = 32, 32
SKFEM_MESH = (14, 12)


def kratos_interpreter() -> str:
    """The first interpreter on this host that can actually run the shipped
    Kratos participant. Never a hard-coded path taken on faith."""
    cands = [sys.executable]
    params = (L.REPO_ROOT / "benchmarks" / "coupling_pairs" /
              "fourc_kratos_cht" / "params.json")
    if params.is_file():
        try:
            p = json.loads(params.read_text()).get("kratos_python")
            if p:
                cands.append(p)
        except (OSError, json.JSONDecodeError):
            pass
    cands += ["/usr/bin/python3", shutil.which("python3") or ""]
    tried = []
    for c in cands:
        if not c or not (Path(c).exists() or shutil.which(c)):
            continue
        try:
            r = subprocess.run([c, "-c", NEEDED], capture_output=True,
                               text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as e:
            tried.append(f"{c}: {e}")
            continue
        if r.returncode == 0:
            return c
        tried.append(f"{c}: {(r.stderr or '').strip().splitlines()[-1][:90]}")
    raise L.Absent("no interpreter on this host can import Kratos with "
                   "ConvectionDiffusionApplication: " + " | ".join(tried))


def stage_kratos(root: Path, name: str, interp: str, p: L.Problem,
                 partner: str) -> dict:
    """The shipped Kratos script, configured for the LEFT subdomain."""
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    text = L.edit(L.shipped("kratos").read_text(), {
        "PARTNER": json.dumps(partner),
        "X0, X1": f"{p.xl}, {p.xi}",
        "H": f"{p.y1 - p.y0}",
        "K": f"{p.kl}",
        "T_OUTER": f"{p.tl}",
        "T_INIT": "310.0",
        "nx, ny": f"{KRATOS_NX}, {KRATOS_NY}",
    })
    (wd / f"participant_{name}.py").write_text(text)
    print(f"participant[{name}]=kratos interpreter={interp}")
    return {"name": name, "command": [interp, f"participant_{name}.py"],
            "work_dir": str(wd), "timeout": 900}


def body() -> None:
    L.require_available("skfem")
    interp = kratos_interpreter()
    print(f"kratos_interpreter_found={bool(interp)}")
    print(f"kratos_runs_outside_the_oasis_interpreter="
          f"{Path(interp).resolve() != Path(sys.executable).resolve()}")

    p = L.DEFAULT
    tag = "pair_kratos_skfem"
    theta = p.theta_opt("left")
    print(f"--- {tag}: left=kratos/dirichlet right=skfem/neumann "
          f"theta={theta:.6f}")
    root = L.workroot(tag)
    left = stage_kratos(root, "left", interp, p, "right")
    right = L.stage(root, "right", "skfem",
                    L.heat_edits(p, "right", "neumann", "left", SKFEM_MESH))
    res = L.pair([left, right], max_iter=200, tol=1e-8,
                 accelerator="constant", theta=theta)
    if not L.check(bool(res.get("converged")), f"{tag}_did_not_converge",
                   str(res.get("error"))[:300]):
        return
    print(f"{tag}_iterations={res['iterations']}")
    print(f"{tag}_residual={res['residual']:.3e}")
    print(f"{tag}_converged=True")
    # The shipped Kratos script recovers its interface flux with a ONE-SIDED
    # difference over one cell. For this problem the exact field is linear in x,
    # so that difference is exact and no allowance is needed; on a problem with
    # curvature it would be first-order and the tolerance would have to reflect
    # the mesh.
    L._check_interface_physics(tag, res, p.t_iface, p.q, 1e-3, 5e-3)
    ex = res["exports"]
    L.assert_against_monolithic(
        tag, 0.5 * sum(L.span(ex["left"]["values"])),
        0.5 * sum(L.span(ex["left"]["normal_fluxes"])), p)
    print("claims_checked=1")


L.main(body)
