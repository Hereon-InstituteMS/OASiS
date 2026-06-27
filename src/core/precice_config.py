"""
preCICE coupling — a first-class member of OASiS.

preCICE (precice.org) is the field-standard library for black-box coupling of
independent simulation codes. In OASiS it is the native coupling path for genuinely
forced multi-paradigm couplings — e.g. a DSMC particle code (SPARTA) coupled to a FEM
solid (FEniCS) for conjugate heat transfer — that the file-handshake driver cannot
express as cleanly.

This module (1) detects the preCICE Python bindings + C++ library, (2) generates valid
preCICE-config.xml for standard coupling scenarios (heat DN, TSI, and generic), and
(3) verifies an end-to-end coupling actually runs. The libprecice shared library lives
at PRECICE_LIB_DIR (default /opt/precice/lib); pyprecice must match its version.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("oasis.precice")

# Location of libprecice.so.* — pyprecice loads it at import time.
PRECICE_LIB_DIR = os.environ.get("PRECICE_LIB_DIR", "/opt/precice/lib")


def _ensure_lib_on_path() -> None:
    """Make libprecice loadable for the pyprecice import.

    Sets LD_LIBRARY_PATH (for child processes) AND ctypes-preloads libprecice into the
    current process with RTLD_GLOBAL — necessary because changing LD_LIBRARY_PATH at
    runtime does NOT affect the already-initialized dynamic linker of this process.
    """
    if not Path(PRECICE_LIB_DIR).exists():
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if PRECICE_LIB_DIR not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (PRECICE_LIB_DIR + ":" + cur).rstrip(":")
    import ctypes
    for cand in ("libprecice.so.3", "libprecice.so"):
        try:
            ctypes.CDLL(str(Path(PRECICE_LIB_DIR) / cand), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def check_precice_available() -> tuple[bool, str]:
    """Check that the preCICE Python bindings + C++ library are usable.

    Returns (ok, message). Sets LD_LIBRARY_PATH so the bindings can find libprecice.
    """
    _ensure_lib_on_path()
    try:
        import precice
        info = precice.get_version_information()
        ver = info.decode().split(";")[0] if isinstance(info, bytes) else str(info)
        return True, f"preCICE {ver} (lib at {PRECICE_LIB_DIR})"
    except ImportError as e:
        return False, (
            f"preCICE Python bindings not importable ({e}). Install a pyprecice whose "
            f"version matches libprecice in {PRECICE_LIB_DIR}: pip install 'pyprecice==<libver>'"
        )
    except Exception as e:
        return False, f"preCICE present but unusable: {e}"


def generate_precice_config(
    participants: list,
    data: list,
    exchanges: list,
    *,
    scheme: str = "serial-explicit",
    dimensions: int = 2,
    time_window: float = 1.0,
    max_time: float = 10.0,
    max_iterations: int = 20,
    convergence_tol: float = 1e-6,
    acceleration: dict = None,
    mapping: str = "nearest-neighbor",
) -> str:
    """Generate a preCICE XML config for an ARBITRARY cross-code coupling.

    Fully general — any number of participants, any data fields, any exchange pattern.
    This is the backend for all coupling scenarios (heat, TSI, FSI, DSMC-FEM, ...).

    Args:
        participants: list of dicts, one per coupled code:
            {"name": str, "mesh": str, "writes": [data_name, ...], "reads": [data_name, ...]}
        data: list of dicts describing exchanged fields:
            {"name": str, "type": "scalar" | "vector"}
        exchanges: list of dicts, one per coupled field transfer:
            {"data": str, "from": participant_name, "to": participant_name}
        scheme: "serial-explicit" | "serial-implicit" | "parallel-explicit" | "parallel-implicit"
        dimensions: spatial dimension of the coupling meshes (2 or 3)
        time_window, max_time: coupling time control
        max_iterations, convergence_tol: implicit-scheme controls (ignored for explicit)
        acceleration: {"type": "aitken"|"IQN-ILS", "data": data_name, "mesh": mesh_name,
                       "initial_relaxation": float} — for implicit schemes
        mapping: "nearest-neighbor" | "nearest-projection" | "rbf"

    Returns:
        Complete preCICE XML configuration as a string.
    """
    implicit = "implicit" in scheme
    names = [p["name"] for p in participants]
    mesh_of = {p["name"]: p["mesh"] for p in participants}
    # writer (participant, mesh) of each data field, from the exchange list
    writer_mesh = {}
    for ex in exchanges:
        writer_mesh[ex["data"]] = (ex["from"], mesh_of[ex["from"]])

    # --- <data> tags ---
    data_xml = "\n".join(
        f'  <data:{d.get("type","scalar")} name="{d["name"]}" />' for d in data)

    # --- <mesh> tags (each participant provides one mesh, using all data it touches) ---
    mesh_xml = []
    for p in participants:
        used = sorted(set(p.get("writes", []) + p.get("reads", [])))
        uses = "".join(f'\n    <use-data name="{dn}" />' for dn in used)
        mesh_xml.append(f'  <mesh name="{p["mesh"]}" dimensions="{dimensions}">{uses}\n  </mesh>')
    mesh_xml = "\n".join(mesh_xml)

    # --- <participant> tags ---
    part_xml = []
    for p in participants:
        lines = [f'  <participant name="{p["name"]}">',
                 f'    <provide-mesh name="{p["mesh"]}" />']
        # meshes this participant must receive (sources of data it reads)
        recv = {}
        for dn in p.get("reads", []):
            src_p, src_m = writer_mesh[dn]
            recv.setdefault((src_p, src_m), []).append(dn)
        for (src_p, src_m) in recv:
            lines.append(f'    <receive-mesh name="{src_m}" from="{src_p}" />')
        for dn in p.get("writes", []):
            lines.append(f'    <write-data name="{dn}" mesh="{p["mesh"]}" />')
        for dn in p.get("reads", []):
            lines.append(f'    <read-data name="{dn}" mesh="{p["mesh"]}" />')
        for (src_p, src_m) in recv:
            lines.append(f'    <mapping:{mapping} direction="read" from="{src_m}" '
                         f'to="{p["mesh"]}" constraint="consistent" />')
        lines.append("  </participant>")
        part_xml.append("\n".join(lines))
    part_xml = "\n".join(part_xml)

    # --- m2n connections (pairwise between participants that exchange) ---
    pairs = set()
    for ex in exchanges:
        pairs.add(tuple(sorted((ex["from"], ex["to"]))))
    m2n_xml = "\n".join(
        f'  <m2n:sockets acceptor="{a}" connector="{b}" exchange-directory="." />'
        for (a, b) in sorted(pairs))

    # --- coupling scheme ---
    exch_xml = "\n".join(
        f'    <exchange data="{ex["data"]}" mesh="{writer_mesh[ex["data"]][1]}" '
        f'from="{ex["from"]}" to="{ex["to"]}" />' for ex in exchanges)
    cs = [f'  <coupling-scheme:{scheme}>',
          f'    <time-window-size value="{time_window}" />',
          f'    <max-time value="{max_time}" />',
          f'    <participants first="{names[0]}" second="{names[1]}" />',
          exch_xml]
    if implicit:
        cs.append(f'    <max-iterations value="{max_iterations}" />')
        conv_data = exchanges[0]["data"]
        cs.append(f'    <relative-convergence-measure limit="{convergence_tol}" '
                  f'data="{conv_data}" mesh="{writer_mesh[conv_data][1]}" />')
        acc = acceleration or {"type": "aitken", "data": conv_data,
                               "mesh": writer_mesh[conv_data][1], "initial_relaxation": 0.5}
        cs.append(f'    <acceleration:{acc["type"]}>')
        cs.append(f'      <data mesh="{acc["mesh"]}" name="{acc["data"]}" />')
        cs.append(f'      <initial-relaxation value="{acc.get("initial_relaxation",0.5)}" />')
        cs.append(f'    </acceleration:{acc["type"]}>')
    cs.append(f'  </coupling-scheme:{scheme}>')
    cs_xml = "\n".join(cs)

    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<precice-configuration>\n'
        '  <!-- generated by OASiS generate_precice_config (general) -->\n'
        f'{data_xml}\n\n{mesh_xml}\n\n{part_xml}\n\n{m2n_xml}\n\n{cs_xml}\n'
        '</precice-configuration>\n'
    )


def generate_heat_coupling_config(
    mesh_name_a: str = "FEniCS-Mesh",
    mesh_name_b: str = "4C-Mesh",
    data_name: str = "Temperature",
    flux_name: str = "Heat-Flux",
    max_iterations: int = 20,
    convergence_tol: float = 1e-6,
    relaxation: float = 0.5,
    time_window: float = 1.0,
    output_dir: str = "precice-output",
) -> str:
    """Generate preCICE XML config for Dirichlet-Neumann heat coupling.

    This produces a configuration equivalent to our MCP-orchestrated DN
    domain decomposition, enabling direct comparison.

    Args:
        mesh_name_a: Mesh name for the Dirichlet participant (FEniCS).
        mesh_name_b: Mesh name for the Neumann participant (4C).
        data_name: Name of the temperature data field.
        flux_name: Name of the heat flux data field.
        max_iterations: Maximum implicit coupling iterations.
        convergence_tol: Convergence criterion for the coupling.
        relaxation: Initial relaxation factor for Aitken acceleration.
        time_window: Size of coupling time window.
        output_dir: Directory for preCICE output.

    Returns:
        Complete preCICE XML configuration as a string.
    """
    return generate_precice_config(
        participants=[
            {"name": "Dirichlet", "mesh": mesh_name_a,
             "writes": [data_name], "reads": [flux_name]},
            {"name": "Neumann", "mesh": mesh_name_b,
             "writes": [flux_name], "reads": [data_name]},
        ],
        data=[{"name": data_name, "type": "scalar"},
              {"name": flux_name, "type": "scalar"}],
        exchanges=[{"data": data_name, "from": "Dirichlet", "to": "Neumann"},
                   {"data": flux_name, "from": "Neumann", "to": "Dirichlet"}],
        scheme="serial-implicit", dimensions=2, time_window=time_window,
        max_time=time_window, max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        acceleration={"type": "aitken", "data": data_name, "mesh": mesh_name_a,
                      "initial_relaxation": relaxation},
    )


def generate_tsi_coupling_config(
    mesh_name_a: str = "Thermal-Mesh",
    mesh_name_b: str = "Structure-Mesh",
    temperature_name: str = "Temperature",
    displacement_name: str = "Displacement",
    heat_flux_name: str = "Heat-Flux",
    traction_name: str = "Force",
    max_iterations: int = 30,
    convergence_tol: float = 1e-6,
    time_window: float = 1.0,
) -> str:
    """Generate preCICE XML for thermal-structural interaction coupling.

    Two-way coupling: temperature and displacement exchanged between solvers.

    Returns:
        Complete preCICE XML configuration string.
    """
    return generate_precice_config(
        participants=[
            {"name": "Thermal", "mesh": mesh_name_a,
             "writes": [temperature_name], "reads": [displacement_name]},
            {"name": "Structure", "mesh": mesh_name_b,
             "writes": [displacement_name], "reads": [temperature_name]},
        ],
        data=[{"name": temperature_name, "type": "scalar"},
              {"name": displacement_name, "type": "vector"}],
        exchanges=[{"data": temperature_name, "from": "Thermal", "to": "Structure"},
                   {"data": displacement_name, "from": "Structure", "to": "Thermal"}],
        scheme="serial-implicit", dimensions=3, time_window=time_window,
        max_time=time_window, max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        acceleration={"type": "aitken", "data": temperature_name, "mesh": mesh_name_a,
                      "initial_relaxation": 0.5},
    )


def save_precice_config(config_xml: str, output_dir: Path, filename: str = "precice-config.xml") -> Path:
    """Save preCICE configuration to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(config_xml)
    logger.info(f"preCICE config saved to {path}")
    return path


# Verification config — built via the GENERAL generator so the general path is exercised.
_VERIFY_CONFIG = generate_precice_config(
    participants=[{"name": "A", "mesh": "A-Mesh", "writes": ["Val"], "reads": []},
                  {"name": "B", "mesh": "B-Mesh", "writes": [], "reads": ["Val"]}],
    data=[{"name": "Val", "type": "scalar"}],
    exchanges=[{"data": "Val", "from": "A", "to": "B"}],
    scheme="serial-explicit", dimensions=2, time_window=1.0, max_time=3.0)

_VERIFY_PARTICIPANT = """\
import sys, numpy as np, precice
name, other = sys.argv[1], sys.argv[2]
p = precice.Participant(name, "precice-config.xml", 0, 1)
mesh = f"{name}-Mesh"
vid = p.set_mesh_vertices(mesh, np.array([[0.0, 0.0]]))
p.initialize()
while p.is_coupling_ongoing():
    dt = p.get_max_time_step_size()
    if name == "B":
        v = p.read_data(mesh, "Val", vid, dt)
        print(f"B received {float(v[0])}", flush=True)
    if name == "A":
        p.write_data(mesh, "Val", vid, np.array([42.0]))
    p.advance(dt)
p.finalize()
"""


def run_precice_coupling(
    participants: list,
    data: list,
    exchanges: list,
    work_dir: Path,
    *,
    scheme: str = "serial-explicit",
    dimensions: int = 2,
    max_time: float = 10.0,
    time_window: float = 1.0,
    timeout: int = 1800,
    extra_env: dict = None,
    **config_kw,
) -> dict:
    """Run a GENERAL preCICE coupling of arbitrary codes, end-to-end.

    OASiS generates the preCICE config and launches every participant's solver command,
    then waits for the coupling to finish. This is the physics-agnostic, code-agnostic
    coupling orchestrator — the same call drives heat DN, TSI, FSI, DSMC<->FEM, etc.

    Args:
        participants: one dict per coupled code, combining the config spec and how to run it:
            {"name": str, "mesh": str, "writes": [data], "reads": [data], "command": [argv...]}
        data:      [{"name": str, "type": "scalar"|"vector"}]
        exchanges: [{"data": str, "from": name, "to": name}]
        work_dir:  directory to run in (config + participant cwd)
        scheme/dimensions/max_time/time_window/**config_kw: passed to generate_precice_config
        extra_env: extra environment for participant processes (e.g. LD_LIBRARY_PATH for
                   libprecice / solver libraries)

    Returns:
        {"converged": bool, "returncodes": {name: rc}, "config": path, "logs": {name: tail}}
    """
    import subprocess
    _ensure_lib_on_path()
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    cfg = generate_precice_config(
        participants=[{k: p[k] for k in ("name", "mesh", "writes", "reads")} for p in participants],
        data=data, exchanges=exchanges, scheme=scheme, dimensions=dimensions,
        max_time=max_time, time_window=time_window, **config_kw)
    cfg_path = work_dir / "precice-config.xml"
    cfg_path.write_text(cfg)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PRECICE_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    if extra_env:
        for k, v in extra_env.items():
            env[k] = v + ":" + env.get(k, "") if k.endswith("PATH") else v
    procs = {}
    for p in participants:
        lf = open(work_dir / f"{p['name']}.out", "w")
        procs[p["name"]] = (subprocess.Popen(p["command"], cwd=work_dir, env=env,
                                             stdout=lf, stderr=subprocess.STDOUT, text=True), lf)
    rcs, logs = {}, {}
    try:
        for name, (proc, lf) in procs.items():
            proc.wait(timeout=timeout)
            rcs[name] = proc.returncode
            lf.close()
            logs[name] = (work_dir / f"{name}.out").read_text(errors="replace")[-600:]
    except subprocess.TimeoutExpired:
        for proc, lf in procs.values():
            proc.kill(); lf.close()
        return {"converged": False, "returncodes": rcs, "config": str(cfg_path),
                "error": f"coupling timed out after {timeout}s", "logs": logs}
    converged = all(rc == 0 for rc in rcs.values())
    return {"converged": converged, "returncodes": rcs, "config": str(cfg_path), "logs": logs}


def verify_precice_coupling(work_dir: Path = None, timeout: int = 60) -> tuple[bool, str]:
    """Run a real 2-participant preCICE coupling and confirm data is exchanged.

    Launches participants A and B (A writes 42.0, B reads it) over 3 time windows via
    the preCICE sockets m2n. Returns (ok, message). This is the complete end-to-end
    verification that preCICE works in this environment, not just that it imports.
    """
    import subprocess, sys, tempfile, time
    _ensure_lib_on_path()
    wd = Path(work_dir or tempfile.mkdtemp(prefix="precice_verify_"))
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "precice-config.xml").write_text(_VERIFY_CONFIG)
    (wd / "participant.py").write_text(_VERIFY_PARTICIPANT)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PRECICE_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    py = sys.executable
    try:
        pa = subprocess.Popen([py, "participant.py", "A", "B"], cwd=wd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pb = subprocess.Popen([py, "participant.py", "B", "A"], cwd=wd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_b, _ = pb.communicate(timeout=timeout)
        pa.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pa.kill(); pb.kill()
        return False, "preCICE coupling timed out (deadlock?)"
    except Exception as e:
        return False, f"preCICE coupling launch failed: {e}"
    got = out_b.count("B received 42.0")
    if pa.returncode == 0 and pb.returncode == 0 and got >= 1:
        return True, f"preCICE coupling verified: B received the coupled value over {got} window(s)"
    return False, f"preCICE coupling failed (rc_A={pa.returncode}, rc_B={pb.returncode}); output:\n{out_b[-500:]}"
