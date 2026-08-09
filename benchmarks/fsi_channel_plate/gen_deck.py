#!/usr/bin/env python3
"""
Generate a 2D monolithic ALE-FSI 4C deck:

  rigid-walled channel, TOP wall is a thin elastic plate

  Fluid     : 0 <= x <= L, 0 <= y <= H            (incompressible Navier-Stokes, ALE)
  Structure : 0 <= x <= L, H <= y <= H + TS       (plane-strain wall element)
  Interface : y = H, 0 <= x <= L                  (node matching)

  Fluid BCs : inlet  x=0  parabolic vx(y) = 6*U*y*(H-y)/H^2, vy = 0 (smooth ramp in time)
              bottom y=0  no-slip
              outlet x=L  "do nothing" (no Dirichlet at all)
  Struct BCs: clamped on x=0 and x=L, top edge y=H+TS traction free

Design-entity (DLINE) numbering -- one global numbering shared by both fields,
exactly as in tests/input_files/fsi_dc_mono_*.4C.yaml:

  DLINE 1  structure clamped ends            (x=0 and x=L)
  DLINE 2  structure FSI interface           (y=H)
  DLINE 3  fluid inlet                       (x=0)
  DLINE 4  fluid bottom no-slip wall         (y=0)
  DLINE 5  fluid FSI interface               (y=H)
  DLINE 6  fluid outlet                      (x=L)  -> ALE Dirichlet only

ALE is cloned from the fluid via CLONING MATERIAL MAP (no ALE ELEMENTS block).

Element node ordering is counter-clockwise (bl, br, tr, tl), which is what the
reference decks use (verified against fsi_dc_mono_ss_ost_ga_eos.4C.yaml).
"""
import argparse
import json
import os

# ----------------------------------------------------------------------------- geometry
L = 1.0        # channel length
H = 0.2        # channel height (fluid)
TS = 0.05      # plate thickness (structure)

U_MEAN = 1.0   # mean inflow velocity
MU_F = 1.0     # fluid dynamic viscosity
RHO_F = 1.0    # fluid density
E_S = 3.0e6    # structure Young's modulus
NU_S = 0.3     # structure Poisson ratio
RHO_S = 1.0    # structure density


def build(nx_f, ny_f, nx_s, ny_s, dt, nstep, tau, kinem, coupalgo, ale_type,
          out_yaml, out_meta, eas="none", young=None, tol=1e-9):
    global E_S
    if young is not None:
        E_S = young
    assert nx_f == nx_s, "interface must be node matching -> nx_f == nx_s"

    # ---------------------------------------------------------------- structure nodes
    # id 1 .. (nx_s+1)*(ny_s+1);  row-major, j (y) outer, i (x) inner
    s_id = {}
    nodes = []          # (id, x, y)
    nid = 0
    for j in range(ny_s + 1):
        y = H + TS * j / ny_s
        for i in range(nx_s + 1):
            x = L * i / nx_s
            nid += 1
            s_id[(i, j)] = nid
            nodes.append((nid, x, y))
    n_struct = nid

    # ---------------------------------------------------------------- fluid nodes
    f_id = {}
    for j in range(ny_f + 1):
        y = H * j / ny_f
        for i in range(nx_f + 1):
            x = L * i / nx_f
            nid += 1
            f_id[(i, j)] = nid
            nodes.append((nid, x, y))
    n_total = nid

    # ---------------------------------------------------------------- elements (CCW)
    struct_ele = []
    eid = 0
    for j in range(ny_s):
        for i in range(nx_s):
            eid += 1
            n = (s_id[(i, j)], s_id[(i + 1, j)], s_id[(i + 1, j + 1)], s_id[(i, j + 1)])
            struct_ele.append(
                f"{eid} WALL QUAD4 {n[0]} {n[1]} {n[2]} {n[3]} MAT 1 "
                f"KINEM {kinem} EAS {eas} THICK 1.0 STRESS_STRAIN plane_strain GP 2 2")
    fluid_ele = []
    for j in range(ny_f):
        for i in range(nx_f):
            eid += 1
            n = (f_id[(i, j)], f_id[(i + 1, j)], f_id[(i + 1, j + 1)], f_id[(i, j + 1)])
            fluid_ele.append(f"{eid} FLUID QUAD4 {n[0]} {n[1]} {n[2]} {n[3]} MAT 2 NA ALE")

    # ---------------------------------------------------------------- design lines
    dl = {k: [] for k in range(1, 7)}
    # 1: structure clamped ends x=0 and x=L (all j, including the interface corners --
    #    the reference deck likewise lets the clamp DLINE share corner nodes with the
    #    FSI interface DLINE)
    for j in range(ny_s + 1):
        dl[1].append(s_id[(0, j)])
        dl[1].append(s_id[(nx_s, j)])
    # 2: structure FSI interface y=H  (j == 0)
    iface_struct = [s_id[(i, 0)] for i in range(nx_s + 1)]
    dl[2] = list(iface_struct)
    # 3: fluid inlet x=0
    dl[3] = [f_id[(0, j)] for j in range(ny_f + 1)]
    # 4: fluid bottom wall y=0
    dl[4] = [f_id[(i, 0)] for i in range(nx_f + 1)]
    # 5: fluid FSI interface y=H (j == ny_f)
    iface_fluid = [f_id[(i, ny_f)] for i in range(nx_f + 1)]
    dl[5] = list(iface_fluid)
    # 6: fluid outlet x=L
    dl[6] = [f_id[(nx_f, j)] for j in range(ny_f + 1)]

    dline_lines = []
    for d in sorted(dl):
        for n in sorted(set(dl[d])):
            dline_lines.append(f"NODE {n} DLINE {d}")

    # ------------------------------------------------------- slave-interface DBC release
    # 4C forbids Dirichlet BCs on the SLAVE side of the FSI interface
    # (4C_fsi_monolithic{fluid,structure}split.cpp).  The reference decks solve this by
    # a DESIGN POINT DIRICH condition with ONOFF all-zero on the offending corner nodes,
    # which overrides the line condition there.
    #   iter_monolithicfluidsplit      -> MASTER = structure, SLAVE = fluid
    #   iter_monolithicstructuresplit  -> MASTER = fluid,     SLAVE = structure
    if coupalgo == "iter_monolithicfluidsplit":
        # only fluid interface node carrying a DBC is the inlet/interface corner (0, H)
        dnode_nodes = [f_id[(0, ny_f)]]
        pnt_numdof, pnt_onoff, pnt_val = 3, "[0, 0, 0]", "[0, 0, 0]"
        predict = "ConstVel"
    elif coupalgo == "iter_monolithicstructuresplit":
        # both structure interface corners are clamped by DLINE 1 -> release them
        dnode_nodes = [s_id[(0, 0)], s_id[(nx_s, 0)]]
        pnt_numdof, pnt_onoff, pnt_val = 2, "[0, 0]", "[0, 0]"
        predict = "ConstDisVelAcc"
    else:
        raise SystemExit(f"unsupported coupalgo {coupalgo}")
    dnode_lines = [f"NODE {n} DNODE 1" for n in dnode_nodes]
    point_dirich = (
        "DESIGN POINT DIRICH CONDITIONS:\n"
        "  - E: 1\n"
        f"    NUMDOF: {pnt_numdof}\n"
        f"    ONOFF: {pnt_onoff}\n"
        f"    VAL: {pnt_val}\n"
        f"    FUNCT: {pnt_onoff}\n")

    t_end = dt * nstep

    # ---------------------------------------------------------------- deck
    def yl(items, indent="  "):
        return "\n".join(f'{indent}- "{s}"' for s in items)

    tol_block = "\n".join(
        f"  {k}: {tol}"
        for k in ["TOL_DIS_RES_L2", "TOL_DIS_RES_INF", "TOL_DIS_INC_L2", "TOL_DIS_INC_INF",
                  "TOL_FSI_RES_L2", "TOL_FSI_RES_INF", "TOL_FSI_INC_L2", "TOL_FSI_INC_INF",
                  "TOL_PRE_RES_L2", "TOL_PRE_RES_INF", "TOL_PRE_INC_L2", "TOL_PRE_INC_INF",
                  "TOL_VEL_RES_L2", "TOL_VEL_RES_INF", "TOL_VEL_INC_L2", "TOL_VEL_INC_INF"])

    # parabolic profile, smooth exponential ramp -> exactly steady for t >> tau
    amp = 6.0 * U_MEAN / (H * H)          # vx = amp * y * (H - y)
    inflow = f"{amp:.10g}*y*({H}-y)*(1.0-exp(-t/{tau}))"

    deck = f"""TITLE:
  - "2D monolithic ALE-FSI: rigid channel with thin elastic TOP plate"
  - "Fluid {nx_f}x{ny_f} QUAD4 on [0,{L}]x[0,{H}], mu={MU_F}, rho={RHO_F}"
  - "Structure {nx_s}x{ny_s} WALL QUAD4 on [0,{L}]x[{H},{H + TS}], E={E_S}, nu={NU_S}, rho={RHO_S}"
  - "Inflow parabolic, U_MEAN={U_MEAN}; outlet do-nothing; run to steady state"
  - "FSI: {coupalgo}, UMFPACK direct"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid_Structure_Interaction"
IO:
  VERBOSITY: "Standard"
  STRUCT_DISP: true
  FLUID_STRESS: true
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
  COMPRESSION_LEVEL: no_compression
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
  PRESSURE: true
  DISPLACEMENT: true
  NODE_GID: true
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "GenAlpha"
  PREDICT: "{predict}"
  LINEAR_SOLVER: 1
  TIMESTEP: {dt}
  NUMSTEP: {nstep}
  MAXTIME: {t_end}
  RESULTSEVERY: 1
STRUCTURAL DYNAMIC/GENALPHA:
  BETA: 0.5
  GAMMA: 1
  ALPHA_M: 0
  ALPHA_F: 0
  RHO_INF: -1
FLUID DYNAMIC:
  PHYSICAL_TYPE: "Incompressible"
  TIMEINTEGR: "One_Step_Theta"
  NONLINITER: Newton
  LINEAR_SOLVER: 1
  THETA: 1
  TIMESTEP: {dt}
  NUMSTEP: {nstep}
  MAXTIME: {t_end}
  RESULTSEVERY: 1
  ITEMAX: 20
FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
  CHARELELENGTH_PC: "root_of_volume"
ALE DYNAMIC:
  ALE_TYPE: {ale_type}
  UPDATEMATRIX: true
  LINEAR_SOLVER: 1
  TIMESTEP: {dt}
  NUMSTEP: {nstep}
  MAXTIME: {t_end}
  RESULTSEVERY: 1
FSI DYNAMIC:
  COUPALGO: "{coupalgo}"
  TIMESTEP: {dt}
  NUMSTEP: {nstep}
  MAXTIME: {t_end}
  RESULTSEVERY: 1
  RESTARTEVERY: {nstep}
FSI DYNAMIC/MONOLITHIC SOLVER:
  ADAPTIVEDIST: 0
  BASETOL: 1e-08
  ITEMAX: 30
  LINEARBLOCKSOLVER: "LinalgSolver"
  LINEAR_SOLVER: 1
  SHAPEDERIVATIVES: true
{tol_block}
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E_S}
      NUE: {NU_S}
      DENS: {RHO_S}
  - MAT: 2
    MAT_fluid:
      DYNVISCOSITY: {MU_F}
      DENSITY: {RHO_F}
  - MAT: 3
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1
      NUE: 0
      DENS: 0
CLONING MATERIAL MAP:
  - SRC_FIELD: "fluid"
    SRC_MAT: 2
    TAR_FIELD: "ale"
    TAR_MAT: 3
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{inflow}"
{point_dirich}DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1, 0, 0]
    FUNCT: [1, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN LINE ALE DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
  - E: 6
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DESIGN FSI COUPLING LINE CONDITIONS:
  - E: 2
    coupling_id: 1
  - E: 5
    coupling_id: 1
DNODE-NODE TOPOLOGY:
{yl(dnode_lines)}
DLINE-NODE TOPOLOGY:
{yl(dline_lines)}
NODE COORDS:
{yl([f"NODE {n} COORD {x:.16e} {y:.16e} {0.0:.16e}" for n, x, y in nodes])}
STRUCTURE ELEMENTS:
{yl(struct_ele)}
FLUID ELEMENTS:
{yl(fluid_ele)}
"""
    with open(out_yaml, "w") as f:
        f.write(deck)

    meta = {
        "deck": os.path.abspath(out_yaml),
        "L": L, "H": H, "TS": TS,
        "U_MEAN": U_MEAN, "MU_F": MU_F, "RHO_F": RHO_F,
        "E_S": E_S, "NU_S": NU_S, "RHO_S": RHO_S,
        "nx_f": nx_f, "ny_f": ny_f, "nx_s": nx_s, "ny_s": ny_s,
        "dt": dt, "nstep": nstep, "t_end": t_end, "tau": tau,
        "kinem": kinem, "coupalgo": coupalgo, "ale_type": ale_type, "eas": eas, "tol": tol,
        "n_struct_nodes": n_struct, "n_total_nodes": n_total,
        "n_struct_ele": nx_s * ny_s, "n_fluid_ele": nx_f * ny_f,
        # global node ids on the interface, ordered by increasing x
        "iface_struct_nodes": iface_struct,
        "iface_struct_x": [L * i / nx_s for i in range(nx_s + 1)],
        "iface_fluid_nodes": iface_fluid,
        "inlet_fluid_nodes": [f_id[(0, j)] for j in range(ny_f + 1)],
        "inlet_fluid_y": [H * j / ny_f for j in range(ny_f + 1)],
        "outlet_fluid_nodes": [f_id[(nx_f, j)] for j in range(ny_f + 1)],
        "inflow_function": inflow,
        "released_dbc_nodes": dnode_nodes,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out_yaml}")
    print(f"  nodes {n_total} (struct 1..{n_struct}, fluid {n_struct+1}..{n_total})"
          f"  elements {nx_s*ny_s + nx_f*ny_f}")
    print(f"  dt={dt} nstep={nstep} t_end={t_end} kinem={kinem} eas={eas} coupalgo={coupalgo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--nyf", type=int, default=10)
    ap.add_argument("--nys", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--nstep", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--kinem", default="nonlinear")
    ap.add_argument("--eas", default="none")
    ap.add_argument("--young", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--coupalgo", default="iter_monolithicstructuresplit")
    ap.add_argument("--ale-type", default="springs_material")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    build(a.nx, a.nyf, a.nx, a.nys, a.dt, a.nstep, a.tau, a.kinem,
          a.coupalgo, a.ale_type, a.out, a.meta, a.eas, a.young, a.tol)
