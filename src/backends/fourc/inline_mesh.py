"""
Inline mesh generation for 4C input files.

Creates NODE COORDS + ELEMENTS sections directly in YAML,
bypassing the need for external Exodus mesh files.
This makes 4C fully self-contained for standard benchmark problems.
"""


def generate_quad4_rectangle(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                              element_section: str = "STRUCTURE",
                              element_type: str = "SOLID QUAD4",
                              element_suffix: str = "MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"):
    """Generate inline QUAD4 mesh on [0,lx]×[0,ly].

    Returns dict with:
        nodes: list of "NODE id COORD x y 0.0" strings
        elements: list of element definition strings
        node_grid: dict (i,j) -> node_id for boundary access
        left_nodes, right_nodes, bottom_nodes, top_nodes: boundary node lists
        all_nodes: all node IDs
        n_nodes, n_elements: counts
    """
    nodes = []
    node_grid = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * lx / nx
            y = j * ly / ny
            nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} 0.0')
            node_grid[(i, j)] = nid
            nid += 1

    elements = []
    eid = 1
    elem_section = element_section.upper()
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elements.append(f'{eid} {element_type} {n1} {n2} {n3} {n4} {element_suffix}')
            eid += 1

    left_nodes = [node_grid[(0, j)] for j in range(ny + 1)]
    right_nodes = [node_grid[(nx, j)] for j in range(ny + 1)]
    bottom_nodes = [node_grid[(i, 0)] for i in range(nx + 1)]
    top_nodes = [node_grid[(i, ny)] for i in range(nx + 1)]
    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_grid": node_grid,
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "geometry_section": f"{elem_section} ELEMENTS",
    }


def matched_poisson_input(nx: int = 32, ny: int = 32) -> str:
    """Poisson -Δu=1 on [0,1]², u=0 on ∂Ω. Matches FEniCS/deal.II setup."""
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson -Δu=1 on [0,1]² — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Boundary topology: 4 lines (edges) + 1 surface (domain)
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_heat_input(nx: int = 32, ny: int = 32, T_left: float = 100.0, T_right: float = 0.0) -> str:
    """Heat conduction T_left on left, T_right on right. Matches FEniCS/deal.II."""
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Heat conduction — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def matched_poisson_rectangle_input(nx: int = 64, ny: int = 32, lx: float = 2.0, ly: float = 1.0) -> str:
    """Poisson -Δu=1 on [0,lx]×[0,ly], u=0 on ∂Ω. Matches FEniCS/deal.II setup."""
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson -Δu=1 on [0,{lx}]×[0,{ly}] — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_heat_rectangle_input(nx: int = 64, ny: int = 32, lx: float = 2.0, ly: float = 1.0,
                                   T_left: float = 100.0, T_right: float = 0.0) -> str:
    """Heat conduction on [0,lx]×[0,ly], T_left on left, T_right on right."""
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Heat conduction on [0,{lx}]×[0,{ly}] — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def generate_l_domain_mesh(n: int = 16):
    """Generate L-shaped domain [-1,1]²\\[0,1]×[-1,0] with QUAD4 elements.

    The L-domain is built from a 2n×2n grid covering [-1,1]², then excluding
    elements in the bottom-right quadrant [0,1]×[-1,0].

    Returns same dict format as generate_quad4_rectangle plus boundary info.
    """
    # Create full grid [-1,1]×[-1,1] with 2n subdivisions
    nn = 2 * n  # total subdivisions per direction
    dx = 2.0 / nn
    dy = 2.0 / nn

    # Generate all nodes (even those in excluded region — we'll skip unused ones)
    # First pass: determine which nodes are used
    used_nodes = set()
    for j in range(nn):
        for i in range(nn):
            # Skip elements in bottom-right quadrant: i >= n and j < n
            if i >= n and j < n:
                continue
            # This element uses 4 corners
            used_nodes.update([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

    # Assign node IDs (sequential)
    node_map = {}
    nid = 1
    for j in range(nn + 1):
        for i in range(nn + 1):
            if (i, j) in used_nodes:
                node_map[(i, j)] = nid
                nid += 1

    # Generate nodes
    nodes = []
    for j in range(nn + 1):
        for i in range(nn + 1):
            if (i, j) in used_nodes:
                x = -1.0 + i * dx
                y = -1.0 + j * dy
                nodes.append(f'NODE {node_map[(i, j)]} COORD {x:.6f} {y:.6f} 0.0')

    # Generate elements
    elements = []
    eid = 1
    for j in range(nn):
        for i in range(nn):
            if i >= n and j < n:
                continue
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append(f'{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std')
            eid += 1

    # Boundary nodes: all nodes on the exterior boundary of the L
    # The L-domain boundary consists of:
    # - Bottom edge: y=-1, x in [-1, 0] (j=0, i=0..n)
    # - Left edge of bottom-right cutout: x=0, y in [-1, 0] (i=n, j=0..n)
    # - Bottom edge of top-right: y=0, x in [0, 1] (j=n, i=n..2n)
    # - Right edge: x=1, y in [0, 1] (i=2n, j=n..2n)
    # - Top edge: y=1, x in [-1, 1] (j=2n, i=0..2n)
    # - Left edge: x=-1, y in [-1, 1] (i=0, j=0..2n)
    boundary_nodes = set()
    # Bottom: y=-1, x in [-1, 0]
    for i in range(n + 1):
        if (i, 0) in node_map:
            boundary_nodes.add(node_map[(i, 0)])
    # Re-entrant vertical: x=0, y in [-1, 0]
    for j in range(n + 1):
        if (n, j) in node_map:
            boundary_nodes.add(node_map[(n, j)])
    # Re-entrant horizontal: y=0, x in [0, 1]
    for i in range(n, nn + 1):
        if (i, n) in node_map:
            boundary_nodes.add(node_map[(i, n)])
    # Right: x=1, y in [0, 1]
    for j in range(n, nn + 1):
        if (nn, j) in node_map:
            boundary_nodes.add(node_map[(nn, j)])
    # Top: y=1, x in [-1, 1]
    for i in range(nn + 1):
        if (i, nn) in node_map:
            boundary_nodes.add(node_map[(i, nn)])
    # Left: x=-1, y in [-1, 1]
    for j in range(nn + 1):
        if (0, j) in node_map:
            boundary_nodes.add(node_map[(0, j)])

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_map": node_map,
        "boundary_nodes": sorted(boundary_nodes),
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
    }


def matched_l_domain_poisson_input(n: int = 16) -> str:
    """Poisson -Δu=1 on L-domain [-1,1]²\\[0,1]×[-1,0], u=0 on ∂Ω.

    Matches FEniCS (Gmsh L-domain) and deal.II (hyper_L) setup.
    """
    mesh = generate_l_domain_mesh(n)

    yaml = '''TITLE:
  - "Poisson on L-domain — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # All boundary nodes go to DLINE 1 (u=0 Dirichlet)
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["boundary_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    # All nodes on DSURFACE 1 for Neumann source
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def generate_hex8_cube(nx: int, ny: int, nz: int,
                       lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
                       element_section: str = "TRANSPORT",
                       element_type: str = "TRANSP HEX8",
                       element_suffix: str = "MAT 1 TYPE Std"):
    """Generate inline HEX8 mesh on [0,lx]×[0,ly]×[0,lz].

    Returns dict with nodes, elements, boundary node sets, etc.
    """
    nodes = []
    node_grid = {}
    nid = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * lx / nx
                y = j * ly / ny
                z = k * lz / nz
                nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} {z:.6f}')
                node_grid[(i, j, k)] = nid
                nid += 1

    elements = []
    eid = 1
    elem_section = element_section.upper()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j, k)]
                n2 = node_grid[(i+1, j, k)]
                n3 = node_grid[(i+1, j+1, k)]
                n4 = node_grid[(i, j+1, k)]
                n5 = node_grid[(i, j, k+1)]
                n6 = node_grid[(i+1, j, k+1)]
                n7 = node_grid[(i+1, j+1, k+1)]
                n8 = node_grid[(i, j+1, k+1)]
                elements.append(
                    f'{eid} {element_type} {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8} {element_suffix}')
                eid += 1

    # Boundary nodes: all nodes on any face of the cube
    boundary_nodes = set()
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                if i == 0 or i == nx or j == 0 or j == ny or k == 0 or k == nz:
                    boundary_nodes.add(node_grid[(i, j, k)])

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_grid": node_grid,
        "boundary_nodes": sorted(boundary_nodes),
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "geometry_section": f"{elem_section} ELEMENTS",
    }


def matched_poisson_3d_input(n: int = 8) -> str:
    """Poisson -Δu=1 on [0,1]³, u=0 on ∂Ω. Matches FEniCS/deal.II 3D setup."""
    mesh = generate_hex8_cube(n, n, n, element_section="TRANSPORT",
                               element_type="TRANSP HEX8",
                               element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson 3D on [0,1]³ — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # All boundary nodes on DSURFACE 1 (u=0 Dirichlet)
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["boundary_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    # All nodes in DVOL 1 for volumetric source
    yaml += 'DVOL-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DVOLUME 1"\n'

    return yaml


def matched_tsi_oneway_input(
    nx: int = 4, ny: int = 4, nz: int = 4,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    E: float = 200e3, nu: float = 0.3, alpha: float = 12e-6,
    T_left: float = 100.0, T_right: float = 0.0, T_ref: float = 0.0,
    conductivity: float = 1.0, capacity: float = 1.0,
    density: float = 1.0,
) -> str:
    """Generate 4C TSI one-way input: thermal expansion of a heated beam.

    3D beam [0,lx]x[0,ly]x[0,lz] with SOLIDSCATRA HEX8 elements.
    Thermal BCs: T_left on x=0, T_right on x=lx, insulated elsewhere.
    Structural BCs: Fix x=0 face (u=0).
    TSI one-way: thermal -> structural (no reverse coupling).

    Used for cross-solver coupling: FEniCS computes temperature, 4C computes
    structural response via native TSI with the same thermal BCs.
    """
    mesh = generate_hex8_cube(
        nx, ny, nz, lx, ly, lz,
        element_section="STRUCTURE",
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )
    ng = mesh["node_grid"]

    # Face node sets for boundary conditions
    left_face = sorted({ng[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1)})
    right_face = sorted({ng[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1)})

    # Temperature function expression for INITIALFIELD
    t_expr = f"{T_left} + ({T_right} - {T_left}) * x / {lx}"

    yaml = f'''TITLE:
  - "TSI one-way: thermal expansion — cross-solver coupling benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 10
  LINEAR_SOLVER: 2
  PREDICT: TangDis
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{t_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    # Node coordinates
    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Surface topology: DSURFACE 1 = left face, DSURFACE 2 = right face
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_elasticity_input(nx: int = 40, ny: int = 4, E: float = 1000.0, nu: float = 0.3,
                               lx: float = 10.0, ly: float = 1.0) -> str:
    """Cantilever beam lx×ly, fixed left, body force (0,-1). Matches FEniCS/deal.II.

    Uses KINEM linear (small-strain St. Venant-Kirchhoff) so that
    the 4C result matches the linear-elasticity formulations the
    FEniCS / deal.II / NGSolve / scikit-fem cantilever tests use.
    With KINEM nonlinear (finite-strain) the slender 10x1 beam
    deflection diverged from the linear answer by 40%+ (audit
    2026-06-01: u_y_max fourc=7.50 vs fenics=12.96 / dealii=13.23
    on a 10x1 cantilever under the same body force).
    """
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="STRUCTURE",
                                     element_type="SOLID QUAD4",
                                     element_suffix="MAT 1 KINEM linear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain")

    yaml = f'''TITLE:
  - "Cantilever {lx}x{ly} — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  # KINEM nonlinear (St. Venant-Kirchhoff finite strain) needs
  # > 2 Newton iters for moderate body loads. With MAXITER=2
  # NOX hits the StatusTest::MaxIters threshold and 4C aborts
  # with MPI_Abort(1). Audit 2026-06-01: the cantilever_10x1
  # e2e test failed in FourC::Solid::Nln::SOLVER::Nox::solve()
  # because the previous default was 2.
  MAXITER: 20
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 0.0
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'''

    # Dirichlet: fix each left-edge node individually via DNODE
    yaml += 'DESIGN POINT DIRICH CONDITIONS:\n'
    for i in range(len(mesh["left_nodes"])):
        yaml += f'''  - E: {i+1}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    # Body force via surface Neumann
    yaml += '''DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DNODE-NODE TOPOLOGY:\n'
    for i, nid in enumerate(mesh["left_nodes"]):
        yaml += f'  - "NODE {nid} DNODE {i+1}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def generate_l_domain_hex8(n: int = 4, lz: float = 0.5,
                           element_type: str = "SOLIDSCATRA HEX8",
                           element_suffix: str = "MAT 1 KINEM linear TYPE Undefined"):
    """Generate 3D L-shaped domain by extruding the 2D L-domain in z.

    L-domain: [-1,1]^2 \\ [0,1]x[-1,0], extruded to thickness lz.
    Uses HEX8 elements (SOLIDSCATRA for TSI or SOLID for pure structure).

    Returns dict with nodes, elements, face node sets, all_nodes, etc.
    """
    nn = 2 * n
    used_nodes_2d = set()
    for j in range(nn):
        for i in range(nn):
            if i >= n and j < n:
                continue
            used_nodes_2d.update([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

    nz = max(n // 2, 1)
    dx = 2.0 / nn
    dy = 2.0 / nn
    dz = lz / nz

    node_map = {}
    nid = 1
    nodes = []
    for k in range(nz + 1):
        for j in range(nn + 1):
            for i in range(nn + 1):
                if (i, j) in used_nodes_2d:
                    x = -1.0 + i * dx
                    y = -1.0 + j * dy
                    z = k * dz
                    node_map[(i, j, k)] = nid
                    nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} {z:.6f}')
                    nid += 1

    elements = []
    eid = 1
    for k in range(nz):
        for j in range(nn):
            for i in range(nn):
                if i >= n and j < n:
                    continue
                n1 = node_map[(i, j, k)]
                n2 = node_map[(i+1, j, k)]
                n3 = node_map[(i+1, j+1, k)]
                n4 = node_map[(i, j+1, k)]
                n5 = node_map[(i, j, k+1)]
                n6 = node_map[(i+1, j, k+1)]
                n7 = node_map[(i+1, j+1, k+1)]
                n8 = node_map[(i, j+1, k+1)]
                elements.append(
                    f'{eid} {element_type} {n1} {n2} {n3} {n4} '
                    f'{n5} {n6} {n7} {n8} {element_suffix}')
                eid += 1

    left_face = sorted({node_map[(0, j, k)]
                        for j in range(nn + 1) for k in range(nz + 1)
                        if (0, j) in used_nodes_2d})
    top_face = sorted({node_map[(i, nn, k)]
                       for i in range(nn + 1) for k in range(nz + 1)
                       if (i, nn) in used_nodes_2d})
    bottom_face = sorted({node_map[(i, 0, k)]
                          for i in range(n + 1) for k in range(nz + 1)
                          if (i, 0) in used_nodes_2d})
    right_face = sorted({node_map[(nn, j, k)]
                         for j in range(n, nn + 1) for k in range(nz + 1)
                         if (nn, j) in used_nodes_2d})

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_map": node_map,
        "left_face": left_face,
        "right_face": right_face,
        "top_face": top_face,
        "bottom_face": bottom_face,
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
    }


def matched_l_bracket_tsi_input(
    n: int = 4, lz: float = 0.5,
    E: float = 200e3, nu: float = 0.3, alpha: float = 12e-6,
    T_hot: float = 100.0, T_cold: float = 0.0, T_ref: float = 0.0,
    conductivity: float = 1.0, capacity: float = 1.0, density: float = 1.0,
) -> str:
    """Generate 4C TSI one-way input on L-shaped bracket.

    Demonstrates thermal stress concentration at the re-entrant corner.
    Thermal BCs: T_hot on left face (x=-1), T_cold on right face (x=1).
    Structural BCs: Fix left face (x=-1).
    """
    mesh = generate_l_domain_hex8(
        n=n, lz=lz,
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )

    t_expr = f"{T_hot} + ({T_cold} - {T_hot}) * (x + 1.0) / 2.0"

    yaml = f'''TITLE:
  - "L-bracket TSI: thermal stress concentration — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 10
  LINEAR_SOLVER: 2
  PREDICT: TangDis
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{t_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_hot}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_cold}]
    FUNCT: [0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["left_face"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in mesh["right_face"]:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_heat_transient_input(nx: int = 16, ny: int = 16,
                                 T_left: float = 100.0,
                                 T_right: float = 0.0,
                                 numstep: int = 10,
                                 timestep: float = 0.01,
                                 theta: float = 0.66) -> str:
    """Transient heat conduction on [0,1]^2 with One-Step-Theta.

    Same setup as matched_heat_input (T_left / T_right Dirichlet ends)
    but integrates the parabolic problem for `numstep` steps from a
    zero initial field, so the catalog's heat_transient_2d variant is
    actually transient instead of routing to the stationary solve.
    Small default mesh (16x16) keeps the run < 5 s.
    """
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Transient heat conduction — One-Step-Theta"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "One_Step_Theta"
  THETA: {theta}
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {numstep * timestep}
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def matched_elasticity_genalpha_input(nx: int = 20, ny: int = 4,
                                      E: float = 1000.0, nu: float = 0.3,
                                      dens: float = 1.0,
                                      numstep: int = 10,
                                      timestep: float = 0.05,
                                      lx: float = 10.0,
                                      ly: float = 1.0) -> str:
    """Transient cantilever under sudden tip-side body load, GenAlpha.

    Same cantilever geometry as matched_elasticity_input, but
    DYNAMICTYPE GenAlpha with non-zero density so the structural
    inertia matters — the catalog's structural_dynamics/genalpha_2d
    variant gets a real transient instead of the placeholder
    template (which aborted in 4C's MatchTree, probe 2026-06-12).
    """
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="STRUCTURE",
                                     element_type="SOLID QUAD4",
                                     element_suffix="MAT 1 KINEM linear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain")

    yaml = f'''TITLE:
  - "Cantilever {lx}x{ly} transient — GenAlpha"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {numstep * timestep}
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: {dens}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'''

    yaml += 'DESIGN POINT DIRICH CONDITIONS:\n'
    for i in range(len(mesh["left_nodes"])):
        yaml += f'''  - E: {i+1}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += '''DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DNODE-NODE TOPOLOGY:\n'
    for i, nid in enumerate(mesh["left_nodes"]):
        yaml += f'  - "NODE {nid} DNODE {i+1}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_elasticity_3d_nonlinear_input(n: int = 4,
                                          E: float = 1000.0,
                                          nu: float = 0.3,
                                          load: float = -10.0) -> str:
    """3D unit-cube compression, St. Venant-Kirchhoff, KINEM nonlinear.

    Left face (x=0) fully fixed, right face (x=1) loaded in -y.
    Small n^3 HEX8 mesh so the nonlinear Newton solve completes in
    seconds. Gives the catalog's solid_mechanics/nonlinear_3d
    variant a real finite-strain solve instead of the placeholder
    template (probe 2026-06-12: MatchTree abort).
    """
    mesh = generate_hex8_cube(n, n, n,
                              element_section="STRUCTURE",
                              element_type="SOLID HEX8",
                              element_suffix="MAT 1 KINEM nonlinear")

    # Face node sets from the structured grid
    grid = mesh["node_grid"]
    left_face = sorted(nid for (i, j, k), nid in grid.items() if i == 0)
    right_face = sorted(nid for (i, j, k), nid in grid.items() if i == n)

    yaml = f'''TITLE:
  - "3D cube nonlinear elasticity — finite strain"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 0.0
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, {load}, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml
