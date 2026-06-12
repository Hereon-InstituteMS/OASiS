"""Discontinuous Galerkin transport templates for deal.II.

Based on deal.II tutorial step-12 (DG advection with upwind flux).
"""


def _dg_transport_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    DG method for advection — based on step-12 pattern.
    """
    refinements = params.get("refinements", 4)
    degree = params.get("degree", 1)
    beta_x = params.get("beta_x", 1.0)
    beta_y = params.get("beta_y", 1.0)
    return f'''\
/* Discontinuous Galerkin for advection — based on deal.II step-12 pattern
 * Solves beta . grad(u) = 0 with upwind flux on unit square.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// deal.II 9.4 renamed the FEInterfaceValues shape accessors:
//   9.3.x:  fe_iv.jump(i, q)               fe_iv.average(i, q)
//   9.4+:   fe_iv.jump_in_shape_values(i,q) fe_iv.average_of_shape_values(i,q)
// Map the new names onto the old ones on 9.3 so the same template
// compiles on both lines (conda-forge currently ships 9.3.2).
#if !DEAL_II_VERSION_GTE(9, 4, 0)
#  define jump_in_shape_values jump
#  define average_of_shape_values average
// NOTE: do NOT blanket-#define normal_vector -> normal here:
// FEFaceValues::normal_vector exists on 9.3 already; only
// FEInterfaceValues lacks it. The interface call site below uses
// an inline version guard instead.
#endif

// Advection velocity field — set for your problem
template <int dim>
Tensor<1, dim> beta(const Point<dim> & /* p */)
{{
  Tensor<1, dim> wind;
  wind[0] = {beta_x};
  wind[1] = {beta_y};
  return wind;
}}

// Inflow boundary condition — set for your problem
template <int dim>
class BoundaryValues : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int) const override
  {{
    // Inflow value — set for your problem
    if (p[0] < 1e-10 || p[1] < 1e-10)
      return 1.0;
    return 0.0;
  }}
}};

// Scratch and copy data for MeshWorker
template <int dim>
struct ScratchData
{{
  ScratchData(const FiniteElement<dim> &fe,
              const unsigned int        quadrature_degree)
    : fe_values(fe,
                QGauss<dim>(quadrature_degree),
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , fe_face_values(fe,
                     QGauss<dim - 1>(quadrature_degree),
                     update_values | update_quadrature_points |
                       update_JxW_values | update_normal_vectors)
    , fe_interface_values(fe,
                          QGauss<dim - 1>(quadrature_degree),
                          update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors)
  {{}}

  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , fe_face_values(scratch_data.fe_face_values.get_fe(),
                     scratch_data.fe_face_values.get_quadrature(),
                     update_values | update_quadrature_points |
                       update_JxW_values | update_normal_vectors)
    , fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors)
  {{}}

  FEValues<dim>          fe_values;
  FEFaceValues<dim>      fe_face_values;
  FEInterfaceValues<dim> fe_interface_values;
}};

struct CopyData
{{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  struct FaceData
  {{
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  }};
  std::vector<FaceData> face_data;
}};

int main()
{{
  const unsigned int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  const unsigned int degree = {degree};
  FE_DGQ<dim>     fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "DG transport: " << dof_handler.n_dofs() << " DOFs, "
            << triangulation.n_active_cells() << " cells" << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity_pattern);
  Vector<double>       solution(dof_handler.n_dofs());
  Vector<double>       system_rhs(dof_handler.n_dofs());

  const BoundaryValues<dim> boundary_function;
  const QGauss<dim>         quadrature(degree + 1);
  const QGauss<dim - 1>     face_quadrature(degree + 1);

  // Cell worker: volume integral beta . grad(phi_j) * phi_i
  const auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {{
    copy.cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    copy.cell_rhs.reinit(fe.n_dofs_per_cell());
    copy.local_dof_indices.resize(fe.n_dofs_per_cell());

    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(copy.local_dof_indices);

    for (unsigned int q = 0; q < scratch.fe_values.n_quadrature_points; ++q)
      {{
        const auto beta_q = beta<dim>(scratch.fe_values.quadrature_point(q));
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            copy.cell_matrix(i, j) += -scratch.fe_values.shape_value(i, q) *
                                        (beta_q * scratch.fe_values.shape_grad(j, q)) *
                                        scratch.fe_values.JxW(q);
      }}
  }};

  // Boundary worker: upwind flux on domain boundary
  const auto boundary_worker = [&](const auto &cell, const unsigned int face_no,
                                    auto &scratch, auto &copy) {{
    scratch.fe_face_values.reinit(cell, face_no);
    const auto &fe_fv = scratch.fe_face_values;

    CopyData::FaceData face_copy;
    face_copy.joint_dof_indices.resize(fe.n_dofs_per_cell());
    cell->get_dof_indices(face_copy.joint_dof_indices);
    face_copy.cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());

    for (unsigned int q = 0; q < fe_fv.n_quadrature_points; ++q)
      {{
        const auto     beta_q    = beta<dim>(fe_fv.quadrature_point(q));
        const double   beta_dot_n = beta_q * fe_fv.normal_vector(q);
        const Point<dim> &q_point  = fe_fv.quadrature_point(q);

        if (beta_dot_n >= 0) // outflow
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              face_copy.cell_matrix(i, j) += beta_dot_n *
                                              fe_fv.shape_value(j, q) *
                                              fe_fv.shape_value(i, q) *
                                              fe_fv.JxW(q);
        else // inflow
          {{
            const double g = boundary_function.value(q_point, 0);
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {{
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                  face_copy.cell_matrix(i, j) += beta_dot_n *
                                                  fe_fv.shape_value(j, q) *
                                                  fe_fv.shape_value(i, q) *
                                                  fe_fv.JxW(q);
                copy.cell_rhs(i) += -beta_dot_n * g *
                                      fe_fv.shape_value(i, q) *
                                      fe_fv.JxW(q);
              }}
          }}
      }}
    copy.face_data.push_back(face_copy);
  }};

  // Face worker: upwind flux on interior faces
  const auto face_worker = [&](const auto &cell, const unsigned int f,
                                const unsigned int sf,
                                const auto &ncell, const unsigned int nf,
                                const unsigned int nsf,
                                auto &scratch, auto &copy) {{
    scratch.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);
    const auto &fe_iv = scratch.fe_interface_values;

    CopyData::FaceData face_copy;
    const unsigned int n_interface_dofs = fe_iv.n_current_interface_dofs();
    face_copy.joint_dof_indices = fe_iv.get_interface_dof_indices();
    face_copy.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

    for (unsigned int q = 0; q < fe_iv.n_quadrature_points; ++q)
      {{
        // get_quadrature_points()[q], not quadrature_point(q): the
        // per-point accessor on FEInterfaceValues only exists since
        // deal.II 9.4; the vector accessor is available on 9.3 too.
        const auto   beta_q    = beta<dim>(fe_iv.get_quadrature_points()[q]);
        // FEInterfaceValues gained normal_vector(q) only in 9.4;
        // 9.3 calls it normal(q). (FEFaceValues::normal_vector
        // exists on both, so no blanket rename is possible.)
#if DEAL_II_VERSION_GTE(9, 4, 0)
        const double beta_dot_n = beta_q * fe_iv.normal_vector(q);
#else
        const double beta_dot_n = beta_q * fe_iv.normal(q);
#endif

        for (unsigned int i = 0; i < n_interface_dofs; ++i)
          for (unsigned int j = 0; j < n_interface_dofs; ++j)
            {{
              // Upwind flux: use value from upwind side
              face_copy.cell_matrix(i, j) +=
                beta_dot_n *
                fe_iv.jump_in_shape_values(j, q) *
                fe_iv.average_of_shape_values(i, q) *
                fe_iv.JxW(q);
              // Stabilization: penalty on jumps
              face_copy.cell_matrix(i, j) +=
                0.5 * std::abs(beta_dot_n) *
                fe_iv.jump_in_shape_values(j, q) *
                fe_iv.jump_in_shape_values(i, q) *
                fe_iv.JxW(q);
            }}
      }}
    copy.face_data.push_back(face_copy);
  }};

  // Copier: distribute to global system
  const auto copier = [&](const CopyData &copy) {{
    for (unsigned int i = 0; i < copy.local_dof_indices.size(); ++i)
      {{
        for (unsigned int j = 0; j < copy.local_dof_indices.size(); ++j)
          system_matrix.add(copy.local_dof_indices[i],
                            copy.local_dof_indices[j],
                            copy.cell_matrix(i, j));
        system_rhs(copy.local_dof_indices[i]) += copy.cell_rhs(i);
      }}
    for (const auto &fd : copy.face_data)
      for (unsigned int i = 0; i < fd.joint_dof_indices.size(); ++i)
        for (unsigned int j = 0; j < fd.joint_dof_indices.size(); ++j)
          system_matrix.add(fd.joint_dof_indices[i],
                            fd.joint_dof_indices[j],
                            fd.cell_matrix(i, j));
  }};

  ScratchData<dim> scratch(fe, degree + 1);
  CopyData         copy;

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                         dof_handler.end(),
                         cell_worker,
                         copier,
                         scratch,
                         copy,
                         MeshWorker::assemble_own_cells |
                           MeshWorker::assemble_boundary_faces |
                           MeshWorker::assemble_own_interior_faces_once,
                         boundary_worker,
                         face_worker);

  // Solve with GMRES: the DG advection matrix is non-symmetric, and
  // plain Richardson iteration with BlockSSOR was observed to throw
  // SolverControl::NoConvergence on it (probe 2026-06-12) — its
  // convergence requires the preconditioned spectrum inside the unit
  // disk, which the upwind flux does not guarantee. GMRES with the
  // same block preconditioner is the robust choice. Tolerance is
  // RELATIVE to the rhs norm (absolute 1e-12 is unreachable noise).
  SolverControl            solver_control(2000, 1e-10 * system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(solver_control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "DG transport solved: " << solver_control.last_step()
            << " iterations" << std::endl;

  // Output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(degree);
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "DG transport: " << dof_handler.n_dofs() << " DOFs complete"
            << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Discontinuous Galerkin for advection (step-12, step-30, step-67)",
    "tutorial_steps": ["step-12 (DG upwind, MeshWorker)", "step-30 (anisotropic refinement)",
                      "step-67 (Euler equations, matrix-free DG)"],
    "function_space": "FE_DGQ<dim>(p) — tensor-product DG, or FE_DGP<dim>(p) — polynomial",
    "solver": "Richardson + block SSOR, or direct (UMFPACK) for small systems",
    "numerical_flux": {
        "upwind": "Use value from element where beta.n > 0 (simplest, diffusive)",
        "Lax-Friedrichs": "0.5*(F_L + F_R) + 0.5*alpha*(u_L - u_R), alpha=max|beta.n|",
        "central": "Average flux (not stable for advection-dominated)",
    },
    "pitfalls": [
        "[Syntax] DG requires flux sparsity pattern: "
        "DoFTools::make_flux_sparsity_pattern. The standard "
        "make_sparsity_pattern misses the off-cell face-coupling "
        "entries. Signal: assembly raises ExcMessage('matrix entry "
        "at i,j does not exist in sparsity pattern') when the "
        "FEInterfaceValues face term writes into the off-cell "
        "block; or, if the assembly succeeds via ExcInvalidIterator, "
        "DataOut shows the DG solution with continuous-Galerkin-"
        "like behaviour at faces because face coupling was dropped.",
        "[Syntax] FEInterfaceValues needed for face integrals "
        "(jump/average operators on cell interfaces). Using "
        "FEValues alone produces only cell-interior contributions. "
        "Signal: SolverGMRES converges but DataOut shows a smooth "
        "(non-DG) solution; jump-across-face values from "
        "VectorTools::integrate_difference vs reference are "
        "1e-8 (effectively zero) where they should be O(1) for "
        "upwind DG. (Note: the real dealii function is "
        "integrate_difference, NOT interpolate_difference; "
        "the latter does not exist in numerics/vector_tools.h.)",
        "[API] MeshWorker::mesh_loop simplifies cell / face / "
        "boundary assembly — without it, the user re-implements "
        "the dispatch logic and typically forgets the periodic-"
        "face case. Signal: assembly compiles and runs but the "
        "global system is non-symmetric AND inconsistent on "
        "periodic boundaries (if any); DataOut shows the solution "
        "with kinks at periodic-face nodes.",
        "[Numerical] PreconditionBlockSSOR with block_size = "
        "dofs_per_cell is the right DG preconditioner — point "
        "Jacobi or scalar SSOR ignore the per-cell block "
        "structure of the DG mass matrix. Signal: SolverGMRES "
        "with PreconditionJacobi reports SolverControl::"
        "last_step() growing linearly with n_cells; switching "
        "to PreconditionBlockSSOR drops iteration count by 10x.",
        "[Numerical] For higher-order DG: FE_DGQHermite gives "
        "better matrix-free performance than FE_DGQ because the "
        "Hermite-like basis preserves face-value continuity, "
        "reducing the cross-face stencil weight. Signal: "
        "MatrixFree::cell_loop wall-time per iteration with "
        "FE_DGQ(6) is 2-3x larger than with FE_DGQHermite(6) at "
        "the same n_dofs.",
        "[Physics] No continuity constraints needed (DG has no "
        "hanging-node constraints), but you DO need to track "
        "non-conforming face DoFs explicitly when refining. "
        "Signal: AffineConstraints::distribute on a DG solution "
        "is a no-op (constraints.size() == 0); attempting to "
        "apply hanging-node constraints raises ExcMessage('DG "
        "discretisation has no hanging-node constraints').",
        "[Physics] Inflow BCs: weakly enforced via numerical "
        "flux on boundary faces (NOT via AffineConstraints — DG "
        "has none). Setting Dirichlet values strongly is a "
        "common bug for users coming from CG. Signal: "
        "VectorTools::interpolate_boundary_values on a DG FE "
        "raises ExcMessage('strong boundary conditions not "
        "supported for DG'); or, if silently ignored, DataOut "
        "shows the prescribed Dirichlet value NOT appearing at "
        "the inflow boundary.",
    ],
}
