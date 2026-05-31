/* Tier-2 for poisson#3: hanging-node constraints not applied.
 *
 * Assemble a Poisson system on a triangulation with hanging
 * nodes (one cell refined more than its neighbour), do NOT call
 * constraints.condense, and try to solve with CG. The matrix
 * is rank-deficient at the hanging nodes; SolverCG fails to
 * converge.
 *
 * We use a small max_iter so SolverControl raises promptly
 * rather than churning.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(2);
  // Refine just ONE cell to create hanging nodes.
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);

  SparseMatrix<double> A(sp);
  Vector<double> rhs(dof_handler.n_dofs());
  Vector<double> sol(dof_handler.n_dofs());

  // Assemble a trivial Laplace bilinear form without
  // applying hanging-node constraints.
  QGauss<2> q(2);
  FEValues<2> fv(fe, q,
                 update_gradients | update_JxW_values);
  std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
  FullMatrix<double> cell_M(fe.dofs_per_cell, fe.dofs_per_cell);
  for (auto cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
  {
    fv.reinit(cell);
    cell_M = 0;
    for (unsigned q_pt = 0; q_pt < q.size(); ++q_pt)
      for (unsigned i = 0; i < fe.dofs_per_cell; ++i)
        for (unsigned j = 0; j < fe.dofs_per_cell; ++j)
          cell_M(i, j) +=
              fv.shape_grad(i, q_pt) * fv.shape_grad(j, q_pt)
              * fv.JxW(q_pt);
    cell->get_dof_indices(dofs);
    for (unsigned i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned j = 0; j < fe.dofs_per_cell; ++j)
        A.add(dofs[i], dofs[j], cell_M(i, j));
  }
  // Add a trivial rhs to make the system non-zero.
  for (unsigned i = 0; i < rhs.size(); ++i) rhs(i) = 1.0;

  // Solve WITHOUT calling constraints.condense — system has
  // unresolved hanging-node DoFs.
  SolverControl ctrl(20, 1e-8);
  SolverCG<Vector<double>> cg(ctrl);
  try {
    cg.solve(A, sol, rhs, PreconditionIdentity());
    std::cout << "CG converged in " << ctrl.last_step()
              << " — unexpected\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr << "SolverControl: " << e.what() << '\n';
    return 1;
  }
}
