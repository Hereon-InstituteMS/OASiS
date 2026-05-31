/* Tier-2 fixture: confirms that a 1-cell hyper_cube without
 * refine_global() yields a 4-DoF system (the well-known
 * 'forgot-to-refine' pitfall for FE_Q(1) on a single quad).
 *
 * Pitfall (poisson #0): 'Must call triangulation.refine_global()
 * before distributing DOFs. Signal: SolverControl reports
 * Convergence step 0 value..., DoFHandler::n_dofs() returns 4.'
 *
 * This fixture deliberately omits refine_global and prints
 * the n_dofs result. The Tier-2 runner matches against the
 * 'n_dofs' and 'DoFHandler' substrings.
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>

using namespace dealii;

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  // DELIBERATELY no tria.refine_global(...) — the pitfall under
  // verification.

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Print the small n_dofs that the pitfall warns about.
  std::cout << "DoFHandler::n_dofs() = "
            << dof_handler.n_dofs() << '\n';
  std::cout << "(expected 4 for un-refined FE_Q(1) on a single "
            << "hyper_cube cell)\n";

  if (dof_handler.n_dofs() != 4u)
  {
    std::cerr << "FIXTURE WARNING: n_dofs is "
              << dof_handler.n_dofs()
              << ", not 4 — pitfall numerics differ on this "
              << "deal.II version\n";
    return 2;
  }
  return 0;
}
