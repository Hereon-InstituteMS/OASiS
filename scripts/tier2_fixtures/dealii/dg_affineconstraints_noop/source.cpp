/* Tier-2: DG has no hanging-node constraints.
 * DoFTools::make_hanging_node_constraints on a DG DoFHandler
 * produces an empty AffineConstraints — even on a non-conforming
 * triangulation that DOES have hanging nodes for a continuous FE.
 *
 * The fixture builds a 2D triangulation with locally-refined
 * cells (true hanging-node configuration), distributes FE_DGQ(1),
 * then asserts AffineConstraints::n_constraints() == 0.
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the pathology —
// the DG space — and distributes a CONTINUOUS FE_Q(1) on the very same
// non-conforming mesh, where make_hanging_node_constraints really does have
// hanging nodes to constrain.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}
}  // namespace

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  std::unique_ptr<FiniteElement<2>> fe;
  if (mutate())
    fe = std::make_unique<FE_Q<2>>(1);
  else
    fe = std::make_unique<FE_DGQ<2>>(1);
  DoFHandler<2> dh(tria);
  dh.distribute_dofs(*fe);

  AffineConstraints<double> ac;
  DoFTools::make_hanging_node_constraints(dh, ac);
  ac.close();

  std::cout << "make_hanging_node_constraints_on_DG: "
            << "n_constraints=" << ac.n_constraints()
            << " (n_active_cells=" << tria.n_active_cells()
            << ", n_dofs=" << dh.n_dofs() << ")\n";

  if (ac.n_constraints() != 0) {
    std::cerr << "FIXTURE FAILED: DG produced "
              << ac.n_constraints() << " constraints — expected 0\n";
    return 2;
  }
  return 0;
}
