/* Tier-2: VectorTools::interpolate_boundary_values on a DG
 * FE_DGQ silently returns an empty map — no exception, no
 * warning, just empty output. The user thinks they have set
 * boundary conditions; in fact the boundary_values map has
 * size 0 and the linear system carries no Dirichlet rows.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <map>

using namespace dealii;

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);

  FE_DGQ<2> fe(1);
  DoFHandler<2> dh(tria);
  dh.distribute_dofs(fe);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
      dh, /*boundary_id=*/0, Functions::ZeroFunction<2>(),
      boundary_values);

  std::cout << "interpolate_boundary_values_on_DG: "
            << "boundary_values_size=" << boundary_values.size()
            << " (n_dofs=" << dh.n_dofs()
            << ", n_active_cells=" << tria.n_active_cells() << ")\n";

  if (!boundary_values.empty()) {
    std::cerr << "FIXTURE FAILED: DG produced "
              << boundary_values.size()
              << " boundary values — expected 0\n";
    return 2;
  }
  return 0;
}
