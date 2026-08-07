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
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the pathology —
// the DG space — and interpolates the same boundary values onto a CONTINUOUS
// FE_Q(1), which has boundary dofs to fill, so the map is no longer empty.
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
  tria.refine_global(2);

  std::unique_ptr<FiniteElement<2>> fe;
  if (mutate())
    fe = std::make_unique<FE_Q<2>>(1);
  else
    fe = std::make_unique<FE_DGQ<2>>(1);
  DoFHandler<2> dh(tria);
  dh.distribute_dofs(*fe);

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
