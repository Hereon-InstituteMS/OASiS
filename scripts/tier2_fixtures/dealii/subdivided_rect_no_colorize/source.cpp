/* Tier-2: subdivided_hyper_rectangle with colorize=false gives
 * all boundary faces id=0. The hyperelasticity pitfall warns
 * users to pass colorize=true to distinguish faces.
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <set>
#include <vector>

using namespace dealii;

int main()
{
  Triangulation<2> tria;
  std::vector<unsigned int> reps = {2, 2};
  GridGenerator::subdivided_hyper_rectangle(
      tria, reps, Point<2>(0, 0), Point<2>(1, 1),
      /*colorize=*/false);

  std::set<types::boundary_id> ids;
  for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    for (unsigned f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        ids.insert(cell->face(f)->boundary_id());

  std::cout << "GridGenerator::subdivided_hyper_rectangle "
            << "(colorize=false): " << ids.size()
            << " unique boundary_id(s)\n";
  for (auto id : ids) {
    std::cout << "  boundary_id = "
              << static_cast<unsigned>(id) << "\n";
  }
  if (ids.size() != 1 || *ids.begin() != 0) {
    std::cerr << "FIXTURE WARNING: expected {0}, got "
              << ids.size() << " ids\n";
    return 2;
  }
  return 0;
}
