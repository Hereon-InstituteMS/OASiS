/* Tier-2: GridGenerator::hyper_rectangle with colorize=true
 * assigns boundary_id k to the face on the k-th coordinate
 * face — distinct ids {0,1,2,3} in 2D. The pitfall's contract
 * is structural and can be verified without solving a PDE.
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the phenomenon
// under test by passing colorize=false, so the generator hands every face the
// same boundary_id and n_distinct_boundary_ids drops to 1.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}
}  // namespace

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_rectangle(
      tria, Point<2>(0, 0), Point<2>(1, 1),
      /*colorize=*/!mutate());

  std::set<types::boundary_id> ids;
  for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    for (unsigned f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        ids.insert(cell->face(f)->boundary_id());

  std::cout << "GridGenerator::hyper_rectangle(colorize=true): "
            << "n_distinct_boundary_ids=" << ids.size() << "\n";
  for (auto id : ids)
    std::cout << "  boundary_id="
              << static_cast<unsigned>(id) << "\n";

  if (ids.size() != 4) {
    std::cerr << "FIXTURE WARNING: expected 4 distinct ids in 2D\n";
    return 2;
  }
  return 0;
}
