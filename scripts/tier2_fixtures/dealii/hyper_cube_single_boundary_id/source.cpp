/* Tier-2 for poisson #1: hyper_cube gives all faces boundary_id=0.
 *
 * deal.II 9.1.1 lacks GridTools::get_boundary_ids (added in 9.4),
 * so iterate cells manually and collect the unique face
 * boundary_ids — same observable, version-portable.
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <set>

using namespace dealii;

int main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);

  std::set<types::boundary_id> ids;
  for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
  {
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
    {
      if (cell->face(f)->at_boundary())
        ids.insert(cell->face(f)->boundary_id());
    }
  }

  // Print the observed boundary_ids — the Signal claim is that
  // GridTools-style enumeration on hyper_cube returns just {0}.
  std::cout << "GridTools-equivalent enumeration: "
            << ids.size() << " unique boundary_id(s):\n";
  for (auto id : ids) {
    std::cout << "  boundary_id = "
              << static_cast<unsigned>(id) << "\n";
  }

  // Confirm the pitfall claim.
  if (ids.size() != 1 || *ids.begin() != 0) {
    std::cerr << "FIXTURE WARNING: expected exactly {0}, got "
              << ids.size() << " ids\n";
    return 2;
  }
  return 0;
}
