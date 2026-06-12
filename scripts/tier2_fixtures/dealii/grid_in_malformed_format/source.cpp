/* Tier-2: GridIn::read_msh on truncated input raises. */

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <sstream>

using namespace dealii;

int main()
{
  // Severely truncated Gmsh header — no closing markers.
  std::istringstream iss("$MeshFormat\n2.2 0 8\n");
  Triangulation<2> tria;
  GridIn<2> grid_in;
  grid_in.attach_triangulation(tria);
  try {
    grid_in.read_msh(iss);
  } catch (const std::exception &e) {
    std::cerr << "GridIn raised: " << e.what() << '\n';
    return 1;
  }
  std::cerr << "ERROR: GridIn accepted truncated mesh\n";
  return 2;
}
