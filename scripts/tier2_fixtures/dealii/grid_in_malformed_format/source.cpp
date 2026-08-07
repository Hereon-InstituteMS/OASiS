/* Tier-2: GridIn::read_msh on truncated input raises. */

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the pathology from
// the input: the same reader is handed a COMPLETE one-quad mesh, so there is
// nothing malformed left to raise on. The fixture STILL PASSES — see
// fixture.json.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}

const char *const WELL_FORMED_MSH =
    "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
    "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
    "$Elements\n1\n1 3 2 1 1 1 2 3 4\n$EndElements\n";
}  // namespace

int main()
{
  // Severely truncated Gmsh header — no closing markers.
  std::istringstream iss(mutate() ? std::string(WELL_FORMED_MSH)
                                  : std::string("$MeshFormat\n2.2 0 8\n"));
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
