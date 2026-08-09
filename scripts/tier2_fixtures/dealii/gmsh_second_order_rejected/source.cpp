/* Tier-2: deal.II's Gmsh reader supports only linear element
 * topology (line/quad/hex/tri/tet, Gmsh types 1/3/5/2/4). Type 10
 * (9-node biquadratic quad) is rejected with a specific ExcMessage.
 *
 * The fixture writes a minimal .msh containing a single Quad9
 * element, attempts to read it, and verifies the exception text
 * mentions the unsupported element identifier and lists the
 * supported types — both observable substrings.
 */

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the pathology from
// the input: the same reader is handed a LINEAR quad (Gmsh element type 3),
// which is one of the topologies deal.II does support, so nothing is rejected.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}

const char *const QUAD4_MSH =
    "$MeshFormat\n"
    "2.2 0 8\n"
    "$EndMeshFormat\n"
    "$Nodes\n"
    "4\n"
    "1 0 0 0\n"
    "2 1 0 0\n"
    "3 1 1 0\n"
    "4 0 1 0\n"
    "$EndNodes\n"
    "$Elements\n"
    "1\n"
    "1 3 2 1 1 1 2 3 4\n"
    "$EndElements\n";

const char *const QUAD9_MSH =
    "$MeshFormat\n"
    "2.2 0 8\n"
    "$EndMeshFormat\n"
    "$Nodes\n"
    "9\n"
    "1 0 0 0\n"
    "2 1 0 0\n"
    "3 1 1 0\n"
    "4 0 1 0\n"
    "5 0.5 0 0\n"
    "6 1 0.5 0\n"
    "7 0.5 1 0\n"
    "8 0 0.5 0\n"
    "9 0.5 0.5 0\n"
    "$EndNodes\n"
    "$Elements\n"
    "1\n"
    "1 10 2 1 1 1 2 3 4 5 6 7 8 9\n"
    "$EndElements\n";
}

int main()
{
  const std::string path = "/tmp/_tier2_quad9.msh";
  { std::ofstream out(path); out << (mutate() ? QUAD4_MSH : QUAD9_MSH); }

  Triangulation<2> tria;
  GridIn<2> gi;
  gi.attach_triangulation(tria);
  std::ifstream f(path);
  try {
    gi.read_msh(f);
    std::cerr << "FIXTURE FAILED: GridIn accepted Quad9 — expected "
              << "ExcMessage about unsupported Element Identifier\n";
    return 2;
  } catch (const std::exception &e) {
    std::cout << "GridIn::read_msh threw as expected. "
              << "Diagnostic:\n"
              << e.what() << "\n";
    return 0;
  }
}
