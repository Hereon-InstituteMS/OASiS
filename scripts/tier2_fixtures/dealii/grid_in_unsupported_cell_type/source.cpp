/* Tier-2 fixture: deal.II 2D GridIn refuses Gmsh meshes that
 * contain only triangles, not quads.
 *
 * The pitfall (linear_elasticity #4): 'deal.II 2D ONLY reads
 * QUADS from Gmsh — no triangles. Always set
 * gmsh.option.setNumber(Mesh.RecombineAll, 1) to produce quads.
 * Signal: ExcMessage from grid_in.cc that the mesh contains
 * unsupported cell type.'
 *
 * This fixture writes a tiny in-memory Gmsh-format mesh with
 * triangles only and tries to read it into Triangulation<2>.
 * GridIn::read_msh raises ExcMessage; we catch it and print the
 * exception so the Tier-2 runner can match against the Signal
 * substrings (ExcMessage / grid_in).
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace dealii;

int main()
{
  // Minimal Gmsh v2 mesh format with three triangles forming a
  // unit square split along the diagonal.
  const std::string gmsh_triangles_only =
      "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
      "$Nodes\n4\n"
      "1 0 0 0\n"
      "2 1 0 0\n"
      "3 1 1 0\n"
      "4 0 1 0\n"
      "$EndNodes\n"
      "$Elements\n2\n"
      "1 2 2 1 1 1 2 3\n"   // triangle (type 2) — NOT a quad
      "2 2 2 1 1 1 3 4\n"
      "$EndElements\n";

  std::istringstream iss(gmsh_triangles_only);
  Triangulation<2> tria;
  GridIn<2> grid_in;
  grid_in.attach_triangulation(tria);
  try
  {
    grid_in.read_msh(iss);
  }
  catch (const std::exception &e)
  {
    // Print both the standard exception name and the message
    // so the matcher can see both "ExcMessage" (from typeid
    // or class context) and "grid_in" (from file location).
    std::cerr << "Caught exception of type: "
              << typeid(e).name() << '\n';
    std::cerr << "what(): " << e.what() << '\n';
    return 1;
  }
  // If we get here, deal.II accepted triangles — pitfall claim
  // is invalid for this version.
  std::cerr << "ERROR: GridIn::read_msh accepted triangles; "
            << "pitfall claim does not hold in this deal.II "
            << "version\n";
  return 2;
}
