/* Tier-2 fixture for linear_elasticity#4 — INVERTED 2026-08-03.
 *
 * The pitfall used to read: 'deal.II 2D ONLY reads QUADS from Gmsh —
 * no triangles. Signal: ExcMessage from grid_in.cc that the mesh
 * contains unsupported cell type.' That has been false since deal.II
 * 9.3 added simplex support, and this fixture already detected it: on
 * the installed 9.8.0-pre it printed "ERROR: GridIn::read_msh
 * accepted triangles; pitfall claim does not hold in this deal.II
 * version" and therefore failed its own expect_in_output.
 *
 * The fixture now asserts the CORRECTED claim: a triangles-only Gmsh
 * mesh reads successfully into Triangulation<2>, and the resulting
 * cells really are triangles (n_vertices() == 3). Measured on
 * deal.II 9.8.0-pre 2026-08-03.
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 feeds the SAME reader a
// quads-only mesh (Gmsh element type 3) instead of the triangles-only one, so
// the simplex cells this fixture claims to read are no longer there.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}
}  // namespace

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

  // The same four nodes, but one QUAD (type 3) — used only by the
  // mutation control.
  const std::string gmsh_quads_only =
      "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
      "$Nodes\n4\n"
      "1 0 0 0\n"
      "2 1 0 0\n"
      "3 1 1 0\n"
      "4 0 1 0\n"
      "$EndNodes\n"
      "$Elements\n1\n"
      "1 3 2 1 1 1 2 3 4\n"   // quad (type 3)
      "$EndElements\n";

  std::istringstream iss(mutate() ? gmsh_quads_only : gmsh_triangles_only);
  Triangulation<2> tria;
  GridIn<2> grid_in;
  grid_in.attach_triangulation(tria);
  try
  {
    grid_in.read_msh(iss);
  }
  catch (const std::exception &e)
  {
    std::cout << "GridIn_read_msh_triangles=REJECTED\n";
    std::cout << "what(): " << e.what() << '\n';
    std::cout << "FIXTURE_MISMATCH\n";
    return 1;
  }

  unsigned int n_tri = 0, n_quad = 0;
  for (const auto &cell : tria.active_cell_iterators())
    (cell->n_vertices() == 3 ? n_tri : n_quad)++;

  std::cout << "GridIn_read_msh_triangles=ACCEPTED\n";
  std::cout << "n_active_cells=" << tria.n_active_cells()
            << " n_triangles=" << n_tri
            << " n_quads=" << n_quad << '\n';
  const bool ok = (tria.n_active_cells() == 2) && (n_tri == 2) &&
                  (n_quad == 0);
  std::cout << (ok ? "FIXTURE_OK" : "FIXTURE_MISMATCH") << '\n';
  return ok ? 0 : 2;
}
