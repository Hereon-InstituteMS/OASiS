// Shared translation unit for the Assert-gated Signal family.
//
// Assert() exists only in a Debug build. In Release the same misuse returns
// silently with a wrong answer, or segfaults. Each probe below is run against
// BOTH libraries by the fixtures that reference it, so the pair is the
// evidence — not one build's behaviour quoted as if it were the other's.
//
// Assert ABORTS (SIGABRT, rc=134); it does not throw. There is deliberately no
// try/catch here, because a catch would see nothing.
//
// usage: assert_family <probe>
//   sparse_add_outside_pattern | hp_active_fe_index | vector_valued_gradients
// Env T2_MUTATE=1 runs the CORRECT variant of the same probe.

#include <deal.II/base/config.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

static void report_build()
{
#ifdef DEBUG
  std::cout << "consumer_DEBUG=1" << std::endl;
#else
  std::cout << "consumer_DEBUG=0" << std::endl;
#endif
  std::cout << "dealii_version=" << DEAL_II_PACKAGE_VERSION << std::endl;
}

// Probe 1 — SparseMatrix::add() at an entry the sparsity pattern does not have.
// Debug: aborts with the "does not exist in the sparsity pattern" report.
// Release: silently DROPS the value, and the matrix norm proves it.
static int sparse_add_outside_pattern()
{
  SparsityPattern sp(3, 3, 1);
  sp.add(0, 0);
  sp.add(1, 1);
  sp.add(2, 2);
  sp.compress();
  SparseMatrix<double> A(sp);
  A = 0.0;
  const unsigned int col = mutate() ? 0u : 2u;   // (0,2) is not in the pattern
  std::cout << "writing_entry=0," << col << std::endl;
  std::cout << "before_add" << std::endl;
  A.add(0, col, 5.0);
  std::cout << "after_add" << std::endl;
  std::cout << "frobenius_norm=" << A.frobenius_norm() << std::endl;
  std::cout << "value_was_stored=" << (A.frobenius_norm() > 1e-12 ? "true"
                                                                  : "false")
            << std::endl;
  return 0;
}

// Probe 2 — an active_fe_index past the end of the hp::FECollection.
// Debug: aborts naming the index and the collection size.
// Release: SEGFAULT (rc=139), no message. This Assert lives in the LIBRARY, so
// -DDEBUG on the consumer does NOT revive it — only a Debug library does.
static int hp_active_fe_index()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  hp::FECollection<2> fes;
  fes.push_back(FE_Q<2>(1));
  fes.push_back(FE_Q<2>(2));
  std::cout << "collection_size=" << fes.size() << std::endl;
  DoFHandler<2> dof(tria);
  const unsigned int idx = mutate() ? 1u : 7u;
  std::cout << "setting_active_fe_index=" << idx << std::endl;
  for (const auto &cell : dof.active_cell_iterators())
    cell->set_active_fe_index(idx);
  std::cout << "before_distribute_dofs" << std::endl;
  dof.distribute_dofs(fes);
  std::cout << "after_distribute_dofs" << std::endl;
  std::cout << "n_dofs=" << dof.n_dofs() << std::endl;
  return 0;
}

// Probe 3 — FEValues::get_function_gradients into a SCALAR-shaped container on
// a vector-valued FESystem.
// Debug: aborts with "Two sizes or dimensions were supposed to be equal".
// Release: returns normally, having filled the container with a mixture of
// different components' derivatives.
static int vector_valued_gradients()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FESystem<2> fe(FE_Q<2>(1), 2);
  DoFHandler<2> dof(tria);
  dof.distribute_dofs(fe);
  Vector<double> sol(dof.n_dofs());
  for (unsigned int i = 0; i < sol.size(); ++i)
    sol(i) = 1.0 + i;
  QGauss<2> quad(2);
  FEValues<2> fev(fe, quad, update_gradients);
  const auto cell = dof.begin_active();
  fev.reinit(cell);
  std::cout << "n_components=" << fe.n_components() << std::endl;
  std::cout << "before_get_function_gradients" << std::endl;
  if (mutate())
    {
      std::vector<std::vector<Tensor<1, 2>>> g(
        quad.size(), std::vector<Tensor<1, 2>>(fe.n_components()));
      fev.get_function_gradients(sol, g);
      std::cout << "container=vector_shaped" << std::endl;
      std::cout << "g00_x=" << g[0][0][0] << std::endl;
    }
  else
    {
      std::vector<Tensor<1, 2>> g(quad.size());
      fev.get_function_gradients(sol, g);
      std::cout << "container=scalar_shaped" << std::endl;
      std::cout << "g0_x=" << g[0][0] << std::endl;
    }
  std::cout << "after_get_function_gradients" << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  report_build();
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "sparse_add_outside_pattern")
    return sparse_add_outside_pattern();
  if (probe == "hp_active_fe_index")
    return hp_active_fe_index();
  if (probe == "vector_valued_gradients")
    return vector_valued_gradients();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
