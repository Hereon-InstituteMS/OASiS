/* Tier-2: SolverControl with max_iter=1 on a non-trivial system
 * raises SolverControl::NoConvergence — verifying the
 * "iterative method failed to converge" Signal family.
 */

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int main()
{
  // 5x5 SPD matrix (tridiagonal 2,-1,-1) — fine for CG, but
  // we cap iterations at 1 so it can't converge.
  const unsigned n = 5;
  DynamicSparsityPattern dsp(n, n);
  for (unsigned i = 0; i < n; ++i)
  {
    dsp.add(i, i);
    if (i > 0) {
      dsp.add(i, i - 1);
      dsp.add(i - 1, i);
    }
  }
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  for (unsigned i = 0; i < n; ++i) {
    A.set(i, i, 2.0);
    if (i > 0) {
      A.set(i, i - 1, -1.0);
      A.set(i - 1, i, -1.0);
    }
  }

  Vector<double> b(n), x(n);
  b(0) = 1.0;

  SolverControl ctrl(1, 1e-15);  // only ONE iteration allowed
  SolverCG<Vector<double>> cg(ctrl);
  try
  {
    cg.solve(A, x, b, PreconditionIdentity());
    std::cout << "Unexpected: SolverCG converged in 1 step\n";
    return 2;
  }
  catch (const std::exception &e)
  {
    // SolverControl::NoConvergence::what() includes the text
    // "Iterative method reported convergence failure in step 1.
    // The residual ... did not converge to ..." which matches
    // the "iterative method" + "SolverControl" Signal family.
    std::cerr << "SolverControl raised: " << e.what() << '\n';
    return 1;
  }
}
