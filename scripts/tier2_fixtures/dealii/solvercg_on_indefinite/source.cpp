/* Tier-2 fixture: SolverCG on a 2x2 indefinite matrix.
 *
 * Pitfall (stokes #0): 'System is INDEFINITE — cannot use
 * SolverCG, use SolverGMRES / SolverMinRes / a direct solver.
 * Signal: SolverCG reports breakdown on iteration 2-3 with a
 * negative inner product...'
 *
 * The simplest indefinite SPD-impostor is [[1, 0], [0, -1]] —
 * symmetric but indefinite. SolverCG either throws
 * SolverControl::NoConvergence or reports the indefiniteness
 * via stderr; either way the captured output should include
 * 'SolverCG'.
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
  // 2x2 indefinite diagonal A = diag(1, -1).
  DynamicSparsityPattern dsp(2, 2);
  dsp.add(0, 0);
  dsp.add(1, 1);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  A.set(0, 0, 1.0);
  A.set(1, 1, -1.0);

  Vector<double> b(2), x(2);
  b(0) = 1.0;
  b(1) = 1.0;

  SolverControl ctrl(50, 1e-10);
  SolverCG<Vector<double>> cg(ctrl);
  try
  {
    cg.solve(A, x, b, PreconditionIdentity());
    std::cout << "SolverCG converged on indefinite matrix in "
              << ctrl.last_step() << " iterations — pitfall "
              << "claim does not hold for this build\n";
    return 2;
  }
  catch (const std::exception &e)
  {
    // SolverCG (with PreconditionIdentity) on the indefinite
    // diag(1, -1) raises SolverControl::NoConvergence after
    // exhausting iterations OR ExcMessage("breakdown") if the
    // inner product goes negative.
    std::cerr << "SolverCG raised: " << e.what() << '\n';
    return 1;
  }
}
