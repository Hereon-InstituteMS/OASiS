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

#include <cstdlib>
#include <iostream>
#include <string>

using namespace dealii;

namespace {
// MUTATION CONTROL (see fixture.json). T2_MUTATE=1 removes the pathology: the
// matrix becomes diag(1, +1), which is SPD, so CG is entitled to converge and
// nothing is raised. The fixture STILL PASSES — see fixture.json.
bool mutate()
{
  const char *v = std::getenv("T2_MUTATE");
  return v != nullptr && std::string(v) == "1";
}
}  // namespace

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
  A.set(1, 1, mutate() ? 1.0 : -1.0);

  Vector<double> b(2), x(2);
  b(0) = 1.0;
  b(1) = 1.0;

  SolverControl ctrl(50, 1e-10);
  SolverCG<Vector<double>> cg(ctrl);
  // Printed BEFORE the solve, deliberately.  On a DEBUG deal.II the failure
  // arrives as Assert(std::abs(alpha) != 0., ExcDivideByZero()) inside
  // solver_cg.h, and Assert aborts -- the catch below is never reached and
  // nothing this fixture writes after the solve exists.  Measured on both
  // builds available here; see fixture.json.
  std::cout << "solvercg_probe_started=1\n";
  std::cout.flush();
  try
  {
    cg.solve(A, x, b, PreconditionIdentity());
    // The single expectation used to be the bare word "SolverCG", which this
    // fixture writes on BOTH paths -- here and in the catch below -- so it
    // matched the indefinite matrix and the SPD mutation identically.  The
    // outcome and deal.II's own exception type are what is asserted now.
    std::cout << "solvercg_converged_on_indefinite=1\n";
    std::cout << "solvercg_last_step=" << ctrl.last_step() << "\n";
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
    std::cout << "solvercg_raised=1\n";
    std::cout << "solvercg_last_step=" << ctrl.last_step() << "\n";
    std::cout << "solvercg_exception_is_no_convergence="
              << (dynamic_cast<const SolverControl::NoConvergence *>(&e)
                      ? 1 : 0)
              << "\n";
    std::cerr << "SolverCG raised: " << e.what() << '\n';
    return 1;
  }
}
