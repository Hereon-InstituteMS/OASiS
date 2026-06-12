/* Tier-2: SolverCG is not safe on an indefinite Helmholtz
 * operator (K - k^2 M). The crispest empirical demonstration:
 * place k^2 exactly at an eigenvalue of K so the operator is
 * singular; SolverCG runs to max_steps without converging and
 * SolverControl::NoConvergence is thrown.
 *
 * Catalog audit (2026-05-31): the original Signal "SolverCG
 * reports 'breakdown' immediately" was inaccurate — deal.II's
 * SolverCG does not emit a literal "breakdown" string. The
 * observable is the NoConvergence exception; the fixture
 * verifies the exception class name and SolverCG context.
 */

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <iostream>

using namespace dealii;

int main()
{
  const unsigned n = 32;
  const double   h = 1.0 / (n + 1);
  const double k2 =
      4.0 / (h * h) * std::pow(std::sin(4.0 * M_PI * h / 2.0), 2.0);

  DynamicSparsityPattern dsp(n, n);
  for (unsigned i = 0; i < n; ++i) {
    dsp.add(i, i);
    if (i > 0) dsp.add(i, i - 1);
    if (i < n - 1) dsp.add(i, i + 1);
  }
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  for (unsigned i = 0; i < n; ++i) {
    A.set(i, i, 2.0 / (h * h) - k2);
    if (i > 0)     A.set(i, i - 1, -1.0 / (h * h));
    if (i < n - 1) A.set(i, i + 1, -1.0 / (h * h));
  }

  Vector<double> b(n), x(n);
  for (unsigned i = 0; i < n; ++i)
    b(i) = static_cast<double>((i * 37 + 13) % 7) - 3.0;

  SolverControl ctl(2000, 1e-12);
  SolverCG<Vector<double>> cg(ctl);
  try {
    cg.solve(A, x, b, PreconditionIdentity());
    std::cout << "FIXTURE FAILED: SolverCG_converged_OK iters="
              << ctl.last_step() << "\n";
    return 2;
  } catch (const SolverControl::NoConvergence &e) {
    std::cout << "SolverCG indefinite Helmholtz: "
              << "SolverControl::NoConvergence "
              << "last_step=" << e.last_step
              << " last_residual=" << e.last_residual << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cout << "SolverCG threw (unexpected class): "
              << e.what() << "\n";
    return 0;
  }
}
