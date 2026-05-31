/* Tier-2 (Debug-build dependent): SparseMatrix::vmult with
 * mismatched sizes raises ExcDimensionMismatch.
 *
 * The Assert macro that triggers this exception is compiled out
 * in Release builds; the deal.II conda binary is Release, so
 * this fixture requires the Debug rebuild at
 * ~/Schreibtisch/dealii-debug.
 */

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int main()
{
  DynamicSparsityPattern dsp(3, 3);
  for (unsigned i = 0; i < 3; ++i) dsp.add(i, i);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  for (unsigned i = 0; i < 3; ++i) A.set(i, i, 1.0);

  // src has wrong size (2 instead of 3).
  Vector<double> src(2), dst(3);
  src(0) = 1.0;
  src(1) = 1.0;
  try {
    A.vmult(dst, src);
  } catch (const std::exception &e) {
    std::cerr << "Caught: " << e.what() << '\n';
    return 1;
  }
  std::cerr << "ERROR: SparseMatrix::vmult accepted mismatched "
            << "src.size()=" << src.size()
            << " for A.n()=" << A.n() << "\n";
  return 2;
}
