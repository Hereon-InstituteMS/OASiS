/* Tier-2 fixture verifying hyperelasticity pitfall #11.
 *
 * The bug: on a vector-valued FESystem, calling the SCALAR
 * fe_values.get_function_gradients(solution, gradients)
 * — instead of fe_values[FEValuesExtractors::Vector(0)]
 *   .get_function_gradients(solution, gradients)
 * — fails to compile because the scalar signature expects a
 * Vector<double> of gradient values, not a vector-of-vectors.
 *
 * Expected behaviour: this file MUST NOT compile. The Signal:
 * clause promises "no matching function for call to
 * 'FEValuesBase<dim>::get_function_gradients'" — the Tier-2
 * runner greps for that substring in g++ output.
 */

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <vector>

using namespace dealii;

template <int dim>
void buggy_assemble(const FEValues<dim> &fe_values,
                    const Vector<double> &solution)
{
  // The SCALAR signature signature wants a std::vector<Tensor<1, dim>>
  // for a SCALAR FE. Calling it with a vector solution on a
  // FESystem(FE_Q, dim) at compile time is what produces the
  // "no matching function for call to" error.
  //
  // Here we deliberately pass a std::vector<Tensor<2, dim>>
  // (rank-2 tensor, as if the FE were vector-valued) — that
  // signature does not exist on the scalar overload, so the
  // compiler must reject it.
  std::vector<Tensor<2, dim>> gradients(1);
  fe_values.get_function_gradients(solution, gradients);
}

int main() { return 0; }
