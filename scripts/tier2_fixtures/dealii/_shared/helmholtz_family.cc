// Shared translation unit for the Helmholtz Signal family.
//
// deal.II here is a REAL-scalar build, so a complex Helmholtz problem is split
// into a 2-component FESystem carrying (u_re, u_im). That split is exactly what
// helmholtz#1 is about.
//
// usage: helmholtz_family <probe>
//   complex_split_coupling
// Env T2_MUTATE=1 runs the CORRECT variant.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

// helmholtz#1 — omitting the imaginary coupling decouples the two components,
// and the absorbing boundary term silently stops absorbing.
static int complex_split_coupling()
{
  const double k = 8.0;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(5);
  FESystem<dim>   fe(FE_Q<dim>(1) ^ 2);   // (u_re, u_im)
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  const bool couple = mutate();
  QGauss<dim>       quad(2);
  QGauss<dim - 1>   fquad(2);
  FEValues<dim>     fev(fe, quad,
                    update_values | update_gradients |
                      update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> ffv(fe, fquad,
                        update_values | update_quadrature_points |
                          update_JxW_values);
  const unsigned int n = fe.dofs_per_cell;
  FullMatrix<double> cm(n, n);
  Vector<double>     cv(n);
  std::vector<types::global_dof_index> local(n);

  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const Point<dim> &x = fev.quadrature_point(q);
          // A source with a non-zero IMAGINARY part.
          const double f_re = std::exp(-200.0 * ((x[0] - 0.5) * (x[0] - 0.5) +
                                                 (x[1] - 0.5) * (x[1] - 0.5)));
          const double f_im = 0.5 * f_re;
          for (unsigned int i = 0; i < n; ++i)
            {
              const unsigned int ci = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < n; ++j)
                {
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  if (ci == cj)
                    cm(i, j) += (fev.shape_grad(i, q) * fev.shape_grad(j, q) -
                                 k * k * fev.shape_value(i, q) *
                                   fev.shape_value(j, q)) *
                                fev.JxW(q);
                }
              cv(i) += (ci == 0 ? f_re : f_im) * fev.shape_value(i, q) *
                       fev.JxW(q);
            }
        }
      // Absorbing boundary term -i k u v dS. In the split system that is
      // PURELY off-diagonal: it maps the real block onto the imaginary one and
      // back. Omitting it is the mistake.
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            ffv.reinit(cell, face);
            for (unsigned int q = 0; q < fquad.size(); ++q)
              for (unsigned int i = 0; i < n; ++i)
                {
                  const unsigned int ci = fe.system_to_component_index(i).first;
                  for (unsigned int j = 0; j < n; ++j)
                    {
                      const unsigned int cj =
                        fe.system_to_component_index(j).first;
                      if (!couple || ci == cj)
                        continue;
                      const double s = (ci == 0 && cj == 1) ? +k : -k;
                      cm(i, j) += s * ffv.shape_value(i, q) *
                                  ffv.shape_value(j, q) * ffv.JxW(q);
                    }
                }
          }
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, cv, local, A, rhs);
    }

  SparseDirectUMFPACK dir;
  dir.initialize(A);
  sol = rhs;
  dir.solve(sol);

  double re_max = 0.0, im_max = 0.0;
  for (const auto &cell : dof.active_cell_iterators())
    {
      cell->get_dof_indices(local);
      for (unsigned int i = 0; i < n; ++i)
        {
          const double v = std::abs(sol(local[i]));
          if (fe.system_to_component_index(i).first == 0)
            re_max = std::max(re_max, v);
          else
            im_max = std::max(im_max, v);
        }
    }
  std::cout << "imaginary_coupling_assembled=" << (couple ? "true" : "false")
            << std::endl;
  std::cout << "n_dofs=" << dof.n_dofs() << std::endl;
  std::cout << "real_component_linfty=" << re_max << std::endl;
  std::cout << "imaginary_component_linfty=" << im_max << std::endl;
  // Without the coupling the two blocks are independent, so the imaginary part
  // is driven ONLY by the imaginary source — it is not zero, which is what the
  // claim says. What IS observable is that the two components are then exact
  // multiples of each other, because the same operator acts on both.
  const double ratio = (re_max > 0.0) ? im_max / re_max : 0.0;
  std::cout << "imag_over_real=" << ratio << std::endl;
  const bool proportional = std::abs(ratio - 0.5) < 1e-8;
  const bool exactly_zero = im_max == 0.0;
  std::cout << "imaginary_component_exactly_zero="
            << (exactly_zero ? "true" : "false") << std::endl;
  std::cout << "components_are_proportional_to_the_source_split="
            << (proportional ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (proportional
                  ? "no_coupling_leaves_the_two_blocks_independent"
                  : "blocks_are_coupled")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "complex_split_coupling")
    return complex_split_coupling();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
