/* Tier-2 for poisson#3: hanging-node constraints not applied.
 *
 * REWRITTEN 2026-08-03 after an execution pass on deal.II 9.8.0-pre.
 * The previous version of this fixture asserted the catalog's old
 * Signal — "the matrix is non-symmetric and CG breaks down at the
 * first iteration" — by solving an unconstrained system with a
 * 20-iteration budget and reporting the resulting
 * SolverControl::NoConvergence. That proved nothing: the same budget
 * fails WITH constraints too, and the premise is false.
 *
 * What actually happens when hanging-node constraints are omitted:
 *   * the assembled matrix stays EXACTLY symmetric,
 *   * SolverCG converges normally, in the same iteration count as
 *     the constrained run,
 *   * only the ANSWER is wrong — the L2 error against the exact
 *     solution stops improving under refinement.
 *
 * This fixture measures all three on the manufactured solution
 * u = sin(pi x) sin(pi y) (f = 2 pi^2 u, homogeneous Dirichlet) over
 * five adaptive cycles that refine the lower-left quadrant, so every
 * cycle has hanging nodes.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>

using namespace dealii;

namespace
{
  const double PI = 3.14159265358979323846;

  struct Exact : Function<2>
  {
    double value(const Point<2> &p, unsigned int) const override
    {
      return std::sin(PI * p[0]) * std::sin(PI * p[1]);
    }
  };

  struct Rhs : Function<2>
  {
    double value(const Point<2> &p, unsigned int) const override
    {
      return 2 * PI * PI * std::sin(PI * p[0]) * std::sin(PI * p[1]);
    }
  };

  struct Result
  {
    double max_asymmetry;
    bool   cg_converged;
    unsigned int cg_steps;
    double l2;
  };

  Result run(bool use_constraints, unsigned int n_cycles)
  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria, 0, 1);
    tria.refine_global(2);
    FE_Q<2>       fe(1);
    DoFHandler<2> dh(tria);
    Result        r{0.0, true, 0, 0.0};

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        if (cycle > 0)
          {
            for (auto &cell : tria.active_cell_iterators())
              if (cell->center()[0] < 0.5 && cell->center()[1] < 0.5)
                cell->set_refine_flag();
            tria.execute_coarsening_and_refinement();
          }
        dh.distribute_dofs(fe);

        AffineConstraints<double> cons;
        if (use_constraints)
          DoFTools::make_hanging_node_constraints(dh, cons);
        cons.close();

        DynamicSparsityPattern dsp(dh.n_dofs());
        DoFTools::make_sparsity_pattern(dh, dsp, cons, false);
        SparsityPattern sp;
        sp.copy_from(dsp);
        SparseMatrix<double> A(sp);
        Vector<double>       b(dh.n_dofs()), x(dh.n_dofs());

        QGauss<2>   quad(3);
        FEValues<2> fv(fe, quad,
                       update_values | update_gradients |
                         update_quadrature_points | update_JxW_values);
        const unsigned int dpc = fe.dofs_per_cell;
        FullMatrix<double> cm(dpc, dpc);
        Vector<double>     cr(dpc);
        std::vector<types::global_dof_index> ldi(dpc);
        Rhs rhs;

        for (auto &cell : dh.active_cell_iterators())
          {
            fv.reinit(cell);
            cm = 0;
            cr = 0;
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                const double f = rhs.value(fv.quadrature_point(q), 0);
                for (unsigned int i = 0; i < dpc; ++i)
                  {
                    for (unsigned int j = 0; j < dpc; ++j)
                      cm(i, j) += fv.shape_grad(i, q) * fv.shape_grad(j, q) *
                                  fv.JxW(q);
                    cr(i) += fv.shape_value(i, q) * f * fv.JxW(q);
                  }
              }
            cell->get_dof_indices(ldi);
            if (use_constraints)
              cons.distribute_local_to_global(cm, cr, ldi, A, b);
            else
              {
                for (unsigned int i = 0; i < dpc; ++i)
                  {
                    for (unsigned int j = 0; j < dpc; ++j)
                      A.add(ldi[i], ldi[j], cm(i, j));
                    b(ldi[i]) += cr(i);
                  }
              }
          }

        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dh, 0,
                                                 Functions::ZeroFunction<2>(),
                                                 bv);
        MatrixTools::apply_boundary_values(bv, A, x, b);

        double asym = 0, nrm = 0;
        for (unsigned int i = 0; i < dh.n_dofs(); ++i)
          for (auto it = A.begin(i); it != A.end(i); ++it)
            {
              const double aij = it->value();
              const double aji = A.el(it->column(), i);
              asym = std::max(asym, std::abs(aij - aji));
              nrm  = std::max(nrm, std::abs(aij));
            }
        r.max_asymmetry = (nrm > 0 ? asym / nrm : 0.0);

        SolverControl sc(2000, 1e-12 * b.l2_norm());
        SolverCG<Vector<double>> cg(sc);
        PreconditionSSOR<SparseMatrix<double>> pre;
        pre.initialize(A, 1.2);
        try
          {
            cg.solve(A, x, b, pre);
            r.cg_converged = true;
            r.cg_steps     = sc.last_step();
          }
        catch (const std::exception &)
          {
            r.cg_converged = false;
          }
        if (use_constraints)
          cons.distribute(x);

        Vector<float> diff(tria.n_active_cells());
        VectorTools::integrate_difference(dh, x, Exact(), diff, QGauss<2>(4),
                                          VectorTools::L2_norm);
        r.l2 = VectorTools::compute_global_error(tria, diff,
                                                 VectorTools::L2_norm);
      }
    return r;
  }
} // namespace

int main()
{
  const unsigned int cycles = 5;
  const Result without = run(false, cycles);
  const Result with    = run(true, cycles);

  std::cout << "hanging_nodes_no_condense (deal.II " << DEAL_II_PACKAGE_VERSION
            << ")\n";
  std::cout << "max_asymmetry_without_constraints=" << without.max_asymmetry
            << "\n";
  std::cout << "CG_converged_normally_without_constraints="
            << (without.cg_converged ? "True" : "False")
            << " steps=" << without.cg_steps << "\n";
  std::cout << "CG_converged_normally_with_constraints="
            << (with.cg_converged ? "True" : "False")
            << " steps=" << with.cg_steps << "\n";
  std::cout << "L2_without_constraints=" << without.l2
            << " L2_with_constraints=" << with.l2 << "\n";
  std::cout << "L2_error_worse_without_constraints="
            << (without.l2 > with.l2 ? "True" : "False") << "\n";

  const bool as_documented =
    (without.max_asymmetry == 0.0) && without.cg_converged &&
    with.cg_converged && (without.l2 > with.l2);
  std::cout << (as_documented ? "FIXTURE_OK" : "FIXTURE_MISMATCH") << "\n";
  return as_documented ? 0 : 1;
}
