// Shared translation unit for the DG-transport Signal family (dg_transport::*).
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: dg_family <probe>
//   gmres_without_face_terms | handrolled_forgets_periodic
//   | block_sweep_regimes | dof_wise_renumbering | dg_strong_dirichlet
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

// ===========================================================================
// Upwind DG advection-reaction:  b.grad(u) + sigma*u = 0 with inflow datum g,
// FE_DGQ elements, weak (numerical-flux) boundary conditions.  The reaction
// term keeps the CELL-LOCAL blocks invertible, which is what lets the
// "no interface terms" variant of dg_transport::1 be handed to a Krylov solver
// at all.
// ===========================================================================
enum Renum
{
  RENUM_NONE,
  RENUM_CUTHILL,
  RENUM_DOWNSTREAM_CELLWISE,
  RENUM_DOWNSTREAM_DOFWISE
};

// FIELD_ROT_CORNER is rotation about the corner (0,0): every characteristic is a
// circular arc that enters through y = 1 and leaves through x = 1, so the problem
// is well posed and NO downstream cell ordering exists -- the entry's case (c).
enum Field
{
  FIELD_CONST,
  FIELD_ROT_CORNER
};

struct DGAR
{
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       rhs, sol;
  Tensor<1, dim>       b;
  double               sigma;
  Field                field;

  DGAR(unsigned int degree, double reaction, double bx = 1.0, double by = 0.3,
       Field f = FIELD_CONST)
    : fe(degree)
    , dof(tria)
    , sigma(reaction)
    , field(f)
  {
    b[0] = bx;
    b[1] = by;
  }

  Tensor<1, dim> beta(const Point<dim> &p) const
  {
    if (field == FIELD_ROT_CORNER)
      {
        Tensor<1, dim> r;
        r[0] = p[1];
        r[1] = -p[0];
        return r;
      }
    return b;
  }

  // inflow datum. DATUM_LAYER (default): 1 below y = 0.5, 0 above -- an O(1)
  // internal layer that a correct upwind DG carries downstream with an O(1) jump
  // across faces. DATUM_LEFT_HALF: 1 for x < 0.5, which is a non-trivial datum
  // on the inflow faces of BOTH fields, including the rotation field whose
  // inflow boundary is y = 1.
  enum Datum
  {
    DATUM_LAYER,
    DATUM_LEFT_HALF
  };
  Datum datum = DATUM_LAYER;

  double g(const Point<dim> &p) const
  {
    if (datum == DATUM_LEFT_HALF)
      return (p[0] < 0.5) ? 1.0 : 0.0;
    return (p[1] < 0.5) ? 1.0 : 0.0;
  }

  void setup(unsigned int refine, Renum renum = RENUM_NONE)
  {
    if (tria.n_active_cells() == 0)
      {
        GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
        tria.refine_global(refine);
      }
    dof.distribute_dofs(fe);
    if (renum == RENUM_CUTHILL)
      DoFRenumbering::Cuthill_McKee(dof);
    else if (renum == RENUM_DOWNSTREAM_CELLWISE)
      DoFRenumbering::downstream(dof, b, false);
    else if (renum == RENUM_DOWNSTREAM_DOFWISE)
      DoFRenumbering::downstream(dof, b, true);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  // with_interior_faces=false reproduces "FEValues alone, no FEInterfaceValues"
  void assemble(bool with_interior_faces)
  {
    A   = 0.0;
    rhs = 0.0;
    const unsigned int n = fe.dofs_per_cell;
    QGauss<dim>        quad(fe.degree + 2);
    QGauss<dim - 1>    fquad(fe.degree + 2);
    FEValues<dim>      fev(fe, quad,
                           update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);
    FEFaceValues<dim>  ffv(fe, fquad,
                           update_values | update_normal_vectors |
                             update_quadrature_points | update_JxW_values);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_quadrature_points | update_JxW_values);
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);

    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            const Tensor<1, dim> bq = beta(fev.quadrature_point(q));
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                cm(i, j) +=
                  (-fev.shape_value(j, q) * (bq * fev.shape_grad(i, q)) +
                   sigma * fev.shape_value(i, q) * fev.shape_value(j, q)) *
                  fev.JxW(q);
          }
        cell->get_dof_indices(local);
        A.add(local, cm);
        rhs.add(local, cv);

        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              ffv.reinit(cell, f);
              cm = 0.0;
              cv = 0.0;
              for (unsigned int q = 0; q < fquad.size(); ++q)
                {
                  const double bn =
                    beta(ffv.quadrature_point(q)) * ffv.normal_vector(q);
                  if (bn > 0)   // outflow: into the operator
                    for (unsigned int i = 0; i < n; ++i)
                      for (unsigned int j = 0; j < n; ++j)
                        cm(i, j) += bn * ffv.shape_value(j, q) *
                                    ffv.shape_value(i, q) * ffv.JxW(q);
                  else          // inflow: datum into the right-hand side
                    for (unsigned int i = 0; i < n; ++i)
                      cv(i) -= bn * g(ffv.quadrature_point(q)) *
                               ffv.shape_value(i, q) * ffv.JxW(q);
                }
              A.add(local, cm);
              rhs.add(local, cv);
            }

        if (!with_interior_faces)
          continue;

        for (const auto f : cell->face_indices())
          {
            if (cell->face(f)->at_boundary())
              continue;
            const auto ncell = cell->neighbor(f);
            if (ncell->active_cell_index() < cell->active_cell_index())
              continue;   // visit each interior face once
            const unsigned int nf = cell->neighbor_of_neighbor(f);
            fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                       numbers::invalid_unsigned_int);
            const unsigned int ni = fiv.n_current_interface_dofs();
            FullMatrix<double> fm(ni, ni);
            fm = 0.0;
            for (unsigned int q = 0; q < fquad.size(); ++q)
              {
                const double bn =
                  beta(fiv.quadrature_point(q)) * fiv.normal(q);
                const bool take_here = (bn > 0);   // upwind
                for (unsigned int i = 0; i < ni; ++i)
                  for (unsigned int j = 0; j < ni; ++j)
                    fm(i, j) += bn * fiv.shape_value(take_here, j, q) *
                                fiv.jump_in_shape_values(i, q) * fiv.JxW(q);
              }
            const auto idx = fiv.get_interface_dof_indices();
            A.add(idx, fm);
          }
      }
  }

  // how many nonzero matrix entries connect dofs living on DIFFERENT cells
  unsigned int inter_cell_couplings() const
  {
    std::vector<unsigned int>            owner(dof.n_dofs(), 0);
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (auto i : local)
          owner[i] = cell->active_cell_index();
      }
    unsigned int cnt = 0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        if (owner[it->column()] != owner[i] && std::abs(it->value()) > 1e-14)
          ++cnt;
    return cnt;
  }

  // mean and max |u^+ - u^-| over the interior faces, the quantity
  // dg_transport::1 says collapses to 1e-8 when the interface terms are missing
  double mean_absolute_interior_jump(double *max_jump = nullptr) const
  {
    if (max_jump != nullptr)
      *max_jump = 0.0;
    QGauss<dim - 1>        fquad(fe.degree + 2);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
    double total = 0.0, length = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      for (const auto f : cell->face_indices())
        {
          if (cell->face(f)->at_boundary())
            continue;
          const auto ncell = cell->neighbor(f);
          if (ncell->active_cell_index() < cell->active_cell_index())
            continue;
          const unsigned int nf = cell->neighbor_of_neighbor(f);
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                     numbers::invalid_unsigned_int);
          const auto idx = fiv.get_interface_dof_indices();
          for (unsigned int q = 0; q < fquad.size(); ++q)
            {
              double jump = 0.0;
              for (unsigned int i = 0; i < idx.size(); ++i)
                jump += sol(idx[i]) * fiv.jump_in_shape_values(i, q);
              total += std::abs(jump) * fiv.JxW(q);
              length += fiv.JxW(q);
              if (max_jump != nullptr)
                *max_jump = std::max(*max_jump, std::abs(jump));
            }
        }
    return total / length;
  }

  double max_abs_solution() const
  {
    double m = 0.0;
    for (unsigned int i = 0; i < sol.size(); ++i)
      m = std::max(m, std::abs(sol(i)));
    return m;
  }
};

// ---------------------------------------------------------------------------
// dg_transport::1 -- "FEValues alone produces only cell-interior
// contributions.  Signal: SolverGMRES converges but DataOut shows a smooth
// (non-DG) solution; jump-across-face values ... are 1e-8 (effectively zero)
// where they should be O(1) for upwind DG."
//
// Both halves of that Signal are measured here: whether GMRES reports
// convergence, and what the interior jumps actually are, against a reference
// assembled WITH FEInterfaceValues in the same invocation.
// ---------------------------------------------------------------------------
static int gmres_without_face_terms()
{
  const bool with_faces = mutate();
  std::cout << "interior_face_terms="
            << (with_faces ? "FEInterfaceValues" : "none_FEValues_only")
            << std::endl;

  DGAR t(1, 1.0);
  t.setup(4);
  t.assemble(with_faces);
  std::cout << "n_active_cells=" << t.tria.n_active_cells()
            << " n_dofs=" << t.dof.n_dofs() << std::endl;
  const unsigned int couplings = t.inter_cell_couplings();
  std::cout << "n_inter_cell_matrix_couplings=" << couplings << std::endl;
  std::cout << "cells_are_coupled=" << (couplings > 0 ? "true" : "false")
            << std::endl;

  // The claim names SolverGMRES, so SolverGMRES is what runs.
  SolverControl control(2000, 1e-10 * std::max(1.0, t.rhs.l2_norm()));
  SolverGMRES<Vector<double>> gmres(control);
  PreconditionJacobi<SparseMatrix<double>> prec;
  prec.initialize(t.A);
  bool converged = true;
  try
    {
      gmres.solve(t.A, t.sol, t.rhs, prec);
    }
  catch (const std::exception &e)
    {
      converged = false;
      const std::string w(e.what());
      std::cout << "gmres_exception_first_line="
                << w.substr(0, std::min<size_t>(w.find('\n'), 200))
                << std::endl;
    }
  std::cout << "gmres_converged=" << (converged ? "true" : "false")
            << std::endl;
  std::cout << "gmres_steps=" << control.last_step()
            << " gmres_last_residual=" << control.last_value() << std::endl;

  double       max_jump = 0.0;
  const double jump     = t.mean_absolute_interior_jump(&max_jump);
  const double scale    = t.max_abs_solution();
  std::cout << "max_abs_solution=" << scale << std::endl;
  std::cout << "mean_absolute_interior_jump=" << jump
            << " max_absolute_interior_jump=" << max_jump << std::endl;
  const double relative_jump = (scale > 0.0) ? jump / scale : 0.0;
  std::cout << "relative_mean_interior_jump=" << relative_jump << std::endl;

  // Reference: the SAME problem assembled WITH the interface terms and solved
  // directly, so "what the jumps should be" is measured, not asserted.
  DGAR r(1, 1.0);
  r.setup(4);
  r.assemble(true);
  SparseDirectUMFPACK direct;
  direct.initialize(r.A);
  direct.vmult(r.sol, r.rhs);
  double       ref_max   = 0.0;
  const double ref_jump  = r.mean_absolute_interior_jump(&ref_max);
  const double ref_scale = r.max_abs_solution();
  std::cout << "reference_relative_mean_interior_jump=" << ref_jump / ref_scale
            << " reference_max_interior_jump=" << ref_max << std::endl;
  std::cout << "reference_max_abs_solution=" << ref_scale << std::endl;

  const bool tiny = relative_jump < 1e-6;
  std::cout << "jump_collapses_to_effectively_zero_as_the_claim_says="
            << (tiny ? "true" : "false") << std::endl;
  // The inflow datum is bounded by 1, so a correct upwind DG answer is too (up
  // to a small overshoot).  This is the real Release-build observable, and it is
  // the opposite of the claim's "smooth non-DG solution".
  const bool blown_up = scale > 10.0 * ref_scale;
  std::cout << "amplitude_exceeds_the_reference_by_an_order_of_magnitude="
            << (blown_up ? "true" : "false") << std::endl;
  std::cout << "cells_uncoupled_and_jumps_are_not_small="
            << ((couplings == 0 && !tiny) ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((couplings == 0)
                  ? "omitting_feinterfacevalues_is_silent_and_leaves_the_cells_"
                    "uncoupled"
                  : "interface_terms_couple_the_cells")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport::3 -- how much PreconditionBlockSSOR buys is not a number.
//
// For each mesh in a refinement sequence the probe measures
//   (i)  the ONE-APPLICATION residual ||b - A*P(b)|| / ||b||, which separates
//        "P is an exact block-triangular solve" (~1e-15) from "P is an ordinary
//        preconditioner" (O(1)), and
//   (ii) the GMRES iteration counts with block SSOR, point SSOR and point
//        Jacobi on the identical operator, rhs and relative tolerance.
// The default run uses the constant field the entry names as downstream-
// compatible; T2_MUTATE=1 uses the corner-rotation field, the entry's case (c),
// where no downstream ordering exists.
// ===========================================================================
struct SweepResult
{
  unsigned int dofs;
  double       one_application_residual;
  unsigned int block_iters, ssor_iters, jacobi_iters;
};

template <class Prec>
static unsigned int gmres_count(const SparseMatrix<double> &A,
                                const Vector<double> &rhs, Prec &prec,
                                unsigned int budget = 5000)
{
  Vector<double>              x(rhs.size());
  SolverControl               control(budget, 1e-8 * rhs.l2_norm());
  SolverGMRES<Vector<double>> gmres(control);
  try
    {
      gmres.solve(A, x, rhs, prec);
    }
  catch (const std::exception &)
    {
      return budget;   // exhausted the budget
    }
  return control.last_step();
}

static SweepResult measure_sweep(unsigned int refine, Field field)
{
  DGAR t(1, 0.0, 1.0, 1.0, field);   // pure transport, no reaction
  t.datum = DGAR::DATUM_LEFT_HALF;
  t.setup(refine);
  t.assemble(true);

  SweepResult r;
  r.dofs = t.dof.n_dofs();

  PreconditionBlockSSOR<SparseMatrix<double>, double> block;
  block.initialize(t.A,
                   PreconditionBlock<SparseMatrix<double>, double>::
                     AdditionalData(t.fe.dofs_per_cell, 1.0));
  Vector<double> z(t.dof.n_dofs()), res(t.dof.n_dofs());
  block.vmult(z, t.rhs);
  t.A.vmult(res, z);
  res -= t.rhs;
  r.one_application_residual = res.l2_norm() / t.rhs.l2_norm();

  r.block_iters = gmres_count(t.A, t.rhs, block);
  PreconditionSSOR<SparseMatrix<double>> pssor;
  pssor.initialize(t.A, 1.0);
  r.ssor_iters = gmres_count(t.A, t.rhs, pssor);
  PreconditionJacobi<SparseMatrix<double>> pjac;
  pjac.initialize(t.A);
  r.jacobi_iters = gmres_count(t.A, t.rhs, pjac);
  return r;
}

static int block_sweep_regimes()
{
  const Field field = mutate() ? FIELD_ROT_CORNER : FIELD_CONST;
  std::cout << "field_under_test="
            << (field == FIELD_CONST ? "constant_beta_1_1"
                                     : "rotation_about_the_corner")
            << std::endl;
  const unsigned int levels[3] = {3, 4, 5};
  SweepResult        r[3];
  for (int k = 0; k < 3; ++k)
    {
      r[k] = measure_sweep(levels[k], field);
      std::cout << "refine=" << levels[k] << " n_dofs=" << r[k].dofs
                << " one_application_residual=" << r[k].one_application_residual
                << " block_ssor_iters=" << r[k].block_iters
                << " point_ssor_iters=" << r[k].ssor_iters
                << " point_jacobi_iters=" << r[k].jacobi_iters
                << " jacobi_over_block="
                << double(r[k].jacobi_iters) / double(r[k].block_iters)
                << std::endl;
    }
  const bool exact = r[0].one_application_residual < 1e-12 &&
                     r[2].one_application_residual < 1e-12;
  std::cout << "block_sweep_is_an_exact_solve=" << (exact ? "true" : "false")
            << std::endl;
  const bool block_flat = r[2].block_iters <= r[0].block_iters + 1;
  std::cout << "block_iteration_count_is_mesh_independent="
            << (block_flat ? "true" : "false") << std::endl;
  const double ratio_coarse =
    double(r[0].jacobi_iters) / double(r[0].block_iters);
  const double ratio_fine = double(r[2].jacobi_iters) / double(r[2].block_iters);
  std::cout << "jacobi_over_block_coarsest=" << ratio_coarse
            << " jacobi_over_block_finest=" << ratio_fine << std::endl;
  const bool ratio_grows = ratio_fine > 3.0 * ratio_coarse;
  std::cout << "speedup_ratio_grows_by_more_than_3x_across_refinements="
            << (ratio_grows ? "true" : "false") << std::endl;
  // The entry's clause (e): point SSOR is not reliably poor, because an SSOR
  // sweep also follows the DoF ordering.
  const bool ssor_close = r[2].ssor_iters < 4 * r[2].block_iters;
  std::cout << "point_ssor_is_within_4x_of_the_block_version_on_the_finest_mesh="
            << (ssor_close ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((exact && ratio_grows)
                  ? "block_sweep_is_a_direct_solve_and_the_ratio_is_a_property_"
                    "of_the_mesh"
                  : "block_sweep_is_an_ordinary_preconditioner")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport::4 -- DoFRenumbering::downstream's third argument.
// dof_wise_renumbering=true interleaves the cell dofs, so the "blocks"
// PreconditionBlock* inverts are no longer cells.
// ===========================================================================
static int dof_wise_renumbering()
{
  const bool dof_wise = !mutate();
  std::cout << "dof_wise_renumbering=" << (dof_wise ? "true" : "false")
            << std::endl;
  DGAR t(1, 0.0, 1.0, 1.0);
  t.datum = DGAR::DATUM_LEFT_HALF;
  t.setup(4, dof_wise ? RENUM_DOWNSTREAM_DOFWISE : RENUM_DOWNSTREAM_CELLWISE);
  t.assemble(true);
  std::cout << "n_dofs=" << t.dof.n_dofs()
            << " block_size=" << t.fe.dofs_per_cell << std::endl;

  // The entry's premise, measured: are a cell's dofs still contiguous?
  unsigned int contiguous = 0, total = 0;
  std::vector<types::global_dof_index> local(t.fe.dofs_per_cell);
  for (const auto &cell : t.dof.active_cell_iterators())
    {
      cell->get_dof_indices(local);
      const auto mn = *std::min_element(local.begin(), local.end());
      const auto mx = *std::max_element(local.begin(), local.end());
      if (mx - mn + 1 == t.fe.dofs_per_cell)
        ++contiguous;
      ++total;
    }
  std::cout << "cells_with_contiguous_dofs=" << contiguous << " of " << total
            << std::endl;
  std::cout << "block_structure_still_matches_the_cells="
            << ((contiguous == total) ? "true" : "false") << std::endl;

  PreconditionBlockSSOR<SparseMatrix<double>, double> block;
  block.initialize(t.A,
                   PreconditionBlock<SparseMatrix<double>, double>::
                     AdditionalData(t.fe.dofs_per_cell, 1.0));
  Vector<double> z(t.dof.n_dofs()), res(t.dof.n_dofs());
  block.vmult(z, t.rhs);
  t.A.vmult(res, z);
  res -= t.rhs;
  const double one_app = res.l2_norm() / t.rhs.l2_norm();
  std::cout << "one_application_residual=" << one_app << std::endl;
  const bool finite_res = std::isfinite(one_app);
  std::cout << "one_application_residual_is_finite="
            << (finite_res ? "true" : "false") << std::endl;
  std::cout << "one_application_residual_is_nan_or_astronomical="
            << ((!finite_res || one_app > 1e6) ? "true" : "false") << std::endl;

  Vector<double>              x(t.dof.n_dofs());
  SolverControl               control(200, 1e-8 * t.rhs.l2_norm());
  SolverGMRES<Vector<double>> gmres(control);
  std::string                 outcome = "converged";
  try
    {
      gmres.solve(t.A, x, t.rhs, block);
    }
  catch (const std::exception &e)
    {
      const std::string w(e.what());
      outcome = "threw";
      std::cout << "gmres_exception_head="
                << w.substr(0, std::min<size_t>(w.size(), 300)) << std::endl;
    }
  std::cout << "gmres_outcome=" << outcome
            << " gmres_last_step=" << control.last_step()
            << " gmres_last_value=" << control.last_value() << std::endl;
  const bool nan_value = !std::isfinite(control.last_value());
  std::cout << "gmres_last_value_is_nan=" << (nan_value ? "true" : "false")
            << std::endl;
  std::cout << "gmres_gave_up_at_step_0_or_1="
            << ((control.last_step() <= 1) ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((!finite_res || one_app > 1e6)
                  ? "dof_wise_renumbering_silently_destroys_the_block_"
                    "preconditioner"
                  : "block_preconditioner_survives_the_renumbering")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport::2 -- MeshWorker::mesh_loop's dispatch versus a hand-rolled
// cell/face/boundary loop on a mesh WITH PERIODIC FACES.
//
// Same operator as above but on a triangulation whose y = 0 and y = 1 faces are
// matched by Triangulation::add_periodicity, with b = (1, 0.5) so the transported
// profile leaves through the top and must re-enter at the bottom.  The
// hand-rolled loop dispatches on cell->face(f)->at_boundary() alone, which is
// still TRUE for a periodic face, so the periodic pair is assembled as a
// physical boundary -- exactly the omission the entry describes.
// ===========================================================================
struct CopyDataFace
{
  FullMatrix<double>                   matrix;
  std::vector<types::global_dof_index> joint_dof_indices;
};

struct CopyDataDG
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<CopyDataFace>            face_data;

  template <class Iterator>
  void reinit(const Iterator &cell, unsigned int n)
  {
    cell_matrix.reinit(n, n);
    cell_matrix = 0.0;
    cell_rhs.reinit(n);
    cell_rhs = 0.0;
    local_dof_indices.resize(n);
    cell->get_dof_indices(local_dof_indices);
    face_data.clear();
  }
};

struct DGPeriodic
{
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       rhs, sol;
  Tensor<1, dim>       b;

  DGPeriodic(unsigned int degree)
    : fe(degree)
    , dof(tria)
  {
    b[0] = 1.0;
    b[1] = 0.5;
  }

  // inflow datum on the LEFT boundary only; a generic inflow face elsewhere
  // gets the usual default of zero.
  static double g_left(const Point<dim> &p)
  {
    return 1.0 + std::sin(2.0 * numbers::PI * p[1]);
  }

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, true);   // colorize: ids 0..3
    std::vector<GridTools::PeriodicFacePair<
      typename Triangulation<dim>::cell_iterator>>
      matched;
    GridTools::collect_periodic_faces(tria, 2, 3, 1, matched);
    tria.add_periodicity(matched);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  void volume_terms(const FEValues<dim> &fev, FullMatrix<double> &cm) const
  {
    const unsigned int n = fe.dofs_per_cell;
    for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          cm(i, j) += -fev.shape_value(j, q) * (b * fev.shape_grad(i, q)) *
                      fev.JxW(q);
  }

  void boundary_terms(const FEFaceValues<dim> &ffv, types::boundary_id id,
                      FullMatrix<double> &cm, Vector<double> &cv) const
  {
    const unsigned int n = fe.dofs_per_cell;
    for (unsigned int q = 0; q < ffv.n_quadrature_points; ++q)
      {
        const double bn = b * ffv.normal_vector(q);
        if (bn > 0)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cm(i, j) +=
                bn * ffv.shape_value(j, q) * ffv.shape_value(i, q) * ffv.JxW(q);
        else
          {
            const double datum =
              (id == 0) ? g_left(ffv.quadrature_point(q)) : 0.0;
            for (unsigned int i = 0; i < n; ++i)
              cv(i) -= bn * datum * ffv.shape_value(i, q) * ffv.JxW(q);
          }
      }
  }

  void interface_terms(const FEInterfaceValues<dim> &fiv,
                       FullMatrix<double> &fm) const
  {
    const unsigned int ni = fiv.n_current_interface_dofs();
    for (unsigned int q = 0; q < fiv.n_quadrature_points; ++q)
      {
        const double bn        = b * fiv.normal(q);
        const bool   take_here = (bn > 0);
        for (unsigned int i = 0; i < ni; ++i)
          for (unsigned int j = 0; j < ni; ++j)
            fm(i, j) += bn * fiv.shape_value(take_here, j, q) *
                        fiv.jump_in_shape_values(i, q) * fiv.JxW(q);
      }
  }

  // The hand-rolled dispatch: at_boundary() decides, and nothing asks about
  // has_periodic_neighbor().
  void assemble_handrolled()
  {
    A   = 0.0;
    rhs = 0.0;
    const unsigned int     n = fe.dofs_per_cell;
    QGauss<dim>            quad(fe.degree + 2);
    QGauss<dim - 1>        fquad(fe.degree + 2);
    FEValues<dim>          fev(fe, quad,
                               update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim>      ffv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_quadrature_points | update_JxW_values);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        volume_terms(fev, cm);
        cell->get_dof_indices(local);
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              ffv.reinit(cell, f);
              boundary_terms(ffv, cell->face(f)->boundary_id(), cm, cv);
            }
        A.add(local, cm);
        rhs.add(local, cv);
        for (const auto f : cell->face_indices())
          {
            if (cell->face(f)->at_boundary())
              continue;
            const auto ncell = cell->neighbor(f);
            if (ncell->active_cell_index() < cell->active_cell_index())
              continue;
            const unsigned int nf = cell->neighbor_of_neighbor(f);
            fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                       numbers::invalid_unsigned_int);
            FullMatrix<double> fm(fiv.n_current_interface_dofs(),
                                  fiv.n_current_interface_dofs());
            fm = 0.0;
            interface_terms(fiv, fm);
            A.add(fiv.get_interface_dof_indices(), fm);
          }
      }
  }

  void assemble_mesh_loop()
  {
    A   = 0.0;
    rhs = 0.0;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const QGauss<dim>     quad(fe.degree + 2);
    const QGauss<dim - 1> fquad(fe.degree + 2);
    MeshWorker::ScratchData<dim> scratch(
      fe, quad, update_values | update_gradients | update_JxW_values, fquad,
      update_values | update_normal_vectors | update_quadrature_points |
        update_JxW_values);
    CopyDataDG copy;

    const auto cell_worker = [&](const Iterator &cell,
                                 MeshWorker::ScratchData<dim> &s,
                                 CopyDataDG                   &c) {
      const FEValues<dim> &fev = s.reinit(cell);
      c.reinit(cell, fe.dofs_per_cell);
      volume_terms(fev, c.cell_matrix);
    };
    const auto boundary_worker = [&](const Iterator     &cell,
                                     const unsigned int  face_no,
                                     MeshWorker::ScratchData<dim> &s,
                                     CopyDataDG                   &c) {
      const FEFaceValues<dim> &ffv = s.reinit(cell, face_no);
      boundary_terms(ffv, cell->face(face_no)->boundary_id(), c.cell_matrix,
                     c.cell_rhs);
    };
    const auto face_worker = [&](const Iterator &cell, const unsigned int f,
                                 const unsigned int sf, const Iterator &ncell,
                                 const unsigned int nf, const unsigned int nsf,
                                 MeshWorker::ScratchData<dim> &s,
                                 CopyDataDG                   &c) {
      const FEInterfaceValues<dim> &fiv =
        s.reinit(cell, f, sf, ncell, nf, nsf);
      c.face_data.emplace_back();
      CopyDataFace &cdf   = c.face_data.back();
      const unsigned int ni = fiv.n_current_interface_dofs();
      cdf.joint_dof_indices = fiv.get_interface_dof_indices();
      cdf.matrix.reinit(ni, ni);
      cdf.matrix = 0.0;
      interface_terms(fiv, cdf.matrix);
    };
    const auto copier = [&](const CopyDataDG &c) {
      A.add(c.local_dof_indices, c.cell_matrix);
      for (unsigned int i = 0; i < c.local_dof_indices.size(); ++i)
        rhs(c.local_dof_indices[i]) += c.cell_rhs(i);
      for (const auto &cdf : c.face_data)
        A.add(cdf.joint_dof_indices, cdf.matrix);
    };

    MeshWorker::mesh_loop(dof.begin_active(), dof.end(), cell_worker, copier,
                          scratch, copy,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker, face_worker);
  }

  void solve()
  {
    SparseDirectUMFPACK direct;
    direct.initialize(A);
    direct.vmult(sol, rhs);
  }

  // matrix entries linking a dof on a cell at y = 0 with a dof on a cell at
  // y = 1 -- the periodic coupling itself
  unsigned int periodic_couplings() const
  {
    std::vector<char> at_bottom(dof.n_dofs(), 0), at_top(dof.n_dofs(), 0);
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        bool bottom = false, top = false;
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              if (cell->face(f)->boundary_id() == 2)
                bottom = true;
              if (cell->face(f)->boundary_id() == 3)
                top = true;
            }
        if (!bottom && !top)
          continue;
        cell->get_dof_indices(local);
        for (auto i : local)
          {
            if (bottom)
              at_bottom[i] = 1;
            if (top)
              at_top[i] = 1;
          }
      }
    unsigned int cnt = 0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        if (std::abs(it->value()) > 1e-14 &&
            ((at_bottom[i] && at_top[it->column()]) ||
             (at_top[i] && at_bottom[it->column()])))
          ++cnt;
    return cnt;
  }

  // mean |u(x, 1) - u(x, 0)| along the periodic pair: the "kink at the
  // periodic face" the entry describes, measured instead of looked at
  double periodic_mismatch() const
  {
    QGauss<dim - 1>   fq(fe.degree + 2);
    FEFaceValues<dim> fv(fe, fq,
                         update_values | update_quadrature_points |
                           update_JxW_values);
    std::map<long long, double> top;
    std::vector<double>         vals(fq.size());
    for (const auto &cell : dof.active_cell_iterators())
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == 3)
          {
            fv.reinit(cell, f);
            fv.get_function_values(sol, vals);
            for (unsigned int q = 0; q < fq.size(); ++q)
              top[llround(fv.quadrature_point(q)[0] * 1e9)] = vals[q];
          }
    double total = 0.0, length = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == 2)
          {
            fv.reinit(cell, f);
            fv.get_function_values(sol, vals);
            for (unsigned int q = 0; q < fq.size(); ++q)
              {
                const long long key = llround(fv.quadrature_point(q)[0] * 1e9);
                auto            it  = top.find(key);
                if (it == top.end())
                  continue;
                total += std::abs(vals[q] - it->second) * fv.JxW(q);
                length += fv.JxW(q);
              }
          }
    return (length > 0.0) ? total / length : -1.0;
  }

  double relative_asymmetry() const
  {
    double num = 0.0, den = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        {
          const double a = it->value();
          const double b2 = A.el(it->column(), i);
          num += (a - b2) * (a - b2);
          den += a * a;
        }
    return std::sqrt(num) / std::sqrt(den);
  }
};

static int handrolled_forgets_periodic()
{
  const bool use_mesh_loop = mutate();
  std::cout << "assembly_driver="
            << (use_mesh_loop ? "MeshWorker_mesh_loop"
                              : "handrolled_at_boundary_dispatch")
            << std::endl;

  DGPeriodic t(1);
  t.setup(4);
  std::cout << "n_active_cells=" << t.tria.n_active_cells()
            << " n_dofs=" << t.dof.n_dofs() << std::endl;
  if (use_mesh_loop)
    t.assemble_mesh_loop();
  else
    t.assemble_handrolled();
  const unsigned int pc = t.periodic_couplings();
  std::cout << "periodic_interface_matrix_couplings=" << pc << std::endl;
  std::cout << "periodic_pair_is_coupled=" << (pc > 0 ? "true" : "false")
            << std::endl;
  const double asym = t.relative_asymmetry();
  std::cout << "relative_asymmetry=" << asym << std::endl;
  t.solve();
  const double mismatch = t.periodic_mismatch();
  std::cout << "mean_periodic_mismatch=" << mismatch << std::endl;

  // Reference: the same problem through mesh_loop, in the same invocation.
  DGPeriodic r(1);
  r.setup(4);
  r.assemble_mesh_loop();
  r.solve();
  const double ref_mismatch = r.periodic_mismatch();
  const double ref_asym     = r.relative_asymmetry();
  std::cout << "reference_mesh_loop_periodic_mismatch=" << ref_mismatch
            << std::endl;
  std::cout << "reference_mesh_loop_periodic_couplings="
            << r.periodic_couplings() << std::endl;
  std::cout << "reference_mesh_loop_relative_asymmetry=" << ref_asym
            << std::endl;
  double diff = 0.0;
  for (unsigned int i = 0; i < t.sol.size(); ++i)
    diff = std::max(diff, std::abs(t.sol(i) - r.sol(i)));
  std::cout << "max_difference_from_the_mesh_loop_answer=" << diff << std::endl;

  const bool kinked = mismatch > 20.0 * std::max(ref_mismatch, 1e-14);
  std::cout << "solution_is_discontinuous_across_the_periodic_pair="
            << (kinked ? "true" : "false") << std::endl;
  // The entry also promises the two assemblies differ in SYMMETRY.  Upwind
  // transport is non-symmetric either way, so that clause cannot tell them
  // apart, and this is the measurement that says so.
  const bool asym_distinguishes =
    std::abs(asym - ref_asym) > 0.1 * std::max(asym, ref_asym);
  std::cout << "non_symmetry_distinguishes_the_two_assemblies="
            << (asym_distinguishes ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((pc == 0)
                  ? "handrolled_dispatch_drops_the_periodic_faces_silently"
                  : "dispatch_covers_the_periodic_faces")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport::7 -- inflow conditions on a DG space are weak, and the strong
// route says nothing at all when it is used by mistake.
// ===========================================================================
static int dg_strong_dirichlet()
{
  const bool continuous = mutate();
  std::cout << "element_under_test=" << (continuous ? "FE_Q_1" : "FE_DGQ_1")
            << std::endl;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
  tria.refine_global(2);
  DoFHandler<dim>                       dof(tria);
  std::unique_ptr<FiniteElement<dim>>   fe;
  if (continuous)
    fe = std::make_unique<FE_Q<dim>>(1);
  else
    fe = std::make_unique<FE_DGQ<dim>>(1);
  dof.distribute_dofs(*fe);
  std::cout << "n_active_cells=" << tria.n_active_cells()
            << " n_dofs=" << dof.n_dofs() << std::endl;

  // route 1: the boundary-value map
  std::map<types::global_dof_index, double> bv;
  std::string                              raised = "nothing";
  try
    {
      VectorTools::interpolate_boundary_values(
        dof, 0, Functions::ConstantFunction<dim>(1.0), bv);
    }
  catch (const std::exception &e)
    {
      const std::string w(e.what());
      raised = w.substr(0, std::min<size_t>(w.size(), 300));
    }
  std::cout << "interpolate_boundary_values_raised=" << raised << std::endl;
  std::cout << "boundary_values_size=" << bv.size() << std::endl;

  // route 2: straight into an AffineConstraints
  AffineConstraints<double> ac;
  std::string               raised2 = "nothing";
  try
    {
      VectorTools::interpolate_boundary_values(
        dof, 0, Functions::ConstantFunction<dim>(1.0), ac);
    }
  catch (const std::exception &e)
    {
      const std::string w(e.what());
      raised2 = w.substr(0, std::min<size_t>(w.size(), 300));
    }
  ac.close();
  std::cout << "constraints_route_raised=" << raised2 << std::endl;
  std::cout << "n_constraints=" << ac.n_constraints() << std::endl;
  const bool empty = (bv.size() == 0 && ac.n_constraints() == 0);
  std::cout << "strong_route_wrote_nothing=" << (empty ? "true" : "false")
            << std::endl;

  // The downstream consequence on the DG space: hand the (empty) map to
  // MatrixTools::apply_boundary_values instead of using the numerical flux, and
  // the prescribed inflow value is nowhere in the answer.
  if (!continuous)
    {
      DGAR t(1, 1.0);
      t.datum = DGAR::DATUM_LEFT_HALF;
      t.setup(2);
      t.assemble(true);
      t.rhs = 0.0;   // no weak inflow: the datum was "set" strongly instead
      std::map<types::global_dof_index, double> sbv;
      VectorTools::interpolate_boundary_values(
        t.dof, 0, Functions::ConstantFunction<dim>(1.0), sbv);
      std::cout << "strong_boundary_values_for_the_transport_solve=" << sbv.size()
                << std::endl;
      SparseDirectUMFPACK direct;
      direct.initialize(t.A);
      direct.vmult(t.sol, t.rhs);
      std::cout << "prescribed_inflow_value=1" << std::endl;
      std::cout << "max_abs_solution_after_the_strong_route="
                << t.max_abs_solution() << std::endl;
      std::cout << "prescribed_value_appears_in_the_answer="
                << ((t.max_abs_solution() > 0.5) ? "true" : "false")
                << std::endl;
    }
  else
    std::cout << "downstream_transport_check=skipped_not_a_dg_element"
              << std::endl;

  std::cout << "VERDICT="
            << (empty ? "strong_dirichlet_on_dg_is_silently_ignored"
                      : "strong_dirichlet_route_wrote_boundary_values")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "gmres_without_face_terms")
    return gmres_without_face_terms();
  if (probe == "handrolled_forgets_periodic")
    return handrolled_forgets_periodic();
  if (probe == "block_sweep_regimes")
    return block_sweep_regimes();
  if (probe == "dof_wise_renumbering")
    return dof_wise_renumbering();
  if (probe == "dg_strong_dirichlet")
    return dg_strong_dirichlet();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
