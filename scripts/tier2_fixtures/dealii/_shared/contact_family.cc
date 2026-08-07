// Shared translation unit for the contact Signal family.
//
// A membrane on (-1,1)^2, -laplace(u) = -2, u = 0 on the boundary, pressed onto a
// RIGID PARABOLOID indenter psi(x) = -0.35 + |x|^2 of finite extent (|x| < 0.9) --
// the Hertz-like shape, so the contact zone is a disc strictly inside the indenter
// and its RADIUS is a number the probes can compare.
//
// usage: contact_family <probe>
//   contact_single_shot | contact_penalty | contact_handmade_rows
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace dealii;
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}
static std::string yesno(bool b)
{
  return b ? "true" : "false";
}

struct Membrane
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  MappingQ1<dim>            mapping;
  DoFHandler<dim>           dof;
  AffineConstraints<double> bc;
  SparsityPattern           sp;
  SparseMatrix<double>      K;
  Vector<double>            F, lumped, psi;
  std::vector<Point<dim>>   pts;
  double                    h = 0.0;

  Membrane()
    : fe(1)
    , dof(tria)
  {}

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, -1.0, 1.0);
    tria.refine_global(refine);
    h = 2.0 / std::pow(2.0, refine);
    dof.distribute_dofs(fe);
    bc.clear();
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), bc);
    bc.close();
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    K.reinit(sp);
    F.reinit(dof.n_dofs());
    lumped.reinit(dof.n_dofs());
    QGauss<dim>   quad(2);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    Vector<double>     cr(n), cl(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cr = 0.0;
        cl = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            {
              for (unsigned int j = 0; j < n; ++j)
                cm(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              cr(i) += -2.0 * fev.shape_value(i, q) * fev.JxW(q);
              cl(i) += fev.shape_value(i, q) * fev.JxW(q);
            }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          {
            for (unsigned int j = 0; j < n; ++j)
              K.add(local[i], local[j], cm(i, j));
            F(local[i]) += cr(i);
            lumped(local[i]) += cl(i);
          }
      }
    pts.resize(dof.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof, pts);
    psi.reinit(dof.n_dofs());
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      // A rigid indenter of finite extent: the paraboloid only exists
      // inside |x| < 0.9, and the load is chosen so that the contact zone
      // is a disc STRICTLY INSIDE it -- otherwise the contact radius is
      // pinned by the geometry and measures nothing.
      psi(i) = (pts[i].norm() < 0.9) ? (-0.35 + pts[i].square()) : -10.0;
  }

  // Solve with the given dofs pinned to the obstacle, via AffineConstraints.
  void solve_with_active_set(const std::set<unsigned int> &active,
                             Vector<double>               &u) const
  {
    AffineConstraints<double> con;
    con.merge(bc);
    for (const unsigned int i : active)
      if (!con.is_constrained(i))
        {
          con.add_line(i);
          con.set_inhomogeneity(i, psi(i));
        }
    con.close();
    SparseMatrix<double> A;
    A.reinit(sp);
    A.copy_from(K);
    Vector<double> b(F);
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      if (con.is_constrained(i))
        b(i) = con.get_inhomogeneity(i);
    for (unsigned int r = 0; r < dof.n_dofs(); ++r)
      {
        if (con.is_constrained(r))
          {
            for (auto it = A.begin(r); it != A.end(r); ++it)
              it->value() = (it->column() == r) ? 1.0 : 0.0;
            continue;
          }
        for (auto it = A.begin(r); it != A.end(r); ++it)
          if (con.is_constrained(it->column()))
            {
              b(r) -= it->value() * con.get_inhomogeneity(it->column());
              it->value() = 0.0;
            }
      }
    SparseDirectUMFPACK inv;
    inv.initialize(A);
    u = b;
    inv.solve(u);
  }

  double contact_radius(const std::set<unsigned int> &active) const
  {
    double r = 0.0;
    for (const unsigned int i : active)
      r = std::max(r, pts[i].norm());
    return r;
  }
};

// ===========================================================================
// contact#0 -- a single-shot active set against the iterated one.
// ===========================================================================
static int contact_single_shot()
{
  const bool iterate = mutate(); // the mistake: predict the set once and solve
  Membrane m;
  m.setup(6);
  std::cout << "n_dofs=" << m.dof.n_dofs() << " element_edge=" << m.h
            << " strategy=" << (iterate ? "iterated_active_set" : "single_shot")
            << std::endl;

  // The initial guess: the unconstrained solve.
  std::set<unsigned int> none;
  Vector<double>         u0(m.dof.n_dofs());
  m.solve_with_active_set(none, u0);
  std::set<unsigned int> predicted;
  for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
    if (!m.bc.is_constrained(i) && u0(i) < m.psi(i))
      predicted.insert(i);
  Vector<double> u_single(m.dof.n_dofs());
  m.solve_with_active_set(predicted, u_single);
  const double r_single = m.contact_radius(predicted);

  // The iterated loop, with step-41's own criterion.
  std::set<unsigned int> active;
  Vector<double>         u(m.dof.n_dofs()), lambda(m.dof.n_dofs());
  unsigned int           iterations = 0;
  for (unsigned int it = 0; it < 40; ++it)
    {
      m.solve_with_active_set(active, u);
      m.K.vmult(lambda, u);
      lambda -= m.F;
      std::set<unsigned int> next;
      for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
        {
          if (m.bc.is_constrained(i))
            continue;
          if (lambda(i) + 100.0 * m.lumped(i) * (m.psi(i) - u(i)) > 0.0)
            next.insert(i);
        }
      if (next == active && it > 0)
        {
          iterations = it;
          break;
        }
      active = next;
    }
  const double r_conv = m.contact_radius(active);
  double worst_single = 0.0, worst_conv = 0.0;
  for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
    {
      worst_single = std::max(worst_single, m.psi(i) - u_single(i));
      worst_conv = std::max(worst_conv, m.psi(i) - u(i));
    }
  std::cout << "single_shot_contact_radius=" << r_single
            << " single_shot_active_dofs=" << predicted.size()
            << " single_shot_worst_penetration=" << worst_single << std::endl;
  std::cout << "iterated_contact_radius=" << r_conv
            << " iterated_active_dofs=" << active.size()
            << " iterated_worst_penetration=" << worst_conv
            << " outer_iterations=" << iterations << std::endl;
  // The outer radius is the entry's own diagnostic. An EQUIVALENT radius from
  // the contact AREA is reported next to it, because a set that reaches the rim
  // of the indenter has the same outer radius however wrong its area is.
  const double area_single = predicted.size() * m.h * m.h;
  const double area_conv = active.size() * m.h * m.h;
  const double req_single = std::sqrt(area_single / numbers::PI);
  const double req_conv = std::sqrt(area_conv / numbers::PI);
  std::cout << "single_shot_equivalent_radius=" << req_single
            << " iterated_equivalent_radius=" << req_conv << std::endl;
  const double rel_outer =
    std::abs(r_single - r_conv) / std::max(1e-300, r_conv);
  const double rel_eq =
    std::abs(req_single - req_conv) / std::max(1e-300, req_conv);
  std::cout << "single_shot_outer_radius_relative_error=" << rel_outer
            << " single_shot_equivalent_radius_relative_error=" << rel_eq
            << std::endl;
  const double req_test = iterate ? req_conv : req_single;
  const double rel_test =
    std::abs(req_test - req_conv) / std::max(1e-300, req_conv);
  const std::size_t n_test = iterate ? active.size() : predicted.size();
  std::cout << "outer_radius_diagnostic_separates_the_two_strategies="
            << yesno(rel_outer > 0.05) << std::endl;
  std::cout << "equivalent_contact_radius_under_test_is_wrong_by_more_than_half="
            << yesno(rel_test > 0.5) << std::endl;
  std::cout << "active_set_under_test_over_predicts_the_contact_area="
            << yesno(n_test > 2 * active.size()) << std::endl;
  std::cout << "iterated_loop_settled_within_ten_outer_iterations="
            << yesno(iterations > 0 && iterations <= 10) << std::endl;
  std::cout << "VERDICT="
            << ((rel_test > 0.5)
                  ? "single_shot_active_set_gives_the_wrong_contact_zone"
                  : "strategy_under_test_finds_the_converged_contact_zone")
            << std::endl;
  return 0;
}

// ===========================================================================
// contact#1 -- the penalty parameter.
// ===========================================================================
static int contact_penalty()
{
  Membrane m;
  m.setup(5);
  const double rule = 1e3 / m.h; // the entry's 1e3 * E / h with E = 1
  const double small = 1.0;
  const double huge = 1e14;
  // NOT the entry's rule: measured below, 1e3*E/h leaves 10% of an element
  // edge of penetration, twice the entry's own 5% criterion. What actually
  // meets it here is about 1e8.
  const double p_test = mutate() ? 1e8 : small;
  std::cout << "n_dofs=" << m.dof.n_dofs() << " element_edge=" << m.h
            << " rule_of_thumb_penalty=" << rule
            << " penalty_under_test=" << p_test << std::endl;

  auto run = [&](double p, double &penetration, unsigned int &cg_steps,
                 bool &cg_ok, double &cond) {
    Vector<double>       u(m.dof.n_dofs());
    SparseMatrix<double> A;
    A.reinit(m.sp);
    Vector<double> b(m.dof.n_dofs());
    for (unsigned int sweep = 0; sweep < 25; ++sweep)
      {
        A.copy_from(m.K);
        b = m.F;
        for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
          if (!m.bc.is_constrained(i) && u(i) < m.psi(i))
            {
              A.add(i, i, p * m.lumped(i));
              b(i) += p * m.lumped(i) * m.psi(i);
            }
        for (unsigned int r = 0; r < m.dof.n_dofs(); ++r)
          if (m.bc.is_constrained(r))
            {
              for (auto it = A.begin(r); it != A.end(r); ++it)
                it->value() = (it->column() == r) ? 1.0 : 0.0;
              b(r) = 0.0;
            }
        SparseDirectUMFPACK inv;
        inv.initialize(A);
        u = b;
        inv.solve(u);
      }
    penetration = 0.0;
    for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
      penetration = std::max(penetration, m.psi(i) - u(i));
    // How hard is the penalised operator to solve iteratively?
    SolverControl control(2000, 1e-10 * std::max(1e-300, b.l2_norm()));
    SolverCG<Vector<double>> cg(control);
    Vector<double>           x(m.dof.n_dofs());
    cg_ok = false;
    try
      {
        cg.solve(A, x, b, PreconditionIdentity());
        cg_ok = true;
      }
    catch (const std::exception &)
      {}
    cg_steps = control.last_step();
    // Condition number from the extreme eigenvalues of a smaller copy.
    cond = 0.0;
    if (m.dof.n_dofs() <= 1200)
      {
        LAPACKFullMatrix<double> D(m.dof.n_dofs(), m.dof.n_dofs());
        for (unsigned int r = 0; r < m.dof.n_dofs(); ++r)
          for (auto it = A.begin(r); it != A.end(r); ++it)
            D(r, it->column()) = it->value();
        D.compute_eigenvalues();
        double mn = 1e300, mx = -1e300;
        for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
          {
            mn = std::min(mn, std::abs(D.eigenvalue(i).real()));
            mx = std::max(mx, std::abs(D.eigenvalue(i).real()));
          }
        cond = mx / std::max(1e-300, mn);
      }
  };

  double       pen = 0.0, cond = 0.0;
  unsigned int steps = 0;
  bool         ok = false;
  for (double p : {small, 1e2, rule, 1e8, huge})
    {
      run(p, pen, steps, ok, cond);
      std::cout << "penalty=" << p << " worst_penetration=" << pen
                << " penetration_over_element_edge=" << pen / m.h
                << " cg_steps=" << steps << " cg_converged=" << yesno(ok)
                << " condition_number=" << cond << std::endl;
      if (p == p_test)
        {
          std::cout << "penetration_under_test_over_element_edge=" << pen / m.h
                    << std::endl;
          std::cout
            << "penetration_under_test_exceeds_five_percent_of_an_element_edge="
            << yesno(pen / m.h > 0.05) << std::endl;
        }
      if (p == rule)
        std::cout
          << "rule_of_thumb_penalty_exceeds_five_percent_of_an_element_edge="
          << yesno(pen / m.h > 0.05) << std::endl;
      if (p == huge)
        {
          std::cout << "huge_penalty_condition_number_above_1e14="
                    << yesno(cond > 1e14) << std::endl;
          std::cout << "huge_penalty_cg_stagnated=" << yesno(!ok) << std::endl;
        }
    }
  double pen_test = 0.0, c2 = 0.0;
  unsigned int s2 = 0;
  bool         ok2 = false;
  run(p_test, pen_test, s2, ok2, c2);
  std::cout << "VERDICT="
            << ((pen_test / m.h > 0.05)
                  ? "penalty_under_test_lets_the_body_penetrate"
                  : "penalty_under_test_holds_the_constraint")
            << std::endl;
  return 0;
}

// ===========================================================================
// contact#2 -- AffineConstraints against hand-modified rows.
// ===========================================================================
static int contact_handmade_rows()
{
  const bool use_affine = mutate(); // the mistake: zero the row, set diag = 1
  Membrane m;
  m.setup(5);
  // A fixed, converged active set, so the only thing that varies is HOW it is
  // imposed.
  std::set<unsigned int> active;
  {
    Vector<double> u(m.dof.n_dofs()), lambda(m.dof.n_dofs());
    for (unsigned int it = 0; it < 40; ++it)
      {
        m.solve_with_active_set(active, u);
        m.K.vmult(lambda, u);
        lambda -= m.F;
        std::set<unsigned int> next;
        for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
          if (!m.bc.is_constrained(i) &&
              lambda(i) + 100.0 * m.lumped(i) * (m.psi(i) - u(i)) > 0.0)
            next.insert(i);
        if (next == active && it > 0)
          break;
        active = next;
      }
  }
  std::cout << "n_dofs=" << m.dof.n_dofs() << " active_dofs=" << active.size()
            << " enforcement_under_test="
            << (use_affine ? "affine_constraints" : "hand_modified_rows")
            << std::endl;

  SparseMatrix<double> A;
  A.reinit(m.sp);
  A.copy_from(m.K);
  Vector<double> b(m.F);
  auto constrained = [&](unsigned int i) {
    return m.bc.is_constrained(i) || active.count(i) > 0;
  };
  auto value_of = [&](unsigned int i) {
    return m.bc.is_constrained(i) ? 0.0 : m.psi(i);
  };
  for (unsigned int i = 0; i < m.dof.n_dofs(); ++i)
    if (constrained(i))
      b(i) = value_of(i);
  for (unsigned int r = 0; r < m.dof.n_dofs(); ++r)
    {
      if (constrained(r))
        {
          for (auto it = A.begin(r); it != A.end(r); ++it)
            it->value() = (it->column() == r) ? 1.0 : 0.0;
          continue;
        }
      if (!use_affine)
        continue; // the hand-modified route stops at the ROWS
      for (auto it = A.begin(r); it != A.end(r); ++it)
        if (constrained(it->column()))
          {
            b(r) -= it->value() * value_of(it->column());
            it->value() = 0.0;
          }
    }

  // Is the resulting operator still symmetric?
  double num = 0.0, den = 0.0;
  for (unsigned int r = 0; r < m.dof.n_dofs(); ++r)
    for (auto it = A.begin(r); it != A.end(r); ++it)
      {
        const double a = it->value();
        const double t = m.sp.exists(it->column(), r) ? A.el(it->column(), r)
                                                      : 0.0;
        num += (a - t) * (a - t);
        den += a * a;
      }
  const double sym = std::sqrt(num / std::max(1e-300, den));
  std::cout << "relative_symmetry_defect_under_test=" << sym << std::endl;
  std::cout << "operator_under_test_is_symmetric=" << yesno(sym < 1e-12)
            << std::endl;

  SolverControl control(3000, 1e-10 * std::max(1e-300, b.l2_norm()));
  SolverCG<Vector<double>> cg(control);
  Vector<double>           x(m.dof.n_dofs());
  bool                     cg_ok = false;
  try
    {
      cg.solve(A, x, b, PreconditionIdentity());
      cg_ok = true;
    }
  catch (const std::exception &)
    {}
  std::cout << "cg_converged=" << yesno(cg_ok)
            << " cg_steps=" << control.last_step()
            << " cg_last_value=" << control.last_value() << std::endl;

  // Does the answer match the AffineConstraints route?
  Vector<double> ref(m.dof.n_dofs());
  m.solve_with_active_set(active, ref);
  SparseDirectUMFPACK inv;
  inv.initialize(A);
  Vector<double> direct(b);
  inv.solve(direct);
  Vector<double> d(direct);
  d -= ref;
  const double rel = d.l2_norm() / std::max(1e-300, ref.l2_norm());
  std::cout << "relative_difference_from_the_affine_constraints_solution=" << rel
            << std::endl;
  std::cout << "solution_under_test_matches_the_affine_constraints_one="
            << yesno(rel < 1e-10) << std::endl;
  std::cout << "VERDICT="
            << ((sym < 1e-12)
                  ? "enforcement_under_test_keeps_the_operator_symmetric"
                  : "hand_modified_rows_destroy_the_symmetry")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  MultithreadInfo::set_thread_limit(1);
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(8);
  if (probe == "contact_single_shot")
    return contact_single_shot();
  if (probe == "contact_penalty")
    return contact_penalty();
  if (probe == "contact_handmade_rows")
    return contact_handmade_rows();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
