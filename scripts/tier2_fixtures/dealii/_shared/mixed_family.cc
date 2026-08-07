// Shared translation unit for the mixed-Laplacian (H(div) / saddle-point)
// Signal family: the step-20 shape, FE_RaviartThomas for the flux and FE_DGQ(0)
// for the pressure on the unit square.
//
// usage: mixed_family <probe>
//   rt_dof_structure | rt_map_support_points | rt_vertex_dof_index
//   | cg_on_mixed_saddle_point
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// Two of the probes CRASH by design and are run in their own processes by the
// fixture's cmd.sh, which pins the exit codes: Assert ABORTS with 134 and a
// missing support-point table dereferences an empty vector and gives 139, so the
// exit code is the observable and a try/catch would see neither.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
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

// A flux field with a known face integral: u = (x, 2y), div u = 3.
class FluxField : public Function<dim>
{
public:
  FluxField()
    : Function<dim>(dim)
  {}
  double value(const Point<dim> &p, const unsigned int c = 0) const override
  {
    return (c == 0) ? p[0] : 2.0 * p[1];
  }
  void vector_value(const Point<dim> &p, Vector<double> &v) const override
  {
    v[0] = p[0];
    v[1] = 2.0 * p[1];
  }
};

// ===========================================================================
// mixed_laplacian#0 -- where an H(div) dof lives.
// ===========================================================================
static int rt_dof_structure()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(1);
  MappingQ1<dim> mapping;

  for (unsigned int deg = 0; deg < 2; ++deg)
    {
      FE_RaviartThomas<dim> fe(deg);
      DoFHandler<dim>       dh(tria);
      dh.distribute_dofs(fe);
      std::cout << "element=FE_RaviartThomas_" << deg
                << " dofs_per_vertex=" << fe.n_dofs_per_vertex()
                << " dofs_per_face=" << fe.n_dofs_per_face(0)
                << " dofs_per_cell=" << fe.n_dofs_per_cell()
                << " has_support_points=" << yesno(fe.has_support_points())
                << " has_generalized_support_points="
                << yesno(fe.has_generalized_support_points()) << std::endl;
    }
  FESystem<dim>   feq(FE_Q<dim>(1), dim);
  DoFHandler<dim> dq(tria);
  dq.distribute_dofs(feq);
  std::cout << "element=FE_Q_1_to_the_dim"
            << " dofs_per_vertex=" << feq.n_dofs_per_vertex()
            << " dofs_per_face=" << feq.n_dofs_per_face(0)
            << " has_support_points=" << yesno(feq.has_support_points())
            << std::endl;
  std::cout << "raviart_thomas_has_no_vertex_dofs="
            << yesno(FE_RaviartThomas<dim>(0).n_dofs_per_vertex() == 0)
            << std::endl;
  std::cout << "h1_vector_element_has_vertex_dofs="
            << yesno(feq.n_dofs_per_vertex() > 0) << std::endl;

  // Interpolate the known flux field and ask what a dof is worth.
  FE_RaviartThomas<dim> fe(0);
  DoFHandler<dim>       dh(tria);
  dh.distribute_dofs(fe);
  Vector<double> v(dh.n_dofs());
  VectorTools::interpolate(dh, FluxField(), v);

  QGauss<dim - 1>   fq(4);
  FEFaceValues<dim> ffv(mapping, fe, fq,
                        update_values | update_normal_vectors |
                          update_JxW_values | update_quadrature_points);
  const FEValuesExtractors::Vector U(0);
  auto cell = dh.begin_active();
  std::vector<types::global_dof_index> local(fe.n_dofs_per_cell());
  cell->get_dof_indices(local);

  double worst_face = 0.0, worst_nodal = 0.0;
  unsigned int vertex_dofs_found = 0;
  for (unsigned int f = 0; f < cell->n_faces(); ++f)
    {
      ffv.reinit(cell, f);
      std::vector<Tensor<1, dim>> vals(fq.size());
      ffv[U].get_function_values(v, vals);
      double discrete = 0.0, exact = 0.0;
      for (unsigned int q = 0; q < fq.size(); ++q)
        {
          discrete += vals[q] * ffv.normal_vector(q) * ffv.JxW(q);
          Vector<double> e(dim);
          FluxField().vector_value(ffv.quadrature_point(q), e);
          Tensor<1, dim> ee;
          ee[0] = e[0];
          ee[1] = e[1];
          exact += ee * ffv.normal_vector(q) * ffv.JxW(q);
        }
      // The FACE dof of RT(0): one per face.
      const unsigned int face_dof = fe.face_to_cell_index(0, f);
      const double from_face_dof = v(local[face_dof]);
      // What a NODAL post-processor collects at the two vertices of this face:
      // n_dofs_per_vertex() of them, which for RT is none.
      double nodal_sum = 0.0;
      for (unsigned int vi = 0; vi < GeometryInfo<dim>::vertices_per_face; ++vi)
        for (unsigned int k = 0; k < fe.n_dofs_per_vertex(); ++k)
          {
            ++vertex_dofs_found;
            nodal_sum += v(cell->face(f)->vertex_dof_index(vi, k));
          }
      const double from_nodal = nodal_sum;
      std::cout << "face=" << f << " exact_flux_integral=" << exact
                << " discrete_flux_integral=" << discrete
                << " value_of_the_face_dof=" << from_face_dof
                << " value_a_nodal_gather_returns=" << from_nodal << std::endl;
      worst_face = std::max(worst_face, std::abs(from_face_dof - exact));
      worst_nodal = std::max(worst_nodal, std::abs(from_nodal - exact));
    }
  std::cout << "dofs_found_at_the_face_vertices=" << vertex_dofs_found
            << std::endl;
  std::cout << "worst_error_of_the_face_dof=" << worst_face
            << " worst_error_of_the_nodal_gather=" << worst_nodal << std::endl;
  const bool under_test_ok = mutate() ? (worst_face < 1e-12)
                                      : (worst_nodal < 1e-12);
  std::cout << "post_processing_under_test="
            << (mutate() ? "face_dof" : "nodal_gather") << std::endl;
  std::cout << "post_processing_under_test_reproduces_the_face_flux_integral="
            << yesno(under_test_ok) << std::endl;
  std::cout << "the_face_dof_is_exactly_the_face_flux_integral="
            << yesno(worst_face < 1e-12) << std::endl;

  // The entry's other half: does DataOut with type_dof_data refuse an RT field?
  std::cout << "before_data_out" << std::endl;
  DataOut<dim> out;
  out.attach_dof_handler(dh);
  std::vector<DataComponentInterpretation::DataComponentInterpretation> ci(
    dim, DataComponentInterpretation::component_is_part_of_vector);
  out.add_data_vector(v, std::vector<std::string>(dim, "flux"),
                      DataOut<dim>::type_dof_data, ci);
  out.build_patches();
  std::ostringstream vtu;
  out.write_vtu(vtu);
  std::cout << "after_data_out vtu_bytes=" << vtu.str().size() << std::endl;
  std::cout << "data_out_with_type_dof_data_on_raviart_thomas_returned_normally=true"
            << std::endl;

  std::cout << "VERDICT="
            << (under_test_ok
                  ? "post_processing_under_test_matches_the_flux_integral"
                  : "nodal_post_processing_of_an_hdiv_field_returns_nothing")
            << std::endl;
  return 0;
}

// The two canonical "treat the dofs as nodal" calls, each fatal.
static int rt_map_support_points()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(1);
  FE_RaviartThomas<dim> fe(0);
  DoFHandler<dim>       dh(tria);
  dh.distribute_dofs(fe);
  std::cout << "n_dofs=" << dh.n_dofs() << std::endl;
  std::cout << "before_map_dofs_to_support_points" << std::endl;
  std::vector<Point<dim>> pts(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh, pts);
  std::cout << "after_map_dofs_to_support_points first=" << pts[0] << std::endl;
  return 0;
}

static int rt_vertex_dof_index()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(1);
  FE_RaviartThomas<dim> fe(0);
  DoFHandler<dim>       dh(tria);
  dh.distribute_dofs(fe);
  std::cout << "n_dofs=" << dh.n_dofs()
            << " dofs_per_vertex=" << fe.n_dofs_per_vertex() << std::endl;
  std::cout << "before_vertex_dof_index" << std::endl;
  const auto c = dh.begin_active();
  std::cout << "after_vertex_dof_index index=" << c->vertex_dof_index(0, 0)
            << std::endl;
  return 0;
}

// ===========================================================================
// mixed_laplacian#1 -- CG on the saddle-point system.
// The step-20 system, assembled with FESystem(RT(0), DG(0)):
//   [ M   B ] [u]   [ -boundary pressure term ]
//   [ B^T 0 ] [p] = [ -f                      ]
// which is SYMMETRIC and INDEFINITE. How indefinite is counted, not asserted:
// the eigenvalues of the dense matrix are computed and the negative ones
// counted, and there are exactly as many as there are pressure dofs.
// ===========================================================================
class RightHandSide : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    // Deliberately NOT a discrete eigenmode of the uniform-grid operator: a
    // sine source would make the Schur-complement CG converge in one step and
    // the iteration count would then say nothing.
    return std::exp(p[0]) * (1.0 + p[1]);
  }
};

static int cg_on_mixed_saddle_point()
{
  const unsigned int refine = 3;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(refine);
  FESystem<dim>   fe(FE_RaviartThomas<dim>(0), 1, FE_DGQ<dim>(0), 1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  MappingQ1<dim> mapping;

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A;
  A.reinit(sp);
  Vector<double> rhs(dof.n_dofs());

  QGauss<dim>       quad(3);
  QGauss<dim - 1>   fquad(3);
  FEValues<dim>     fev(mapping, fe, quad,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> ffv(mapping, fe, fquad,
                        update_values | update_normal_vectors |
                          update_quadrature_points | update_JxW_values);
  const FEValuesExtractors::Vector U(0);
  const FEValuesExtractors::Scalar P(dim);
  const unsigned int n = fe.n_dofs_per_cell();
  FullMatrix<double> cm(n, n);
  Vector<double>     cr(n);
  std::vector<types::global_dof_index> local(n);
  const RightHandSide f;

  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cr = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const double fq = f.value(fev.quadrature_point(q));
          for (unsigned int i = 0; i < n; ++i)
            {
              const Tensor<1, dim> ui = fev[U].value(i, q);
              const double         divi = fev[U].divergence(i, q);
              const double         pi = fev[P].value(i, q);
              for (unsigned int j = 0; j < n; ++j)
                {
                  const Tensor<1, dim> uj = fev[U].value(j, q);
                  const double         divj = fev[U].divergence(j, q);
                  const double         pj = fev[P].value(j, q);
                  cm(i, j) += (ui * uj - pi * divj - divi * pj) * fev.JxW(q);
                }
              cr(i) += -pi * fq * fev.JxW(q);
            }
        }
      cell->get_dof_indices(local);
      for (unsigned int i = 0; i < n; ++i)
        {
          for (unsigned int j = 0; j < n; ++j)
            A.add(local[i], local[j], cm(i, j));
          rhs(local[i]) += cr(i);
        }
    }

  // Split the dofs into the two blocks.
  const ComponentMask   umask = fe.component_mask(U);
  const IndexSet        uset = DoFTools::extract_dofs(dof, umask);
  std::vector<int>      is_u(dof.n_dofs(), 0);
  for (const auto i : uset)
    is_u[i] = 1;
  unsigned int nu = uset.n_elements();
  unsigned int np = dof.n_dofs() - nu;
  std::cout << "n_dofs=" << dof.n_dofs() << " velocity_dofs=" << nu
            << " pressure_dofs=" << np << std::endl;

  // How indefinite is it? Count the negative eigenvalues.
  {
    LAPACKFullMatrix<double> D(dof.n_dofs(), dof.n_dofs());
    for (unsigned int r = 0; r < dof.n_dofs(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        D(r, it->column()) = it->value();
    D.compute_eigenvalues();
    unsigned int neg = 0, pos = 0;
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      {
        if (D.eigenvalue(i).real() < 0)
          ++neg;
        else
          ++pos;
      }
    std::cout << "negative_eigenvalues=" << neg << " positive_eigenvalues=" << pos
              << std::endl;
    std::cout << "matrix_is_indefinite=" << yesno(neg > 0 && pos > 0)
              << std::endl;
    std::cout << "negative_eigenvalue_count_equals_the_pressure_dof_count="
              << yesno(neg == np) << std::endl;
  }

  // (1) CG straight on the full system -- what the entry says diverges.
  unsigned int cg_steps = 0;
  double       cg_value = 0.0;
  bool         cg_ok = false;
  std::string  cg_what = "none";
  {
    SolverControl control(2000, 1e-8 * rhs.l2_norm());
    SolverCG<Vector<double>> cg(control);
    Vector<double>           x(dof.n_dofs());
    try
      {
        cg.solve(A, x, rhs, PreconditionIdentity());
        cg_ok = true;
      }
    catch (const SolverControl::NoConvergence &)
      {
        cg_what = "SolverControl_NoConvergence";
      }
    catch (const std::exception &)
      {
        cg_what = "other_std_exception";
      }
    cg_steps = control.last_step();
    cg_value = control.last_value();
    std::cout << "cg_on_the_full_system converged=" << yesno(cg_ok)
              << " last_step=" << cg_steps << " last_value=" << cg_value
              << std::endl;
    std::cout << "cg_failure_kind=" << cg_what << std::endl;
    std::cout << "cg_last_value_is_not_a_number="
              << yesno(!std::isfinite(cg_value)) << std::endl;
  }

  // (2) MINRES on the same full system.
  bool minres_ok = false;
  unsigned int minres_steps = 0;
  {
    SolverControl control(20000, 1e-8 * rhs.l2_norm());
    SolverMinRes<Vector<double>> mr(control);
    Vector<double>               x(dof.n_dofs());
    try
      {
        mr.solve(A, x, rhs, PreconditionIdentity());
        minres_ok = true;
      }
    catch (const std::exception &)
      {}
    minres_steps = control.last_step();
    std::cout << "minres_on_the_full_system converged=" << yesno(minres_ok)
              << " last_step=" << minres_steps << std::endl;
  }

  // (3) The Schur complement S = B^T M^{-1} B, and CG on it.
  bool schur_ok = false;
  unsigned int schur_steps = 0;
  {
    std::vector<unsigned int> uidx, pidx;
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      (is_u[i] ? uidx : pidx).push_back(i);
    FullMatrix<double> M(nu, nu), B(nu, np);
    for (unsigned int a = 0; a < nu; ++a)
      {
        for (unsigned int b = 0; b < nu; ++b)
          M(a, b) = A.el(uidx[a], uidx[b]);
        for (unsigned int b = 0; b < np; ++b)
          B(a, b) = A.el(uidx[a], pidx[b]);
      }
    FullMatrix<double> Minv(M);
    Minv.gauss_jordan();
    FullMatrix<double> MinvB(nu, np), S(np, np);
    Minv.mmult(MinvB, B);
    B.Tmmult(S, MinvB); // S = B^T M^{-1} B, symmetric positive definite
    Vector<double> bu(nu), bp(np), t(nu), sr(np);
    for (unsigned int a = 0; a < nu; ++a)
      bu(a) = rhs(uidx[a]);
    for (unsigned int b = 0; b < np; ++b)
      bp(b) = rhs(pidx[b]);
    Minv.vmult(t, bu);
    B.Tvmult(sr, t);
    sr -= bp; // B^T M^{-1} bu - bp
    SolverControl control(2000, 1e-10 * std::max(1e-300, sr.l2_norm()));
    SolverCG<Vector<double>> cg(control);
    Vector<double>           p(np);
    try
      {
        cg.solve(S, p, sr, PreconditionIdentity());
        schur_ok = true;
      }
    catch (const std::exception &)
      {}
    schur_steps = control.last_step();
    Vector<double> chk(np);
    S.vmult(chk, p);
    chk -= sr;
    std::cout << "cg_on_the_schur_complement converged=" << yesno(schur_ok)
              << " last_step=" << schur_steps << " initial_rhs_norm="
              << sr.l2_norm() << " final_relative_residual="
              << chk.l2_norm() / std::max(1e-300, sr.l2_norm()) << std::endl;

    // Does the Schur route give the SAME pressure as a direct solve of the
    // whole system? Otherwise "converged" would prove nothing.
    SparseDirectUMFPACK direct;
    direct.initialize(A);
    Vector<double> full(rhs);
    direct.solve(full);
    double num = 0.0, den = 0.0;
    for (unsigned int b = 0; b < np; ++b)
      {
        num += (full(pidx[b]) - p(b)) * (full(pidx[b]) - p(b));
        den += full(pidx[b]) * full(pidx[b]);
      }
    const double rel = std::sqrt(num / std::max(1e-300, den));
    std::cout << "schur_pressure_vs_direct_solve_relative_difference=" << rel
              << std::endl;
    std::cout << "schur_complement_pressure_matches_the_direct_solve="
              << yesno(rel < 1e-8) << std::endl;
    schur_ok = schur_ok && rel < 1e-8;
  }

  const bool under_test_ok = mutate() ? schur_ok : cg_ok;
  std::cout << "solver_under_test="
            << (mutate() ? "cg_on_the_schur_complement" : "cg_on_the_full_system")
            << std::endl;
  std::cout << "solver_under_test_converged=" << yesno(under_test_ok)
            << std::endl;
  std::cout << "schur_complement_cg_converged=" << yesno(schur_ok) << std::endl;
  std::cout << "minres_on_the_full_system_converged=" << yesno(minres_ok)
            << std::endl;
  std::cout << "VERDICT="
            << (under_test_ok
                  ? "solver_under_test_handled_the_saddle_point_system"
                  : "cg_cannot_solve_the_indefinite_mixed_system")
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
  if (probe == "rt_dof_structure")
    return rt_dof_structure();
  if (probe == "rt_map_support_points")
    return rt_map_support_points();
  if (probe == "rt_vertex_dof_index")
    return rt_vertex_dof_index();
  if (probe == "cg_on_mixed_saddle_point")
    return cg_on_mixed_saddle_point();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
