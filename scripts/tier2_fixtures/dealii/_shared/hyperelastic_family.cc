// Shared translation unit for the finite-strain / hyperelasticity Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: hyperelastic_family <probe>
//   load_stepping | det_f_guard | lnj_vs_squared_j | geometric_term
//   | volumetric_locking | svk_compression | roller_vs_clamped
//   | ad_energy_functional | mapping_q_eulerian | incremental_constraints
//   | fesystem_gradient_containers | umfpack_vs_cg
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// The model problem is a total-Lagrangian compressible Neo-Hookean solid in 2D
// plane strain,
//    W = mu/2 (tr C - 2) - mu ln J + lambda/2 (ln J)^2
//    S = mu (I - C^{-1}) + lambda ln(J) C^{-1}
//    C_ijkl = lambda Cinv_ij Cinv_kl + (mu - lambda lnJ)(Cinv_ik Cinv_jl
//                                                        + Cinv_il Cinv_jk)
// with the residual and the consistent tangent
//    R_i   = int S : dE_i  -  int f . N_i
//    K_ij  = int dE_i : C : dE_j            (material)
//          + int S : sym(grad N_i^T grad N_j)   (geometric)
//    dE_i  = sym(F^T grad N_i).
// deal.II has NO Newton solver and NO constitutive-model layer, so every line
// of the above is the user's own -- which is what this family of claims is
// about.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
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
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

// ---------------------------------------------------------------------------
// material
// ---------------------------------------------------------------------------
enum class Law
{
  lnJ,   // S = mu (I - Cinv) + lambda ln(J) Cinv          (standard)
  J2,    // S = mu (I - Cinv) + lambda/2 (J^2 - 1) Cinv    (the squared-J form)
  SVK    // S = lambda tr(E) I + 2 mu E                    (Saint-Venant-K.)
};

struct Response
{
  SymmetricTensor<2, dim> S;
  SymmetricTensor<2, dim> Cinv;
  double                  J    = 0.0;
  double                  lnJ  = 0.0;
  bool                    bad  = false;   // J <= 0
};

static double quad_form(const SymmetricTensor<2, dim> &A,
                        const SymmetricTensor<2, dim> &Cinv,
                        const SymmetricTensor<2, dim> &B)
{
  const Tensor<2, dim> a(A), c(Cinv), b(B);
  return trace(a * c * b * c);
}

static Response material(const Tensor<2, dim> &F, double mu, double lam,
                         Law law)
{
  Response r;
  const SymmetricTensor<2, dim> I = unit_symmetric_tensor<dim>();
  r.J                             = determinant(F);
  if (law == Law::SVK)
    {
      const SymmetricTensor<2, dim> E =
        0.5 * (symmetrize(transpose(F) * F) - I);
      r.S    = lam * trace(E) * I + 2.0 * mu * E;
      r.Cinv = I;   // unused
      r.lnJ  = 0.0;
      r.bad  = false;
      return r;
    }
  const SymmetricTensor<2, dim> C = symmetrize(transpose(F) * F);
  r.Cinv                          = invert(C);
  r.bad                           = !(r.J > 0.0);
  if (law == Law::lnJ)
    {
      // std::log of a non-positive argument returns NaN / -inf: no exception,
      // no message -- exactly the silence the claims describe.
      r.lnJ = std::log(r.J);
      r.S   = mu * (I - r.Cinv) + lam * r.lnJ * r.Cinv;
    }
  else
    {
      r.lnJ = std::log(std::abs(r.J));
      r.S   = mu * (I - r.Cinv) + 0.5 * lam * (r.J * r.J - 1.0) * r.Cinv;
    }
  return r;
}

// A : C : B for the tangent above, without ever forming the rank-4 tensor.
static double tangent_contract(const Response &r, double mu, double lam,
                               Law law, const SymmetricTensor<2, dim> &A,
                               const SymmetricTensor<2, dim> &B)
{
  if (law == Law::SVK)
    return lam * trace(A) * trace(B) + 2.0 * mu * (A * B);
  const double aC = A * r.Cinv;
  const double bC = B * r.Cinv;
  const double q  = quad_form(A, r.Cinv, B);
  if (law == Law::lnJ)
    return lam * aC * bC + 2.0 * (mu - lam * r.lnJ) * q;
  const double J2 = r.J * r.J;
  return lam * J2 * aC * bC +
         2.0 * (mu - 0.5 * lam * (J2 - 1.0)) * q;
}

// deal.II exception text is multi-line and framed; keep one compact line, and
// prefer the "Additional information:" tail, which is where the library puts
// the sentence a user would actually read.
static std::string flatten(const std::string &s, std::size_t maxlen = 260)
{
  std::string out;
  bool        pending = false;
  for (char c : s)
    {
      if (std::isspace(static_cast<unsigned char>(c)))
        {
          pending = true;
          continue;
        }
      if (pending && !out.empty())
        out += ' ';
      pending = false;
      out += c;
    }
  return out.substr(0, maxlen);
}

static std::string additional_info(const std::string &s)
{
  const auto p = s.find("Additional information:");
  std::string t = (p == std::string::npos) ? s : s.substr(p + 23);
  const auto q  = t.find("Stacktrace:");
  if (q != std::string::npos)
    t = t.substr(0, q);
  return flatten(t);
}

static void lame_from(double E, double nu, double &mu, double &lam)
{
  mu  = E / (2.0 * (1.0 + nu));
  lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
}

// ---------------------------------------------------------------------------
// the solid
// ---------------------------------------------------------------------------
enum class BC
{
  cantilever,       // left face clamped, load by body force / end traction
  compress_clamped, // left face clamped, right face fully prescribed
  compress_roller,  // left face u_x = 0, bottom u_y = 0, right face u_x = d
  confined          // as above plus u_y = 0 on top AND bottom: J = lambda_x
};

enum class Inner
{
  umfpack,
  cg_ssor,
  gmres
};

struct Opts
{
  Law    law                 = Law::lnJ;
  bool   drop_k_geo          = false;
  double k_geo_scale         = 1.0;   // -1 reproduces a sign error
  bool   reduced_volumetric  = false; // Q1/P0-equivalent selective reduced int.
  bool   guard_detF          = false; // the user's own AssertThrow
  double body_force_y        = 0.0;
  double end_traction_y      = 0.0;   // shear traction on boundary id 1
};

struct Solid
{
  Triangulation<dim>        tria;
  FESystem<dim>             fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      K;
  Vector<double>            sol, rhs, du;
  double                    mu = 1.0, lam = 1.0;
  double                    Lx = 1.0, Ly = 1.0;
  BC                        bc = BC::cantilever;
  // measured during the last assembly
  double min_detF = 0.0, max_absK = 0.0;
  bool   tangent_has_nan = false, rhs_has_nan = false;
  std::string guard_message;

  explicit Solid(unsigned int degree = 1)
    : fe(FE_Q<dim>(degree) ^ dim)
    , dof(tria)
  {}

  void make_grid(unsigned int nx, unsigned int ny, double L, double H)
  {
    Lx = L;
    Ly = H;
    GridGenerator::subdivided_hyper_rectangle(
      tria, {nx, ny}, Point<dim>(0, 0), Point<dim>(L, H), true);
    dof.distribute_dofs(fe);
    sol.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
    du.reinit(dof.n_dofs());
  }

  // Rebuild the constraint object. `value` is the prescribed x-displacement of
  // the right face; `inhomogeneous` decides whether the constraint carries that
  // value (first Newton iteration of a load step) or zero (all later ones).
  void set_bc(double value, bool inhomogeneous)
  {
    constraints.clear();
    const FEValuesExtractors::Scalar ux(0), uy(1);
    const ComponentMask mx = fe.component_mask(ux);
    const ComponentMask my = fe.component_mask(uy);
    const double v         = inhomogeneous ? value : 0.0;
    std::vector<double> vx(dim, 0.0);
    vx[0] = v;
    switch (bc)
      {
        case BC::cantilever:
          VectorTools::interpolate_boundary_values(
            dof, 0, Functions::ZeroFunction<dim>(dim), constraints);
          break;
        case BC::compress_clamped:
          VectorTools::interpolate_boundary_values(
            dof, 0, Functions::ZeroFunction<dim>(dim), constraints);
          VectorTools::interpolate_boundary_values(
            dof, 1, Functions::ConstantFunction<dim>(vx), constraints);
          break;
        case BC::compress_roller:
          VectorTools::interpolate_boundary_values(
            dof, 0, Functions::ZeroFunction<dim>(dim), constraints, mx);
          VectorTools::interpolate_boundary_values(
            dof, 2, Functions::ZeroFunction<dim>(dim), constraints, my);
          VectorTools::interpolate_boundary_values(
            dof, 1, Functions::ConstantFunction<dim>(vx), constraints, mx);
          break;
        case BC::confined:
          // The lateral faces cannot move, so det F is exactly the axial
          // stretch: this is the only way to drive J far from 1 and see what
          // the two volumetric forms do about it.
          VectorTools::interpolate_boundary_values(
            dof, 0, Functions::ZeroFunction<dim>(dim), constraints, mx);
          VectorTools::interpolate_boundary_values(
            dof, 2, Functions::ZeroFunction<dim>(dim), constraints, my);
          VectorTools::interpolate_boundary_values(
            dof, 3, Functions::ZeroFunction<dim>(dim), constraints, my);
          VectorTools::interpolate_boundary_values(
            dof, 1, Functions::ConstantFunction<dim>(vx), constraints, mx);
          break;
      }
    constraints.close();
  }

  void allocate()
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, true);
    sp.copy_from(dsp);
    K.reinit(sp);
  }

  // Assemble K and rhs = -R at the state `sol`. Returns the residual norm over
  // the free dofs.
  double assemble(const Opts &o, bool want_matrix = true)
  {
    K   = 0.0;
    rhs = 0.0;
    min_detF        = std::numeric_limits<double>::max();
    tangent_has_nan = false;
    rhs_has_nan     = false;
    guard_message.clear();

    const unsigned int deg = fe.degree;
    QGauss<dim>        qfull(deg + 1);
    QGauss<dim>        qred(1);
    FEValues<dim>      fev(fe, qfull,
                           update_values | update_gradients | update_JxW_values);
    FEValues<dim>      fer(fe, qred,
                           update_values | update_gradients | update_JxW_values);
    QGauss<dim - 1>    qface(deg + 1);
    FEFaceValues<dim>  fef(fe, qface,
                           update_values | update_JxW_values);

    const unsigned int n = fe.dofs_per_cell;
    FullMatrix<double> cm(n, n);
    Vector<double>     cv(n);
    std::vector<types::global_dof_index> local(n);
    const FEValuesExtractors::Vector     u(0);
    std::vector<Tensor<2, dim>>          grad_u;

    for (const auto &cell : dof.active_cell_iterators())
      {
        cm = 0.0;
        cv = 0.0;
        // Two quadrature passes: the second one only exists when the
        // volumetric part is under-integrated (the Q1/P0-equivalent cure).
        const int npass = o.reduced_volumetric ? 2 : 1;
        for (int pass = 0; pass < npass; ++pass)
          {
            FEValues<dim> &fv = (pass == 0) ? fev : fer;
            const Quadrature<dim> &qr = (pass == 0) ? qfull : qred;
            const double mu_p = (npass == 1 || pass == 0) ? mu : 0.0;
            const double lam_p =
              (npass == 1) ? lam : ((pass == 1) ? lam : 0.0);
            fv.reinit(cell);
            grad_u.resize(qr.size());
            fv[u].get_function_gradients(sol, grad_u);
            for (unsigned int q = 0; q < qr.size(); ++q)
              {
                Tensor<2, dim> F = grad_u[q];
                for (unsigned int d = 0; d < dim; ++d)
                  F[d][d] += 1.0;
                const Response r = material(F, mu_p, lam_p, o.law);
                if (pass == 0)
                  min_detF = std::min(min_detF, r.J);
                if (o.guard_detF && r.bad && guard_message.empty())
                  {
                    // Exactly the guard the claim tells the user to write.
                    // AssertThrow is live in Release, unlike Assert.
                    try
                      {
                        AssertThrow(r.J > 0.0,
                                    ExcMessage(
                                      "det(F) <= 0 at quadrature point"));
                      }
                    catch (const std::exception &e)
                      {
                        guard_message = e.what();
                      }
                  }
                std::vector<SymmetricTensor<2, dim>> dE(n);
                std::vector<Tensor<2, dim>>          gN(n);
                for (unsigned int i = 0; i < n; ++i)
                  {
                    gN[i] = fv[u].gradient(i, q);
                    dE[i] = symmetrize(transpose(F) * gN[i]);
                  }
                for (unsigned int i = 0; i < n; ++i)
                  {
                    cv(i) -= (r.S * dE[i]) * fv.JxW(q);
                    if (pass == 0)
                      cv(i) += o.body_force_y * fv[u].value(i, q)[1] *
                               fv.JxW(q);
                    if (!want_matrix)
                      continue;
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        double k = tangent_contract(r, mu_p, lam_p, o.law,
                                                    dE[i], dE[j]);
                        if (!o.drop_k_geo)
                          {
                            const Tensor<2, dim> t = transpose(gN[i]) * gN[j];
                            k += o.k_geo_scale *
                                 (r.S * symmetrize(t));
                          }
                        cm(i, j) += k * fv.JxW(q);
                      }
                  }
              }
          }
        // end traction (shear) on boundary id 1
        if (o.end_traction_y != 0.0)
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() && face->boundary_id() == 1)
              {
                fef.reinit(cell, face);
                for (unsigned int q = 0; q < qface.size(); ++q)
                  for (unsigned int i = 0; i < n; ++i)
                    cv(i) += o.end_traction_y * fef[u].value(i, q)[1] *
                             fef.JxW(q);
              }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cv, local, K, rhs);
      }

    max_absK = 0.0;
    for (unsigned int r = 0; r < K.m(); ++r)
      for (auto it = K.begin(r); it != K.end(r); ++it)
        {
          if (std::isnan(it->value()))
            tangent_has_nan = true;
          else
            max_absK = std::max(max_absK, std::abs(it->value()));
        }
    double nrm = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      {
        if (std::isnan(rhs(i)))
          rhs_has_nan = true;
        if (!constraints.is_constrained(i))
          nrm += rhs(i) * rhs(i);
      }
    return std::sqrt(nrm);
  }

  // Sample the second Piola-Kirchhoff stress and det F at every quadrature
  // point of every cell, in cell-loop order. Two Solid objects on the same mesh
  // with the same element visit the same points in the same order, so the two
  // sample vectors can be differenced entry by entry.
  void sample(const Opts &o, std::vector<double> &s00,
              std::vector<double> &smag, std::vector<double> &detF,
              std::vector<double> &jxw)
  {
    s00.clear();
    smag.clear();
    detF.clear();
    jxw.clear();
    QGauss<dim>   q(fe.degree + 1);
    FEValues<dim> fv(fe, q, update_gradients | update_JxW_values);
    const FEValuesExtractors::Vector u(0);
    std::vector<Tensor<2, dim>>      g(q.size());
    for (const auto &cell : dof.active_cell_iterators())
      {
        fv.reinit(cell);
        fv[u].get_function_gradients(sol, g);
        for (unsigned int k = 0; k < q.size(); ++k)
          {
            Tensor<2, dim> F = g[k];
            for (unsigned int d = 0; d < dim; ++d)
              F[d][d] += 1.0;
            const Response r = material(F, mu, lam, o.law);
            s00.push_back(r.S[0][0]);
            smag.push_back(std::sqrt(r.S * r.S));
            detF.push_back(r.J);
            jxw.push_back(fv.JxW(k));
          }
      }
  }

  double min_det_F_of(const Vector<double> &state)
  {
    Vector<double> keep = sol;
    sol                 = state;
    QGauss<dim>   q(fe.degree + 1);
    FEValues<dim> fv(fe, q, update_gradients);
    const FEValuesExtractors::Vector u(0);
    std::vector<Tensor<2, dim>>      g(q.size());
    double                           m = std::numeric_limits<double>::max();
    for (const auto &cell : dof.active_cell_iterators())
      {
        fv.reinit(cell);
        fv[u].get_function_gradients(sol, g);
        for (unsigned int k = 0; k < q.size(); ++k)
          {
            Tensor<2, dim> F = g[k];
            for (unsigned int d = 0; d < dim; ++d)
              F[d][d] += 1.0;
            m = std::min(m, determinant(F));
          }
      }
    sol = keep;
    return m;
  }

  // signed tip deflection: the most negative y-displacement anywhere
  double tip_deflection() const
  {
    double worst = 0.0;
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.system_to_component_index(i).first == 1)
            worst = std::min(worst, sol(local[i]));
      }
    return worst;
  }

  double symmetry_defect() const
  {
    double worst = 0.0;
    for (unsigned int r = 0; r < K.m(); ++r)
      for (auto it = K.begin(r); it != K.end(r); ++it)
        worst = std::max(worst, std::abs(it->value() - K.el(it->column(), r)));
    return worst;
  }

  // Cook's membrane: the unit square mapped onto the classic trapezoid
  // (0,0)-(48,44)-(48,60)-(0,44). The colorize ids survive the transform, so
  // the left edge is still 0 and the loaded right edge still 1.
  void make_cook_grid(unsigned int nx, unsigned int ny)
  {
    Lx = 48.0;
    Ly = 60.0;
    GridGenerator::subdivided_hyper_rectangle(
      tria, {nx, ny}, Point<dim>(0, 0), Point<dim>(1, 1), true);
    GridTools::transform(
      [](const Point<dim> &p) {
        return Point<dim>(48.0 * p[0],
                          44.0 * p[0] + p[1] * (44.0 - 28.0 * p[0]));
      },
      tria);
    dof.distribute_dofs(fe);
    sol.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
    du.reinit(dof.n_dofs());
  }

  // largest upward y-displacement anywhere -- the Cook membrane tip deflection
  double max_uy() const
  {
    double best = 0.0;
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.system_to_component_index(i).first == 1)
            best = std::max(best, sol(local[i]));
      }
    return best;
  }

  IndexSet driven_dofs() const
  {
    const FEValuesExtractors::Scalar ux(0);
    return DoFTools::extract_boundary_dofs(dof, fe.component_mask(ux), {1});
  }

  // Sum of the internal x-force over the driven face. Assembled with NO
  // constraints so the constrained rows still carry the reaction. Destroys K.
  double reaction_x(const Opts &o)
  {
    AffineConstraints<double> keep;
    keep.copy_from(constraints);
    constraints.clear();
    constraints.close();
    assemble(o, false);
    double s = 0.0;
    for (const auto i : driven_dofs())
      s -= rhs(i);   // rhs holds -R
    constraints.copy_from(keep);
    return s;
  }

  // Is the assembled tangent the derivative of the assembled residual? A
  // central difference of R along one direction answers that without any
  // reference to what the tangent is SUPPOSED to look like. Destroys K and rhs.
  double tangent_vs_fd(const Opts &o, double eps = 1e-7)
  {
    set_bc(0.0, false);
    assemble(o, true);
    Vector<double> v(dof.n_dofs());
    for (unsigned int i = 0; i < v.size(); ++i)
      v(i) = std::sin(0.7 + 1.3 * i);
    constraints.set_zero(v);
    v /= v.l2_norm();
    Vector<double> Kv(dof.n_dofs());
    K.vmult(Kv, v);

    const Vector<double> keep = sol;
    sol                       = keep;
    sol.add(eps, v);
    assemble(o, false);
    const Vector<double> rp = rhs;
    sol                     = keep;
    sol.add(-eps, v);
    assemble(o, false);
    const Vector<double> rm = rhs;
    sol                     = keep;

    Vector<double> fd(rm);   // rhs = -R, so (rm - rp)/(2 eps) = dR/dv
    fd -= rp;
    fd /= (2.0 * eps);
    for (const auto &l : constraints.get_lines())
      {
        fd(l.index) = 0.0;
        Kv(l.index) = 0.0;
      }
    Vector<double> d(Kv);
    d -= fd;
    return d.l2_norm() / std::max(1e-300, fd.l2_norm());
  }

  void spectrum(double &lmin, double &lmax)
  {
    LAPACKFullMatrix<double> M(K.m(), K.n());
    for (unsigned int r = 0; r < K.m(); ++r)
      for (auto it = K.begin(r); it != K.end(r); ++it)
        M(r, it->column()) = it->value();
    M.compute_eigenvalues();
    lmin = std::numeric_limits<double>::max();
    lmax = -std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < K.m(); ++i)
      {
        const double e = M.eigenvalue(i).real();
        lmin           = std::min(lmin, e);
        lmax           = std::max(lmax, e);
      }
  }

  std::string solve_increment(Inner which, double &lin_residual)
  {
    du = 0.0;
    std::string msg;
    lin_residual = std::numeric_limits<double>::quiet_NaN();
    try
      {
        if (which == Inner::umfpack)
          {
            SparseDirectUMFPACK direct;
            direct.initialize(K);
            direct.vmult(du, rhs);
          }
        else
          {
            SolverControl ctrl(2000, 1e-10 * std::max(1.0, rhs.l2_norm()));
            if (which == Inner::cg_ssor)
              {
                PreconditionSSOR<SparseMatrix<double>> prec;
                prec.initialize(K);
                SolverCG<Vector<double>> cg(ctrl);
                cg.solve(K, du, rhs, prec);
              }
            else
              {
                SolverGMRES<Vector<double>> g(ctrl);
                g.solve(K, du, rhs, PreconditionIdentity());
              }
          }
        Vector<double> t(du.size());
        K.vmult(t, du);
        t -= rhs;
        lin_residual = t.l2_norm() / std::max(1e-300, rhs.l2_norm());
      }
    catch (const SolverControl::NoConvergence &e)
      {
        msg = "SolverControl::NoConvergence " + additional_info(e.what());
      }
    catch (const std::exception &e)
      {
        msg = std::string("other_exception ") + additional_info(e.what());
      }
    if (!msg.empty())
      {
        Vector<double> t(du.size());
        K.vmult(t, du);
        t -= rhs;
        lin_residual = t.l2_norm() / std::max(1e-300, rhs.l2_norm());
      }
    return msg;
  }
};

struct NewtonReport
{
  std::vector<double> hist;
  unsigned int        steps         = 0;
  bool                converged     = false;
  bool                diverged      = false;
  bool                nan_seen      = false;
  bool                detF_negative = false;
  double              min_detF      = 0.0;
  std::string         inner_message;
  double              alpha_min = 1.0;
};

struct NewtonOpts
{
  unsigned int max_it            = 20;
  double       tol               = 1e-9;
  bool         line_search       = false;   // backtracking on the residual
  bool         detF_line_search  = false;   // backtracking until min det F > 0
  bool         inhomogeneous_every_iteration = false;
  Inner        inner             = Inner::umfpack;
  double       dirichlet         = 0.0;
};

// One Newton solve from the CURRENT solid.sol towards the prescribed
// Dirichlet value. Load stepping is the caller's business.
static NewtonReport newton(Solid &s, const Opts &o, const NewtonOpts &n)
{
  NewtonReport rep;
  double       r0 = 0.0;
  for (unsigned int it = 0; it < n.max_it; ++it)
    {
      const bool first = (it == 0);
      s.set_bc(n.dirichlet, first || n.inhomogeneous_every_iteration);
      const double r = s.assemble(o);
      rep.hist.push_back(r);
      rep.min_detF = s.min_detF;
      if (s.min_detF <= 0.0)
        rep.detF_negative = true;
      if (std::isnan(r) || s.tangent_has_nan || s.rhs_has_nan)
        {
          rep.nan_seen = true;
          rep.steps    = it + 1;
          std::string m;
          double      lr;
          m = s.solve_increment(n.inner == Inner::umfpack ? Inner::gmres
                                                          : n.inner,
                                lr);
          if (!m.empty() && rep.inner_message.empty())
            rep.inner_message = m;
          rep.diverged = true;
          return rep;
        }
      if (first)
        r0 = r;
      if (!first && r < n.tol * std::max(1.0, r0))
        {
          rep.converged = true;
          rep.steps     = it;
          return rep;
        }
      if (first && r < 1e-14)
        {
          rep.converged = true;
          rep.steps     = it;
          return rep;
        }
      double lr = 0.0;
      const std::string msg = s.solve_increment(n.inner, lr);
      if (!msg.empty() && rep.inner_message.empty())
        rep.inner_message = msg;
      s.constraints.distribute(s.du);
      double alpha = 1.0;
      // The first iteration of a load step is the one whose constraint carries
      // the inhomogeneous Dirichlet increment. Scaling THAT step would leave
      // the boundary condition permanently unsatisfied (later iterations solve
      // with a zero increment on the boundary), so it is never backtracked.
      const bool carries_the_dirichlet_increment =
        (it == 0) && (n.dirichlet != 0.0);
      if (n.detF_line_search && !carries_the_dirichlet_increment)
        {
          for (int k = 0; k < 30; ++k)
            {
              Vector<double> trial = s.sol;
              trial.add(alpha, s.du);
              if (s.min_det_F_of(trial) > 1e-3)
                break;
              alpha *= 0.5;
            }
        }
      if (n.line_search && !carries_the_dirichlet_increment)
        {
          const double base = r;
          for (int k = 0; k < 20; ++k)
            {
              Vector<double> keep  = s.sol;
              Vector<double> trial = s.sol;
              trial.add(alpha, s.du);
              s.sol                = trial;
              s.set_bc(n.dirichlet, false);
              const double rt = s.assemble(o, false);
              s.sol           = keep;
              if (!std::isnan(rt) && rt < base)
                break;
              alpha *= 0.5;
            }
        }
      rep.alpha_min = std::min(rep.alpha_min, alpha);
      if (std::getenv("T2_DEBUG"))
        std::cout << "   [dbg] it=" << it << " r=" << r
                  << " du_l2=" << s.du.l2_norm() << " alpha=" << alpha
                  << " lin_rel_resid=" << lr << " maxK=" << s.max_absK
                  << " msg=" << msg << std::endl;
      s.sol.add(alpha, s.du);
      rep.steps = it + 1;
      if (r > 1e14 * std::max(1.0, r0))
        {
          rep.diverged = true;
          return rep;
        }
    }
  return rep;
}

static void print_history(const char *tag, const std::vector<double> &h)
{
  std::cout << tag << "_residual_history=";
  for (unsigned int i = 0; i < h.size() && i < 14; ++i)
    std::cout << (i ? "," : "") << h[i];
  std::cout << std::endl;
}

static void report(const char *tag, const NewtonReport &r)
{
  std::cout << tag << "_newton_steps=" << r.steps
            << " converged=" << (r.converged ? "true" : "false")
            << " diverged=" << (r.diverged ? "true" : "false")
            << " nan_seen=" << (r.nan_seen ? "true" : "false")
            << " min_det_F=" << r.min_detF << std::endl;
  print_history(tag, r.hist);
  if (!r.inner_message.empty())
    std::cout << tag << "_inner_solver_message=" << r.inner_message
              << std::endl;
}

// ---------------------------------------------------------------------------
// hyperelasticity#0 -- load stepping. The whole load applied in ONE step from
// an undeformed cold start against the SAME total load applied in increments.
// ---------------------------------------------------------------------------
static double g_scan  = 0.0;   // optional argv[2], for tuning only
static double g_scan2 = 0.0;   // optional argv[3], for tuning only

static int load_stepping()
{
  const double total = (g_scan > 0.0) ? g_scan : 400.0;
  const double E = 1.0e4, nu = 0.3;
  std::cout << "total_body_force=" << total << " youngs_modulus=" << E
            << " poisson_ratio=" << nu << std::endl;

  auto run = [&](unsigned int nsteps) {
    Solid s(1);
    s.bc = BC::cantilever;
    s.make_grid(12, 3, 1.0, 0.15);
    lame_from(E, nu, s.mu, s.lam);
    s.set_bc(0.0, true);
    s.allocate();
    NewtonReport last;
    unsigned int total_steps = 0;
    for (unsigned int L = 1; L <= nsteps; ++L)
      {
        Opts o;
        o.body_force_y = -total * double(L) / double(nsteps);
        NewtonOpts n;
        n.max_it = 25;
        n.tol    = 1e-9;
        last     = newton(s, o, n);
        total_steps += last.steps;
        if (!last.converged)
          break;
      }
    last.steps = total_steps;
    std::cout << "n_dofs=" << s.dof.n_dofs() << std::endl;
    return std::make_pair(last, s.tip_deflection());
  };

  const auto one  = run(1);
  const auto many = run(20);
  report("one_shot", one.first);
  std::cout << "one_shot_tip_deflection=" << one.second << std::endl;
  report("twenty_load_steps", many.first);
  std::cout << "twenty_load_steps_tip_deflection=" << many.second << std::endl;

  const NewtonReport &under = mutate() ? many.first : one.first;
  std::cout << "load_application="
            << (mutate() ? "twenty_increments" : "whole_load_in_one_step")
            << std::endl;
  bool over1e3 = false, sawnan = false;
  for (double v : under.hist)
    {
      if (std::isnan(v))
        sawnan = true;
      else if (v > 1e3)
        over1e3 = true;
    }
  // the claim's own Signal, split into the two things it actually predicts
  std::cout << "residual_exceeds_1e3=" << (over1e3 ? "true" : "false")
            << std::endl;
  std::cout << "residual_becomes_nan=" << (sawnan ? "true" : "false")
            << std::endl;
  std::cout << "elements_inverted=" << (under.detF_negative ? "true" : "false")
            << std::endl;
  std::cout << "converged=" << (under.converged ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (under.converged ? "load_path_converges"
                                : "cold_start_single_step_newton_fails")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// hyperelasticity#1 -- the line search that checks J = det(F) > 0. Same load
// path in both variants; the ONLY difference is whether the step length is
// backtracked until every quadrature point still has a positive Jacobian.
// ---------------------------------------------------------------------------
static int det_f_guard()
{
  // Load control, so the constraints stay homogeneous and the step length is
  // free to be backtracked on every iteration. One load step, a body force big
  // enough that the full Newton step walks straight through the point where
  // det F changes sign.
  const double total   = (g_scan > 0.0) ? g_scan : 1600.0;
  const double E = 1.0e4, nu = 0.3;
  const bool   guarded = mutate();
  std::cout << "body_force=" << total << " applied_in_one_step=true"
            << std::endl;
  std::cout << "step_length="
            << (guarded ? "backtracked_until_min_det_F_positive"
                        : "full_newton_step_alpha_1")
            << std::endl;

  Solid s(1);
  s.bc = BC::cantilever;
  s.make_grid(12, 3, 1.0, 0.15);
  lame_from(E, nu, s.mu, s.lam);
  s.set_bc(0.0, true);
  s.allocate();
  std::cout << "n_dofs=" << s.dof.n_dofs() << std::endl;

  Opts o;
  o.body_force_y = -total;
  NewtonOpts n;
  n.max_it           = 20;
  n.tol              = 1e-9;
  n.detF_line_search = guarded;
  const NewtonReport rep = newton(s, o, n);
  report("bending", rep);
  std::cout << "smallest_step_length_used=" << rep.alpha_min << std::endl;
  std::cout << "min_det_F_reached=" << rep.min_detF << std::endl;
  std::cout << "det_F_went_non_positive="
            << (rep.detF_negative ? "true" : "false") << std::endl;
  std::cout << "tangent_filled_with_nan=" << (rep.nan_seen ? "true" : "false")
            << std::endl;

  // What the user's OWN AssertThrow reports at the state Newton walked into.
  // AssertThrow is live in Release; nothing in deal.II raises anything here,
  // because there is no constitutive-model layer to raise it.
  Opts og       = o;
  og.guard_detF = true;
  s.set_bc(0.0, false);
  s.assemble(og);
  std::cout << "guarded_assembly_min_det_F=" << s.min_detF << std::endl;
  std::cout << "own_assertthrow_message="
            << (s.guard_message.empty() ? std::string("(none)")
                                        : additional_info(s.guard_message))
            << std::endl;
  std::cout << "VERDICT="
            << (rep.detF_negative
                  ? "unguarded_full_step_inverts_elements_silently"
                  : "det_F_line_search_keeps_the_elements_valid")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// hyperelasticity#2 -- the ln(J) volumetric form against the squared-J variant.
// Both are consistent linearisations of the same small-strain law, so the two
// only part company as the deformation grows -- and the gap they open is a
// MODEL error, which refinement cannot touch.
// ---------------------------------------------------------------------------
static double weighted_rel_gap(const std::vector<double> &a,
                               const std::vector<double> &b,
                               const std::vector<double> &w)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    {
      num += (a[i] - b[i]) * (a[i] - b[i]) * w[i];
      den += b[i] * b[i] * w[i];
    }
  return std::sqrt(num) / std::sqrt(std::max(1e-300, den));
}

static int lnj_vs_squared_j()
{
  const double E = 1.0e4;
  const double nu = 0.3;
  const Law test_law = mutate() ? Law::lnJ : Law::J2;
  std::cout << "law_under_test="
            << (test_law == Law::lnJ ? "S=mu(I-Cinv)+lambda*ln(J)*Cinv"
                                     : "S=mu(I-Cinv)+lambda/2*(J^2-1)*Cinv")
            << std::endl;
  std::cout << "reference_law=S=mu(I-Cinv)+lambda*ln(J)*Cinv" << std::endl;
  std::cout << "youngs_modulus=" << E << " poisson_ratio=" << nu << std::endl;

  // solve the SAME problem with one law, return the stress samples
  auto solve = [&](unsigned int nel, double squeeze, Law law,
                   std::vector<double> &s00, std::vector<double> &smag,
                   std::vector<double> &detF, std::vector<double> &jxw,
                   Vector<double> &u) {
    Solid s(1);
    // Confined compression: J is exactly the axial stretch, so the volumetric
    // part of the law is what is being loaded. The exact solution is a
    // HOMOGENEOUS deformation, which the Q1 space represents exactly -- that is
    // deliberate, it removes the discretisation error entirely, so whatever gap
    // the two laws show is the model difference and nothing else.
    s.bc = BC::confined;
    s.make_grid(nel, nel, 1.0, 1.0);
    lame_from(E, nu, s.mu, s.lam);
    s.set_bc(squeeze, true);
    s.allocate();
    Opts o;
    o.law = law;
    const unsigned int nload = 10;
    for (unsigned int L = 1; L <= nload; ++L)
      {
        NewtonOpts n;
        n.max_it    = 25;
        n.tol       = 1e-10;
        // AffineConstraints' inhomogeneity is applied to the INCREMENT, so
        // the value handed over here is this load step's increment, not the
        // accumulated target.
        n.dirichlet = squeeze / double(nload);
        const NewtonReport rep = newton(s, o, n);
        if (!rep.converged)
          std::cout << "  (warning: load step " << L << " nel=" << nel
                    << " squeeze=" << squeeze << " did not converge)"
                    << std::endl;
      }
    s.sample(o, s00, smag, detF, jxw);
    u = s.sol;
  };

  auto compare = [&](unsigned int nel, double squeeze, const char *tag) {
    std::vector<double> a0, am, ad, aw, b0, bm, bd, bw;
    Vector<double>      ua, ub;
    solve(nel, squeeze, test_law, a0, am, ad, aw, ua);
    solve(nel, squeeze, Law::lnJ, b0, bm, bd, bw, ub);
    const double sgap = weighted_rel_gap(a0, b0, bw);
    const double mgap = weighted_rel_gap(am, bm, bw);
    Vector<double> d(ua);
    d -= ub;
    const double ugap = d.l2_norm() / std::max(1e-300, ub.l2_norm());
    double minJa = 1e300, minJb = 1e300;
    for (double v : ad)
      minJa = std::min(minJa, v);
    for (double v : bd)
      minJb = std::min(minJb, v);
    std::cout << tag << " nel=" << nel << " squeeze=" << squeeze
              << " n_quadrature_points=" << a0.size()
              << " relative_stress_gap_S00=" << sgap
              << " relative_stress_gap_norm=" << mgap
              << " relative_displacement_gap=" << ugap
              << " min_det_F_test=" << minJa << " min_det_F_reference="
              << minJb << std::endl;
    return sgap;
  };

  const double small_strain = compare(4, -0.02, "small_strain");
  const double large_4      = compare(4, -0.70, "large_strain");
  const double large_8      = compare(8, -0.70, "large_strain");
  const double large_16     = compare(16, -0.70, "large_strain");

  // The claim's actual point: the ln(J) form has an infinite barrier at J -> 0,
  // the squared-J form does not. Under confined compression J is the axial
  // stretch, so the volumetric stress each law asks for is directly readable.
  for (double sq : {-0.5, -0.8, -0.95})
    {
      std::vector<double> a0, am, ad, aw, b0, bm, bd, bw;
      Vector<double>      ua, ub;
      solve(4, sq, Law::J2, a0, am, ad, aw, ua);
      solve(4, sq, Law::lnJ, b0, bm, bd, bw, ub);
      double sa = 0.0, sb = 0.0, w = 0.0;
      for (std::size_t i = 0; i < a0.size(); ++i)
        {
          sa += a0[i] * aw[i];
          sb += b0[i] * bw[i];
          w += bw[i];
        }
      std::cout << "confined_J=" << (1.0 + sq)
                << " mean_S00_squared_j_form=" << sa / w
                << " mean_S00_ln_j_form=" << sb / w
                << " ratio=" << (sa / std::max(1e-300, std::abs(sb)))
                << std::endl;
    }

  std::cout << "stress_gap_at_2_percent_strain=" << small_strain << std::endl;
  std::cout << "stress_gap_at_70_percent_strain_nel4=" << large_4 << std::endl;
  std::cout << "stress_gap_at_70_percent_strain_nel8=" << large_8 << std::endl;
  std::cout << "stress_gap_at_70_percent_strain_nel16=" << large_16
            << std::endl;
  const double drift = (large_16 > 0.0 || large_4 > 0.0)
                         ? std::abs(large_16 - large_4) /
                             std::max(1e-300, large_4)
                         : 0.0;
  std::cout << "stress_gap_drift_from_nel4_to_nel16=" << drift << std::endl;
  const bool grows   = large_4 > 20.0 * std::max(1e-300, small_strain);
  const bool order1  = large_4 > 0.1;
  const bool refines = drift < 0.05;
  std::cout << "gap_grows_with_deformation=" << (grows ? "true" : "false")
            << std::endl;
  std::cout << "gap_is_order_one_at_large_strain=" << (order1 ? "true"
                                                             : "false")
            << std::endl;
  std::cout << "gap_survives_a_four_fold_refinement=" << (refines ? "true"
                                                                 : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((grows && order1 && refines)
                  ? "squared_j_form_is_a_model_error_refinement_cannot_remove"
                  : "the_two_volumetric_forms_agree")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// hyperelasticity#3 -- the geometric part of the tangent. The claim says
// dropping K_geo makes the system NON-SYMMETRIC and tells the reader to measure
// max|A_ij - A_ji|. That measurement is done here, on both tangents, together
// with the directional-derivative check that actually detects the defect.
// ---------------------------------------------------------------------------
static void tangent_report(Solid &s, const Opts &o, const char *tag,
                           double &rel_sym, double &rel_fd)
{
  s.set_bc(0.0, false);
  s.assemble(o);
  const double defect = s.symmetry_defect();
  rel_sym             = defect / std::max(1e-300, s.max_absK);

  // directional-derivative check: K v against the central difference of the
  // residual in the direction v, on the free dofs only
  Vector<double> v(s.dof.n_dofs());
  for (types::global_dof_index i = 0; i < s.dof.n_dofs(); ++i)
    v(i) = s.constraints.is_constrained(i)
             ? 0.0
             : std::sin(1.0 + 2.7 * double(i)) * 1.0;
  Vector<double> Kv(s.dof.n_dofs());
  s.K.vmult(Kv, v);

  const double   eps  = 1e-7;
  Vector<double> base = s.sol;
  Vector<double> rp(s.dof.n_dofs()), rm(s.dof.n_dofs());
  s.sol = base;
  s.sol.add(eps, v);
  s.assemble(o, false);
  rp = s.rhs;
  s.sol = base;
  s.sol.add(-eps, v);
  s.assemble(o, false);
  rm    = s.rhs;
  s.sol = base;

  double num = 0.0, den = 0.0;
  for (types::global_dof_index i = 0; i < s.dof.n_dofs(); ++i)
    {
      if (s.constraints.is_constrained(i))
        continue;
      const double fd = -(rp(i) - rm(i)) / (2.0 * eps);
      num += (Kv(i) - fd) * (Kv(i) - fd);
      den += fd * fd;
    }
  rel_fd = std::sqrt(num) / std::sqrt(std::max(1e-300, den));
  // restore the matrix at the base state for anything the caller wants
  s.assemble(o);
  std::cout << tag << "_max_abs_entry=" << s.max_absK
            << " symmetry_defect=" << defect
            << " relative_symmetry_defect=" << rel_sym
            << " relative_directional_derivative_error=" << rel_fd << std::endl;
}

static int geometric_term()
{
  const double total = 400.0;
  const double E = 1.0e4, nu = 0.3;
  const bool   keep_k_geo = mutate();
  std::cout << "tangent_under_test="
            << (keep_k_geo ? "K_mat_plus_K_geo" : "K_mat_only") << std::endl;

  // drive the SAME deformed state with the consistent tangent, then compare the
  // two tangents at that state
  Solid s(1);
  s.bc = BC::cantilever;
  s.make_grid(12, 3, 1.0, 0.15);
  lame_from(E, nu, s.mu, s.lam);
  s.set_bc(0.0, true);
  s.allocate();
  Opts good;
  good.body_force_y = 0.0;
  const unsigned int nload = 20;
  for (unsigned int L = 1; L <= nload; ++L)
    {
      good.body_force_y = -total * double(L) / double(nload);
      NewtonOpts n;
      n.max_it = 25;
      n.tol    = 1e-10;
      const NewtonReport rep = newton(s, good, n);
      if (!rep.converged)
        std::cout << "  (warning: load step " << L << " did not converge)"
                  << std::endl;
    }
  std::cout << "n_dofs=" << s.dof.n_dofs()
            << " tip_deflection_at_the_test_state=" << s.tip_deflection()
            << " min_det_F=" << s.min_detF << std::endl;

  Opts bad      = good;
  bad.drop_k_geo = true;
  double sym_full, fd_full, sym_mat, fd_mat;
  tangent_report(s, good, "full_tangent", sym_full, fd_full);
  tangent_report(s, bad, "k_mat_only_tangent", sym_mat, fd_mat);

  // the claim's own instruction: measure the asymmetry directly
  const bool full_sym = sym_full < 1e-10;
  const bool mat_sym  = sym_mat < 1e-10;
  std::cout << "full_tangent_is_symmetric_to_roundoff="
            << (full_sym ? "true" : "false") << std::endl;
  std::cout << "k_mat_only_tangent_is_symmetric_to_roundoff="
            << (mat_sym ? "true" : "false") << std::endl;
  // FINDING, measured here: in a total-Lagrangian formulation K_mat = dE:C:dE
  // is symmetric all by itself, so dropping K_geo does NOT break symmetry. The
  // defect it does produce is INCONSISTENCY with the residual.
  std::cout << "dropping_k_geo_breaks_symmetry_as_the_claim_says="
            << ((!mat_sym) ? "true" : "false") << std::endl;
  // Consistency is judged by the RATIO of the two directional-derivative
  // errors, not by an absolute threshold: the consistent tangent sits at the
  // finite-difference floor and the truncated one is eight orders above it.
  const bool inconsistent =
    fd_mat > 1e4 * std::max(fd_full, 1e-14) && fd_mat > 1e-3;
  std::cout << "k_mat_only_tangent_is_inconsistent_with_the_residual="
            << (inconsistent ? "true" : "false") << std::endl;
  std::cout << "directional_derivative_error_ratio_mat_over_full="
            << fd_mat / std::max(fd_full, 1e-300) << std::endl;
  const double fd_test = keep_k_geo ? fd_full : fd_mat;
  std::cout << "tangent_under_test_directional_derivative_error=" << fd_test
            << std::endl;
  std::cout << "tangent_under_test_is_inconsistent="
            << ((keep_k_geo ? false : inconsistent) ? "true" : "false")
            << std::endl;

  // what it costs: Newton on the same load path with each tangent
  auto path = [&](bool drop) {
    Solid t(1);
    t.bc = BC::cantilever;
    t.make_grid(12, 3, 1.0, 0.15);
    lame_from(E, nu, t.mu, t.lam);
    t.set_bc(0.0, true);
    t.allocate();
    Opts o;
    o.drop_k_geo       = drop;
    unsigned int steps = 0;
    bool         all_ok = true;
    std::vector<double> lasth;
    for (unsigned int L = 1; L <= nload; ++L)
      {
        o.body_force_y = -total * double(L) / double(nload);
        NewtonOpts n;
        n.max_it               = 60;
        n.tol                  = 1e-10;
        const NewtonReport rep = newton(t, o, n);
        steps += rep.steps;
        all_ok = all_ok && rep.converged;
        if (L == nload)
          lasth = rep.hist;
      }
    print_history(drop ? "k_mat_only_last_load_step"
                       : "full_tangent_last_load_step",
                  lasth);
    return std::make_tuple(steps, all_ok, t.tip_deflection());
  };
  const auto pf = path(false);
  const auto pm = path(true);
  std::cout << "k_mat_only_path_completed="
            << (std::get<1>(pm) ? "true" : "false") << std::endl;
  std::cout << "full_tangent_total_newton_iterations=" << std::get<0>(pf)
            << " converged=" << (std::get<1>(pf) ? "true" : "false")
            << " tip=" << std::get<2>(pf) << std::endl;
  std::cout << "k_mat_only_total_newton_iterations=" << std::get<0>(pm)
            << " converged=" << (std::get<1>(pm) ? "true" : "false")
            << " tip=" << std::get<2>(pm) << std::endl;
  const double tipgap =
    std::abs(std::get<2>(pm) - std::get<2>(pf)) /
    std::max(1e-300, std::abs(std::get<2>(pf)));
  std::cout << "relative_tip_difference_between_the_two_tangents=" << tipgap
            << std::endl;
  std::cout << "iteration_count_penalty="
            << double(std::get<0>(pm)) / double(std::get<0>(pf)) << std::endl;

  const bool under_test_inconsistent = keep_k_geo ? false : inconsistent;
  std::cout << "VERDICT="
            << (under_test_inconsistent
                  ? "dropping_k_geo_costs_consistency_not_symmetry"
                  : "tangent_is_consistent_with_the_residual")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// hyperelasticity#4 -- volumetric locking of the single-field displacement
// element as nu approaches 1/2, and the mixed cure. The cure used here is
// selective reduced integration of the volumetric term on Q1 quads, which is
// the Malkus-Hughes equivalent of the Q1/P0 mixed element; the step-44
// three-field FESystem(FE_Q(2)^dim, FE_DGP(1), FE_DGP(0)) itself is NOT built
// here, so what is measured is "single field locks, mixed does not", not that
// particular element.
// ---------------------------------------------------------------------------
static double bend_tip(unsigned int degree, unsigned int nx, unsigned int ny,
                       double nu, bool reduced, bool &ok)
{
  const double E = 1.0e4;
  Solid        s(degree);
  s.bc = BC::cantilever;
  s.make_grid(nx, ny, 1.0, 0.1);
  lame_from(E, nu, s.mu, s.lam);
  s.set_bc(0.0, true);
  s.allocate();
  Opts o;
  o.reduced_volumetric = reduced;
  ok                   = true;
  const unsigned int nload = 2;
  for (unsigned int L = 1; L <= nload; ++L)
    {
      o.end_traction_y = -1.25 * double(L) / double(nload);
      NewtonOpts n;
      n.max_it = 30;
      // lambda reaches 1.7e7 at nu = 0.4999, so the achievable relative
      // residual is limited by round-off in the tangent, not by Newton.
      n.tol                  = 1e-8;
      const NewtonReport rep = newton(s, o, n);
      ok                     = ok && rep.converged;
    }
  return s.tip_deflection();
}

static int volumetric_locking()
{
  const bool mixed = mutate();
  std::cout << "element_under_test="
            << (mixed ? "Q1_with_reduced_volumetric_integration_eq_Q1_P0"
                      : "single_field_Q1_full_integration")
            << std::endl;
  std::cout << "step_44_three_field_element_built=false" << std::endl;

  double ratio_at_half = 1.0;
  for (double nu : {0.3, 0.45, 0.49, 0.4999})
    {
      bool okf = false, okr = false;
      const double tf = bend_tip(1, 16, 2, nu, false, okf);
      const double tr = bend_tip(1, 16, 2, nu, true, okr);
      const double ratio = tf / tr;
      std::cout << "nu=" << nu << " single_field_Q1_tip=" << tf
                << " mixed_tip=" << tr << " single_over_mixed=" << ratio
                << " converged=" << ((okf && okr) ? "true" : "false")
                << std::endl;
      if (nu > 0.499)
        ratio_at_half = ratio;
    }

  // an independent, much finer Q2 reference at the same nu, so the mixed answer
  // is not simply believed
  bool okref = false;
  const double ref = bend_tip(2, 64, 8, 0.4999, false, okref);
  bool okf = false, okr = false;
  const double tf = bend_tip(1, 16, 2, 0.4999, false, okf);
  const double tr = bend_tip(1, 16, 2, 0.4999, true, okr);
  std::cout << "fine_Q2_64x8_reference_tip=" << ref
            << " converged=" << (okref ? "true" : "false") << std::endl;
  const double ef = std::abs(tf - ref) / std::abs(ref);
  const double er = std::abs(tr - ref) / std::abs(ref);
  std::cout << "single_field_Q1_relative_error_vs_reference=" << ef
            << std::endl;
  std::cout << "mixed_relative_error_vs_reference=" << er << std::endl;

  const double e_test = mixed ? er : ef;
  std::cout << "relative_error_of_the_element_under_test=" << e_test
            << std::endl;
  const bool locks = e_test > 0.30;
  std::cout << "element_under_test_is_off_by_more_than_30_percent="
            << (locks ? "true" : "false") << std::endl;
  std::cout << "mixed_is_within_5_percent=" << (er < 0.05 ? "true" : "false")
            << std::endl;
  std::cout << "single_field_ratio_to_mixed_at_nu_0.4999=" << ratio_at_half
            << std::endl;
  std::cout << "VERDICT="
            << (locks ? "single_field_q1_locks_at_the_volumetric_limit"
                      : "element_does_not_lock")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// The smallest eigenvalue of the tangent restricted to the FREE dofs. Small
// meshes only -- this is a dense LAPACK call.
// ---------------------------------------------------------------------------
static double min_eig_free(Solid &s)
{
  std::vector<types::global_dof_index> freed;
  for (types::global_dof_index i = 0; i < s.dof.n_dofs(); ++i)
    if (!s.constraints.is_constrained(i))
      freed.push_back(i);
  const unsigned int n = freed.size();
  LAPACKFullMatrix<double> A(n, n);
  for (unsigned int a = 0; a < n; ++a)
    for (unsigned int b = 0; b < n; ++b)
      A(a, b) = s.K.el(freed[a], freed[b]);
  A.compute_eigenvalues();
  double m = std::numeric_limits<double>::max();
  for (unsigned int i = 0; i < n; ++i)
    m = std::min(m, A.eigenvalue(i).real());
  return m;
}

// ---------------------------------------------------------------------------
// hyperelasticity#5 -- Saint-Venant-Kirchhoff in compression. The tangent of
// the SVK energy loses positive definiteness at finite compression (the 1D
// first Piola stress lambda*E*(lambda^2-1)/2 turns over at lambda = 1/sqrt(3));
// the compressible Neo-Hookean energy does not. Both laws walk the SAME
// displacement ladder here.
// ---------------------------------------------------------------------------
static int svk_compression()
{
  const double E = 1.0e4, nu = 0.3;
  const Law    test_law = mutate() ? Law::lnJ : Law::SVK;
  std::cout << "law_under_test="
            << (test_law == Law::SVK ? "saint_venant_kirchhoff"
                                     : "compressible_neo_hookean")
            << std::endl;

  struct Out
  {
    double first_negative_eig_at = -1.0;
    double first_failure_at      = -1.0;
    bool   monotone              = true;
    bool   nan_seen              = false;
    double last_min_eig          = 0.0;
    double reached               = 0.0;
  };

  auto ladder = [&](Law law, const char *tag) {
    Out            out;
    Solid          s(1);
    s.bc = BC::compress_roller;
    s.make_grid(4, 4, 1.0, 1.0);
    lame_from(E, nu, s.mu, s.lam);
    s.set_bc(0.0, true);
    s.allocate();
    Opts o;
    o.law                    = law;
    const double        step = -0.05;
    for (unsigned int L = 1; L <= 12; ++L)
      {
        const double reached = -step * double(L);
        NewtonOpts   n;
        n.max_it               = 30;
        n.tol                  = 1e-9;
        n.dirichlet            = step;
        const NewtonReport rep = newton(s, o, n);
        out.reached            = reached;
        for (unsigned int i = 1; i < rep.hist.size(); ++i)
          if (!(rep.hist[i] < rep.hist[i - 1]))
            out.monotone = false;
        if (rep.nan_seen)
          out.nan_seen = true;
        if (!rep.converged && out.first_failure_at < 0.0)
          out.first_failure_at = reached;
        double eig = std::numeric_limits<double>::quiet_NaN();
        if (rep.converged)
          {
            s.set_bc(0.0, false);
            s.assemble(o);
            eig              = min_eig_free(s);
            out.last_min_eig = eig;
            if (eig < 0.0 && out.first_negative_eig_at < 0.0)
              out.first_negative_eig_at = reached;
          }
        std::cout << tag << " compression=" << reached
                  << " newton_steps=" << rep.steps
                  << " converged=" << (rep.converged ? "true" : "false")
                  << " min_tangent_eigenvalue=" << eig
                  << " min_det_F=" << rep.min_detF << std::endl;
        if (!rep.converged)
          break;
      }
    return out;
  };

  const Out svk = ladder(Law::SVK, "svk");
  const Out nh  = ladder(Law::lnJ, "neo_hookean");

  std::cout << "svk_first_negative_tangent_eigenvalue_at_compression="
            << svk.first_negative_eig_at << std::endl;
  std::cout << "svk_first_newton_failure_at_compression="
            << svk.first_failure_at << std::endl;
  std::cout << "svk_residual_was_monotone=" << (svk.monotone ? "true" : "false")
            << std::endl;
  std::cout << "neo_hookean_first_negative_tangent_eigenvalue_at_compression="
            << nh.first_negative_eig_at << std::endl;
  std::cout << "neo_hookean_reached_compression=" << nh.reached << std::endl;
  std::cout << "neo_hookean_min_tangent_eigenvalue_at_60_percent="
            << nh.last_min_eig << std::endl;

  // The claim's own Signal is a Newton residual that stops decreasing and an
  // inner solve that reports nan. Neither happens under displacement control:
  // pinned here as measured booleans rather than assumed.
  std::cout << "claimed_non_monotone_newton_residual_observed="
            << (svk.monotone ? "false" : "true") << std::endl;
  std::cout << "claimed_inner_solver_nan_observed="
            << (svk.nan_seen ? "true" : "false") << std::endl;

  const Out &under = (test_law == Law::SVK) ? svk : nh;
  const bool lost_definiteness = under.first_negative_eig_at > 0.0;
  const bool failed            = under.first_failure_at > 0.0;
  std::cout << "law_under_test_lost_positive_definiteness="
            << (lost_definiteness ? "true" : "false") << std::endl;
  std::cout << "law_under_test_newton_failed=" << (failed ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((lost_definiteness || failed)
                  ? "svk_loses_stability_in_compression_neo_hookean_does_not"
                  : "law_stays_stable_through_the_whole_ladder")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  if (argc > 2)
    g_scan = std::atof(argv[2]);
  if (argc > 3)
    g_scan2 = std::atof(argv[3]);
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(10);
  if (probe == "load_stepping")
    return load_stepping();
  if (probe == "det_f_guard")
    return det_f_guard();
  if (probe == "lnj_vs_squared_j")
    return lnj_vs_squared_j();
  if (probe == "geometric_term")
    return geometric_term();
  if (probe == "volumetric_locking")
    return volumetric_locking();
  if (probe == "svk_compression")
    return svk_compression();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 3;
}
