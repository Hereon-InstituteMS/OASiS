// Shared translation unit for the DG-transport Signal family.
//
// One upwind-DG transport solver on the unit square, exactly the shape of
// step-12: cell term -(u, beta.grad v), an upwind numerical flux on interior
// faces, and weak inflow/outflow terms on the domain boundary. Every probe
// below bends one part of it and measures what actually changes.
//
// The SAME machinery serves the dg_transport, dg_advection_reaction and
// advection_dg topics, so eleven fixture directories share this one build.
//
// SOME ASSERTS LIVE IN THE COMPILED LIBRARY (lac/sparse_matrix.h is a header,
// but the Assert is compiled with the consumer's NDEBUG, so the sparsity-pattern
// guard needs the Debug LIBRARY only for the deal.II-internal parts). Assert
// ABORTS (SIGABRT, rc=134); it does not throw, so the exit code is the
// observable and a try/catch would see nothing. The fixtures that care run this
// program against BOTH libraries and pin the pair.
//
// usage: dgtransport_family <probe>
//   feinterface_face_terms | mesh_loop_dispatch | block_preconditioner_ratio
//   | dof_wise_renumbering | strong_bc_on_dg | hier_bc_crash | bern_bc_crash
//   | central_flux | iteration_count_vs_h | flux_sparsity_pattern
//   | flipped_face_sides | renumbering_and_gmres
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_hierarchical.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
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

static std::string yesno(bool b)
{
  return b ? "true" : "false";
}

// ===========================================================================
// The advection field and the data.
// ===========================================================================
// field 0 : beta = (1, 1)              -- default numbering IS downstream
// field 1 : beta = (1, -1)             -- default numbering is NOT downstream
// field 2 : beta = (y, -x)             -- rotation about the corner (0,0):
//                                         every characteristic enters through
//                                         the top/left edge and leaves through
//                                         the bottom/right edge
// field 3 : beta = (y-1/2, -(x-1/2))   -- rotation about the middle: the
//                                         interior characteristics are CLOSED
static Tensor<1, dim> beta_at(const Point<dim> &p, int field)
{
  Tensor<1, dim> b;
  switch (field)
    {
      case 0:
        b[0] = 1.0;
        b[1] = 1.0;
        break;
      case 1:
        b[0] = 1.0;
        b[1] = -1.0;
        break;
      case 2:
        b[0] = p[1];
        b[1] = -p[0];
        break;
      default:
        b[0] = p[1] - 0.5;
        b[1] = -(p[0] - 0.5);
        break;
    }
  return b;
}

// prob 0 "layer"        : g = 1 on the whole inflow boundary, sigma given, f=0
// prob 1 "manufactured" : u = sin(pi x) sin(pi y) + 1/4 with sigma = 1 and the
//                         matching source, so an L2 rate can be measured
// prob 2 "periodic"     : y-periodic, u = sin(2 pi (y - x)), sigma = 0, f = 0,
//                         inflow datum defined on the x = 0 edge ONLY
enum Prob
{
  P_LAYER = 0,
  P_MANUF = 1,
  P_PERIODIC = 2
};

static double exact_u(const Point<dim> &p, int prob)
{
  if (prob == P_MANUF)
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) + 0.25;
  if (prob == P_PERIODIC)
    return std::sin(2.0 * numbers::PI * (p[1] - p[0]));
  return 1.0;
}

// The boundary datum a hand-written boundary_worker would use: it is the datum
// of the PHYSICAL inflow boundary, and it knows nothing about a periodic edge.
static double inflow_g(const Point<dim> &p, int prob)
{
  if (prob == P_MANUF)
    return exact_u(p, prob);
  if (prob == P_PERIODIC)
    return (p[0] < 1e-10) ? std::sin(2.0 * numbers::PI * p[1]) : 0.0;
  return 1.0;
}

static double source_f(const Point<dim> &p, int prob, int field, double sigma)
{
  if (prob != P_MANUF)
    return 0.0;
  const Tensor<1, dim> b = beta_at(p, field);
  const double gx = numbers::PI * std::cos(numbers::PI * p[0]) *
                    std::sin(numbers::PI * p[1]);
  const double gy = numbers::PI * std::sin(numbers::PI * p[0]) *
                    std::cos(numbers::PI * p[1]);
  return b[0] * gx + b[1] * gy + sigma * exact_u(p, prob);
}

class ExactFunction : public Function<dim>
{
public:
  explicit ExactFunction(int prob)
    : prob(prob)
  {}
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return exact_u(p, prob);
  }

private:
  const int prob;
};

// ===========================================================================
// Options and the solver.
// ===========================================================================
struct Opt
{
  unsigned int refine = 4;
  unsigned int degree = 1;
  int          field = 0;
  int          prob = P_LAYER;
  double       sigma = 0.0;
  bool         face_terms = true;   // interior-face (FEInterfaceValues) terms
  bool         central_flux = false;
  bool         flip_normal = false; // take n from the other side, keep [[v]]
  bool         flux_pattern = true; // make_flux_sparsity_pattern vs the cell one
  bool         use_mesh_loop = false;
  bool         periodic = false;
  bool         weak_inflow = true;  // put the inflow datum in the rhs at all
  int          renumber = 0;        // 0 none 1 Cuthill-McKee 2 downstream
                                    // cell-wise 3 downstream dof-wise
  MeshWorker::AssembleFlags extra_flags =
    MeshWorker::assemble_own_interior_faces_once;
};

struct DG
{
  Opt                  o;
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  MappingQ1<dim>       mapping;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       sol, rhs;
  std::vector<unsigned int> dof_cell;   // DG: every dof belongs to one cell
  std::vector<Point<dim>>   cell_center; // per cell index used above

  unsigned long n_cell_visits = 0, n_boundary_visits = 0, n_interior_visits = 0;
  unsigned long n_dropped_entries = 0;

  explicit DG(const Opt &opt)
    : o(opt)
    , fe(opt.degree)
    , dof(tria)
  {}

  void setup()
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
    if (o.periodic)
      {
        std::vector<GridTools::PeriodicFacePair<
          typename Triangulation<dim>::cell_iterator>>
          pairs;
        // boundary ids from hyper_cube(..., colorize=true): 2 is y=0, 3 is y=1
        GridTools::collect_periodic_faces(tria, 2, 3, 1, pairs);
        tria.add_periodicity(pairs);
      }
    tria.refine_global(o.refine);
    dof.distribute_dofs(fe);
    if (o.renumber == 1)
      DoFRenumbering::Cuthill_McKee(dof);
    else if (o.renumber == 2 || o.renumber == 3)
      {
        Tensor<1, dim> d = beta_at(Point<dim>(0.5, 0.5), o.field);
        DoFRenumbering::downstream(dof, d, o.renumber == 3);
      }

    DynamicSparsityPattern dsp(dof.n_dofs());
    if (o.flux_pattern)
      DoFTools::make_flux_sparsity_pattern(dof, dsp);
    else
      DoFTools::make_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    A.reinit(sp);
    sol.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());

    dof_cell.assign(dof.n_dofs(), 0);
    cell_center.clear();
    std::vector<types::global_dof_index> local(fe.n_dofs_per_cell());
    unsigned int c = 0;
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (const auto i : local)
          dof_cell[i] = c;
        cell_center.push_back(cell->center());
        ++c;
      }
  }

  unsigned long pattern_nonzeros(bool flux) const
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    if (flux)
      DoFTools::make_flux_sparsity_pattern(dof, dsp);
    else
      DoFTools::make_sparsity_pattern(dof, dsp);
    return dsp.n_nonzero_elements();
  }

  // ---- one interior-face contribution, shared by both dispatch styles ----
  void face_term(FEInterfaceValues<dim> &fiv, FullMatrix<double> &M) const
  {
    const unsigned int n = fiv.n_current_interface_dofs();
    M.reinit(n, n);
    M = 0.0;
    const auto &q = fiv.get_quadrature_points();
    const auto &JxW = fiv.get_JxW_values();
    const auto &nor = fiv.get_normal_vectors();
    for (unsigned int p = 0; p < q.size(); ++p)
      {
        Tensor<1, dim> nv = nor[p];
        if (o.flip_normal)
          nv *= -1.0;
        const double bdn = beta_at(q[p], o.field) * nv;
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              const double trace =
                o.central_flux ? fiv.average_of_shape_values(j, p)
                               : fiv.shape_value((bdn > 0), j, p);
              M(i, j) +=
                fiv.jump_in_shape_values(i, p) * trace * bdn * JxW[p];
            }
      }
  }

  void add_face_matrix(const FullMatrix<double>                   &M,
                       const std::vector<types::global_dof_index> &idx)
  {
    for (unsigned int i = 0; i < idx.size(); ++i)
      for (unsigned int j = 0; j < idx.size(); ++j)
        {
          if (M(i, j) == 0.0)
            continue;
          if (!sp.exists(idx[i], idx[j]))
            ++n_dropped_entries;
          A.add(idx[i], idx[j], M(i, j));
        }
  }

  // ---- hand-rolled dispatch: cells, then at_boundary faces, then the rest ---
  void assemble_hand_rolled()
  {
    QGauss<dim>       quad(fe.degree + 2);
    QGauss<dim - 1>   fquad(fe.degree + 2);
    FEValues<dim>     fev(mapping, fe, quad,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> ffv(mapping, fe, fquad,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors);
    FEInterfaceValues<dim> fiv(mapping, fe, fquad,
                               update_values | update_quadrature_points |
                                 update_JxW_values | update_normal_vectors);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n), fm;
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);

    for (const auto &cell : dof.active_cell_iterators())
      {
        ++n_cell_visits;
        fev.reinit(cell);
        cm = 0.0;
        cr = 0.0;
        for (unsigned int p = 0; p < quad.size(); ++p)
          {
            const Tensor<1, dim> b = beta_at(fev.quadrature_point(p), o.field);
            const double f = source_f(fev.quadrature_point(p), o.prob, o.field,
                                      o.sigma);
            for (unsigned int i = 0; i < n; ++i)
              {
                for (unsigned int j = 0; j < n; ++j)
                  cm(i, j) += (-b * fev.shape_grad(i, p) *
                                 fev.shape_value(j, p) +
                               o.sigma * fev.shape_value(i, p) *
                                 fev.shape_value(j, p)) *
                              fev.JxW(p);
                cr(i) += f * fev.shape_value(i, p) * fev.JxW(p);
              }
          }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          {
            for (unsigned int j = 0; j < n; ++j)
              A.add(local[i], local[j], cm(i, j));
            rhs(local[i]) += cr(i);
          }

        for (const unsigned int f : cell->face_indices())
          {
            // THE DISPATCH. A hand-written loop asks at_boundary() and stops
            // there; a periodic face answers YES to that question.
            if (cell->at_boundary(f))
              {
                ++n_boundary_visits;
                ffv.reinit(cell, f);
                cm = 0.0;
                cr = 0.0;
                for (unsigned int p = 0; p < fquad.size(); ++p)
                  {
                    const double bdn =
                      beta_at(ffv.quadrature_point(p), o.field) *
                      ffv.normal_vector(p);
                    if (bdn > 0)
                      {
                        for (unsigned int i = 0; i < n; ++i)
                          for (unsigned int j = 0; j < n; ++j)
                            cm(i, j) += ffv.shape_value(i, p) *
                                        ffv.shape_value(j, p) * bdn *
                                        ffv.JxW(p);
                      }
                    else if (o.weak_inflow)
                      {
                        const double g =
                          inflow_g(ffv.quadrature_point(p), o.prob);
                        for (unsigned int i = 0; i < n; ++i)
                          cr(i) += -ffv.shape_value(i, p) * g * bdn *
                                   ffv.JxW(p);
                      }
                  }
                for (unsigned int i = 0; i < n; ++i)
                  {
                    for (unsigned int j = 0; j < n; ++j)
                      A.add(local[i], local[j], cm(i, j));
                    rhs(local[i]) += cr(i);
                  }
                continue;
              }
            if (!o.face_terms)
              continue;
            const auto ncell = cell->neighbor(f);
            if (ncell->has_children())
              continue;
            if (cell->id() < ncell->id())
              {
                ++n_interior_visits;
                fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell,
                           cell->neighbor_of_neighbor(f),
                           numbers::invalid_unsigned_int);
                face_term(fiv, fm);
                add_face_matrix(fm, fiv.get_interface_dof_indices());
              }
          }
      }
  }

  // ---- MeshWorker::mesh_loop dispatch ----
  struct Scratch
  {
    Scratch(const Mapping<dim>        &m,
            const FiniteElement<dim>  &f,
            const Quadrature<dim>     &q,
            const Quadrature<dim - 1> &fq)
      : fev(m, f, q,
            update_values | update_gradients | update_quadrature_points |
              update_JxW_values)
      , fiv(m, f, fq,
            update_values | update_quadrature_points | update_JxW_values |
              update_normal_vectors)
    {}
    Scratch(const Scratch &s)
      : fev(s.fev.get_mapping(), s.fev.get_fe(), s.fev.get_quadrature(),
            s.fev.get_update_flags())
      , fiv(s.fiv.get_mapping(), s.fiv.get_fe(), s.fiv.get_quadrature(),
            s.fiv.get_update_flags())
    {}
    FEValues<dim>          fev;
    FEInterfaceValues<dim> fiv;
  };
  struct CopyFace
  {
    FullMatrix<double>                   m;
    std::vector<types::global_dof_index> idx;
  };
  struct Copy
  {
    FullMatrix<double>                   cm;
    Vector<double>                       cr;
    std::vector<types::global_dof_index> idx;
    std::vector<CopyFace>                faces;
  };

  void assemble_mesh_loop()
  {
    using Iter = typename DoFHandler<dim>::active_cell_iterator;
    QGauss<dim>     quad(fe.degree + 2);
    QGauss<dim - 1> fquad(fe.degree + 2);
    const unsigned int n = fe.n_dofs_per_cell();

    const auto cell_worker = [&](const Iter &cell, Scratch &s, Copy &c) {
      ++n_cell_visits;
      c.cm.reinit(n, n);
      c.cr.reinit(n);
      c.idx.resize(n);
      c.faces.clear();
      cell->get_dof_indices(c.idx);
      s.fev.reinit(cell);
      for (unsigned int p = 0; p < quad.size(); ++p)
        {
          const Tensor<1, dim> b = beta_at(s.fev.quadrature_point(p), o.field);
          const double f =
            source_f(s.fev.quadrature_point(p), o.prob, o.field, o.sigma);
          for (unsigned int i = 0; i < n; ++i)
            {
              for (unsigned int j = 0; j < n; ++j)
                c.cm(i, j) += (-b * s.fev.shape_grad(i, p) *
                                 s.fev.shape_value(j, p) +
                               o.sigma * s.fev.shape_value(i, p) *
                                 s.fev.shape_value(j, p)) *
                              s.fev.JxW(p);
              c.cr(i) += f * s.fev.shape_value(i, p) * s.fev.JxW(p);
            }
        }
    };

    const auto boundary_worker =
      [&](const Iter &cell, const unsigned int &face_no, Scratch &s, Copy &c) {
        ++n_boundary_visits;
        s.fiv.reinit(cell, face_no);
        const FEFaceValuesBase<dim> &ff = s.fiv.get_fe_face_values(0);
        for (unsigned int p = 0; p < ff.n_quadrature_points; ++p)
          {
            const double bdn = beta_at(ff.quadrature_point(p), o.field) *
                               ff.normal_vector(p);
            if (bdn > 0)
              {
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int j = 0; j < n; ++j)
                    c.cm(i, j) += ff.shape_value(i, p) * ff.shape_value(j, p) *
                                  bdn * ff.JxW(p);
              }
            else if (o.weak_inflow)
              {
                const double g = inflow_g(ff.quadrature_point(p), o.prob);
                for (unsigned int i = 0; i < n; ++i)
                  c.cr(i) += -ff.shape_value(i, p) * g * bdn * ff.JxW(p);
              }
          }
      };

    const auto face_worker = [&](const Iter &cell, const unsigned int &f,
                                 const unsigned int &sf, const Iter &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf, Scratch &s, Copy &c) {
      ++n_interior_visits;
      s.fiv.reinit(cell, f, sf, ncell, nf, nsf);
      c.faces.emplace_back();
      CopyFace &cf = c.faces.back();
      cf.idx = s.fiv.get_interface_dof_indices();
      face_term(s.fiv, cf.m);
    };

    const auto copier = [&](const Copy &c) {
      for (unsigned int i = 0; i < c.idx.size(); ++i)
        {
          for (unsigned int j = 0; j < c.idx.size(); ++j)
            A.add(c.idx[i], c.idx[j], c.cm(i, j));
          rhs(c.idx[i]) += c.cr(i);
        }
      for (const auto &cf : c.faces)
        add_face_matrix(cf.m, cf.idx);
    };

    Scratch scratch(mapping, fe, quad, fquad);
    Copy    copy;
    MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells |
                                      MeshWorker::assemble_boundary_faces;
    if (o.face_terms)
      flags = flags | o.extra_flags;
    // Serial, single thread: the visit counters above must not race.
    MeshWorker::mesh_loop(dof.begin_active(), dof.end(), cell_worker, copier,
                          scratch, copy, flags, boundary_worker, face_worker,
                          1, 1);
  }

  void assemble()
  {
    A = 0.0;
    rhs = 0.0;
    n_cell_visits = n_boundary_visits = n_interior_visits = 0;
    n_dropped_entries = 0;
    if (o.use_mesh_loop)
      assemble_mesh_loop();
    else
      assemble_hand_rolled();
  }

  void solve_direct()
  {
    SparseDirectUMFPACK inv;
    inv.initialize(A);
    sol = rhs;
    inv.solve(sol);
  }

  double l2_error() const
  {
    Vector<double> diff(tria.n_active_cells());
    VectorTools::integrate_difference(mapping, dof, sol, ExactFunction(o.prob),
                                      diff, QGauss<dim>(fe.degree + 3),
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(tria, diff,
                                             VectorTools::L2_norm);
  }

  unsigned long cross_cell_entries() const
  {
    unsigned long c = 0;
    for (unsigned int r = 0; r < dof.n_dofs(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        if (it->value() != 0.0 && dof_cell[r] != dof_cell[it->column()])
          ++c;
    return c;
  }

  // Entries whose two dofs sit in cells more than half a domain apart in y:
  // only a periodic neighbour pair can produce one.
  unsigned long periodic_entries() const
  {
    unsigned long c = 0;
    for (unsigned int r = 0; r < dof.n_dofs(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        if (it->value() != 0.0 &&
            std::abs(cell_center[dof_cell[r]][1] -
                     cell_center[dof_cell[it->column()]][1]) > 0.5)
          ++c;
    return c;
  }

  // Mean |[[u]]| over the interior faces of the solution, normalised by face
  // length: the quantity dg_transport#1 says should be O(1) for upwind DG.
  double mean_interior_jump() const
  {
    QGauss<dim - 1>        fq(fe.degree + 2);
    FEInterfaceValues<dim> fiv(mapping, fe, fq,
                               update_values | update_JxW_values);
    double integral = 0.0, length = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            continue;
          const auto ncell = cell->neighbor(f);
          if (ncell->has_children() || !(cell->id() < ncell->id()))
            continue;
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell,
                     cell->neighbor_of_neighbor(f),
                     numbers::invalid_unsigned_int);
          std::vector<double> here(fq.size()), there(fq.size());
          fiv.get_fe_face_values(0).get_function_values(sol, here);
          fiv.get_fe_face_values(1).get_function_values(sol, there);
          for (unsigned int p = 0; p < fq.size(); ++p)
            {
              integral +=
                std::abs(here[p] - there[p]) * fiv.get_JxW_values()[p];
              length += fiv.get_JxW_values()[p];
            }
        }
    return (length > 0.0) ? integral / length : 0.0;
  }

  double symmetry_defect() const
  {
    double num = 0.0, den = 0.0;
    for (unsigned int r = 0; r < dof.n_dofs(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        {
          const double a = it->value();
          const double b = sp.exists(it->column(), r)
                             ? A.el(it->column(), r)
                             : 0.0;
          num += (a - b) * (a - b);
          den += a * a;
        }
    return (den > 0.0) ? std::sqrt(num / den) : 0.0;
  }
};

// ===========================================================================
// Solver helpers.
// ===========================================================================
struct IterResult
{
  unsigned int steps = 0;
  double       value = 0.0;
  bool         converged = false;
};

enum PrecKind
{
  PREC_NONE = 0,
  PREC_JACOBI,
  PREC_SSOR,
  PREC_BLOCK_SSOR,
  PREC_BLOCK_JACOBI
};

static const char *prec_name(PrecKind k)
{
  switch (k)
    {
      case PREC_NONE:
        return "none";
      case PREC_JACOBI:
        return "point_jacobi";
      case PREC_SSOR:
        return "point_ssor";
      case PREC_BLOCK_SSOR:
        return "block_ssor";
      default:
        return "block_jacobi";
    }
}

template <class Prec>
static IterResult gmres_with(const SparseMatrix<double> &A,
                             const Vector<double>       &b,
                             const Prec                 &prec,
                             unsigned int                budget)
{
  IterResult r;
  Vector<double> x(b.size());
  SolverControl control(budget, 1e-8 * std::max(1e-300, b.l2_norm()));
  SolverGMRES<Vector<double>> solver(control);
  try
    {
      solver.solve(A, x, b, prec);
      r.converged = true;
    }
  catch (const std::exception &)
    {
      r.converged = false;
    }
  r.steps = control.last_step();
  r.value = control.last_value();
  return r;
}

static IterResult gmres_run(const DG &d, PrecKind kind, unsigned int budget)
{
  const unsigned int bs = d.fe.n_dofs_per_cell();
  switch (kind)
    {
      case PREC_NONE:
        return gmres_with(d.A, d.rhs, PreconditionIdentity(), budget);
      case PREC_JACOBI:
        {
          PreconditionJacobi<SparseMatrix<double>> p;
          p.initialize(d.A);
          return gmres_with(d.A, d.rhs, p, budget);
        }
      case PREC_SSOR:
        {
          PreconditionSSOR<SparseMatrix<double>> p;
          p.initialize(d.A);
          return gmres_with(d.A, d.rhs, p, budget);
        }
      case PREC_BLOCK_SSOR:
        {
          PreconditionBlockSSOR<SparseMatrix<double>, double> p;
          p.initialize(d.A,
                       typename PreconditionBlockSSOR<SparseMatrix<double>,
                                                      double>::AdditionalData(
                         bs, 1.0));
          return gmres_with(d.A, d.rhs, p, budget);
        }
      default:
        {
          PreconditionBlockJacobi<SparseMatrix<double>, double> p;
          p.initialize(d.A,
                       typename PreconditionBlockJacobi<SparseMatrix<double>,
                                                        double>::AdditionalData(
                         bs, 1.0));
          return gmres_with(d.A, d.rhs, p, budget);
        }
    }
}

// || b - A P(b) || / || b ||: at ~1e-15 the "preconditioner" is an exact solve.
static double one_application_residual(const DG &d, PrecKind kind)
{
  const unsigned int bs = d.fe.n_dofs_per_cell();
  Vector<double>     y(d.rhs.size()), r(d.rhs.size());
  if (kind == PREC_BLOCK_SSOR)
    {
      PreconditionBlockSSOR<SparseMatrix<double>, double> p;
      p.initialize(d.A,
                   typename PreconditionBlockSSOR<SparseMatrix<double>,
                                                  double>::AdditionalData(bs,
                                                                          1.0));
      p.vmult(y, d.rhs);
    }
  else if (kind == PREC_BLOCK_JACOBI)
    {
      PreconditionBlockJacobi<SparseMatrix<double>, double> p;
      p.initialize(d.A,
                   typename PreconditionBlockJacobi<
                     SparseMatrix<double>, double>::AdditionalData(bs, 1.0));
      p.vmult(y, d.rhs);
    }
  else if (kind == PREC_SSOR)
    {
      PreconditionSSOR<SparseMatrix<double>> p;
      p.initialize(d.A);
      p.vmult(y, d.rhs);
    }
  else
    {
      PreconditionJacobi<SparseMatrix<double>> p;
      p.initialize(d.A);
      p.vmult(y, d.rhs);
    }
  d.A.vmult(r, y);
  r.sadd(-1.0, 1.0, d.rhs);
  return r.l2_norm() / std::max(1e-300, d.rhs.l2_norm());
}

// ===========================================================================
// dg_transport#1 -- FEInterfaceValues is what couples the cells.
// The claim's Signal says the face jumps would come out at ~1e-8 ("a smooth,
// non-DG solution") without the face terms. Measured here against a reference
// assembly that HAS the face terms.
// ===========================================================================
static int feinterface_face_terms()
{
  Opt ref;
  ref.refine = 4;
  ref.prob = P_MANUF;
  ref.sigma = 1.0;
  ref.face_terms = true;
  Opt test = ref;
  test.face_terms = mutate(); // the mistake: FEValues only, no interface terms

  DG a(test), b(ref);
  a.setup();
  a.assemble();
  a.solve_direct();
  b.setup();
  b.assemble();
  b.solve_direct();

  std::cout << "n_dofs=" << a.dof.n_dofs()
            << " face_terms_under_test=" << yesno(test.face_terms) << std::endl;
  std::cout << "cross_cell_matrix_entries_under_test=" << a.cross_cell_entries()
            << " reference=" << b.cross_cell_entries() << std::endl;
  const double ja = a.mean_interior_jump(), jb = b.mean_interior_jump();
  std::cout << "mean_interior_face_jump_under_test=" << ja
            << " reference=" << jb << std::endl;
  std::cout << "l2_error_under_test=" << a.l2_error()
            << " reference=" << b.l2_error() << std::endl;
  const bool couples = a.cross_cell_entries() > 0;
  const bool collapsed = ja < 1e-8;
  const bool bigger = ja > 10.0 * jb;
  std::cout << "matrix_under_test_couples_different_cells=" << yesno(couples)
            << std::endl;
  std::cout << "jump_under_test_collapsed_below_1e_8=" << yesno(collapsed)
            << std::endl;
  std::cout << "jump_under_test_is_more_than_ten_times_the_reference="
            << yesno(bigger) << std::endl;
  std::cout << "VERDICT="
            << (!couples && bigger
                  ? "cell_only_assembly_leaves_cells_uncoupled_and_the_jumps_grow"
                  : "cells_are_coupled")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport#2 -- the dispatch a hand-written loop gets wrong.
// A y-periodic mesh: at_boundary() says YES on the periodic edge, so the
// hand-rolled loop puts inflow/outflow flux terms there. mesh_loop asks
// has_periodic_neighbor() and treats the same face as interior.
// ===========================================================================
static int mesh_loop_dispatch()
{
  Opt ref;
  ref.refine = 4;
  ref.prob = P_PERIODIC;
  ref.periodic = true;
  ref.sigma = 0.0;
  ref.use_mesh_loop = true;
  Opt test = ref;
  test.use_mesh_loop = mutate(); // the mistake: hand-rolled dispatch

  DG a(test), b(ref);
  a.setup();
  a.assemble();
  a.solve_direct();
  b.setup();
  b.assemble();
  b.solve_direct();

  std::cout << "n_dofs=" << a.dof.n_dofs()
            << " used_mesh_loop_under_test=" << yesno(test.use_mesh_loop)
            << std::endl;
  std::cout << "cell_visits_under_test=" << a.n_cell_visits
            << " boundary_face_visits_under_test=" << a.n_boundary_visits
            << " interior_face_visits_under_test=" << a.n_interior_visits
            << std::endl;
  std::cout << "boundary_face_visits_mesh_loop=" << b.n_boundary_visits
            << " interior_face_visits_mesh_loop=" << b.n_interior_visits
            << std::endl;

  // How many faces are there really? Count them off the triangulation.
  unsigned long at_boundary_faces = 0, interior_faces = 0, periodic_sides = 0;
  for (const auto &cell : b.tria.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      {
        if (cell->at_boundary(f))
          ++at_boundary_faces;
        else
          ++interior_faces;
        if (cell->has_periodic_neighbor(f))
          ++periodic_sides;
      }
  interior_faces /= 2; // each non-boundary face was seen from both cells
  const unsigned long periodic_pairs = periodic_sides / 2;
  std::cout << "faces_reporting_at_boundary_true=" << at_boundary_faces
            << " non_boundary_faces=" << interior_faces
            << " periodic_face_pairs=" << periodic_pairs << std::endl;
  std::cout << "at_boundary_is_true_on_a_periodic_face="
            << yesno(periodic_pairs > 0 && at_boundary_faces > 2 * periodic_pairs)
            << std::endl;

  // Does mesh_loop visit an interior face once, or twice? Measure both flags.
  Opt both = ref;
  both.extra_flags = MeshWorker::assemble_own_interior_faces_both;
  DG c(both);
  c.setup();
  c.assemble();
  std::cout << "interior_face_visits_flag_once=" << b.n_interior_visits
            << " interior_face_visits_flag_both=" << c.n_interior_visits
            << std::endl;
  const bool once_is_once =
    b.n_interior_visits == interior_faces + periodic_pairs;
  const bool both_is_twice = c.n_interior_visits == 2 * b.n_interior_visits;
  std::cout << "mesh_loop_visits_each_interior_face_exactly_once_with_the_once_flag="
            << yesno(once_is_once) << std::endl;
  std::cout << "mesh_loop_visits_each_interior_face_twice_with_the_both_flag="
            << yesno(both_is_twice) << std::endl;

  const unsigned long pa = a.periodic_entries(), pb = b.periodic_entries();
  std::cout << "matrix_entries_across_the_periodic_boundary_under_test=" << pa
            << " mesh_loop=" << pb << std::endl;
  const double ea = a.l2_error(), eb = b.l2_error();
  std::cout << "l2_error_under_test=" << ea << " mesh_loop=" << eb << std::endl;
  const double sa = a.symmetry_defect(), sb = b.symmetry_defect();
  std::cout << "relative_symmetry_defect_under_test=" << sa
            << " mesh_loop=" << sb << std::endl;

  std::cout << "operator_under_test_couples_across_the_periodic_boundary="
            << yesno(pa > 0) << std::endl;
  std::cout << "l2_error_under_test_is_order_one=" << yesno(ea > 0.1)
            << std::endl;
  std::cout << "reference_mesh_loop_operator_is_also_non_symmetric="
            << yesno(sb > 0.1) << std::endl;
  std::cout << "VERDICT="
            << ((pa == 0 && ea > 0.1)
                  ? "hand_rolled_dispatch_treats_the_periodic_face_as_a_boundary"
                  : "periodic_face_was_coupled")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport#3 -- how much PreconditionBlockSSOR buys is not a number.
// ===========================================================================
static int block_preconditioner_ratio()
{
  const int renum = mutate() ? 2 : 0; // T2_MUTATE: DoFRenumbering::downstream
  std::cout << "renumbering=" << (renum == 2 ? "downstream_cell_wise" : "none")
            << std::endl;

  double res_a = 0.0, res_b = 0.0;
  for (int field = 0; field < 2; ++field)
    {
      Opt o;
      o.refine = 4;
      o.field = field;
      o.prob = P_LAYER;
      o.renumber = renum;
      DG d(o);
      d.setup();
      d.assemble();
      const double r = one_application_residual(d, PREC_BLOCK_SSOR);
      const IterResult g = gmres_run(d, PREC_BLOCK_SSOR, 5000);
      const IterResult j = gmres_run(d, PREC_JACOBI, 5000);
      std::cout << "field=" << (field == 0 ? "beta_1_1" : "beta_1_minus1")
                << " one_application_residual=" << r
                << " block_ssor_steps=" << g.steps
                << " point_jacobi_steps=" << j.steps << std::endl;
      if (field == 0)
        res_a = r;
      else
        res_b = r;
    }

  // (c) curved but well posed: rotation about a corner, three levels.
  bool grows = false;
  unsigned int prev = 0;
  double last_res_c = 0.0;
  for (unsigned int refine = 3; refine <= 5; ++refine)
    {
      Opt o;
      o.refine = refine;
      o.field = 2;
      o.prob = P_LAYER;
      o.renumber = renum;
      DG d(o);
      d.setup();
      d.assemble();
      last_res_c = one_application_residual(d, PREC_BLOCK_SSOR);
      const IterResult b = gmres_run(d, PREC_BLOCK_SSOR, 5000);
      const IterResult s = gmres_run(d, PREC_SSOR, 5000);
      const IterResult j = gmres_run(d, PREC_JACOBI, 5000);
      std::cout << "rotation_about_a_corner refine=" << refine
                << " n_dofs=" << d.dof.n_dofs()
                << " one_application_residual=" << last_res_c
                << " block_ssor_steps=" << b.steps
                << " point_ssor_steps=" << s.steps
                << " point_jacobi_steps=" << j.steps << std::endl;
      if (refine > 3 && b.steps > prev)
        grows = true;
      prev = b.steps;
    }

  // (d) closed characteristics: rotation about the middle of the box.
  bool exhausted = false;
  {
    Opt o;
    o.refine = 4;
    o.field = 3;
    o.prob = P_LAYER;
    o.renumber = renum;
    DG d(o);
    d.setup();
    d.assemble();
    for (PrecKind k : {PREC_JACOBI, PREC_SSOR, PREC_BLOCK_SSOR})
      {
        const IterResult r = gmres_run(d, k, 5000);
        std::cout << "closed_characteristics prec=" << prec_name(k)
                  << " steps=" << r.steps << " converged=" << yesno(r.converged)
                  << std::endl;
        if (!r.converged)
          exhausted = true;
      }
  }

  std::cout << "downstream_compatible_case_one_application_residual_is_machine_precision="
            << yesno(res_a < 1e-12) << std::endl;
  std::cout << "non_downstream_case_one_application_residual_is_order_one="
            << yesno(res_b > 1e-3) << std::endl;
  std::cout << "curved_field_one_application_residual_is_order_one="
            << yesno(last_res_c > 1e-3) << std::endl;
  std::cout << "curved_field_block_ssor_iteration_count_grows_with_refinement="
            << yesno(grows) << std::endl;
  std::cout << "closed_characteristics_exhausted_a_preconditioner_budget="
            << yesno(exhausted) << std::endl;
  std::cout << "VERDICT="
            << ((res_a < 1e-12 && res_b > 1e-3)
                  ? "block_sweep_is_an_exact_solve_or_a_preconditioner_depending_on_the_ordering"
                  : "one_regime_only")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport#4 -- dof_wise_renumbering=true interleaves the cell dofs, so the
// "blocks" of PreconditionBlock* are no longer cells.
// ===========================================================================
static int dof_wise_renumbering()
{
  Opt o;
  o.refine = 4;
  o.field = 0;
  o.prob = P_LAYER;
  o.renumber = mutate() ? 2 : 3; // 3 = dof_wise_renumbering=true, the mistake
  std::cout << "dof_wise_renumbering=" << yesno(o.renumber == 3) << std::endl;
  DG d(o);
  d.setup();
  d.assemble();

  // Are a cell's dofs still contiguous? Measure it, do not assume.
  unsigned long noncontiguous = 0;
  std::vector<types::global_dof_index> local(d.fe.n_dofs_per_cell());
  for (const auto &cell : d.dof.active_cell_iterators())
    {
      cell->get_dof_indices(local);
      auto mn = *std::min_element(local.begin(), local.end());
      auto mx = *std::max_element(local.begin(), local.end());
      if (mx - mn + 1 != local.size())
        ++noncontiguous;
    }
  std::cout << "cells_whose_dofs_are_not_contiguous=" << noncontiguous
            << " of " << d.tria.n_active_cells() << std::endl;
  std::cout << "cell_dofs_are_contiguous=" << yesno(noncontiguous == 0)
            << std::endl;

  std::cout << "before_preconditioner" << std::endl;
  const double r = one_application_residual(d, PREC_BLOCK_SSOR);
  std::cout << "after_preconditioner one_application_residual=" << r
            << std::endl;
  const IterResult g = gmres_run(d, PREC_BLOCK_SSOR, 200);
  std::cout << "gmres_last_step=" << g.steps << " gmres_last_value=" << g.value
            << " converged=" << yesno(g.converged) << std::endl;
  const bool finite = std::isfinite(r) && r < 1e10;
  std::cout << "one_application_residual_is_finite_and_moderate=" << yesno(finite)
            << std::endl;
  // The entry also promised a NaN in SolverControl::last_value() and a
  // NoConvergence at step 0-1. Both halves are printed rather than assumed.
  std::cout << "gmres_last_value_is_not_finite=" << yesno(!std::isfinite(g.value))
            << std::endl;
  std::cout << "gmres_reported_a_failure=" << yesno(!g.converged) << std::endl;
  std::cout << "gmres_converged_anyway_with_the_broken_blocks="
            << yesno(g.converged) << std::endl;
  std::cout << "the_run_returned_without_an_exception_or_an_abort=true"
            << std::endl;
  std::cout << "VERDICT="
            << (!finite ? "dof_wise_renumbering_destroys_the_block_structure_silently"
                        : "block_structure_survived")
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_transport#7 -- inflow data on a DG space is weak, and the strong route is
// not refused, it is ignored.
// ===========================================================================
static int strong_bc_on_dg()
{
  Opt o;
  o.refine = 4;
  o.field = 0;
  o.prob = P_LAYER;   // g = 1 on the whole inflow boundary
  o.sigma = 0.0;
  o.weak_inflow = mutate(); // the mistake: no weak inflow flux at all
  DG d(o);
  d.setup();

  std::cout << "before_interpolate_boundary_values" << std::endl;
  std::map<types::global_dof_index, double> bv;
  VectorTools::interpolate_boundary_values(
    d.dof, 0, Functions::ConstantFunction<dim>(1.0), bv);
  AffineConstraints<double> c;
  VectorTools::interpolate_boundary_values(
    d.dof, 0, Functions::ConstantFunction<dim>(1.0), c);
  c.close();
  std::cout << "after_interpolate_boundary_values boundary_values_size="
            << bv.size() << " n_constraints=" << c.n_constraints() << std::endl;
  std::cout << "dg_strong_boundary_call_returned_normally=true" << std::endl;
  std::cout << "dg_boundary_values_map_is_empty=" << yesno(bv.empty())
            << std::endl;
  std::cout << "dg_affineconstraints_got_nothing=" << yesno(c.n_constraints() == 0)
            << std::endl;

  d.assemble();
  d.solve_direct();
  c.distribute(d.sol); // the CG habit: harmless and useless here
  std::cout << "used_weak_inflow_flux=" << yesno(o.weak_inflow) << std::endl;
  std::cout << "solution_linfty=" << d.sol.linfty_norm()
            << " rhs_linfty=" << d.rhs.linfty_norm() << std::endl;
  const bool appears = d.sol.linfty_norm() > 0.5;
  std::cout << "prescribed_inflow_value_appears_in_the_solution=" << yesno(appears)
            << std::endl;
  std::cout << "VERDICT="
            << (appears ? "inflow_datum_entered_through_the_flux"
                        : "strong_dirichlet_on_dg_leaves_the_inflow_datum_out")
            << std::endl;
  return 0;
}

// The contrast dg_transport#7 draws: a CONTINUOUS non-interpolatory element.
template <class FE>
static int noninterp_bc(FE &fe, const char *name)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(2);
  DoFHandler<dim> dh(tria);
  dh.distribute_dofs(fe);
  std::cout << "element=" << name << " n_dofs=" << dh.n_dofs()
            << " has_support_points=" << yesno(fe.has_support_points())
            << std::endl;
  std::cout << "before_interpolate_boundary_values" << std::endl;
  std::map<types::global_dof_index, double> bv;
  VectorTools::interpolate_boundary_values(
    dh, 0, Functions::ConstantFunction<dim>(1.0), bv);
  std::cout << "after_interpolate_boundary_values size=" << bv.size()
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_advection_reaction#0 -- the central flux.
// The claim says the amplitude "grows like ~exp(t) regardless of mesh size".
// The growth rate of the semi-discrete system du/dt = -M^{-1} A u is the
// largest real part of an eigenvalue of -M^{-1}A, so that is what is measured,
// at two mesh sizes, together with an actual time integration.
// ===========================================================================
static void mass_matrix(const DG &d, FullMatrix<double> &M)
{
  M.reinit(d.dof.n_dofs(), d.dof.n_dofs());
  M = 0.0;
  QGauss<dim>   quad(d.fe.degree + 2);
  FEValues<dim> fev(d.mapping, d.fe, quad,
                    update_values | update_JxW_values);
  const unsigned int n = d.fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local(n);
  for (const auto &cell : d.dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cell->get_dof_indices(local);
      for (unsigned int p = 0; p < quad.size(); ++p)
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            M(local[i], local[j]) +=
              fev.shape_value(i, p) * fev.shape_value(j, p) * fev.JxW(p);
    }
}

// One semi-discrete operator L = -M^{-1} A, its rightmost eigenvalue, and the
// L2 norm left in the box at t = 1 after the pulse has been advected out.
struct SemiDiscrete
{
  double max_real = 0.0;
  double residue = 0.0;
  unsigned int n_dofs = 0;
};

static SemiDiscrete semidiscrete(unsigned int refine, bool central)
{
  Opt o;
  o.refine = refine;
  o.field = 0;
  o.prob = P_LAYER;
  o.sigma = 0.0;
  o.central_flux = central;
  o.weak_inflow = false; // homogeneous inflow: a pure stability question
  DG d(o);
  d.setup();
  d.assemble();
  const unsigned int n = d.dof.n_dofs();

  FullMatrix<double> M, Ad(n, n);
  mass_matrix(d, M);
  Ad = 0.0;
  for (unsigned int r = 0; r < n; ++r)
    for (auto it = d.A.begin(r); it != d.A.end(r); ++it)
      Ad(r, it->column()) = it->value();
  FullMatrix<double> Minv(M);
  Minv.gauss_jordan();
  FullMatrix<double> L(n, n);
  Minv.mmult(L, Ad);
  L *= -1.0; // du/dt = -M^{-1} A u

  LAPACKFullMatrix<double> LA(n, n);
  for (unsigned int i = 0; i < n; ++i)
    for (unsigned int j = 0; j < n; ++j)
      LA(i, j) = L(i, j);
  LA.compute_eigenvalues();
  double mx = -1e300;
  for (unsigned int i = 0; i < n; ++i)
    mx = std::max(mx, LA.eigenvalue(i).real());

  // RK4 from a smooth bump, T = 1. beta = (1,1) carries the bump out of the
  // box well before t = 1, so the exact answer at t = 1 is zero and whatever
  // norm is left is numerical residue.
  Vector<double> u(n), k1(n), k2(n), k3(n), k4(n), t(n);
  std::vector<types::global_dof_index> local(d.fe.n_dofs_per_cell());
  for (const auto &cell : d.dof.active_cell_iterators())
    {
      cell->get_dof_indices(local);
      const Point<dim> cc = cell->center();
      const double v = std::exp(-40.0 * ((cc[0] - 0.3) * (cc[0] - 0.3) +
                                         (cc[1] - 0.3) * (cc[1] - 0.3)));
      for (const auto i : local)
        u(i) = v;
    }
  const double u0 = u.l2_norm();
  const double dt = 0.2 / (std::pow(2.0, refine) * (o.degree + 1) * 4.0);
  const unsigned int nsteps = static_cast<unsigned int>(1.0 / dt);
  for (unsigned int s = 0; s < nsteps; ++s)
    {
      L.vmult(k1, u);
      t = u;
      t.add(0.5 * dt, k1);
      L.vmult(k2, t);
      t = u;
      t.add(0.5 * dt, k2);
      L.vmult(k3, t);
      t = u;
      t.add(dt, k3);
      L.vmult(k4, t);
      u.add(dt / 6.0, k1);
      u.add(dt / 3.0, k2);
      u.add(dt / 3.0, k3);
      u.add(dt / 6.0, k4);
      if (!std::isfinite(u.l2_norm()))
        break;
    }
  SemiDiscrete r;
  r.max_real = mx;
  r.residue = u.l2_norm() / u0;
  r.n_dofs = n;
  return r;
}

static int central_flux()
{
  const bool central = !mutate(); // the mistake: 0.5*(u^+ + u^-)
  std::cout << "flux_under_test=" << (central ? "central" : "upwind")
            << std::endl;
  SemiDiscrete t[2], up[2];
  unsigned int level = 0;
  for (unsigned int refine = 3; refine <= 4; ++refine, ++level)
    {
      t[level] = semidiscrete(refine, central);
      up[level] = semidiscrete(refine, false);
      std::cout << "refine=" << refine << " n_dofs=" << t[level].n_dofs
                << " max_real_eigenvalue_under_test=" << t[level].max_real
                << " residue_at_t_equals_one_under_test=" << t[level].residue
                << std::endl;
      std::cout << "refine=" << refine
                << " max_real_eigenvalue_upwind=" << up[level].max_real
                << " residue_at_t_equals_one_upwind=" << up[level].residue
                << std::endl;
    }
  const bool positive = t[0].max_real > 0.0 || t[1].max_real > 0.0;
  const bool grew = t[0].residue > 2.0 || t[1].residue > 2.0;
  // Does the damping of the rightmost mode survive refinement, or go to zero?
  const bool damping_vanishes =
    std::abs(t[1].max_real) < 0.5 * std::abs(t[0].max_real);
  const bool residue_worse =
    t[0].residue > 10.0 * up[0].residue && t[1].residue > 10.0 * up[1].residue;
  std::cout << "largest_real_part_under_test_is_positive=" << yesno(positive)
            << std::endl;
  std::cout << "amplitude_under_test_grew_at_either_mesh_size=" << yesno(grew)
            << std::endl;
  std::cout << "damping_of_the_rightmost_mode_under_test_halves_under_refinement="
            << yesno(damping_vanishes) << std::endl;
  std::cout << "residue_under_test_is_more_than_ten_times_the_upwind_residue="
            << yesno(residue_worse) << std::endl;
  std::cout << "VERDICT="
            << (positive
                  ? "flux_under_test_has_growing_modes"
                  : (residue_worse
                       ? "flux_under_test_is_stable_but_leaves_a_residue_upwind_removes"
                       : "flux_under_test_behaves_like_upwind"))
            << std::endl;
  return 0;
}

// ===========================================================================
// dg_advection_reaction#1 -- is the iteration count h-independent?
// ===========================================================================
static int iteration_count_vs_h()
{
  const PrecKind under_test = mutate() ? PREC_BLOCK_SSOR : PREC_JACOBI;
  const int renum = mutate() ? 2 : 0;
  std::cout << "preconditioner_under_test=" << prec_name(under_test)
            << " renumbering=" << (renum == 2 ? "downstream" : "none")
            << std::endl;
  std::vector<unsigned int> test_steps, none_steps, block_steps;
  bool cg_failed = false;
  for (unsigned int refine = 2; refine <= 5; ++refine)
    {
      Opt o;
      o.refine = refine;
      o.field = 2; // curved field: no ordering makes the sweep an exact solve
      o.prob = P_LAYER;
      o.renumber = renum;
      DG d(o);
      d.setup();
      d.assemble();
      const IterResult t = gmres_run(d, under_test, 5000);
      const IterResult n = gmres_run(d, PREC_NONE, 5000);
      Opt o2 = o;
      o2.renumber = 2;
      DG e(o2);
      e.setup();
      e.assemble();
      const IterResult b = gmres_run(e, PREC_BLOCK_SSOR, 5000);
      test_steps.push_back(t.steps);
      none_steps.push_back(n.steps);
      block_steps.push_back(b.steps);
      std::cout << "refine=" << refine << " n_dofs=" << d.dof.n_dofs()
                << " under_test_steps=" << t.steps
                << " unpreconditioned_steps=" << n.steps
                << " block_ssor_downstream_steps=" << b.steps << std::endl;
      if (refine == 3)
        {
          SolverControl control(2000, 1e-8 * d.rhs.l2_norm());
          SolverCG<Vector<double>> cg(control);
          try
            {
              Vector<double> x(d.rhs.size());
              cg.solve(d.A, x, d.rhs, PreconditionIdentity());
              std::cout << "cg_on_the_nonsymmetric_operator=converged"
                        << std::endl;
            }
          catch (const std::exception &e)
            {
              cg_failed = true;
              std::cout << "cg_on_the_nonsymmetric_operator=failed"
                        << std::endl;
            }
        }
    }
  const bool test_grows = test_steps.back() > 2 * test_steps.front();
  const bool none_grows = none_steps.back() > 2 * none_steps.front();
  unsigned int bmax = *std::max_element(block_steps.begin(), block_steps.end());
  unsigned int bmin = *std::min_element(block_steps.begin(), block_steps.end());
  std::cout << "iteration_count_under_test_grows_with_refinement="
            << yesno(test_grows) << std::endl;
  std::cout << "unpreconditioned_iteration_count_grows_with_refinement="
            << yesno(none_grows) << std::endl;
  std::cout << "cg_on_the_nonsymmetric_dg_operator_failed=" << yesno(cg_failed)
            << std::endl;
  std::cout << "block_ssor_with_downstream_ordering_is_mesh_independent="
            << yesno(bmax <= bmin + 2) << std::endl;
  std::cout << "VERDICT="
            << (test_grows ? "iteration_count_under_test_is_not_mesh_independent"
                           : "iteration_count_under_test_is_mesh_independent")
            << std::endl;
  return 0;
}

// ===========================================================================
// advection_dg#0 -- the sparsity pattern a DG assembly needs.
// ===========================================================================
static int flux_sparsity_pattern()
{
  Opt o;
  o.refine = 3;
  o.field = 0;
  o.prob = P_MANUF;
  o.sigma = 1.0;
  o.flux_pattern = mutate(); // the mistake: DoFTools::make_sparsity_pattern
  DG d(o);
  d.setup();
  std::cout << "used_flux_sparsity_pattern=" << yesno(o.flux_pattern) << std::endl;
  std::cout << "cell_pattern_nonzeros=" << d.pattern_nonzeros(false)
            << " flux_pattern_nonzeros=" << d.pattern_nonzeros(true)
            << std::endl;
  const double ratio =
    double(d.pattern_nonzeros(true)) / double(d.pattern_nonzeros(false));
  std::cout << "flux_over_cell_pattern_ratio=" << ratio << std::endl;
  std::cout << "flux_pattern_is_substantially_larger=" << yesno(ratio > 2.0)
            << std::endl;
  std::cout << "before_assembly" << std::endl;
  d.assemble();
  std::cout << "after_assembly silently_dropped_face_entries="
            << d.n_dropped_entries << std::endl;
  std::cout << "face_entries_were_silently_dropped="
            << yesno(d.n_dropped_entries > 0) << std::endl;
  d.solve_direct();
  std::cout << "l2_error=" << d.l2_error() << std::endl;
  std::cout << "VERDICT="
            << (d.n_dropped_entries > 0
                  ? "cell_only_pattern_drops_every_face_entry"
                  : "no_entry_was_dropped")
            << std::endl;
  return 0;
}

// ===========================================================================
// advection_dg#2 -- the normal taken from the wrong side.
// ===========================================================================
static int flipped_face_sides()
{
  const bool flip = !mutate();
  std::cout << "normal_taken_from_the_other_side=" << yesno(flip) << std::endl;
  double err[3], ref[3];
  for (unsigned int k = 0; k < 3; ++k)
    {
      Opt o;
      o.refine = 3 + k;
      o.field = 0;
      o.prob = P_MANUF;
      o.sigma = 1.0;
      o.flip_normal = flip;
      DG a(o);
      a.setup();
      a.assemble();
      a.solve_direct();
      err[k] = a.l2_error();
      Opt g = o;
      g.flip_normal = false;
      DG b(g);
      b.setup();
      b.assemble();
      b.solve_direct();
      ref[k] = b.l2_error();
      std::cout << "refine=" << o.refine << " n_dofs=" << a.dof.n_dofs()
                << " l2_error_under_test=" << err[k]
                << " l2_error_correct=" << ref[k]
                << " solution_linfty_under_test=" << a.sol.linfty_norm()
                << std::endl;
      if (k == 0)
        {
          // Is the flipped operator the TRANSPOSE of the correct one?
          double num = 0.0, den = 0.0;
          for (unsigned int r = 0; r < a.dof.n_dofs(); ++r)
            for (auto it = a.A.begin(r); it != a.A.end(r); ++it)
              {
                const double t = b.sp.exists(it->column(), r)
                                   ? b.A.el(it->column(), r)
                                   : 0.0;
                num += (it->value() - t) * (it->value() - t);
                den += it->value() * it->value();
              }
          const double rel = std::sqrt(num / std::max(1e-300, den));
          std::cout << "relative_distance_to_the_transpose_of_the_correct_matrix="
                    << rel << std::endl;
          std::cout << "flipped_matrix_is_the_transpose_of_the_correct_one="
                    << yesno(rel < 1e-10) << std::endl;
        }
    }
  const double rate_test =
    std::log(err[0] / err[2]) / std::log(4.0);
  const double rate_ref = std::log(ref[0] / ref[2]) / std::log(4.0);
  std::cout << "observed_rate_under_test=" << rate_test
            << " observed_rate_correct=" << rate_ref << std::endl;
  const bool ref_ok = rate_ref > 1.7;
  const bool test_lost = rate_test < 0.5;
  const bool norm_grew = err[2] > err[0];
  std::cout << "correct_operator_shows_the_expected_order_two_rate="
            << yesno(ref_ok) << std::endl;
  std::cout << "under_test_rate_is_not_even_half_an_order=" << yesno(test_lost)
            << std::endl;
  std::cout << "under_test_error_grew_under_refinement=" << yesno(norm_grew)
            << std::endl;
  std::cout << "VERDICT="
            << (test_lost ? "wrong_side_normal_loses_the_convergence_rate"
                          : "rate_preserved")
            << std::endl;
  return 0;
}

// ===========================================================================
// advection_dg#3 -- streamline ordering of the dofs.
// ===========================================================================
static int renumbering_and_gmres()
{
  const int under_test = mutate() ? 2 : 1; // 1 Cuthill-McKee, 2 downstream
  std::cout << "ordering_under_test="
            << (under_test == 2 ? "downstream" : "cuthill_mckee") << std::endl;
  unsigned int none_steps[3];
  unsigned int block_steps[3];
  const char  *names[3] = {"default", "cuthill_mckee", "downstream"};
  for (int r = 0; r < 3; ++r)
    {
      Opt o;
      o.refine = 4;
      o.field = 2; // curved: the sweep is a real preconditioner here
      o.prob = P_LAYER;
      o.renumber = r; // 0 default, 1 Cuthill-McKee, 2 downstream cell-wise
      DG d(o);
      d.setup();
      d.assemble();
      const IterResult n = gmres_run(d, PREC_NONE, 5000);
      const IterResult b = gmres_run(d, PREC_BLOCK_SSOR, 5000);
      none_steps[r] = n.steps;
      block_steps[r] = b.steps;
      std::cout << "ordering=" << names[r]
                << " unpreconditioned_gmres_steps=" << n.steps
                << " block_ssor_gmres_steps=" << b.steps << std::endl;
    }
  const bool none_same =
    none_steps[0] == none_steps[1] && none_steps[1] == none_steps[2];
  std::cout << "unpreconditioned_gmres_count_is_the_same_under_every_ordering="
            << yesno(none_same) << std::endl;
  const bool prec_differs = block_steps[1] != block_steps[2];
  std::cout << "block_preconditioned_count_depends_on_the_ordering="
            << yesno(prec_differs) << std::endl;
  const unsigned int test = block_steps[under_test];
  const unsigned int other = block_steps[under_test == 2 ? 1 : 2];
  std::cout << "block_ssor_steps_under_test=" << test
            << " block_ssor_steps_other_ordering=" << other << std::endl;
  std::cout << "ordering_under_test_is_the_slower_one=" << yesno(test > other)
            << std::endl;
  std::cout << "VERDICT="
            << (none_same
                  ? "ordering_changes_nothing_without_a_preconditioner"
                  : "ordering_changed_the_unpreconditioned_count")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  // Single-threaded: MeshWorker::mesh_loop runs its workers through WorkStream,
  // and the face-visit counters below are plain unsigned longs.
  MultithreadInfo::set_thread_limit(1);
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(8);
  if (probe == "feinterface_face_terms")
    return feinterface_face_terms();
  if (probe == "mesh_loop_dispatch")
    return mesh_loop_dispatch();
  if (probe == "block_preconditioner_ratio")
    return block_preconditioner_ratio();
  if (probe == "dof_wise_renumbering")
    return dof_wise_renumbering();
  if (probe == "strong_bc_on_dg")
    return strong_bc_on_dg();
  if (probe == "hier_bc_crash")
    {
      FE_Q_Hierarchical<dim> fe(2);
      return noninterp_bc(fe, "FE_Q_Hierarchical_2");
    }
  if (probe == "bern_bc_crash")
    {
      FE_Bernstein<dim> fe(2);
      return noninterp_bc(fe, "FE_Bernstein_2");
    }
  if (probe == "central_flux")
    return central_flux();
  if (probe == "iteration_count_vs_h")
    return iteration_count_vs_h();
  if (probe == "flux_sparsity_pattern")
    return flux_sparsity_pattern();
  if (probe == "flipped_face_sides")
    return flipped_face_sides();
  if (probe == "renumbering_and_gmres")
    return renumbering_and_gmres();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
