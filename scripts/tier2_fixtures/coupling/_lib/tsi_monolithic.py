"""The MONOLITHIC reference for the two-way TSI fixtures.

FIXTURE-SIDE ONLY. This file lives under scripts/tier2_fixtures/ and is never
served through any tool: it computes the answer the coupled runs are graded
against, so an agent that could reach it could quote it back instead of solving
anything. The served side is data/coupling_participants/participant_tsi_*.py,
which contain no reference values at all.

WHAT IT IS. The same one-step coupled thermoelastic problem the partitioned runs
solve, assembled as ONE linear system in (u, T) and solved in ONE code with no
partitioning, no exchange and no iteration:

    | K_uu            B  | |u|   | -beta*T_ref * int tr(eps(v)) |
    |                    | | | = |                              |
    | T_ref/dt * B^T  A_T| |T|   | rho_c/dt * (T_old, s)        |

with
    K_uu  = int [ 2 mu eps(u):eps(v) + lam tr(eps(u)) tr(eps(v)) ]
    B     = -int beta * T * tr(eps(v))
    A_T   = int [ rho_c/dt * T*s + k grad(T).grad(s) ]

`coupling` scales the OFF-DIAGONAL BLOCK B^T only — the mechanical -> thermal
direction. coupling=0 is the one-way reference (temperature drives deformation,
nothing comes back); coupling=1 is two-way. Everything else is identical, so the
difference between the two is exactly the reverse direction's contribution and
nothing else.

WHY THIS IS AN INDEPENDENT CHECK AND NOT A RESTATEMENT. It shares the physics
with the participants and nothing else: one mesh instead of two, one assembly
instead of two, no interpolation between non-matching meshes, no fixed-point
iteration, no relaxation. Every mechanism that a partitioned coupling can get
silently wrong — a sign on the exchanged quantity, a unit, an interpolation that
does not preserve the field, a fixed point that is not the solution — is absent
here. A coupled run that agrees with this has none of them.

AND THE REFERENCE ITSELF IS CHECKED, by an identity that involves no reference
values. With no y-variation in the boundary temperature the body is in uniaxial
strain, so sigma_xx = 0 gives tr(eps) = beta*theta/(lam+2mu) pointwise, and the
energy equation collapses to the SAME equation with rho_c replaced by
rho_c*(1+delta), delta = T_ref*beta^2/(rho_c*(lam+2mu)) — the classical
thermoelastic coupling parameter. So the two-way solve at rho_c must equal the
ONE-WAY solve at rho_c*(1+delta), to solver accuracy. That is a statement about
the physics that a wrong sign, a wrong beta or a dropped T_ref all break, and it
is checkable without knowing any answer. `check_effective_capacity_identity`
runs it.

Run as a script it reads `mono_config.json` from its working directory and
writes `monolithic.json` (temperature_change on the mesh nodes, the shape the
`couple` tool's `monolithic=` argument consumes) plus `monolithic_full.json`
(the strain field and the scalar QoIs, for the fixture harness).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TsiProblem:
    """One coupled thermoelastic step. SI units throughout.

    The initial state is a body at the uniform temperature `t_old`, in
    mechanical equilibrium with it. That equilibrium is uniaxial strain with
    tr(eps) = beta*(t_old - t_ref)/(lam + 2 mu) — an exact solution of the
    discrete problem as well as the continuous one, which is why `evol_old` can
    be a constant rather than a field.
    """
    lx: float = 0.02
    ly: float = 0.005
    e_mod: float = 2.1e11
    nu: float = 0.3
    alpha: float = 1.2e-4      # thermal expansion coefficient, 1/K
    k_cond: float = 52.0
    rho_c: float = 3.297e6
    dt: float = 1.0
    t_ref: float = 293.0
    t_old: float = 303.0
    t_hot: float = 323.0
    t_hot_dy: float = 10.0
    t_cold: float = 303.0

    @property
    def lam(self) -> float:
        return self.e_mod * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def mu(self) -> float:
        return self.e_mod / (2.0 * (1.0 + self.nu))

    @property
    def beta(self) -> float:
        """Thermal stress modulus (3 lam + 2 mu) alpha, Pa/K."""
        return (3.0 * self.lam + 2.0 * self.mu) * self.alpha

    @property
    def pwave(self) -> float:
        return self.lam + 2.0 * self.mu

    @property
    def delta(self) -> float:
        """The classical thermoelastic coupling parameter. It is the whole size
        of the mechanical -> thermal direction: the reverse coupling multiplies
        the effective heat capacity by (1 + delta) in uniaxial strain."""
        return self.t_ref * self.beta ** 2 / (self.rho_c * self.pwave)

    @property
    def evol_old(self) -> float:
        return self.beta * (self.t_old - self.t_ref) / self.pwave


STEEL = TsiProblem(alpha=1.2e-5)          # delta ~ 0.012 — a real metal
STRONG = TsiProblem(alpha=1.2e-4)         # delta ~ 1.25  — exaggerated on purpose


def solve_monolithic(p: TsiProblem, nx: int = 80, ny: int = 20,
                     coupling: float = 1.0, rho_c_scale: float = 1.0,
                     alpha_mech: float | None = None) -> dict:
    """Assemble and solve the coupled system in one piece. Returns node
    coordinates, theta = T - t_ref, the volumetric strain and the scalar QoIs.

    `alpha_mech` lets the MECHANICAL expansion coefficient differ from the
    thermal one; it exists only so a mutation control can be checked against a
    reference that did NOT move with it.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from skfem import (Basis, BilinearForm, ElementTriP1, ElementTriP2,
                       ElementVector, LinearForm, MeshTri, asm, condense, solve)
    from skfem.helpers import ddot, dot, grad, sym_grad, trace

    lam, mu = p.lam, p.mu
    beta = p.beta
    beta_m = beta if alpha_mech is None else (3.0 * lam + 2.0 * mu) * alpha_mech
    rho_c = p.rho_c * rho_c_scale
    io = 4

    m = MeshTri.init_tensor(np.linspace(0.0, p.lx, nx + 1),
                            np.linspace(0.0, p.ly, ny + 1))
    ub = Basis(m, ElementVector(ElementTriP2()), intorder=io)
    tb = Basis(m, ElementTriP1(), intorder=io)

    @BilinearForm
    def a_uu(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return 2.0 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

    @BilinearForm
    def b_uT(T, v, w):
        return -beta_m * T * trace(sym_grad(v))

    @BilinearForm
    def a_TT(T, s, w):
        return rho_c / p.dt * T * s + p.k_cond * dot(grad(T), grad(s))

    @BilinearForm
    def c_Tu(u, s, w):
        return coupling * p.t_ref * beta / p.dt * trace(sym_grad(u)) * s

    @LinearForm
    def f_u(v, w):
        return -beta_m * p.t_ref * trace(sym_grad(v))

    @LinearForm
    def f_T(s, w):
        # rho_c/dt*(T_old, s)  +  coupling*T_ref*beta/dt*(e_old, s):
        # the reverse term is (e - e_old), and e_old moves to the right side.
        return (rho_c / p.dt * p.t_old
                + coupling * p.t_ref * beta / p.dt * p.evol_old) * s

    @BilinearForm
    def mass(a, b, w):
        return a * b

    @LinearForm
    def evol_rhs(q, w):
        return trace(sym_grad(w["uh"])) * q

    Kuu, Att = asm(a_uu, ub), asm(a_TT, tb)
    B, C = asm(b_uT, tb, ub), asm(c_Tu, ub, tb)
    fu, ft = asm(f_u, ub), asm(f_T, tb)
    nu_, nt_ = ub.N, tb.N
    A = sp.bmat([[Kuu, B], [C, Att]], format="csr")
    F = np.concatenate([fu, ft])

    tol = 1e-12
    dx0 = ub.get_dofs(lambda x: np.abs(x[0]) < tol).all("u^1")
    dy0 = ub.get_dofs(lambda x: np.abs(x[1]) < tol).all("u^2")
    dy1 = ub.get_dofs(lambda x: np.abs(x[1] - p.ly) < tol).all("u^2")
    Du = np.unique(np.concatenate([dx0, dy0, dy1]))

    hot = tb.get_dofs(lambda x: np.abs(x[0]) < tol).all()
    cold = tb.get_dofs(lambda x: np.abs(x[0] - p.lx) < tol).all()
    x0 = np.zeros(nu_ + nt_)
    x0[nu_ + hot] = p.t_hot + p.t_hot_dy * (tb.doflocs[1][hot] / p.ly)
    x0[nu_ + cold] = p.t_cold
    D = np.concatenate([Du, nu_ + np.unique(np.concatenate([hot, cold]))])

    sol = solve(*condense(A, F, x=x0, D=D),
                solver=lambda K, b: spl.spsolve(K, b))
    u, T = sol[:nu_], sol[nu_:]
    evol = solve(asm(mass, tb), asm(evol_rhs, tb, uh=ub.interpolate(u)))
    ix = ub.split_indices()[0]
    return {"coordinates": tb.doflocs.T.tolist(),
            "theta": (T - p.t_ref).tolist(),
            "evol": evol.tolist(),
            "n": int(nt_),
            "ux_max": float(np.max(np.abs(u[ix]))),
            "theta_mean": float(np.mean(T - p.t_ref)),
            "evol_mean": float(np.mean(evol)),
            "delta": p.delta}


# ── the identity that checks the reference itself ─────────────────────────

def check_effective_capacity_identity(p: TsiProblem, nx: int = 80, ny: int = 20
                                      ) -> tuple[float, float]:
    """With no y-variation the body is in uniaxial strain, and the two-way
    problem at rho_c is EXACTLY the one-way problem at rho_c*(1+delta).

    Returns (relative deviation of that identity, relative size of the reverse
    direction itself). The second is the scale the first must be judged
    against: an identity that holds to 1e-8 is only evidence if the effect it
    is testing is orders of magnitude larger than that.
    """
    q = TsiProblem(**{**asdict(p), "t_hot_dy": 0.0})
    two = np.asarray(solve_monolithic(q, nx, ny, coupling=1.0)["theta"])
    one_eff = np.asarray(solve_monolithic(
        q, nx, ny, coupling=0.0, rho_c_scale=1.0 + q.delta)["theta"])
    one = np.asarray(solve_monolithic(q, nx, ny, coupling=0.0)["theta"])
    scale = float(np.max(np.abs(two))) or 1.0
    return (float(np.max(np.abs(two - one_eff))) / scale,
            float(np.max(np.abs(two - one))) / scale)


# ── run as a script: the `monolithic=` reference the couple tool consumes ──

CONFIG = "mono_config.json"


def write_reference(work_dir: Path, p: TsiProblem, nx: int, ny: int,
                    coupling: float = 1.0) -> list[str]:
    """Put a runnable, self-contained reference solve into `work_dir` and return
    the command that produces monolithic.json there."""
    import shutil
    import sys
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), work_dir / "tsi_monolithic.py")
    (work_dir / CONFIG).write_text(json.dumps(
        {"problem": asdict(p), "nx": nx, "ny": ny, "coupling": coupling},
        indent=2, sort_keys=True))
    return [sys.executable, "tsi_monolithic.py"]


def _main() -> None:
    cfg = json.loads(Path(CONFIG).read_text())
    p = TsiProblem(**cfg["problem"])
    out = solve_monolithic(p, int(cfg["nx"]), int(cfg["ny"]),
                           coupling=float(cfg.get("coupling", 1.0)))
    Path("monolithic_full.json").write_text(json.dumps(out))
    # The shape `couple(monolithic=...)` reads. field_name MUST match what the
    # thermal participant exports or the tool compares nothing and says so.
    Path("monolithic.json").write_text(json.dumps({
        "field_name": "temperature_change",
        "n_points": out["n"],
        "coordinates": out["coordinates"],
        "values": out["theta"]}, indent=2))


if __name__ == "__main__":
    _main()
