"""Make the discrete-residual check reachable from a run.

`core/residual_check.py` can tell a solve from a forgery — it assembles the
stated operator on the submitted mesh and measures how far the submitted field
is from satisfying it, which separates the two by many orders of magnitude. It
was unreachable: no tool took a problem statement, so nothing could ever call
it, and the verification gate's strongest check sat in the repository doing
nothing.

This module is the bridge. A run may declare the PDE it claims to solve, in the
same terms the problem statement is already given in — an operator family, a
source term, a coefficient, the domain's measure. None of that is an answer:
the source term is what the engineer is handed, not what they compute. That
distinction is the whole reason this check is admissible in a gate that must
never see an exact solution. A residual needs f, not u.

WHAT A DECLARATION LOOKS LIKE

    {"operator": "diffusion",
     "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
     "coefficient": "1.0",
     "dim": 2,
     "domain_measure": 1.0}

`source` and `coefficient` are expressions in x, y, z evaluated with numpy and
nothing else — no builtins, no imports, no attribute access. They are numeric
expressions, not a scripting surface.

WHAT IT CANNOT DO
Only the operator families residual_check implements are supported (scalar
diffusion on simplices, P1, Dirichlet data on the outer boundary). Anything else
returns UNSUPPORTED, never a pass — "not checked" must never be reachable as
"checked and fine", because a gate that silently approves what it cannot examine
is worse than one that admits the gap.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from .residual_check import check_residual

# Everything an expression may name. Deliberately small: these are the functions
# a source term is written with, and nothing here can reach the filesystem, the
# interpreter, or any object's attributes.
_SAFE_NAMES = {
    name: getattr(np, name) for name in
    ("sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
     "exp", "log", "log10", "sqrt", "abs", "sign", "minimum", "maximum",
     "power", "floor", "ceil")
}
_SAFE_NAMES.update(pi=np.pi, e=np.e)

SUPPORTED_OPERATORS = ("diffusion",)

# Nodes only appear in a legitimate numeric expression. Attribute access,
# subscripting, comprehensions, lambdas, calls to anything not in _SAFE_NAMES,
# and every statement form are absent by construction.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.FloorDiv, ast.Tuple,
)


class ResidualSpecError(ValueError):
    """The declaration itself is malformed — refused before anything runs."""


def _compile_expression(text: str, label: str):
    """Compile a numeric expression, refusing anything that is not one."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ResidualSpecError(f"{label}: not a valid expression ({exc.msg})")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ResidualSpecError(
                f"{label}: {type(node).__name__} is not allowed here; a source "
                f"term is a numeric expression in x, y, z, not a program")
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id not in _SAFE_NAMES:
                got = getattr(fn, "id", type(fn).__name__)
                raise ResidualSpecError(
                    f"{label}: '{got}' is not an available function; use one of "
                    + ", ".join(sorted(_SAFE_NAMES)))
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES \
                and node.id not in ("x", "y", "z"):
            raise ResidualSpecError(
                f"{label}: unknown name '{node.id}'; expressions may use the "
                f"coordinates x, y, z and the standard functions")
    return compile(tree, f"<{label}>", "eval")


def _evaluator(code, dim: int):
    """Turn a compiled expression into f(X) over coordinates of shape (dim, n)."""
    def evaluate(coords):
        env = dict(_SAFE_NAMES)
        env["x"] = coords[0]
        env["y"] = coords[1] if len(coords) > 1 else np.zeros_like(coords[0])
        env["z"] = coords[2] if len(coords) > 2 else np.zeros_like(coords[0])
        value = eval(code, {"__builtins__": {}}, env)   # noqa: S307
        # A constant expression evaluates to a scalar; the forms need an array.
        return value * np.ones_like(coords[0]) if np.isscalar(value) else value
    return evaluate


def parse_spec(spec: str | dict) -> dict:
    """Validate a residual declaration, raising ResidualSpecError if malformed."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ResidualSpecError(f"not valid JSON: {exc}")
    if not isinstance(spec, dict):
        raise ResidualSpecError("expected a JSON object describing the problem")

    operator = str(spec.get("operator", "diffusion")).lower()
    if operator not in SUPPORTED_OPERATORS:
        raise ResidualSpecError(
            f"operator '{operator}' is not one OASiS can assemble; supported: "
            + ", ".join(SUPPORTED_OPERATORS))
    if "source" not in spec:
        raise ResidualSpecError(
            "a residual needs the problem's source term; give `source` as an "
            "expression in x, y, z (use \"0\" for a source-free problem)")

    dim = int(spec.get("dim", 2))
    if dim not in (2, 3):
        raise ResidualSpecError(f"dim must be 2 or 3, got {dim}")

    measure = spec.get("domain_measure")
    return {
        "operator": operator,
        "dim": dim,
        "source_code": _compile_expression(str(spec["source"]), "source"),
        "coefficient_code": (
            _compile_expression(str(spec["coefficient"]), "coefficient")
            if spec.get("coefficient") not in (None, "", "1", "1.0") else None),
        "domain_measure": float(measure) if measure is not None else None,
        "field": str(spec.get("field", "")),
        "result_file": spec.get("result_file"),
    }


def check_run_residual(spec: str | dict, result_files) -> dict:
    """Measure whether the run's field satisfies the problem it declared.

    Returns a dict the verification gate can act on. `verdict` is one of:
      SOLVES        — the field satisfies the discrete system to solver tolerance
      DOES_NOT_SOLVE— it does not; this field was not obtained by solving this
                      problem on this mesh
      UNSUPPORTED   — OASiS cannot assemble this problem, so nothing was checked
      REFUSED       — the declaration or the artefacts were unusable

    Never raises: a gate that dies on a malformed declaration is a gate an agent
    can switch off by malforming one.
    """
    from .fabrication_gate import select_artefact
    from .mesh_independence import read_nodal_mesh

    try:
        parsed = parse_spec(spec)
    except ResidualSpecError as exc:
        return {"verdict": "REFUSED", "detail": str(exc)}

    candidates = [Path(p) for p in (result_files or [])]
    explicit = Path(parsed["result_file"]) if parsed["result_file"] else None
    chosen, why = select_artefact(candidates, explicit)
    if chosen is None:
        return {"verdict": "REFUSED",
                "detail": f"no single result artefact to check: {why}"}

    try:
        points, cells, fields = read_nodal_mesh(chosen)
    except Exception as exc:
        return {"verdict": "REFUSED",
                "detail": f"{chosen.name} could not be read: {exc}"}
    if not fields:
        return {"verdict": "REFUSED",
                "detail": f"{chosen.name} carries no nodal field"}

    wanted = parsed["field"]
    if wanted and wanted not in fields:
        return {"verdict": "REFUSED",
                "detail": (f"field '{wanted}' is not in {chosen.name}; OASiS "
                           f"does not substitute another")}
    name = wanted or next(iter(fields))
    values = np.asarray(fields[name], float)
    if values.ndim > 1:
        return {"verdict": "UNSUPPORTED",
                "detail": (f"'{name}' is a vector field; the residual check "
                           f"covers scalar diffusion only")}

    dim = parsed["dim"]
    verdict = check_residual(
        points, cells, values, dim=dim,
        source_fn=_evaluator(parsed["source_code"], dim),
        coeff_fn=(_evaluator(parsed["coefficient_code"], dim)
                  if parsed["coefficient_code"] is not None else None),
        domain_measure_expected=parsed["domain_measure"])

    if not verdict.supported:
        return {"verdict": "UNSUPPORTED", "detail": verdict.detail,
                "field": name, "artefact": chosen.name}
    return {
        "verdict": "SOLVES" if verdict.solver_like else "DOES_NOT_SOLVE",
        "relative_residual": verdict.relative_residual,
        "n_interior_dofs": verdict.n_interior,
        "domain_measure": verdict.domain_measure,
        "field": name,
        "artefact": chosen.name,
        "detail": verdict.detail,
    }
