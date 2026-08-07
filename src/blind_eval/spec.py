"""The blind problem specification — the *only* container a prompt is built from.

Structural blindness lives here.  ``BlindSpec`` is a frozen dataclass with a
**closed** field set.  There is no field that could hold an exact solution, so
``u_exact`` cannot travel into a prompt even if a future contributor tries to
put it there: :func:`BlindSpec.from_payload` copies *only* the declared fields
and raises on anything it does not recognise.

Threat model
------------
A well-posed boundary-value problem determines its own solution — that is what
"well-posed" means.  Blindness therefore cannot mean "the prompt does not imply
the answer".  It means:

    **the answer is not obtainable in closed form from the prompt text.**

So the harness forbids (a) any closed-form expression for the solution field,
(b) any phrasing that names or requests an exact/analytic/manufactured
solution, and (c) any numeric value equal to a true probe value.  It
deliberately does *not* forbid trigonometric or exponential functions in the
source term: the source term is derived from the hidden solution and will
almost always contain them.  That distinction is enforced symbolically in
:mod:`blind_eval.leakgate`, never by pattern-matching function names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


# Field names that must never appear in a blind payload.  This list is a
# tripwire for developer error, not the enforcement mechanism -- enforcement is
# the closed field set of BlindSpec itself.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "u_exact", "exact", "exact_solution", "manufactured", "manufactured_solution",
    "solution", "reference", "reference_values", "answer", "answers",
    "probe_reference", "probe_values", "u_exact_components", "p_exact",
    "key", "keys", "secret",
})


@dataclass(frozen=True)
class Probe:
    """A point (or functional) at which the agent must report a value."""

    name: str
    kind: str                      # "point" | "functional"
    quantity: str                  # e.g. "u", "u_y", "sigma_xx", "mean(u)"
    location: tuple | None = None  # coordinates for kind == "point"
    detail: str = ""               # human-readable definition for functionals

    def render(self) -> str:
        if self.kind == "point":
            loc = ", ".join(f"{c:g}" for c in (self.location or ()))
            return f"`{self.name}` = {self.quantity} at ({loc})"
        return f"`{self.name}` = {self.quantity}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class BlindSpec:
    """Everything the agent is allowed to see.  Nothing else exists here.

    Adding a field to this class is the *only* way to widen what a prompt can
    contain, which makes widening a reviewable, deliberate act.
    """

    task_id: str
    title: str
    domain: str                       # prose description of the geometry
    pde: str                          # the PDE, written out
    coefficients: dict                # name -> value/description
    source_term: dict                 # component name -> expression string
    boundary_conditions: list         # list of prose/numeric BC statements
    mesh_sequence: list               # refinement levels, coarse -> fine
    probes: list                      # list[Probe]
    result_keys: list                 # exact RESULT line names the agent must emit
    backends: list = field(default_factory=list)
    initial_conditions: list = field(default_factory=list)
    time_grid: dict = field(default_factory=dict)
    discretisation: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def field_names(cls) -> frozenset:
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BlindSpec":
        """Build from a JSON payload, admitting only declared fields.

        Unknown keys are a hard error rather than a silent drop: a silent drop
        would let a leak sit unnoticed in the build artefact, and the build
        artefact is what auditors read.
        """
        allowed = cls.field_names()
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "blind payload carries fields BlindSpec cannot represent: "
                f"{sorted(unknown)}.  If one of these is solution data the "
                "derivation step has a bug; if it is genuinely public, add it "
                "to BlindSpec deliberately."
            )
        forbidden = set(payload) & FORBIDDEN_PAYLOAD_KEYS
        if forbidden:
            raise ValueError(f"forbidden key(s) in blind payload: {sorted(forbidden)}")
        probes = [p if isinstance(p, Probe) else Probe(**p)
                  for p in payload.get("probes", [])]
        data = dict(payload)
        data["probes"] = probes
        return cls(**data)

    def to_payload(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "probes":
                v = [{"name": p.name, "kind": p.kind, "quantity": p.quantity,
                      "location": list(p.location) if p.location else None,
                      "detail": p.detail} for p in v]
            out[f.name] = v
        return out
