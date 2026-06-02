"""FEBio generator registry — maps physics_variant -> generator function.

Mirrors the layout used by skfem, fenics, ngsolve etc.: one file per
FEBio physics module exposes a per-module ``GENERATORS`` + ``KNOWLEDGE``
dict, and this aggregator merges them into the top-level ``GENERATORS``
and ``KNOWLEDGE`` dicts that ``backend.py`` consumes.

When adding a new FEBio physics module (multiphasic, fluid, fluid-FSI,
etc.), drop a new file alongside this one and add it to the imports and
the merge loops below.
"""
from .linear_elasticity import GENERATORS as _le_gen, KNOWLEDGE as _le_kn
from .hyperelasticity import GENERATORS as _he_gen, KNOWLEDGE as _he_kn
from .biphasic import GENERATORS as _bi_gen, KNOWLEDGE as _bi_kn
from .heat import GENERATORS as _ht_gen, KNOWLEDGE as _ht_kn


GENERATORS: dict[str, callable] = {}
for _g in (_le_gen, _he_gen, _bi_gen, _ht_gen):
    GENERATORS.update(_g)


KNOWLEDGE: dict[str, dict] = {}
for _k in (_le_kn, _he_kn, _bi_kn, _ht_kn):
    KNOWLEDGE.update(_k)
