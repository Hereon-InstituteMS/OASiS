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
from .multiphasic import GENERATORS as _mp_gen, KNOWLEDGE as _mp_kn
from .fluid import GENERATORS as _fl_gen, KNOWLEDGE as _fl_kn
from .fluid_fsi import GENERATORS as _fsi_gen, KNOWLEDGE as _fsi_kn
from .rigid_body import GENERATORS as _rb_gen, KNOWLEDGE as _rb_kn
from .viscoelasticity import GENERATORS as _ve_gen, KNOWLEDGE as _ve_kn
from .plasticity import GENERATORS as _pl_gen, KNOWLEDGE as _pl_kn
from .fiber_reinforced import GENERATORS as _fr_gen, KNOWLEDGE as _fr_kn
from .active_contraction import GENERATORS as _ac_gen, KNOWLEDGE as _ac_kn
from .biphasic_fsi import GENERATORS as _bfs_gen, KNOWLEDGE as _bfs_kn
from .polar_fluid import GENERATORS as _pf_gen, KNOWLEDGE as _pf_kn
from .damage import GENERATORS as _dm_gen, KNOWLEDGE as _dm_kn
from .growth_remodeling import GENERATORS as _gr_gen, KNOWLEDGE as _gr_kn


GENERATORS: dict[str, callable] = {}
for _g in (_le_gen, _he_gen, _bi_gen, _ht_gen,
           _mp_gen, _fl_gen, _fsi_gen, _rb_gen,
           _ve_gen, _pl_gen, _fr_gen, _ac_gen,
           _bfs_gen, _pf_gen, _dm_gen, _gr_gen):
    GENERATORS.update(_g)


KNOWLEDGE: dict[str, dict] = {}
for _k in (_le_kn, _he_kn, _bi_kn, _ht_kn,
           _mp_kn, _fl_kn, _fsi_kn, _rb_kn,
           _ve_kn, _pl_kn, _fr_kn, _ac_kn,
           _bfs_kn, _pf_kn, _dm_kn, _gr_kn):
    KNOWLEDGE.update(_k)
