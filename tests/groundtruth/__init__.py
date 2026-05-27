"""Source-code ground-truth probes for catalog-consistency tests.

Each module here fetches a small slice of a solver's source code (either from
a local checkout pointed to by ``<SOLVER>_ROOT`` or from raw.githubusercontent.com)
and extracts the canonical names that the solver's input parser actually
accepts.  The test suite in ``tests/test_catalog_consistency.py`` then
cross-checks those against what the MCP knowledge base claims.

The catalog can drift in two directions:
  * the catalog promises a keyword the solver no longer accepts (BREAKING)
  * the solver accepts a keyword the catalog never advertised (DEAD ENTRY -- agent never finds it)

This package surfaces both.
"""
