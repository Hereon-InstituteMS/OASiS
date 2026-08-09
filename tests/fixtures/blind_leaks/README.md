# Frozen real leaks — regression fixtures for the leak gate

These are the **original** campaign-3 `task.txt` / `key.json` pairs for the three
instances that disclosed their solution through the published source term, as
they stood before Amendment 1 rebuilt them (see `campaign3_blind/DESIGN.md`).

They are kept because the proof that the gate works must not disappear when the
defect it found is fixed. Testing the gate against the live campaign was
self-defeating: the moment B1, B2 and D3 were rebuilt, the tests that proved the
gate fires on a real leak had nothing left to fire on, and quietly errored.

| file | leak | severity |
|---|---|---|
| `B1_*` | one PRINTED source term is `12*pi**2 * u_exact` | naked-eye |
| `D3_*` | two/three printed terms sum to `pi**2 * u_A`, `4*pi**2 * u_B` | naked-eye |
| `B2_*` | `4*pi**2 * u_exact` survives expansion only | one `expand()` |

`DESIGN.md` admitted B1 and D3; B2 was found by the gate.

The `exact_solution` values in here are **no longer the solution to anything**:
the live B1, B2 and D3 use different manufactured fields. Nothing is disclosed
by committing them, and freezing them is what makes the gate's proof permanent.
