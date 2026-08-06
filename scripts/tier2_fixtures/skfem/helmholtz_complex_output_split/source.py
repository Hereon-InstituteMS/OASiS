"""Tier-2: complex Helmholtz output must be split into .real and np.abs before writing.

Claim: skfem helmholtz#6 — output the REAL PART and the MAGNITUDE |u| for
visualisation, because the solution is complex; writing ``u.tofile(...)``
without the ``.real`` / ``np.abs`` split writes complex128, which ParaView
writers "either reject (`Cannot write complex array`) or truncate to the real
part silently, discarding amplitude information".

Wrong variant: the raw complex nodal vector is handed to (a) ``ndarray.tofile``
and (b) the meshio / ``Mesh.save`` VTU + XDMF writers, and the consequences are
recorded.  Then the documented fix (``{"u_real": u.real, "u_abs": np.abs(u)}``)
is written and round-tripped to show both fields survive and that the real part
alone does NOT carry the amplitude.

Observed on skfem 12.0.1 / meshio 5.3.5 / numpy 1.26.4 (2026-08-06) — the
pathology is real but the claim's quoted texts are BOTH wrong:
  * meshio does not say "Cannot write complex array"; it raises
    ``KeyError: dtype('complex128')`` for .vtu (and ``KeyError: 'complex128'``
    for .xdmf), and ``skfem.Mesh.save`` propagates the same KeyError;
  * it does not truncate silently either — no file is produced at all.
  * ``tofile`` is the silent half: it writes double the bytes, and a
    real-valued reader gets a doubled array with real and imaginary parts
    INTERLEAVED, not the real part.

Mutation control: with T2_MUTATE=1 the payload handed to ``tofile`` and to the
meshio / ``Mesh.save`` writers becomes the real-valued magnitude ``np.abs(u)``
instead of the raw complex vector — the documented ``.real`` / ``np.abs`` split
applied at the pathology site.  Nothing is doubled and nothing is rejected any
more, so 'payload_dtype=complex128', 'tofile_bytes_is_double=True',
'float64_readback_len_is_double=True', 'float64_readback_is_interleaved=True',
'vtu_writer_error_type=KeyError', "dtype('complex128')",
'skfem_save_error_type=KeyError', 'writers_reject_complex128=True',
'vtu_file_written_anyway=False' and 'silently_truncated_to_real_part=False' all
disappear and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import meshio
import numpy as np
from skfem import Basis, ElementTriP1, MeshTri

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The mistake: hand the raw complex array to the writers instead of splitting
# it into a real part and a magnitude.
WRITE_PAYLOAD = "complex" if not MUTATE else "abs"


def payload(u: np.ndarray) -> np.ndarray:
    return u if WRITE_PAYLOAD == "complex" else np.abs(u)


def main() -> int:
    ok = True
    tmp = tempfile.mkdtemp(prefix="skfem_helmholtz_out_")
    try:
        m = MeshTri().refined(2)
        b = Basis(m, ElementTriP1())
        # a complex "Helmholtz solution": amplitude 2, phase 3*x
        u = 2.0 * np.exp(-3j * m.p[0])
        u = u.astype(complex)
        print(f"mesh_class={type(m).__name__} n_dofs={b.N}")
        print(f"solution_dtype={u.dtype}")

        data = payload(u)
        print(f"payload_dtype={data.dtype}")

        # --- WRONG variant (a): tofile, then read as a real-valued consumer --
        raw = os.path.join(tmp, "u.bin")
        data.tofile(raw)
        n_bytes = os.path.getsize(raw)
        print(f"tofile_bytes={n_bytes} real_only_bytes={u.real.nbytes}")
        print(f"tofile_bytes_is_double={n_bytes == 2 * u.real.nbytes}")
        back = np.fromfile(raw, dtype=np.float64)
        print(f"float64_readback_len={back.size} n_nodes={u.size}")
        doubled = back.size == 2 * u.size
        interleaved = (doubled
                       and bool(np.allclose(back[0::2], u.real))
                       and bool(np.allclose(back[1::2], u.imag)))
        equals_real = back.size == u.size and bool(np.allclose(back, u.real))
        print(f"float64_readback_len_is_double={doubled}")
        print(f"float64_readback_is_interleaved={interleaved}")
        print(f"float64_readback_equals_real_part={equals_real}")
        if not (doubled and interleaved):
            print("FAIL: tofile did not produce a doubled interleaved buffer, "
                  "so the silent half of the pitfall did not occur",
                  file=sys.stderr)
            ok = False

        # --- WRONG variant (b): hand complex point_data to the writers ------
        points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
        cells = [("triangle", m.t.T)]
        results = {}
        for suffix in (".vtu", ".xdmf"):
            path = os.path.join(tmp, "out" + suffix)
            try:
                meshio.Mesh(points, cells,
                            point_data={"u": data}).write(path)
                results[suffix] = ("", os.path.exists(path))
            except Exception as exc:             # noqa: BLE001 - want the type
                results[suffix] = (f"{type(exc).__name__}: {exc}",
                                   os.path.exists(path))
        vtu_msg, vtu_file = results[".vtu"]
        xdmf_msg, xdmf_file = results[".xdmf"]
        print(f"vtu_writer_error={vtu_msg}")
        print(f"xdmf_writer_error={xdmf_msg}")
        print(f"vtu_writer_error_type={vtu_msg.split(':')[0]}")

        sk_path = os.path.join(tmp, "sk.vtu")
        try:
            m.save(sk_path, point_data={"u": data})
            sk_msg = ""
        except Exception as exc:                 # noqa: BLE001 - want the type
            sk_msg = f"{type(exc).__name__}: {exc}"
        print(f"skfem_save_error_type={sk_msg.split(':')[0]}")

        rejected = bool(vtu_msg) and bool(xdmf_msg) and bool(sk_msg)
        print(f"writers_reject_complex128={rejected}")
        print(f"vtu_file_written_anyway={vtu_file}")
        if not rejected:
            print("FAIL: a writer accepted the complex array, so the pitfall "
                  "did not occur", file=sys.stderr)
            ok = False

        # the claim's two quoted behaviours, checked verbatim
        claimed = "cannot write complex array"
        print("claimed_cannot_write_complex_array_text_present="
              f"{claimed in (vtu_msg + xdmf_msg + sk_msg).lower()}")
        print(f"silently_truncated_to_real_part={vtu_file}")

        # --- RIGHT variant: split into real part and magnitude --------------
        good = os.path.join(tmp, "good.vtu")
        meshio.Mesh(points, cells,
                    point_data={"u_real": u.real,
                                "u_abs": np.abs(u)}).write(good)
        rd = meshio.read(good)
        got_real = np.asarray(rd.point_data["u_real"]).ravel()
        got_abs = np.asarray(rd.point_data["u_abs"]).ravel()
        split_ok = (bool(np.allclose(got_real, u.real))
                    and bool(np.allclose(got_abs, np.abs(u))))
        print(f"split_write_roundtrip_ok={split_ok}")
        print(f"split_write_dtype={got_abs.dtype}")
        if not split_ok:
            print("FAIL: the documented .real / np.abs split did not round "
                  "trip", file=sys.stderr)
            ok = False

        # amplitude really is lost if only the real part is kept: |u| is
        # constant here while the real part swings through zero.
        amp_lost = not np.allclose(np.abs(u), np.abs(u.real))
        print(f"real_part_alone_loses_amplitude={amp_lost}")
        print(f"abs_is_constant={float(np.ptp(np.abs(u))) < 1e-12}")
        print(f"real_part_ptp_gt_1={float(np.ptp(u.real)) > 1.0}")
        if not amp_lost:
            print("FAIL: the real part carried the full amplitude, so the "
                  "split would be pointless here", file=sys.stderr)
            ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
