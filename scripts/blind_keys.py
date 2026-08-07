#!/usr/bin/env python3
"""Key custody lifecycle for the blind campaign.

    commit      write the timestamped SHA-256 commitment into the repo
    verify      re-check key files against a committed manifest
    encrypt     AES-256-GCM the key files (passphrase typed, never stored)
    seal        chmod 000 the keys directory
    unseal      restore access for grading
    status      what state are the keys in right now
    preflight   every gate that must pass before a campaign may start

Order of operations for a campaign:

    1. blind_keys.py commit      -- hashes go into git BEFORE anything runs
    2. blind_keys.py encrypt     -- type the passphrase; it touches no disk
    3. blind_keys.py seal        -- keys become unreachable
    4. blind_keys.py preflight   -- refuses if any control is missing
    5. ... run the campaign ...
    6. blind_keys.py unseal      -- then grade with blind_grade.py --with-key

Step 1 comes first on purpose.  The commitment must be made while the keys are
still readable, and committing it before any run is what makes it evidence:
it proves the solutions were fixed in advance rather than derived after the
results were seen.

``verify`` is deliberately dependency-light (hashlib and json only) so a third
party can run it against a key file we hand them, without installing anything.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CAMPAIGN = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
KEYS = CAMPAIGN / "keys"
MANIFEST = REPO / "data" / "blind_key_commitment.json"

# The builders hold every hidden field as a literal, and they sit two levels
# above the agent's working directory (runs/<cell>/work). The agent has bash and
# an absolute-path read_file, so `cat ../../build_extra.py` hands over eight of
# the eleven solutions. Sealing keys/ while leaving these readable is custody
# theatre. DESIGN.md flags this ("the builders holding the solutions in
# plaintext sit above the agent's working directory") without acting on it.
BUILDER_SOURCES = [CAMPAIGN / n for n in
                   ("build_problems.py", "build_extra.py", "build_coupled.py")]

# Markers that identify a file as solution-bearing derivation source.
DERIVATION_MARKERS = ("def problem_", "def coupled_", "diffusion_source",
                      "elasticity_source", "_elastic_body_force",
                      "nonlinear_diffusion_source")


def _vault():
    from blind_eval import keyvault
    return keyvault


# ── verify: standalone, no third-party imports ────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify(args):
    man = json.loads(Path(args.manifest).read_text())
    root = Path(args.keys)
    ok, bad, missing = [], [], []
    pw = None
    if args.decrypt:
        pw = getpass.getpass("key passphrase (never written to disk): ")
    for e in man["entries"]:
        p = root / e["path"]
        # The commitment is over the PLAINTEXT key, because that is what
        # "the solutions were fixed in advance" means. Once the keys are
        # encrypted the plaintext is gone from disk, so re-deriving the hash
        # needs an in-memory decrypt -- which is what --decrypt does.
        if not p.is_file() and args.decrypt:
            enc = root / (e["path"] + ".enc")
            if enc.is_file():
                try:
                    from blind_eval import keyvault
                    plain = keyvault.decrypt_bytes(enc.read_bytes(), pw)
                    (ok if hashlib.sha256(plain).hexdigest() == e["sha256"]
                     else bad).append(e["path"])
                except PermissionError:
                    missing.append(f"{e['path']} (sealed — unseal first)")
                except Exception as exc:
                    bad.append(f"{e['path']} (decrypt failed: {type(exc).__name__})")
                continue
        if not p.is_file():
            missing.append(e["path"])
            continue
        try:
            (ok if _sha256(p) == e["sha256"] else bad).append(e["path"])
        except PermissionError:
            missing.append(f"{e['path']} (unreadable — keys are sealed)")
    recomputed = hashlib.sha256(
        json.dumps(man["entries"], sort_keys=True).encode()).hexdigest()
    self_ok = recomputed == man.get("manifest_sha256")
    print(f"manifest      : {args.manifest}")
    print(f"committed at  : {man.get('generated_utc')}")
    print(f"self-consistent: {self_ok}")
    print(f"matched       : {len(ok)}")
    print(f"MISMATCHED    : {bad or 'none'}")
    print(f"missing       : {missing or 'none'}")
    verdict = "PASS" if self_ok and not bad and not missing else "FAIL"
    print(f"VERDICT       : {verdict}")
    return 0 if verdict == "PASS" else 1


def cmd_commit(args):
    kv = _vault()
    if not KEYS.is_dir():
        print(f"no keys directory at {KEYS}")
        return 1
    man = kv.build_manifest(
        KEYS, campaign="campaign3_blind",
        note="Commitment published before any evaluation run. The solutions "
             "themselves are never published; these hashes prove they were "
             "fixed in advance. Verify with: scripts/blind_keys.py verify "
             "--manifest <this file> --keys <key dir>")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"wrote {MANIFEST}")
    print(f"  {len(man['entries'])} key files, generated {man['generated_utc']}")
    print(f"  manifest_sha256 = {man['manifest_sha256']}")
    print("\nCommit this file to git BEFORE running the campaign.")
    return 0


def cmd_encrypt(args):
    kv = _vault()
    pw = getpass.getpass("key passphrase (never written to disk): ")
    if getpass.getpass("confirm: ") != pw:
        print("passphrases differ")
        return 1
    done = kv.encrypt_tree(KEYS, pw)
    print(f"encrypted {len(done)} key files; plaintext removed")
    for d in done:
        print("  ", d)
    print("\nRe-run 'commit' if you want the manifest to cover the ciphertext.")
    return 0


def cmd_seal(args):
    kv = _vault()
    mode = kv.seal(KEYS)
    print(f"sealed {KEYS} -> {mode}")
    for src in BUILDER_SOURCES:
        if src.is_file():
            os.chmod(src, 0o000)
            print(f"sealed {src.name} -> ---------- "
                  f"(holds hidden fields as literals)")
    v = kv.verify_unreadable(KEYS)
    print(json.dumps(v, indent=2))
    return 0 if v.get("sealed") else 1


def cmd_unseal(args):
    kv = _vault()
    print(f"unsealed {KEYS} -> {kv.unseal(KEYS)}")
    for src in BUILDER_SOURCES:
        if src.exists():
            os.chmod(src, 0o600)
            print(f"unsealed {src.name}")
    return 0


def cmd_exposure(args):
    """Is any CURRENT solution recoverable from anything the agent can read?

    Sealing the key directory proves nothing on its own: the question is
    whether the answers survive anywhere in the tree the agent can reach.  Two
    rules, because a literal search alone is not enough — a builder writes
    ``x * (1 - x) * y * (1 - y) * sp.cos(...)`` while the key stores
    ``x*y*(1-x)*(1-y)*cos(2*pi*x)``, so string matching misses it entirely.

      STRUCTURAL  a readable Python file that imports sympy and carries
                  derivation markers is solution-bearing whatever its formatting
      LITERAL     a readable file containing a key's exact-solution string,
                  whitespace-normalised
    """
    kv = _vault()
    root = Path(args.root) if args.root else CAMPAIGN
    exact_strings = []
    if not kv.is_sealed(KEYS):
        enc = any(KEYS.rglob("*.enc"))
        # run_blind.py's preflight runs this sweep as a subprocess with no TTY.
        # Prompting there raised EOFError, the subprocess exited non-zero, and
        # the preflight refused EVERY campaign -- fail-closed, but on a bug
        # rather than a finding, which is how a control gets switched off. Take
        # the passphrase from the environment if the operator exported it, and
        # otherwise run the STRUCTURAL rule alone and say so.
        pw = os.environ.get("OASIS_KEY_PASSPHRASE")
        if enc and pw is None:
            if sys.stdin is not None and sys.stdin.isatty():
                pw = getpass.getpass("key passphrase (to know what to look "
                                     "for; blank to skip the literal rule): ")
                pw = pw or None
            else:
                print("no TTY and no OASIS_KEY_PASSPHRASE: encrypted keys "
                      "cannot be opened, so the LITERAL rule is disabled. The "
                      "STRUCTURAL rule still applies. Coverage is PARTIAL.")
        for kp in sorted(list(KEYS.rglob("key.json.enc")) + list(KEYS.rglob("key.json"))):
            try:
                k = kv.load_key(kp, pw)
            except Exception:
                continue
            ex = k.get("exact_solution")
            for g in (ex.values() if isinstance(ex, dict) else [ex]):
                for c in (g if isinstance(g, list) else [g]):
                    if isinstance(c, str):
                        exact_strings.append(c)
    else:
        print("keys sealed — literal rule disabled; structural rule still applies")

    def norm(s):
        return "".join(s.split())

    findings = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or KEYS in p.parents or p == KEYS:
            continue
        try:
            if p.stat().st_size > 8_000_000:
                continue
            text = p.read_text(errors="ignore")
        except (PermissionError, OSError):
            continue                      # unreadable is the desired state
        if p.suffix == ".py" and "sympy" in text and any(
                m in text for m in DERIVATION_MARKERS):
            findings.append((p, "STRUCTURAL", "sympy derivation source: "
                                              "holds hidden fields as literals"))
            continue
        flat = norm(text)
        for e in exact_strings:
            if len(norm(e)) > 10 and norm(e) in flat:
                findings.append((p, "LITERAL", f"contains {e[:60]}"))
                break

    for p, rule, why in findings:
        print(f"  [{rule}] {p}\n           {why}")
    print(f"\n{len(findings)} agent-readable file(s) expose a solution "
          f"under {root}")
    print("VERDICT:", "FAIL" if findings else "PASS — nothing reachable leaks")
    return 1 if findings else 0


def cmd_status(args):
    kv = _vault()
    import stat as _stat
    exists = KEYS.exists()
    print(f"keys dir      : {KEYS}")
    print(f"exists        : {exists}")
    if exists:
        print(f"mode          : {_stat.filemode(KEYS.stat().st_mode)}")
    print(f"sealed        : {kv.is_sealed(KEYS)}")
    if not kv.is_sealed(KEYS) and exists:
        enc = list(KEYS.rglob("*.enc"))
        plain = [p for p in KEYS.rglob("*.json") if not p.name.endswith(".enc")]
        print(f"encrypted keys: {len(enc)}")
        print(f"PLAINTEXT keys: {len(plain)}"
              + ("  <-- readable answers on disk" if plain else ""))
    print(f"manifest      : {MANIFEST} "
          f"({'present' if MANIFEST.is_file() else 'MISSING'})")
    return 0


def cmd_audit(args):
    """Sweep every problem through the leak gate, decrypting in memory if needed.

    Works whatever state the keys are in (except sealed), so the audit does not
    have to be sequenced before encryption and cannot quietly become a no-op.
    Coverage is reported, because a sweep that examined nothing must not read
    as a pass.
    """
    kv = _vault()
    from blind_eval.leakgate import scan
    if kv.is_sealed(KEYS):
        print("keys are sealed; unseal first (scripts/blind_keys.py unseal)")
        return 2
    problems = sorted(p for p in (CAMPAIGN / "problems").iterdir()
                      if (p / "task.txt").is_file())
    encrypted = any((KEYS / p.name / "key.json.enc").is_file() for p in problems)
    pw = getpass.getpass("key passphrase: ") if encrypted else None

    checked, leaking = 0, []
    for pdir in problems:
        kdir = KEYS / pdir.name
        kpath = next((q for q in (kdir / "key.json.enc", kdir / "key.json")
                      if q.is_file()), None)
        if kpath is None:
            print(f"  {pdir.name:4} NO KEY — not audited")
            continue
        key = kv.load_key(kpath, pw)
        rep = scan((pdir / "task.txt").read_text(), key, pdir.name)
        checked += 1
        worst = [f for f in rep.findings
                 if f.severity in ("CRITICAL", "HIGH", "MEDIUM")]
        print(f"  {pdir.name:4} {'LEAK' if worst else 'clean':5} "
              f"{','.join(f.rule for f in worst) or ''}")
        for f in worst:
            print(f"        [{f.severity}] {f.detail[:160]}")
        if worst:
            leaking.append(pdir.name)
    print(f"\naudited {checked}/{len(problems)} problems")
    if checked != len(problems):
        print("INCOMPLETE — unaudited problems are not known to be blind")
        return 1
    print(f"VERDICT: {'FAIL — ' + ', '.join(leaking) if leaking else 'PASS — all blind'}")
    return 1 if leaking else 0


def cmd_preflight(args):
    """Every control that must hold before a campaign may start."""
    kv = _vault()
    checks = {}
    checks["manifest_committed"] = MANIFEST.is_file()
    if checks["manifest_committed"]:
        man = json.loads(MANIFEST.read_text())
        checks["manifest_self_consistent"] = (
            hashlib.sha256(json.dumps(man["entries"], sort_keys=True).encode())
            .hexdigest() == man.get("manifest_sha256"))
    else:
        checks["manifest_self_consistent"] = False
    plain = ([p for p in KEYS.rglob("*.json") if not p.name.endswith(".enc")]
             if KEYS.is_dir() else [])
    checks["no_plaintext_keys"] = not plain
    checks["keys_sealed"] = kv.is_sealed(KEYS)
    checks["builder_sources_sealed"] = all(
        (not src.exists()) or not os.access(src, os.R_OK)
        for src in BUILDER_SOURCES)
    if checks["keys_sealed"]:
        checks["seal_verified_by_execution"] = bool(
            kv.verify_unreadable(KEYS).get("sealed"))
    else:
        checks["seal_verified_by_execution"] = False

    width = max(len(k) for k in checks)
    for k, v in checks.items():
        print(f"  {k:<{width}}  {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nPREFLIGHT: {'PASS — campaign may start' if ok else 'FAIL — do not run'}")
    if plain:
        print(f"  {len(plain)} plaintext key file(s) still on disk")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("commit").set_defaults(fn=cmd_commit)
    sub.add_parser("encrypt").set_defaults(fn=cmd_encrypt)
    sub.add_parser("seal").set_defaults(fn=cmd_seal)
    sub.add_parser("unseal").set_defaults(fn=cmd_unseal)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    sub.add_parser("audit").set_defaults(fn=cmd_audit)
    ex = sub.add_parser("exposure")
    ex.add_argument("--root", default=None,
                    help="tree to scan (default: the campaign directory)")
    ex.set_defaults(fn=cmd_exposure)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", default=str(MANIFEST))
    v.add_argument("--keys", default=str(KEYS))
    v.add_argument("--decrypt", action="store_true",
                   help="keys are encrypted: decrypt in memory to re-derive the "
                        "committed plaintext hashes (prompts for the passphrase)")
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
