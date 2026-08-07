"""Key custody: encryption, filesystem sealing, and a public hash commitment.

Three independent controls, because each one alone has a hole the others cover.

1. **Not reachable** (primary).  ``keys/`` is ``chmod 000`` for the duration of
   a run.  This is what actually stops a runtime agent, which reads the disk as
   the same user we do.  It is a real control against an agent that wanders,
   and *not* a security boundary against one that decides to attack: the owner
   of a directory can chmod it back.  Stated plainly rather than dressed up.

2. **Not readable** (defence in depth).  Key files are AES-256-GCM encrypted
   under a passphrase that is on **no disk** while runs happen — not in a
   script, not in an env file, not in a config.  It is typed interactively at
   grading time, which is a separate, later, manual step.  So even a process
   that defeats control 1 finds ciphertext.

3. **Not changeable after the fact** (integrity).  A timestamped SHA-256
   manifest of every key file is committed to the repo *before* the campaign
   runs.  The solutions stay private; the hashes prove they were fixed in
   advance.  This answers the question a hostile reviewer will certainly ask —
   "how do we know you did not derive the solutions after seeing the results?"
   — checkably rather than on trust.

Threat model
------------
Keys are never published.  The exposure is purely local: stop a runtime agent
from reading answers off disk.  There is no published ciphertext facing offline
brute force, which is why a short passphrase is acceptable here and would not
be if the artefact were public.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"OASISKEY1"
SALT_BYTES, NONCE_BYTES = 16, 12
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
ENC_SUFFIX = ".enc"


# ── encryption ────────────────────────────────────────────────────────
def _derive(passphrase: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R,
                  p=SCRYPT_P).derive(passphrase.encode("utf-8"))


def encrypt_bytes(plaintext: bytes, passphrase: str) -> bytes:
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    ct = AESGCM(_derive(passphrase, salt)).encrypt(nonce, plaintext, MAGIC)
    return MAGIC + salt + nonce + ct


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    """Decrypt in memory.  Callers must not write the result to the run tree."""
    if not blob.startswith(MAGIC):
        raise ValueError("not an OASiS sealed key (bad magic)")
    off = len(MAGIC)
    salt = blob[off:off + SALT_BYTES]
    nonce = blob[off + SALT_BYTES:off + SALT_BYTES + NONCE_BYTES]
    ct = blob[off + SALT_BYTES + NONCE_BYTES:]
    return AESGCM(_derive(passphrase, salt)).decrypt(nonce, ct, MAGIC)


def load_key(path: Path, passphrase: str) -> dict:
    """Read a key, encrypted or not, and return the parsed JSON. In memory only."""
    path = Path(path)
    if path.suffix == ENC_SUFFIX or path.name.endswith(ENC_SUFFIX):
        return json.loads(decrypt_bytes(path.read_bytes(), passphrase))
    return json.loads(path.read_text())


def encrypt_tree(keys_dir: Path, passphrase: str, remove_plaintext: bool = True) -> list:
    """Encrypt every ``*.json`` under ``keys_dir`` to ``*.json.enc``."""
    keys_dir = Path(keys_dir)
    done = []
    for p in sorted(keys_dir.rglob("*.json")):
        blob = encrypt_bytes(p.read_bytes(), passphrase)
        enc = p.with_suffix(p.suffix + ENC_SUFFIX)
        enc.write_bytes(blob)
        # round-trip before destroying the only copy
        if decrypt_bytes(enc.read_bytes(), passphrase) != p.read_bytes():
            enc.unlink()
            raise RuntimeError(f"round-trip failed for {p}; plaintext kept")
        if remove_plaintext:
            p.unlink()
        done.append(str(enc.relative_to(keys_dir)))
    return done


# ── filesystem sealing ────────────────────────────────────────────────
def seal(keys_dir: Path) -> str:
    keys_dir = Path(keys_dir)
    for p in sorted(keys_dir.rglob("*"), reverse=True):
        try:
            os.chmod(p, 0o000)
        except OSError:
            pass
    os.chmod(keys_dir, 0o000)
    return stat.filemode(keys_dir.stat().st_mode)


def unseal(keys_dir: Path, mode: int = 0o700) -> str:
    """Restore access, top-down.

    A sealed directory cannot be descended into, so anything that globs first
    and chmods second (``rglob``, ``os.walk`` over a 000 tree) silently restores
    nothing below the top level and leaves the keys unreadable to the grader.
    Each directory must be opened up *before* its children are visited.
    """
    keys_dir = Path(keys_dir)
    os.chmod(keys_dir, mode)
    stack = [keys_dir]
    while stack:
        d = stack.pop()
        for entry in os.scandir(d):
            p = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                os.chmod(p, 0o700)
                stack.append(p)
            else:
                os.chmod(p, 0o600)
    return stat.filemode(keys_dir.stat().st_mode)


def is_sealed(keys_dir: Path) -> bool:
    """True only if the directory exists AND cannot be listed.

    The pre-existing ``keys_are_sealed()`` returned True when the directory was
    missing or empty, so a deleted keys tree read as "sealed" and the campaign
    would have started with nothing to grade against.  Absence is not a seal.
    """
    keys_dir = Path(keys_dir)
    if not keys_dir.exists():
        return False
    try:
        next(iter(os.scandir(keys_dir)))
        return False
    except PermissionError:
        return True
    except StopIteration:
        return False          # empty directory is not a sealed directory


def verify_unreadable(keys_dir: Path, timeout: int = 30) -> dict:
    """Prove, by execution, that a separate process cannot read the keys.

    Runs the checks in a child process with the same privileges the agent has,
    because a claim about permissions that has not been executed is a guess.
    """
    keys_dir = Path(keys_dir).resolve()
    probe = f"""
import os, glob, json, sys
d = {str(keys_dir)!r}
res = {{}}
try:
    os.listdir(d); res["listdir"] = "SUCCEEDED (not sealed)"
except PermissionError as e:
    res["listdir"] = f"PermissionError: {{e.strerror}}"
except Exception as e:
    res["listdir"] = f"{{type(e).__name__}}: {{e}}"
hits = []
for pat in ("*/key.json", "*/key.json.enc", "*/*.json", "*/*.enc"):
    try:
        hits += glob.glob(os.path.join(d, pat))
    except Exception:
        pass
res["glob_hits"] = len(hits)
opened = []
for p in hits[:20]:
    try:
        with open(p, "rb") as fh:
            fh.read(64)
        opened.append(p)
    except Exception:
        pass
res["files_opened"] = len(opened)
print(json.dumps(res))
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, timeout=timeout)
    try:
        child = json.loads(r.stdout.strip() or "{}")
    except json.JSONDecodeError:
        child = {"raw": r.stdout, "stderr": r.stderr}
    # a shell attempt too: the agent has bash
    sh = subprocess.run(["bash", "-lc", f"cat {keys_dir}/*/key.json* 2>&1 | head -c 200"],
                        capture_output=True, text=True, timeout=timeout)
    child["shell_cat"] = (sh.stdout or sh.stderr).strip()[:200]
    child["sealed"] = (child.get("files_opened") == 0
                       and child.get("glob_hits") == 0
                       and "PermissionError" in str(child.get("listdir", "")))
    return child


# ── hash commitment ───────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(keys_dir: Path, campaign: str, note: str = "") -> dict:
    """Timestamped SHA-256 commitment over every key artefact.

    Both the plaintext key (if present) and the ciphertext are hashed.  The
    ciphertext hash is a commitment nobody can attack offline; the plaintext
    hash is what a third party checks once a key file is handed to them.
    Publishing the plaintext hash does not practically reveal the solution:
    reproducing the exact bytes of the key file requires already knowing it,
    plus its exact serialisation.
    """
    keys_dir = Path(keys_dir)
    entries = []
    for p in sorted(keys_dir.rglob("*")):
        if not p.is_file():
            continue
        entries.append({
            "path": str(p.relative_to(keys_dir)),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "encrypted": p.name.endswith(ENC_SUFFIX),
        })
    man = {
        "schema": "oasis-blind-key-commitment/1",
        "campaign": campaign,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "keys_dir": str(keys_dir),
        "algorithm": "SHA-256 over the key file as stored",
        "encryption": f"AES-256-GCM, scrypt(n={SCRYPT_N},r={SCRYPT_R},p={SCRYPT_P})",
        "note": note,
        "entries": entries,
    }
    man["manifest_sha256"] = hashlib.sha256(
        json.dumps(man["entries"], sort_keys=True).encode()).hexdigest()
    return man


def verify_manifest(manifest: dict, keys_dir: Path) -> dict:
    """Third-party check: do the key files on disk match the published hashes?"""
    keys_dir = Path(keys_dir)
    ok, bad, missing = [], [], []
    for e in manifest["entries"]:
        p = keys_dir / e["path"]
        if not p.is_file():
            missing.append(e["path"])
            continue
        try:
            got = sha256_file(p)
        except PermissionError:
            missing.append(f"{e['path']} (unreadable — unseal the keys first)")
            continue
        (ok if got == e["sha256"] else bad).append(e["path"])
    recomputed = hashlib.sha256(
        json.dumps(manifest["entries"], sort_keys=True).encode()).hexdigest()
    return {
        "manifest_self_consistent": recomputed == manifest.get("manifest_sha256"),
        "matched": ok, "mismatched": bad, "missing": missing,
        "verdict": ("PASS" if not bad and not missing
                    and recomputed == manifest.get("manifest_sha256") else "FAIL"),
    }
