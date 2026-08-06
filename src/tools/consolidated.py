"""
Consolidated MCP tools for the OASiS.

Reduces 48 tools → ~12 tools by combining related functionality.
Fewer tools = faster schema loading = faster agent response.
"""

import json
import os
import re
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from core.backend import detect_template_language
from core.registry import get_backend, available_backends, all_backends
from core.fabrication_gate import inspect_result_artefacts
from core.critic_gate import (CriticRegistry, CriticGateError,
                              setup_digest)
from core.quality_checks import check_result_files_finite, check_summary_finite

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "simulation_outputs"
_COUPLING_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "coupling"
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
_jobs: dict = {}

# HOE ablation toggle (MCP_NO_PITFALL_DB condition): when set, every
# knowledge surface strips pitfall-DB content so the agent operates as
# if the component were absent. "Pitfall DB" covers, in spirit:
#   - per-backend pitfall lists (incl. their Signal: failure anchors)
#   - general input-format pitfalls
#   - post-mortem records (the Signal-retrieval audit trail)
#   - the cross-backend collation catalog (backends/_cross.py)
# Off by default; never affects normal MCP usage.
_ABLATE_PITFALLS = os.environ.get("OFA_DISABLE_PITFALLS", "0") == "1"

_PITFALL_KEYS = ("pitfalls", "notes", "pitfall_db_entries",
                 "general_pitfalls", "common_pitfalls")

# The MANDATORY pre-execution critic is unconditional.
#
# An OFA_DISABLE_CRITIC environment ablation used to lift it, stamping an
# unreviewed run VERIFIED. That is removed. An environment variable that turns a
# mandatory gate off is a bypass, and a gate with a bypass cannot support the
# claim that OASiS results are critic-reviewed: anything that sets the variable
# — a stray export, a harness default, a copied shell script — silently
# converts every verdict into an unreviewed one that still reads as VERIFIED.
# The evaluation it existed for (a critic-ablation arm) is not run; the design
# is OASiS or no OASiS.

# The critic requirement was a boolean the AGENT passed: an audit showed a run
# stamped VERIFIED with critic_approved=True and no critic anywhere in the
# process. The server now keeps its own record. A review must be SUBMITTED for
# the exact setup being run, and the run tools consult this registry INSTEAD of
# trusting the flag.
_CRITIC_REGISTRY = CriticRegistry()


def _critic_state(solver: str, setup_text: str, *, token: str = "",
                  job_id: str = "") -> tuple[bool, str]:
    """Has an independent critic reviewed THIS setup, on this server's record?

    Only a review submitted through `submit_critic_review` counts, and only for
    the exact deck being run: a digest mismatch means the input changed after it
    was reviewed, so "review a clean setup, then run a different one" is blocked.

    Two routes, both requiring a server-side record:

      * a TOKEN issued for this review — validated by the registry as known,
        unexpired, UNUSED, and matching this solver and this deck. Single-use,
        so it also blocks replaying one review across many runs.
      * no token — the deck is matched against submitted reviews by digest. This
        still proves a review of this exact setup exists; it does NOT bound how
        many runs one review covers. Callers that want that bound pass a token.

    What neither route can do is judge whether the critique was any GOOD; the
    server is not an oracle for review quality. It enforces that a substantive
    review of this setup happened and is auditable, which is the part that can
    be enforced in software rather than requested in a prompt.

    The agent's own `critic_approved` flag is deliberately not an input here: it
    is a self-report, and replacing the self-report is the entire point.
    """
    digest = setup_digest(solver, setup_text)
    if token:
        try:
            _CRITIC_REGISTRY.consume(token, digest=digest, solver=solver,
                                     job_id=job_id)
            return True, "reviewed (critic token redeemed; single use)"
        except CriticGateError as exc:
            return False, f"critic review token refused: {exc}"
    for rec in _CRITIC_REGISTRY.records():
        if rec.solver == solver and rec.digest == digest and not rec.expired():
            return True, "reviewed (submitted review matches this setup)"
    if any(r.solver == solver for r in _CRITIC_REGISTRY.records()):
        return False, ("a critic review exists for this solver but NOT for this "
                       "setup: the input changed after it was reviewed")
    return False, "no critic review is on record for this setup"


def _attest_run_quantities(work_dir, job_id: str) -> dict:
    """Compute the run's headline numbers from the run's OWN data output.

    An audit demonstrated the gap this closes: the gate bound its verdict to the
    RUN but never to a NUMBER, so a plausible invented value attached to a real,
    clean run passed everything. Nothing recomputed it, because nothing could —
    the number existed only in the agent's narration.

    So OASiS computes them itself, from the solver's data artefacts and never
    from anything the agent wrote: a value is derived from the mesh and nodal
    field the run actually produced, and carries the file it came from and that
    file's hash. The agent no longer has to state a number, which is the point —
    fabrication-by-assertion stops being a thing an agent can usefully do,
    because the authoritative value is already in the result next to its
    provenance.

    This is not the same as making fabrication impossible. It binds a number to
    a FILE, not to a PROBLEM: a field that is well-formed but does not solve the
    stated equations still attests fine. Separating solving from forging needs
    the discrete residual (core/residual_check.py), which requires the problem's
    source term and is therefore opt-in per run.

    Never raises: an unattestable run is reported as unattestable, with the
    reason, and the run still returns.
    """
    quantities = {}
    try:
        from core.attestation import AttestationError, attest_quantity
    except Exception as exc:                      # pragma: no cover
        return {"available": False,
                "why": f"attestation unavailable: {exc}"[:300]}
    for quantity in ("l2_norm", "max_abs"):
        try:
            att = attest_quantity(work_dir, job_id, quantity)
        except AttestationError as exc:
            quantities[quantity] = {"available": False, "why": str(exc)[:300]}
        except Exception as exc:
            quantities[quantity] = {
                "available": False,
                "why": f"could not be computed from the run's data: {exc}"[:300]}
        else:
            quantities[quantity] = {
                "available": True,
                "value": att.value,
                "field": att.field,
                "from_file": Path(att.source_file).name,
                "sha256": att.source_sha256[:16],
                "n_points": att.n_points,
                "computed_by": att.computed_by,
            }
    quantities["note"] = (
        "Computed by OASiS from this run's own data output. Report these "
        "rather than numbers read out of a script's print statements, and "
        "never a number you did not obtain from the run.")
    return quantities


def _check_declared_pde(spec: str, out_files) -> dict:
    """Does the field the run produced actually SOLVE the problem it declared?

    Every other check in this gate inspects the run: did it complete, did it
    write output, is that output finite, is the mesh structurally sane. All of
    them are satisfied by a field that is well-formed and wrong — an audit built
    one in eight lines that was MORE accurate than a genuine solve, ran 82x
    faster, and passed a mesh-independence study. As data it is impeccable. The
    only property that separates it from a solve is whether it satisfies the
    equations, and that is what this measures.

    Opt-in, because it needs the problem's source term and OASiS cannot infer
    one. That is not a leak: f is the problem statement, not its solution, and a
    residual is computed from f alone. A gate that needed the answer could not
    verify a real engineering problem, where there isn't one.

    Never raises — a gate an agent can disable by malforming its input is not a
    gate.
    """
    try:
        from core.residual_gate import check_run_residual
        return check_run_residual(spec, out_files)
    except Exception as exc:                       # pragma: no cover
        return {"verdict": "REFUSED",
                "detail": f"the residual check could not run: {exc}"[:300]}


def _residual_coverage_note(result: dict) -> str:
    """Say, in the verdict itself, whether anything checked that this output
    solves anything.

    The residual check is opt-in — it needs the problem's source term, which
    only the caller has. So a run that skips it still passes every other check
    and still reads VERIFIED. If the verdict said nothing, those two cases would
    be indistinguishable in the one place an agent actually looks, and the
    strongest check in the gate would quietly become optional in practice rather
    than in principle. Naming the gap is what keeps it a gap instead of a hole.
    """
    verdict = (result.get("residual_check") or {}).get("verdict")
    if verdict == "SOLVES":
        return ("The output also SATISFIES the equations the run declared "
                "(relative residual "
                f"{result['residual_check'].get('relative_residual'):.2e}), so "
                "it was obtained by solving them rather than merely being a "
                "well-formed field.")
    if verdict == "INCONCLUSIVE":
        return ("NOTE: whether this output solves the declared problem could "
                "NOT be established — its residual falls in the band where a "
                "loosely-converged solve and a very fine-mesh analytic field "
                "are indistinguishable ("
                + str(result["residual_check"].get("detail", ""))[:200]
                + "). It is neither certified nor rejected.")
    if verdict in ("UNSUPPORTED", "REFUSED"):
        return ("NOTE: OASiS could not check whether this output solves the "
                "declared problem ("
                + str(result["residual_check"].get("detail", ""))[:160]
                + "), so this verdict covers the run, not the physics.")
    return ("NOTE: nothing here checked whether this output satisfies any "
            "equations — the run declared no problem to check against. Pass "
            "verify_pde with the problem's source term to have OASiS assemble "
            "it and measure the residual; a field that is finite, structurally "
            "sane and solves nothing passes every other check in this gate.")


def _residual_blocks_verification(result: dict) -> bool:
    """True only when the residual check positively established the field does
    not solve the declared problem. UNSUPPORTED and REFUSED must never block —
    they mean OASiS did not check, and 'not checked' is not evidence of guilt
    any more than it is evidence of innocence."""
    return (result.get("residual_check") or {}).get("verdict") == "DOES_NOT_SOLVE"


# Qualifiers a request can carry that MUST NOT be silently dropped, each with
# the variant-name tokens that satisfy it and the ones that contradict it.
_VARIANT_QUALIFIERS = [
    ("3d", ("3d", "three_d", "3D"), ("2d", "1d")),
    ("2d", ("2d", "two_d", "plane"), ("3d", "1d")),
    ("transient", ("transient", "unsteady", "time_dependent", "dynamic"),
     ("steady", "stationary", "static")),
    ("steady", ("steady", "stationary", "static"),
     ("transient", "unsteady", "dynamic")),
    ("nonlinear", ("nonlinear", "non_linear"), ()),
]
_QUALIFIER_WORDS = {
    "3d": ("3d", "three-dimensional", "three dimensional"),
    "2d": ("2d", "two-dimensional", "plane stress", "plane strain"),
    "transient": ("transient", "unsteady", "time-dependent", "time dependent",
                  "time-varying", "evolving"),
    "steady": ("steady", "stationary", "static", "steady-state"),
    "nonlinear": ("nonlinear", "non-linear", "large deformation", "finite strain"),
}


def _select_template_variant(query: str, variants: list[str]) -> tuple[str, str]:
    """Choose the template variant the request actually asked for.

    THE BUG THIS FIXES. Three call sites read `template_variants[0]` and nothing
    else, so `prepare_simulation(solver, "3d linear elasticity")` returned the
    2D plane-stress template and said nothing about it. A usability measurement
    found this on four of six realistic tasks: a correct deck already existed as
    a working generator and no tool could reach it. That is not a knowledge gap
    — adding prose cannot fix it, because a weak model handed a deck labelled
    "2D (plane stress)" for a 3D task will ship the 2D deck whatever text sits
    above it.

    Returns (variant, note). The note is never empty: it always names the
    alternatives, and when the request carried a qualifier that NO variant
    satisfies it says so in those words rather than substituting quietly.
    Substituting quietly is the failure mode — a weak model cannot detect it.
    """
    if not variants:
        return "", ""
    q = (query or "").lower()
    asked = [name for name, words in _QUALIFIER_WORDS.items()
             if any(w in q for w in words)]

    chosen, unmet = variants[0], []
    for name in asked:
        satisfies, contradicts = next(
            (s, c) for n, s, c in _VARIANT_QUALIFIERS if n == name)
        hit = next((v for v in variants
                    if any(t in v.lower() for t in satisfies)
                    and not any(t in v.lower() for t in contradicts)), None)
        if hit:
            chosen = hit
        else:
            unmet.append(name)

    bits = []
    if unmet:
        bits.append(
            "⚠ You asked for " + " and ".join(f"**{u}**" for u in unmet)
            + f", and no template variant provides it. Serving `{chosen}`, "
            f"which does NOT satisfy that — adapt it rather than running it "
            f"as-is.")
    elif asked:
        bits.append(f"Selected `{chosen}` for: {', '.join(asked)}.")
    if len(variants) > 1:
        others = [v for v in variants if v != chosen]
        bits.append(f"Other variants available: {', '.join(others)} — request "
                    f"one by name via `examples(action='template', "
                    f"variant='<name>')`.")
    return chosen, (" ".join(bits))


def _file_fingerprint(path: Path) -> str:
    """Content hash of one file, or size+mtime when it is too big to hash cheaply.

    Either way a change is visible, which is all the digest needs.
    """
    import hashlib
    try:
        st = path.stat()
        if st.st_size > 4 << 20:
            return f"size:{st.st_size}:mtime:{st.st_mtime_ns}"
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        return f"unreadable:{type(e).__name__}"


def _participant_fingerprints(participants_json: str, monolithic_json: str = "") -> dict:
    """Fingerprint the FILES a coupling actually executes.

    THE BYPASS THIS CLOSES. A participant spec names a command
    (["python", "run.py"]) and a work_dir; the physics lives entirely in
    run.py, which the digest never saw. So a review of one coupling approved
    any other coupling that reused the same file names: reviewing a correct
    setup and then rewriting the participant script produced a completely
    different answer, stamped VERIFIED, with the note "submitted review matches
    this setup". Demonstrated with a 225x change in the coupled result.

    Every command token that resolves to an existing file (absolute, or
    relative to that participant's work_dir) is fingerprinted, as are its
    declared data_files and the monolithic reference command. A file that does
    not exist yet is recorded as absent, so creating it afterwards changes the
    digest — a participant whose script does not exist cannot have been
    reviewed.

    WHAT THIS STILL DOES NOT COVER, established by running it. The fingerprint
    reaches exactly the paths the SPEC NAMES. Anything a script opens at RUNTIME
    is invisible, and rewriting it leaves the digest identical while the coupled
    answer changes completely. Four routes were demonstrated, each taking a
    reviewed x=2.666667 to x=334.666665 with the verdict still reading VERIFIED,
    "an independent critic reviewed this exact setup":

      * `from model import step` — an ordinary helper module beside run.py. No
        trickery at all, and the most likely shape of real code;
      * `exec(open("physics.py").read())`;
      * a path built from an environment variable, or by `os.path.join`;
      * a path found by `glob`.

    A symlink repointed at different content IS caught (the hash follows the
    link), as is a rewritten `data_files` entry and a rewritten monolithic
    reference. Closing the rest needs the participants' whole working trees
    fingerprinted, or an import trace; neither is done here. So `couple` states
    the scope in its served coverage (`_DIGEST_SCOPE_LIMIT`) rather than letting
    VERIFIED imply more than the digest supports, and declaring such a file in
    `data_files` is the supported way to bring it inside.
    """
    out: dict = {}
    try:
        specs = json.loads(participants_json) if participants_json else []
    except (json.JSONDecodeError, TypeError):
        return out
    if monolithic_json:
        try:
            m = json.loads(monolithic_json)
            if isinstance(m, dict):
                specs = list(specs) + [dict(m, name="__monolithic__")]
        except (json.JSONDecodeError, TypeError):
            pass
    if not isinstance(specs, list):
        return out
    for s in specs:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "?"))
        wd = Path(str(s.get("work_dir", "."))).expanduser()
        seen: dict = {}
        for token in list(s.get("command") or []) + list(s.get("data_files") or []):
            tok = str(token)
            for cand in (Path(tok).expanduser(), wd / tok):
                try:
                    if cand.is_file():
                        seen[tok] = _file_fingerprint(cand)
                        break
                except OSError:
                    continue
            else:
                # Only record a miss for tokens that LOOK like files, so option
                # flags do not turn into noise that changes with argv order.
                if "/" in tok or "." in tok:
                    seen[tok] = "absent"
        if seen:
            out[name] = seen
    return out


def _coupling_setup_text(**kwargs) -> str:
    """Canonical text a coupling review is bound to.

    ONE definition, used by the coupling tools and by `submit_critic_review`, so
    the digest a review is issued for and the digest a run is checked against
    cannot drift apart. Every argument that changes what is solved belongs in
    here — including the CONTENTS of the participant scripts, which is what the
    coupling tools actually execute and what the spec only names.
    """
    payload = dict(kwargs)
    if payload.get("participants") or payload.get("monolithic"):
        fp = _participant_fingerprints(str(payload.get("participants") or ""),
                                       str(payload.get("monolithic") or ""))
        if fp:
            payload["__participant_files__"] = fp
    return json.dumps(payload, sort_keys=True)


_DIGEST_SCOPE_LIMIT = (
    "review-to-run binding SCOPE: the review is bound to the participant spec "
    "and to the CONTENTS of every file the spec names (each command token that "
    "is a file, every `data_files` entry, the monolithic reference). It is NOT "
    "bound to files a script opens at runtime — a helper module it imports, a "
    "path built from an environment variable or by os.path.join, a file found by "
    "glob. Rewriting one of those changes the coupled answer and leaves this "
    "verdict's digest identical, so VERIFIED here does not certify that part of "
    "the setup was reviewed. Declare such files in `data_files` to bring them "
    "inside the digest.")


_MONOLITHIC_NOT_SUPPLIED = (
    "monolithic consistency: NOT CHECKED — no un-split reference solve was "
    "supplied. Every other check here is internal to the coupling: it can tell "
    "you the iteration converged, conserved and stayed finite, and all of that "
    "is true of a coupling in which both sides consistently use the wrong units "
    "or apply the interface condition with the wrong sign. If this problem can "
    "be solved un-split in ONE code, pass `monolithic` and OASiS will compare "
    "the two answers; that is the strongest verification available here and it "
    "needs no external benchmark.")


def _run_monolithic_check(monolithic: str, exports: dict,
                          rtol: float = 0.05) -> tuple[dict, list[str], list[str]]:
    """Re-solve the coupled problem un-split, in one code, and compare.

    `monolithic` is a JSON {"command":[argv...], "work_dir":str, "timeout":int}.
    The command must write <work_dir>/monolithic.json in InterfaceData shape,
    sampled on the same interface the participants export. Every participant's
    exported `values` is then compared against the monolithic field interpolated
    onto that participant's own interface coordinates.

    Returns (report, findings, checks_not_run). A monolithic solve that itself
    fails is reported as NOT CHECKED, never as agreement: the coupling is not
    guilty because its reference could not be produced, and it is not innocent
    either.
    """
    import subprocess
    import numpy as _np
    from core.quality_checks import check_monolithic_consistency

    if not (monolithic or "").strip():
        return {"status": "not supplied"}, [], [_MONOLITHIC_NOT_SUPPLIED]
    try:
        spec = json.loads(monolithic)
        cmd = list(spec["command"])
        wd = Path(spec["work_dir"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return ({"status": "bad spec", "detail": str(e)}, [],
                [f"monolithic consistency: NOT CHECKED — the `monolithic` spec "
                 f"could not be read ({e}); expected JSON with `command` and "
                 "`work_dir`."])
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "monolithic.json"
    if out.exists():
        out.unlink()
    try:
        p = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True,
                           timeout=int(spec.get("timeout", 3600)))
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        return ({"status": "reference solve failed", "detail": str(e)[:200]}, [],
                [f"monolithic consistency: NOT CHECKED — the un-split reference "
                 f"solve did not complete ({type(e).__name__}: {str(e)[:120]})."])
    if p.returncode != 0 or not out.exists():
        return ({"status": "reference solve failed",
                 "returncode": p.returncode,
                 "detail": (p.stderr or p.stdout or "")[-300:]}, [],
                [f"monolithic consistency: NOT CHECKED — the un-split reference "
                 f"solve exited {p.returncode} / wrote no monolithic.json, so "
                 "there is nothing to compare the coupled answer against."])
    try:
        ref = json.loads(out.read_text())
        ref_vals = _np.asarray(ref["values"], float).ravel()
        ref_co = _np.atleast_2d(_np.asarray(ref.get("coordinates", []), float))
    except Exception as e:
        return ({"status": "reference unreadable", "detail": str(e)[:200]}, [],
                [f"monolithic consistency: NOT CHECKED — monolithic.json could "
                 f"not be read as InterfaceData ({e})."])
    if ref_vals.size == 0 or not _np.all(_np.isfinite(ref_vals)):
        return ({"status": "reference invalid"}, [],
                ["monolithic consistency: NOT CHECKED — the reference solve's "
                 "values are empty or non-finite."])

    ref_field = str(ref.get("field_name", "") or "")
    report: dict = {"status": "checked", "reference_points": int(ref_vals.size),
                    "reference_field": ref_field}
    findings: list[str] = []
    # Nothing here can tell an independent un-split solve from a script that
    # echoes the coupled answer back: both write the same file. Agreement is
    # therefore only as good as the reference, and that has to be said, because
    # a reference that is wrong in the same way as the coupling turns the
    # strongest check in this tool into a rubber stamp.
    not_run: list[str] = [
        "monolithic reference INDEPENDENCE: OASiS ran the command it was given "
        "and compared the numbers; it cannot tell a genuine un-split solve from "
        "one that re-reads or reproduces the coupled answer. Agreement below is "
        "evidence only if the reference solves the problem on its own."]
    for name, ex in exports.items():
        vals = _np.asarray(ex.get("values", []), float).ravel()
        if vals.size == 0:
            not_run.append(f"monolithic consistency for {name}: it exported no values")
            continue
        # Compare like with like. Participants on the two sides of an interface
        # do not always export the same quantity (a Dirichlet-Neumann pair
        # exports the same temperature; a force/displacement pair does not), and
        # comparing a displacement against a temperature reference would report
        # a large, entirely meaningless disagreement.
        got_field = str(ex.get("field_name", "") or "")
        if ref_field and got_field and got_field != ref_field:
            not_run.append(
                f"monolithic consistency for {name}: it exports {got_field!r} "
                f"while the reference solve provides {ref_field!r}, so there is "
                "nothing to compare. Have the reference write the field this "
                "participant exports if you want it covered.")
            continue
        target = vals
        if ref_vals.size == vals.size:
            ref_at = ref_vals
        else:
            co = _np.atleast_2d(_np.asarray(ex.get("coordinates", []), float))
            if ref_co.size == 0 or co.size == 0 or len(ref_co) != ref_vals.size:
                not_run.append(
                    f"monolithic consistency for {name}: the reference has "
                    f"{ref_vals.size} point(s) and this participant exports "
                    f"{vals.size}, and there are not enough coordinates to map "
                    "one onto the other")
                continue
            from core.field_transfer import InterfaceData, interpolate_to_points
            ref_at = _np.asarray(interpolate_to_points(
                InterfaceData(coordinates=ref_co, values=ref_vals,
                              field_name=str(ref.get("field_name", "ref"))),
                co), float).ravel()
        if ref_at.size != target.size:
            not_run.append(f"monolithic consistency for {name}: shapes did not align")
            continue
        denom = float(_np.linalg.norm(ref_at)) or 1e-30
        rel_l2 = float(_np.linalg.norm(target - ref_at)) / denom
        report[name] = {"coupled_mean": float(_np.mean(target)),
                        "monolithic_mean": float(_np.mean(ref_at)),
                        "relative_l2": rel_l2}
        findings += check_monolithic_consistency(
            float(_np.mean(target)), float(_np.mean(ref_at)), rtol,
            qoi=f"{name} interface mean")
        if rel_l2 > rtol:
            findings.append(
                f"{name}: coupled interface field differs from the un-split "
                f"monolithic re-solve by {rel_l2:.1%} in relative L2 > {rtol:.0%} "
                "— the coupled result is likely WRONG even though the iteration "
                "converged.")
    return report, findings, not_run


def _stamp_verification(result: dict, *, evidence_ok: bool, reason: str = "",
                        critic_approved: bool = False,
                        solver: str | None = None,
                        setup_text: str | None = None,
                        critic_token: str = "",
                        job_id: str = "") -> dict:
    """Attach OASiS's verification-gate verdict to a run/coupling result in place.

    A result is trustworthy ONLY when it (1) passes the numerical checks — the
    run completed, produced output, and that output is finite / converged /
    balanced — AND (2) has been reviewed by OASiS's independent critic. OASiS
    *verifies* and checks integrity; it does not *validate* — physical validity
    stays the engineer's task.

    SCOPE, STATED HONESTLY. These checks bind the verdict to the RUN. They do
    NOT bind it to a reported NUMBER: nothing here recomputes a value the agent
    states, so a plausible invented number attached to a real run still passes.
    An audit demonstrated exactly that against a live backend. Earlier wording
    here claimed attestation "binds every reported number to run evidence";
    that was false and is removed rather than softened. Binding a value to the
    data it came from requires computing it from the run's own output (see
    core/attestation.py) and checking the field satisfies the discrete problem
    (see core/residual_check.py); until those are wired into this path, the
    verdict means "a real, clean run happened", not "this number came from it".

    Enforcement is by VERDICT, never by error: an unverified run still returns
    its output, but is never labelled trustworthy, so a confidently-wrong or
    fabricated claim can't be reported as a result. The critic is mandatory,
    unconditionally — there is no ablation switch that lifts it.

    THE CRITIC IS RESOLVED FROM THE SERVER'S RECORD, NOT FROM THE ARGUMENT.
    Pass `solver` and `setup_text` and this function asks the critic registry
    whether that exact deck was reviewed. A caller that does NOT identify its
    setup gets NOT VERIFIED: the gate fails closed, because a tool that cannot
    say what it ran cannot have had that thing reviewed. `critic_approved` is
    retained only to record what the agent CLAIMED, so a claim with no matching
    review can be named as such in the verdict.

    evidence_ok: True iff a real run backs this result AND the gate's numerical
        checks passed (execution completed, output/logs produced, converged,
        finite, interface balanced — as applicable to the calling tool).
    reason: short cause shown when evidence_ok is False.
    critic_approved: what the agent asserted. Recorded, never trusted.
    solver, setup_text: identify the deck whose review is being looked up.
    critic_token: optional single-use token from `submit_critic_review`.
    job_id: recorded against a redeemed token for audit.
    """
    if solver is not None and setup_text is not None:
        critic_ok, critic_note = _critic_state(
            solver, setup_text, token=critic_token, job_id=job_id)
    else:
        critic_ok, critic_note = False, (
            "not checked — this tool did not identify its setup to the "
            "verification gate, so no review could be looked up")
    if critic_approved and not critic_ok:
        critic_note += ("; the call declared critic_approved=True, which OASiS "
                        "does not accept as evidence — a review must be on "
                        "record via submit_critic_review")
    if not evidence_ok:
        result["trustworthy_result"] = False
        result["verification"] = (
            "NOT VERIFIED — "
            + (reason or "the result is not bound to a check-passing run")
            + ". Per OASiS attestation this claim must NOT be reported as a "
            "result; revise the setup and re-run.")
    elif not critic_ok:
        result["trustworthy_result"] = False
        result["verification"] = (
            "NOT VERIFIED — the automated checks passed, but OASiS's MANDATORY "
            "independent critic has not reviewed this setup, and OASiS treats no "
            "result as trustworthy until it has (" + critic_note + "). Spawn a "
            "critic to challenge the parameters, units, discretisation, problem "
            "statement and boundary conditions and to cross-check against "
            "literature/benchmarks, then call submit_critic_review with what it "
            "found and re-run. Asserting critic_approved=True does not work: "
            "OASiS looks the review up rather than taking your word for it.")
    else:
        result["trustworthy_result"] = True
        result["verification"] = (
            "VERIFIED — an independent critic reviewed this exact setup ("
            + critic_note + ") and the run passed OASiS's verification-gate "
            "numerical checks. This is verification, not validation: confirm "
            "physical validity against reality yourself. "
            + _residual_coverage_note(result))
    result["critic_review"] = critic_note
    return result


def _short_reason(msg: str, limit: int = 240) -> str:
    """Collapse a multi-line availability/error message (often a raw import
    traceback) to a concise one-liner for user-facing surfaces. A backend that
    isn't installed should read as one clear line, not a 30-frame stack dump that
    floods the backend list. Keeps the actual exception (last non-empty line) and
    any install/try hint; the full trace stays available via `developer`/logs.
    """
    if not msg:
        return msg or ""
    lines = [ln.strip() for ln in str(msg).strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0][:limit]
    tail = lines[-1]
    if tail.lower().startswith("traceback"):
        tail = lines[-2] if len(lines) > 1 else tail
    hint = next((ln for ln in lines
                 if ln.lower().startswith(("install", "try", "run ", "set ", "conda ", "pip "))),
                "")
    out = tail if (not hint or hint == tail) else f"{tail}  ({hint})"
    return out[:limit]


def _strip_pitfalls(obj):
    """Recursively remove pitfall-DB keys from nested dicts/lists.

    No-op when _ABLATE_PITFALLS is False.
    """
    if not _ABLATE_PITFALLS:
        return obj
    if isinstance(obj, dict):
        return {k: _strip_pitfalls(v) for k, v in obj.items()
                if k not in _PITFALL_KEYS}
    if isinstance(obj, list):
        return [_strip_pitfalls(x) for x in obj]
    return obj


async def _run_with_progress(ctx: Context, coro, message_prefix: str = "Running"):
    """Run a coroutine while sending periodic MCP progress keepalives.

    This prevents the MCP client from timing out on long-running simulations
    (DUNE JIT compilation, 4C FSI, deal.II builds can take minutes).
    """
    import asyncio

    task = asyncio.create_task(coro)
    elapsed = 0
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5
                try:
                    await ctx.report_progress(
                        elapsed, total=None,
                        message=f"{message_prefix} ({elapsed}s elapsed)"
                    )
                except Exception:
                    pass  # progress reporting is best-effort
    except Exception:
        if not task.done():
            task.cancel()
        raise
    return task.result()


def _stub_template_tag(content: str, fmt: str) -> str:
    """Return a `" — ⚠ STUB"` marker if `content` looks like a
    placeholder template (a single comment line, or fewer than
    ~150 chars of non-comment body), otherwise empty.

    The catalog ships 9 fourc physics rows whose generators
    return only a one-line comment (`# Foo template — use ...`)
    because no full template has been written yet. Surfacing
    those as plain `## Template` sections in prepare_simulation
    output misleads the LLM: the heading promises a runnable
    template, but the body is a 50-80 char placeholder.

    Detection rule: strip every line that begins with `#`
    (YAML / Python comment) or is whitespace-only; if what
    remains is shorter than 150 chars, treat as a stub. The
    `fmt` argument tells us which comment character to honour
    — for the (rare) non-comment-character formats (`json`,
    `cpp`), we still apply the size heuristic but skip the
    comment-stripping step. (Audit 2026-06-02.)
    """
    if not isinstance(content, str):
        return ""
    # Central stub detection: catches print-and-exit (deal.II), availability-probe
    # (Kratos), <...>-placeholder decks (4C), and comment-only templates — fakes that
    # advertise physics but don't solve. (Extends the original size heuristic.)
    try:
        from core.quality_checks import is_stub_output
        reason = is_stub_output(content)
        if reason:
            return f" — ⚠ STUB (not a runnable deck: {reason})"
    except Exception:
        pass
    if fmt in ("yaml", "yml", "python", "py"):
        non_comment_lines = [
            ln for ln in content.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        body = "\n".join(non_comment_lines)
    else:
        body = content
    if len(body.strip()) < 150:
        return " — ⚠ STUB (catalog placeholder — no full template yet)"
    return ""


_PHYSICS_SYNONYMS = {
    # ── Heat / thermal conduction ──────────────────────────────────
    # canonical key 'heat' exists in: fourc, fenics, ngsolve, kratos,
    # dealii, dune, skfem, febio (all 8 backends)
    "thermal": "heat",
    "conduction": "heat",
    "temperature": "heat",
    "heat_transfer": "heat",
    "heat_conduction": "heat",
    "heat_flow": "heat",
    "thermal_conduction": "heat",
    "thermal_diffusion": "heat",
    "fourier": "heat",
    # transient flavour (canonical: heat_transient OR time_dependent_heat)
    "unsteady_heat": "heat_transient",
    "transient_heat": "heat_transient",
    "time_heat": "heat_transient",
    "dynamic_thermal": "heat_transient",
    "transient_thermal": "heat_transient",
    "time_dependent_thermal": "time_dependent_heat",

    # ── Linear elasticity / small-strain mechanics ─────────────────
    "elasticity": "linear_elasticity",
    "structural": "linear_elasticity",
    "structural_mechanics": "linear_elasticity",
    "structural_2d": "linear_elasticity",
    "structural_3d": "linear_elasticity",
    "solid": "linear_elasticity",
    "solid_mechanics": "linear_elasticity",
    "mechanics": "linear_elasticity",
    "small_strain": "linear_elasticity",
    "hooke": "linear_elasticity",
    "hookean": "linear_elasticity",
    "linear_solid": "linear_elasticity",
    "plane_strain": "linear_elasticity",
    "plane_stress": "linear_elasticity",
    "elastic": "linear_elasticity",
    "fea": "linear_elasticity",
    "statics": "linear_elasticity",
    "elasticity_2d": "linear_elasticity",
    "elasticity_3d": "linear_elasticity",

    # ── Hyperelasticity / large-deformation solid ──────────────────
    "nonlinear_elasticity": "hyperelasticity",
    "large_deformation": "hyperelasticity",
    "large_strain": "hyperelasticity",
    "neo_hookean": "hyperelasticity",
    "neohookean": "hyperelasticity",
    "mooney_rivlin": "hyperelasticity",
    "ogden": "hyperelasticity",
    "finite_strain": "hyperelasticity",
    "finite_deformation": "hyperelasticity",
    "hyperelastic": "hyperelasticity",
    "hyperelastic_solid": "hyperelasticity",
    "geometric_nonlinearity": "hyperelasticity",
    "nonlinear_solid": "hyperelasticity",

    # ── Plasticity / elasto-plastic ────────────────────────────────
    "elasto_plasticity": "plasticity",
    "elastoplasticity": "plasticity",
    "elasto_plastic": "plasticity",
    "yield": "plasticity",
    "yielding": "plasticity",
    "mohr_coulomb": "plasticity",
    "drucker_prager": "plasticity",
    "von_mises": "plasticity",
    "j2_plasticity": "plasticity",
    "soil_plasticity": "plasticity",
    "metal_plasticity": "plasticity",
    "return_mapping": "plasticity",
    "plastic_flow": "plasticity",

    # ── Stokes (creeping / mixed) flow ─────────────────────────────
    "stokes_flow": "stokes",
    "creeping_flow": "stokes",
    "mixed_stokes": "stokes",
    "taylor_hood": "stokes",
    "low_reynolds": "stokes",

    # ── Navier-Stokes / CFD ────────────────────────────────────────
    "cfd": "navier_stokes",
    "flow": "navier_stokes",
    "fluid_dynamics": "navier_stokes",
    "ns": "navier_stokes",
    "incompressible": "navier_stokes",
    "incompressible_flow": "navier_stokes",
    "viscous_flow": "navier_stokes",
    "fluid_flow": "navier_stokes",
    "fluid_mechanics": "navier_stokes",
    "internal_flow": "navier_stokes",
    "channel_flow": "navier_stokes",
    "external_flow": "navier_stokes",
    "laminar_flow": "navier_stokes",
    # transient flavour
    "transient_ns": "time_dependent_ns",
    "unsteady_ns": "time_dependent_ns",
    "unsteady_navier_stokes": "time_dependent_ns",
    "vortex_shedding": "time_dependent_ns",

    # ── Maxwell / electromagnetism ─────────────────────────────────
    "magnetostatics": "maxwell",
    "electromagnetics": "maxwell",
    "em": "maxwell",
    "magnetic": "maxwell",
    "eddy_current": "maxwell",
    "eddy_current_problem": "maxwell",
    "nedelec": "maxwell",
    "electromagnetic": "maxwell",
    "h_curl": "maxwell",
    "electric_field": "maxwell",
    "magnetic_field": "maxwell",
    "electrostatics": "maxwell",
    "electrodynamics": "maxwell",

    # ── Helmholtz / time-harmonic acoustics ────────────────────────
    "acoustics": "helmholtz",
    "acoustic": "helmholtz",
    "sound": "helmholtz",
    "frequency_domain": "helmholtz",
    "time_harmonic": "helmholtz",
    "scattering": "helmholtz",

    # ── Wave equation (second-order, time-domain) ──────────────────
    "wave_equation": "wave",
    "second_order_wave": "wave",
    "elastic_wave": "wave",
    "transient_wave": "time_dependent_wave",
    "unsteady_wave": "time_dependent_wave",

    # ── Eigenvalue / modal analysis ────────────────────────────────
    "vibration": "eigenvalue",
    "modal": "eigenvalue",
    "frequencies": "eigenvalue",
    "modes": "eigenvalue",
    "natural_frequencies": "eigenvalue",
    "eigenmode": "eigenvalue",
    "eigenfrequency": "eigenvalue",
    "buckling": "eigenvalue",
    "linear_buckling": "eigenvalue",

    # ── Poisson / Laplace / scalar elliptic ────────────────────────
    "diffusion": "poisson",
    "laplace": "poisson",
    "scalar": "poisson",
    "scalar_pde": "poisson",
    "steady_diffusion": "poisson",
    "electrostatic_field": "poisson",
    "elliptic": "poisson",

    # ── Convection-diffusion / scalar transport ────────────────────
    "transport": "convection_diffusion",
    "advection": "convection_diffusion",
    "advection_diffusion": "convection_diffusion",
    "scalar_transport": "convection_diffusion",
    "mass_transport": "convection_diffusion",
    "contaminant_transport": "convection_diffusion",
    "cd": "convection_diffusion",

    # ── DG (discontinuous Galerkin) ────────────────────────────────
    "discontinuous_galerkin": "dg_methods",
    "dg": "dg_methods",
    "ipdg": "dg_methods",
    "sipg": "dg_methods",
    "nipg": "dg_methods",
    "interior_penalty": "dg_methods",

    # ── Biharmonic / plate bending ─────────────────────────────────
    "plate": "biharmonic",
    "kirchhoff": "biharmonic",
    "kirchhoff_love": "biharmonic",
    "fourth_order": "biharmonic",
    "bending": "biharmonic",
    "kirchhoff_plate": "biharmonic",

    # ── Adaptive refinement (dealii / skfem / dune) ────────────────
    # canonical varies: adaptive_refinement (dealii), adaptive_poisson
    # (skfem, dune), hp_adaptive (dealii). _fuzzy_match_physics routes
    # the synonym only if it exists in this backend's catalog, so
    # mapping to adaptive_refinement first is safe — fall-through
    # picks the right one per backend.
    "amr": "hp_adaptive",
    "refinement": "hp_adaptive",
    "adaptive": "hp_adaptive",
    "h_refinement": "hp_adaptive",
    "p_refinement": "hp_adaptive",
    "hp_refinement": "hp_adaptive",
    "hp": "hp_adaptive",
    "error_estimator": "error_estimation",
    "kelly_estimator": "error_estimation",
    "kelly": "error_estimation",
    "adaptive_mesh": "hp_adaptive",
    "mesh_refinement": "hp_adaptive",

    # ── Phase field / fracture / damage ────────────────────────────
    "cahn_hilliard": "phase_field",
    "allen_cahn": "phase_field",
    "phase_field_fracture": "phase_field",
    "brittle_fracture": "fracture",
    "crack": "fracture",
    "crack_propagation": "fracture",
    "fracture_mechanics": "fracture",
    "damage_mechanics": "damage",
    "continuum_damage": "damage",

    # ── Topology / shape optimization ──────────────────────────────
    "topopt": "topology_optimization",
    "topology": "topology_optimization",
    "topology_opt": "topology_optimization",
    "shape_opt": "shape_optimization",
    "shape_optimisation": "shape_optimization",
    "structural_optimization": "topology_optimization",
    "compliance_minimization": "topology_optimization",

    # ── Contact / friction ─────────────────────────────────────────
    "friction": "contact",
    "contact_mechanics": "contact",
    "frictional_contact": "contact",
    "hertz": "contact",
    "mortar_contact": "contact",
    "node_to_surface": "contact",
    "surface_to_surface": "contact",

    # ── FSI / TSI / multiphysics coupling ──────────────────────────
    "fluid_structure": "fsi",
    "fluid_structure_interaction": "fsi",
    "thermo_structural": "thermal_structural",
    "thermomechanical": "thermal_structural",
    "multiphysics": "fsi",
    "coupling": "fsi",
    "thermal_solid_interaction": "tsi",
    "structural_thermal_interaction": "tsi",
    "soil_structure": "ssi",
    "structure_soil_interaction": "ssi",

    # ── Porous media / geomechanics ────────────────────────────────
    "poroelasticity": "porous_media",
    "poro": "porous_media",
    "consolidation": "porous_media",
    "terzaghi": "porous_media",
    "biot": "porous_media",
    "geomechanics": "porous_media",
    "saturated_porous": "porous_media",
    "unsaturated_porous": "porous_media",

    # ── Particle methods: peridynamics, SPH, DEM, MPM ──────────────
    "peridynamics": "particle_pd",
    "pd": "particle_pd",
    "bond_based": "particle_pd",
    "state_based": "particle_pd",
    "ordinary_state_based": "particle_pd",
    "non_ordinary_state_based": "particle_pd",
    "nosbpd": "particle_pd",
    "sph": "particle_sph",
    "smoothed_particle": "particle_sph",
    "smoothed_particle_hydrodynamics": "particle_sph",
    "discrete_element": "dem",
    "discrete_element_method": "dem",
    "granular": "dem",
    "material_point": "mpm",
    "material_point_method": "mpm",
    "particle_in_cell": "mpm",
    "pic": "mpm",
    "lagrangian_particles": "particle_sph",

    # ── Multiphase / free surface / VOF / level-set ────────────────
    "two_phase": "multiphase",
    "multi_phase": "multiphase",
    "vof": "multiphase",
    "volume_of_fluid": "multiphase",
    "immiscible": "multiphase",
    "interface": "multiphase",
    "free_surface_flow": "free_surface",
    "level_set_method": "level_set",
    "droplet": "droplet_dynamics",

    # ── Reaction-diffusion / chemical kinetics ─────────────────────
    "rd": "reaction_diffusion",
    "reaction_diffusion_system": "reaction_diffusion",
    "fitzhugh_nagumo": "reaction_diffusion",
    "gray_scott": "reaction_diffusion",
    "schnakenberg": "reaction_diffusion",
    "chemical_kinetics": "reaction_diffusion",

    # ── Structural dynamics / transient solid ──────────────────────
    "dynamics": "structural_dynamics",
    "transient_structural": "structural_dynamics",
    "dynamic_analysis": "structural_dynamics",
    "time_domain_structural": "structural_dynamics",
    "implicit_dynamics": "structural_dynamics",
    "explicit_dynamics": "structural_dynamics",
    "structural_transient": "structural_dynamics",

    # ── Schrödinger / quantum ──────────────────────────────────────
    "quantum": "schrodinger",
    "quantum_mechanics": "schrodinger",
    "wavefunction": "schrodinger",
    "eigenstate": "schrodinger",

    # ── MHD ────────────────────────────────────────────────────────
    "magnetohydrodynamics": "mhd",
    "magneto_hydrodynamics": "mhd",
    "plasma": "mhd",

    # ── Beams / shells / membranes ─────────────────────────────────
    "beam": "beams",
    "beam_element": "beams",
    "timoshenko": "beams",
    "euler_bernoulli": "beams",
    "shell_element": "shell",
    "kirchhoff_love_shell": "shell",
    "reissner_mindlin": "shell",
    "membrane_element": "membrane",

    # ── Cardiac / cardiovascular ───────────────────────────────────
    "cardiac": "cardiac_monodomain",
    "electrophysiology": "cardiac_monodomain",
    "monodomain": "cardiac_monodomain",
    "bidomain": "cardiac_monodomain",
    "cardiovascular": "cardiovascular0d",
    "windkessel": "cardiovascular0d",
    "lumped_parameter": "cardiovascular0d",
    "0d_model": "cardiovascular0d",

    # ── XFEM ───────────────────────────────────────────────────────
    "extended_fem": "xfem_fluid",
    "xfem": "xfem_fluid",
    "level_set_fem": "xfem_fluid",
    "embedded_interface": "xfem_fluid",

    # ── Reduced-order / multiscale ─────────────────────────────────
    "rom": "rom",
    "reduced_order": "rom",
    "reduced_order_modeling": "rom",
    "pod": "rom",
    "homogenization": "multiscale",
    "fe_squared": "multiscale",
    "fe2": "multiscale",

    # (Removed dead 'optimal_control' aliases — no backend provides that
    # canonical, even as reference knowledge, so they routed users to nothing.)

    # ── Matrix-free / multigrid (solver-level not physics, but
    #     dealii exposes them as physics keys) ─────────────────────
    "matrix_free_fe": "matrix_free",
    "geometric_multigrid": "multigrid",
    "algebraic_multigrid": "multigrid",
    "amg": "multigrid",
    "gmg": "multigrid",

    # ── HDG / HDivDiv / mixed methods ──────────────────────────────
    "hdg": "hdivdiv",
    "hybridizable_dg": "hdivdiv",
    "hellinger_reissner": "hdivdiv",
    "raviart_thomas": "mixed_poisson",
    "rt": "mixed_poisson",
    "bdm": "mixed_poisson",
    "mixed_finite_element": "mixed_poisson",
    "mixed_method": "mixed_poisson",
    "h_div_conforming": "mixed_poisson",

    # ── Hydraulics / shallow water ─────────────────────────────────
    "shallow_water_equations": "shallow_water",
    "saint_venant": "shallow_water",
    "swe": "shallow_water",

    # ── ALE ────────────────────────────────────────────────────────
    "arbitrary_lagrangian_eulerian": "ale",
    "moving_mesh": "ale",
}


# Queries shorter than this never participate in loose substring
# matches — short tokens collide with too many physics names /
# descriptions ('ns' is a substring of 'transient', 'em' of
# 'eigenvalue', 'pd' of 'pde'). For short tokens we trust ONLY
# exact-name and synonym-map matches. (Audit 2026-06-02.)
_MIN_LOOSE_MATCH_LEN = 4


def _fuzzy_match_physics(backend, query: str) -> str:
    """Fuzzy-match a physics query to an actual physics name in a backend.

    Resolution order (audit 2026-06-02):
      1. Empty -> return empty so caller can surface availables list.
      2. Exact physics-name match.
      3. Synonym map (e.g. 'ns' -> 'navier_stokes', 'em' -> 'maxwell',
         'thermal' -> 'heat'). Synonyms run BEFORE substring matching
         because short-token substrings collide constantly:
         'ns' is a substring of 'transient', 'em' of 'eigenvalue',
         'pd' of 'nonlinear_pde'. Without this ordering, LLMs that
         type the canonical shorthand silently got the wrong physics.
      4. Query is substring of a physics name (only if len >= 4).
      5. Physics name is substring of the query (only if the
         physics name is itself >= 4 chars — otherwise tiny names
         like 'pd' match every query containing those letters).
      6. Query is substring of a physics description (last resort,
         len >= 4).
      7. Fallthrough: return original so caller can produce a
         "no information found" message.
    """
    query_lower = query.lower().strip()

    # Empty / whitespace-only query — return verbatim so the
    # caller's "no information found" path can surface the
    # full available-physics list. Without this guard, the
    # next substring check matches "" to the FIRST physics
    # in the catalog (because "" is a substring of every
    # string) and the LLM silently sees prepare_simulation
    # output for a physics it never asked for. (Audit
    # 2026-06-01.)
    if not query_lower:
        return query_lower

    # 1. Direct match.
    for p in backend.supported_physics():
        if p.name == query_lower:
            return p.name

    # 2. Synonym map — BEFORE the substring scan so short
    # canonical shorthands ('ns', 'em', 'pd') route to the
    # right physics. Only return the synonym if it actually
    # exists in this backend's catalog; otherwise fall through
    # to the loose matchers (a backend that has 'maxwell' but
    # not the synonym should still match via substring).
    mapped = _PHYSICS_SYNONYMS.get(query_lower)
    if mapped:
        for p in backend.supported_physics():
            if p.name == mapped:
                return p.name

    # 3. Loose substring of physics name (only for non-short
    # queries — see _MIN_LOOSE_MATCH_LEN rationale above).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.name.lower():
                return p.name

    # 4. Physics name is substring of query (only when the
    # physics name itself is non-trivial). Without the length
    # guard, a 2-char catalog entry like 'pd' matches every
    # query containing those letters, which is the same
    # collision class we just guarded the other direction
    # against.
    for p in backend.supported_physics():
        if (len(p.name) >= _MIN_LOOSE_MATCH_LEN
                and p.name.lower() in query_lower):
            return p.name

    # 5. Loose substring of physics description (last resort,
    # same length guard).
    if len(query_lower) >= _MIN_LOOSE_MATCH_LEN:
        for p in backend.supported_physics():
            if query_lower in p.description.lower():
                return p.name

    # 6. Token-overlap fallback. Compound synonyms like
    # 'rarefied_gas_dynamics' -> 'rarefied_flow' or 'hypersonic_cht'
    # -> 'hypersonic_flow' share a DISCRIMINATING token ('rarefied',
    # 'hypersonic', 'conjugate') but neither string is a substring of
    # the other, so steps 3-5 miss them. Match on shared significant
    # tokens (>= 5 chars to avoid 'flow'/'heat'/'grid' collisions) and
    # pick the physics with the most shared tokens. Discovered when a
    # 72B agent asked SPARTA for 'rarefied_gas_dynamics' and got a raw
    # "unknown physics" error instead of the worked rarefied_flow deck.
    q_tokens = {t for t in re.split(r"[^a-z0-9]+", query_lower) if len(t) >= 5}
    if q_tokens:
        best, best_n = None, 0
        for p in backend.supported_physics():
            p_tokens = {t for t in re.split(r"[^a-z0-9]+", p.name.lower())
                        if len(t) >= 5}
            n = len(q_tokens & p_tokens)
            if n > best_n:
                best, best_n = p.name, n
        if best:
            return best

    # Nothing matched — return original so the caller can
    # surface the "no information found" message with the
    # available-physics list.
    return query_lower


def _list_alternative_solvers(current_solver: str, physics: str) -> str:
    """List other backends that also support this physics (informational).

    This helps the agent know what alternatives exist if the chosen solver
    runs into issues, without being prescriptive about which to use.
    """
    alternatives = []
    for b in all_backends():
        if b.name() == current_solver:
            continue
        status, _ = b.check_availability()
        for p in b.supported_physics():
            if p.name == physics or physics in p.name or p.name in physics:
                # Tag unavailable backends so the LLM knows
                # they would need to be installed first. Hiding
                # them silently (the old available_backends()
                # behaviour) made dune-fem and febio
                # alternatives invisible. (Audit 2026-06-02.)
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                alternatives.append(
                    f"- **{b.display_name()}**{tag}: {p.description}")
                break
    if not alternatives:
        return ""
    return "Other solvers that support this physics:\n" + "\n".join(alternatives)


def _narrow_coupling_by_signal(groups: dict, signal: str) -> dict:
    """Keep the coupling entries whose recorded symptom matches an observed one.

    DELEGATES to `core.pitfall_index` whenever that module is importable. That
    is the canonical matcher for the whole corpus — it folds quoting,
    whitespace and case, stems inflections, and carries a domain synonym table
    — and a second implementation of the same thing that drifted would be worse
    than none, because a symptom query that quietly matches differently for
    coupling than for every backend is a trap rather than a feature.

    The local fallback below exists only for trees where that module is not
    present yet. It mirrors the canonical matcher's TIERS — verbatim in the
    recorded symptom, verbatim anywhere in the entry, every distinctive query
    word present, and a labelled-weak majority overlap — because a fallback
    that is quietly STRICTER is the more dangerous kind of wrong. Measured
    here: a first version tested only for a full token subset, so paraphrased
    queries the canonical matcher surfaces as weak leads ("the interface flux
    balances to roundoff but the result is wrong") came back as "no recorded
    failure mode matches", and for a silent-wrong mode an authoritative-sounding
    absence is exactly the answer that gets an agent to trust a converged run.
    It carries no synonym table, so it can still never claim a match the
    canonical matcher would not.

    RESULTS ARE RANKED, and ties break towards the SHORTER entry. Unranked
    output meant the first entry of the first group won every tie, and the
    longest, most-general entry collects the most token matches — so the one
    entry that mentions everything was answering queries that belonged to its
    neighbours. Preferring the entry with the smaller vocabulary is a
    specificity tie-break, and the canonical matcher has the same tie problem
    (measured: a query of three generic words returned 19 candidates ranked by
    corpus order).

    Never returns an empty result silently: a query that matches nothing comes
    back with a note saying so, because an empty answer reads as "nothing is
    known about that", which is a different and much more dangerous claim.
    """
    try:                                       # canonical path
        from core import pitfall_index
    except ImportError:
        pitfall_index = None

    if pitfall_index is not None:
        result = pitfall_index.narrow(groups, signal=signal)
        kept = {}
        for e in result["entries"]:
            kept.setdefault(e["physics"], []).append(
                f"{e['text']}  <- match: {e.get('match', '?')}")
        if not kept:
            return {"no_match": [
                f"No recorded coupling failure mode matches {signal!r}. That is "
                "informative but not conclusive: it means this symptom is not "
                "catalogued, NOT that your coupling is right. "
                "knowledge(topic='pitfalls', solver='coupling') returns all "
                f"{result['total_available']} entries."]}
        modes = result.get("match_modes") or {}
        note = (f"signal={signal!r}: {result['shown']} of "
                f"{result['total_available']} entries, best match first. For "
                f"the complete set: knowledge(topic='pitfalls', "
                f"solver='coupling')")
        if modes and set(modes) <= {"some_tokens"}:
            note += (". EVERY match below is a partial word overlap, not a "
                     "match on a recorded symptom — treat them as leads to "
                     "read, not as an identification of your failure.")
        kept["_filter"] = [note]
        return kept

    import re as _re

    def _norm(s: str) -> str:
        s = s.lower().replace("’", "'").replace("‘", "'")
        s = s.replace("“", '"').replace("”", '"')
        s = _re.sub(r"[\\'\"`]+", "", s)
        return _re.sub(r"\s+", " ", s).strip()

    _STOP = set("a an the and or but if of in on at to for from with by is are "
                "was were be it its this that these those as not no so than "
                "then there when where which what how why all any both each "
                "more most other some such only same too very can will just "
                "should now use used using you your we our error warning "
                "message output file files line lines code".split())
    def _toks(s: str) -> set[str]:
        return {w for w in _re.findall(r"[a-z_][a-z0-9_]{2,}", s)
                if w not in _STOP}

    q = _norm(signal)
    qt = _toks(q)
    scored: list[tuple[float, int, str, str, str]] = []
    for group, entries in groups.items():
        for text in entries:
            whole = _norm(text)
            sig = whole.split("signal:", 1)[1] if "signal:" in whole else ""
            wt = _toks(whole)
            hit = qt & wt
            frac = len(hit) / len(qt) if qt else 0.0
            if sig and q in sig:
                score, mode = 1.0, "matches recorded symptom"
            elif q in whole:
                score, mode = 0.85, "matches entry text"
            elif qt and frac >= 1.0:
                score, mode = 0.7, "all query terms present"
            elif frac >= 0.6 and len(hit) >= 2:
                # Labelled WEAK, never presented as an identification. Below
                # 0.6, or on a single word, an overlap is a coincidence.
                score, mode = 0.3 + 0.3 * frac, "WEAK: partial term overlap"
            else:
                continue
            # Tie-break on entry vocabulary size: the shorter entry is the more
            # specific one, and without this the longest entry wins every tie.
            scored.append((score, len(wt), group, text, mode))
    scored.sort(key=lambda t: (-t[0], t[1]))
    kept: dict[str, list[str]] = {}
    for _score, _n, group, text, mode in scored:
        kept.setdefault(group, []).append(f"{text}  <- match: {mode}")
    total = sum(len(v) for v in groups.values())
    if not kept:
        return {"no_match": [
            f"No recorded coupling failure mode matches {signal!r}. That is "
            "informative but not conclusive: it means this symptom is not "
            "catalogued, NOT that your coupling is right. "
            f"knowledge(topic='pitfalls', solver='coupling') returns all "
            f"{total} entries."]}
    shown = sum(len(v) for v in kept.values())
    note = (f"signal={signal!r}: {shown} of {total} entries, best match first. "
            f"For the complete set: knowledge(topic='pitfalls', "
            f"solver='coupling')")
    if all(m.startswith("WEAK") for *_ , m in scored):
        note += (". EVERY match below is a partial word overlap, not a match on "
                 "a recorded symptom — treat them as leads to read, not as an "
                 "identification of your failure.")
    kept["_filter"] = [note]
    return kept


def _load_matching_postmortems(solver: str = "", physics: str = "",
                               signal: str = "") -> list[dict]:
    """Load post-mortem JSONs from data/postmortems/, filtered.

    The post-mortems directory is the audit trail behind the pitfall
    DB. Each record explains WHY a pitfall was added — the surface
    symptom that was observed, the root cause, the Table-1 category,
    the exact pitfall entries shipped, and the detection path the
    agent now has. The Open-FEM-Agent paper's §3.2 / §5
    self-correction loop depends on the agent being able to retrieve
    these at planning time.

    Filters (any can be empty, treated as "match all"):
      * solver  — exact match against the post-mortem's `backend`
                  field (case-insensitive). NOT a fuzzy match because
                  the post-mortem's audit value depends on knowing
                  it's about THIS backend, not a similar one.
      * physics — substring match against the `physics` field. A
                  batch post-mortem like
                  "poisson, heat, helmholtz, eigenvalue" matches any
                  of its members.
      * signal  — substring match across each `pitfall_db_entries`
                  string. Useful when the post-execution critic
                  sees a specific error and wants to find the
                  matching post-mortem.

    Returns the post-mortems as parsed dicts. Sorted by `date`
    descending so the most-recent record comes first — typically
    the most-relevant for the current agent session.

    Files under ``data/postmortems/candidates/`` are NOT included
    here. Candidates are the pre-review staging area
    (Open-FEM-Agent §3.2 autonomous-growth path) — promotion to a
    formal post-mortem is a deliberate review step (#46).
    """
    pm_dir = Path(__file__).resolve().parents[2] / "data" / "postmortems"
    if not pm_dir.is_dir():
        return []
    solver_l = solver.lower().strip()
    physics_l = physics.lower().strip()
    signal_l = signal.lower().strip()
    out: list[dict] = []
    for path in pm_dir.glob("*.json"):
        if path.name.startswith("_"):
            # Skip schema / index files.
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if solver_l and str(doc.get("backend", "")).lower() != solver_l:
            continue
        if physics_l and physics_l not in \
                str(doc.get("physics", "")).lower():
            continue
        if signal_l:
            entries = doc.get("pitfall_db_entries", []) or []
            if not any(signal_l in str(e).lower() for e in entries):
                continue
        out.append(doc)
    out.sort(key=lambda d: str(d.get("date", "")), reverse=True)
    return out


def _make_input_snapshot(input_content: str, solver: str = "",
                         extra: dict | None = None) -> dict:
    """Create a sanitised snapshot of simulation input for diff capture.

    Captures structure (length, line count, key patterns) without leaking content.
    """
    import hashlib
    snap = {
        "solver": solver,
        "input_length": len(input_content),
        "input_lines": input_content.count("\n") + 1,
        "input_hash": hashlib.sha256(input_content.encode()).hexdigest()[:12],
    }
    if extra:
        snap.update(extra)
    return snap


def register_consolidated_tools(mcp: FastMCP):
    """Register all consolidated tools — ~12 tools instead of 48."""

    # Session journal — records events for knowledge capture
    from core.session_journal import get_journal as _get_journal

    # ═══════════════════════════════════════════════════════════
    # 1. KNOWLEDGE (replaces 13 separate knowledge tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def knowledge(topic: str, solver: str = "", physics: str = "",
                  signal: str = "") -> str:
        """Get knowledge about solvers, physics, materials, coupling,
        post-mortems, or input formats.

        This is the single entry point for ALL domain knowledge — the
        catalog, the pitfall database, AND the post-mortem record
        store. Wiring post-mortems through this same tool closes the
        self-improvement loop: every prepare_simulation call also
        surfaces the relevant post-mortems so the critic-gate can
        retrieve them at planning time (Open-FEM-Agent §3.2 / §5
        self-correction loop).

        Args:
            topic: What you want to know. Options:
                - "physics" — physics-specific knowledge + matching
                  post-mortems (needs solver + physics)
                - "pitfalls" — all known pitfalls for a solver.
                  `solver='coupling'` is accepted here and is not the
                  name of a backend. It returns the cross-code
                  coupling failure modes — the ten that converge to a
                  confidently wrong answer, the participant-contract
                  errors, the limits of each verification check, and
                  the capability claims that are NEGATIVE. Pass
                  `signal=` with the message you actually saw, which
                  is what the entries are indexed on, or `physics=`
                  as a topic filter.
                - "postmortems" — formal post-mortem records under
                  data/postmortems/*.json, filtered by solver +
                  physics + optional signal pattern. These are the
                  audit-trail entries that record WHY each pitfall
                  exists; the critic-gate should retrieve them when
                  the agent's plan touches the matching (solver,
                  physics) area.
                - "materials" — material catalog for a solver
                - "coupling" — cross-solver coupling knowledge
                - "tsi" — thermo-structural interaction patterns
                - "precice" — preCICE comparison
                - "input_guide" — how to write input files for a solver
                - "solver_guidance" — which solver to use for a physics type
                - "hardware" — parallelism, GPU, and hardware acceleration capabilities
                - "overview" — backend-level reference catalog (element
                  families, mesh types, solver catalogue, unique
                  features). The content under the special "_general"
                  knowledge key — for dealii ~5 KB, fenics / ngsolve /
                  skfem / kratos / dune ~1-2 KB each. Needs solver=...
                - "cross_backend" — collation pitfalls that surface ONLY
                  when porting a problem between two backends (units
                  conventions, Tet10/Hex20 node ordering, 'linear
                  elastic' semantic drift across backends, Dirichlet
                  strong-vs-penalty enforcement, restart file
                  incompatibility, MPI launch idioms). Pass the
                  optional `physics` arg as a topic filter
                  (e.g. 'units', 'mesh', 'bc', 'restart', 'mpi') to
                  narrow the response. These pitfalls belong to no
                  single backend's catalog because they only fire on
                  the delta between two.
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'navier_stokes')
            signal: The error text you actually observed. Paste it raw
                — quoting, case and whitespace differences are folded.
                Filters post-mortem records, and filters the coupling
                failure modes under solver='coupling'. Every result
                says which match mode produced it, so a partial
                word-overlap is labelled rather than presented as an
                identification. No match means the symptom is not
                catalogued; it does NOT mean the setup is right.
        """
        _get_journal().record("knowledge_lookup", "knowledge",
                              solver=solver, physics=physics,
                              notes=f"topic={topic}")
        if topic == "physics" and solver and physics:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            k = backend.get_knowledge(physics)
            if not k:
                return f"No knowledge for '{physics}' in {solver}"
            k = _strip_pitfalls(k)
            result = json.dumps(k, indent=2, default=str)
            # Append real test file references
            from tools.knowledge import _find_reference_test_files
            ref = _find_reference_test_files(solver, physics)
            if ref:
                result += f"\n\n{ref}"
            # Append post-mortem BREADCRUMBS (ids only) — not full
            # records — at plan time. Rationale (senior-AI-scientist
            # critic, 2026-05-31): full post-mortems include
            # surface_symptom / root_cause / agent_detection_after_fix,
            # which are diagnostic fields for human review (#46), not
            # pre-execution guidance. Auto-including them at plan
            # time produces linear token bloat in N_postmortems and
            # competes with the catalog for the agent's attention.
            # The pitfall_db_entries the catalog already exposes ARE
            # the pre-execution actionable content; the full
            # post-mortem belongs to the post-execution critic when
            # it has a Signal: to match. Agent can fetch the full
            # record explicitly via
            # `knowledge(topic="postmortems", solver=..., signal=...)`.
            postmortems = ([] if _ABLATE_PITFALLS
                           else _load_matching_postmortems(solver, physics, ""))
            if postmortems:
                breadcrumbs = [
                    {"id": pm.get("id", "?"),
                     "categories": pm.get("categories", []),
                     "date": pm.get("date", "")}
                    for pm in postmortems
                ]
                result += (
                    f"\n\n## Post-mortem breadcrumbs "
                    f"({len(postmortems)} record"
                    f"{'' if len(postmortems) == 1 else 's'} — "
                    f"fetch full records via knowledge"
                    f"(topic='postmortems', solver=..., signal=...)"
                    f" when a post-execution Signal needs lookup):\n"
                    + json.dumps(breadcrumbs, indent=2))
            return result

        elif topic == "postmortems":
            if _ABLATE_PITFALLS:
                return ("No post-mortems found. data/postmortems/*.json is "
                        "the canonical store; absence here means the failure "
                        "mode has not yet been audited.")
            postmortems = _load_matching_postmortems(solver, physics, signal)
            if not postmortems:
                what = ", ".join(
                    f"{k}={v!r}" for k, v in
                    {"solver": solver, "physics": physics,
                     "signal": signal}.items() if v)
                return (f"No post-mortems found"
                        f"{' for ' + what if what else ''}. "
                        f"data/postmortems/*.json is the canonical "
                        f"store; absence here means the failure mode "
                        f"has not yet been audited.")
            return json.dumps(postmortems, indent=2)

        elif topic == "pitfalls" and solver:
            # Ablation: when OFA_DISABLE_PITFALLS=1, refuse to surface
            # pitfalls so the agent has no shortcut to known-bug knowledge.
            if _ABLATE_PITFALLS:
                return f"No pitfalls available for {solver}"
            # COUPLING IS NOT A BACKEND, and that is exactly why it had no
            # pitfalls surface: `get_backend('coupling')` returns None, so this
            # branch fell straight through to "No pitfalls found for coupling"
            # while ~145 kB of coupling knowledge sat behind topic='coupling'
            # with no [Category] tag and no Signal: clause anywhere in it. An
            # agent whose coupling had just failed could not look its symptom
            # up — at the exact moment it most needed the knowledge, symptom
            # lookup returned nothing. Served here in the same shape every
            # backend uses ({group: [entry, ...]}), so the narrowing layer
            # applies to it unchanged when it lands.
            if solver.strip().lower() in ("coupling", "couple", "coupled"):
                from backends.coupling import get_coupling_pitfalls
                entries = get_coupling_pitfalls(physics or None)
                if signal:
                    entries = _narrow_coupling_by_signal(entries, signal)
                return json.dumps(entries, indent=2)
            # Backend is the source of truth for pitfalls (Table-1
            # promoted, post-execution-critic-actionable). The
            # deep_knowledge fallback was inverted historically —
            # it returned prose entries even for backends whose
            # generators had been Table-1 promoted, breaking the
            # alignment between prepare_simulation and discover.
            # Now backend is consulted FIRST; deep_knowledge is
            # only used as a supplement for physics the backend
            # does not enumerate (rare in practice).
            backend = get_backend(solver)
            all_pitfalls = {}
            if backend:
                for p in backend.supported_physics():
                    k = backend.get_knowledge(p.name)
                    if k and "pitfalls" in k:
                        all_pitfalls[p.name] = k["pitfalls"]
            try:
                from tools.deep_knowledge import _4C_KNOWLEDGE, _FENICS_KNOWLEDGE
                dicts = {"fourc": _4C_KNOWLEDGE, "4c": _4C_KNOWLEDGE,
                         "fenics": _FENICS_KNOWLEDGE, "fenicsx": _FENICS_KNOWLEDGE}
                d = dicts.get(solver.lower(), {})
                for k, v in d.items():
                    if (isinstance(v, dict) and "pitfalls" in v
                            and k not in all_pitfalls):
                        all_pitfalls[k] = v["pitfalls"]
            except ImportError:
                pass
            if not backend:
                if all_pitfalls:
                    return json.dumps(all_pitfalls, indent=2)
            if backend:
                # Also include general input-format pitfalls (e.g., ExodusII
                # block IDs, FUNCT syntax, shared-node NUMDOF conflict)
                general_k = backend.get_knowledge("input_format")
                if isinstance(general_k, dict):
                    gp = general_k.get("general_pitfalls")
                    if gp:
                        all_pitfalls["general_input_format"] = gp
                    et = general_k.get("element_type_per_physics")
                    if et:
                        all_pitfalls["element_types"] = et
                # Include community-contributed knowledge
                community = _load_community_knowledge(solver)
                if community:
                    all_pitfalls["community_contributed"] = [
                        {"title": c["title"], "description": c.get("description", ""),
                         "category": c.get("category", ""), "confidence": c.get("confidence", 0)}
                        for c in community
                    ]
                return json.dumps(all_pitfalls, indent=2)
            return f"No pitfalls found for {solver}"

        elif topic == "materials" and solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            materials = {}
            for p in backend.supported_physics():
                k = backend.get_knowledge(p.name)
                if k and "materials" in k:
                    materials[p.name] = k["materials"]
            return json.dumps(materials, indent=2) if materials else f"No material catalog for {solver}"

        elif topic == "overview" and solver:
            # Surface the backend-level "_general" reference catalog
            # (element families, mesh types, solver catalogue, unique
            # features). Discovered 2026-06-02: get_knowledge('_general')
            # returns substantive reference content for 6 of 8 backends
            # (dealii 5.2 KB; fenics/ngsolve/skfem/kratos/dune 1-2 KB
            # each) but was NOT exposed via any `knowledge(topic=...)`
            # surface — LLMs had no way to discover it existed.
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            general = backend.get_knowledge("_general")
            if not isinstance(general, dict) or not general or "error" in general:
                return (f"No backend-level overview catalog for "
                        f"{solver} (get_knowledge('_general') is "
                        "empty or returned an error).")
            return json.dumps({solver: general}, indent=2)

        elif topic == "coupling":
            from tools.knowledge import register_knowledge_tools
            # Return coupling knowledge directly
            return _get_coupling_knowledge()

        elif topic == "tsi":
            return _get_tsi_knowledge()

        elif topic == "precice":
            return _get_precice_knowledge()

        elif topic == "input_guide" and solver:
            from tools.examples_search import (
                _4C_INPUT_GUIDE, _FENICS_INPUT_GUIDE, _DEALII_INPUT_GUIDE,
                _FEBIO_INPUT_GUIDE, _DUNE_INPUT_GUIDE,
            )
            guides = {"fourc": _4C_INPUT_GUIDE, "4c": _4C_INPUT_GUIDE,
                      "fenics": _FENICS_INPUT_GUIDE, "dealii": _DEALII_INPUT_GUIDE,
                      "febio": _FEBIO_INPUT_GUIDE,
                      "dune": _DUNE_INPUT_GUIDE, "dune-fem": _DUNE_INPUT_GUIDE,
                      "dunefem": _DUNE_INPUT_GUIDE}
            return guides.get(solver.lower(), f"No input guide for {solver}")

        elif topic == "solver_guidance" and physics:
            # Show ALL registered backends so the LLM can learn
            # which solvers offer the physics in principle —
            # even when not installed yet — and decide whether
            # to install one. Tag unavailable backends so the
            # LLM does not try to run_simulation on them.
            # (Audit 2026-06-02; same hide-unavailable bug as
            # discover('list').)
            results = {}
            for b in all_backends():
                for p in b.supported_physics():
                    if p.name == physics:
                        status, _ = b.check_availability()
                        key = (b.display_name() if status.value == "available"
                               else f"{b.display_name()} [{status.value}]")
                        results[key] = {
                            "variants": p.template_variants,
                            "elements": p.element_types,
                            "dims": p.spatial_dims,
                        }
            return json.dumps(results, indent=2) if results else f"No solver supports '{physics}'"

        elif topic == "hardware":
            hw = {
                "FEniCSx (dolfinx)": {
                    "parallelism": "MPI (first-class, domain decomposition via PETSc)",
                    "gpu": "No native GPU. PETSc can use GPU backends (CUDA/HIP) for linear algebra if compiled with Kokkos/CUDA support, but this is not standard.",
                    "threading": "Limited — PETSc threading for assembly",
                    "typical_scale": "Millions of DOFs on HPC clusters",
                },
                "deal.II": {
                    "parallelism": "MPI (p4est for distributed meshes) + threading (TBB/std::thread)",
                    "gpu": "Yes — matrix-free GPU kernels via CUDA and portable backends. GPU support for matrix-free operators is a key feature (step-64 tutorial).",
                    "threading": "SharedMemory::TBB or std::thread for assembly",
                    "typical_scale": "Billions of DOFs demonstrated (matrix-free, GPU)",
                },
                "4C Multiphysics": {
                    "parallelism": "MPI (domain decomposition) + OpenMP threading",
                    "gpu": "No GPU for linear algebra (Epetra-based, CPU-only). Optional ArborX (Kokkos) for GPU-accelerated geometric search only. Tpetra (GPU-capable) not yet integrated.",
                    "threading": "OpenMP (set OMP_NUM_THREADS)",
                    "typical_scale": "Millions of DOFs on MPI clusters",
                    "note": "Trilinos 16.2.0 is the last supported version due to Epetra dependency",
                },
                "NGSolve": {
                    "parallelism": "MPI (via NGSolve's own parallel framework) + shared-memory task parallelism",
                    "gpu": "Experimental CUDA support for some operations. Not production-ready for most users.",
                    "threading": "Task-based parallelism (Netgen's built-in scheduler)",
                    "typical_scale": "Millions of DOFs",
                },
                "scikit-fem": {
                    "parallelism": "Serial only (no MPI). NumPy/SciPy vectorisation for assembly.",
                    "gpu": "No GPU support. Pure Python/NumPy.",
                    "threading": "NumPy BLAS threading only",
                    "typical_scale": "Tens of thousands of DOFs (prototyping)",
                },
                "Kratos Multiphysics": {
                    "parallelism": "MPI (Trilinos-based) + OpenMP for shared memory",
                    "gpu": "Limited — some GPU acceleration via Trilinos/Kokkos for linear algebra. Not all applications support it.",
                    "threading": "OpenMP",
                    "typical_scale": "Millions of DOFs",
                },
                "DUNE-fem": {
                    "parallelism": "MPI (DUNE grid parallelism via ALUGrid/YaspGrid)",
                    "gpu": "No native GPU support in DUNE-fem. DUNE-copasi has experimental GPU work.",
                    "threading": "Limited",
                    "typical_scale": "Moderate (research scale)",
                },
                "FEBio": {
                    "parallelism": (
                        "Shared-memory only (OpenMP). No MPI domain "
                        "decomposition: a single FEBio process drives "
                        "the whole simulation. Multi-physics coupling "
                        "(biphasic, multiphasic, fluid-solid mixture) "
                        "is monolithic in the solver, not via "
                        "subdomain decomposition."),
                    "gpu": (
                        "No GPU support. FEBio's linear-algebra "
                        "back-end is CPU only (PARDISO / MUMPS / "
                        "Skyline). GPU acceleration is on the wishlist "
                        "but not implemented as of FEBio 4.x."),
                    "threading": (
                        "OpenMP across element assembly + PARDISO's "
                        "internal threading. Set OMP_NUM_THREADS for "
                        "assembly; the linear solver uses its own "
                        "OMP_NUM_THREADS or MKL_NUM_THREADS pool."),
                    "typical_scale": (
                        "Hundreds of thousands of DOFs on a workstation; "
                        "millions are routinely run but FEBio targets "
                        "biomechanical models (single bones, soft "
                        "tissue, biphasic cartilage) rather than HPC "
                        "scale."),
                    "note": (
                        "FEBio's strength is biomechanics-specific "
                        "physics (biphasic / multiphasic mixtures, "
                        "active contraction, fiber materials, "
                        "growth-remodeling). It is NOT a general-"
                        "purpose FEM code; do not pick it for "
                        "Navier-Stokes / electromagnetics / "
                        "geomechanics."),
                },
                "SPARTA (DSMC)": {
                    "parallelism": (
                        "MPI-first: the simulation grid is spatially "
                        "decomposed across ranks; SPARTA scales to "
                        "thousands of cores and is a production HPC DSMC "
                        "code (run: mpirun -np N spa_mpi -in <script>)."),
                    "gpu": (
                        "Yes — via the Kokkos package (build spa_kokkos): "
                        "CUDA (NVIDIA) and HIP (AMD) backends run the "
                        "particle move/collide/surface kernels on GPU. "
                        "Enable with '-k on g 1 -sf kk' package flags."),
                    "threading": (
                        "OpenMP or Kokkos (OpenMP) for shared-memory "
                        "parallelism; typically combined with MPI (MPI+X)."),
                    "typical_scale": (
                        "Billions of simulator particles on HPC clusters; "
                        "particle count (fnum) trades statistical noise "
                        "against cost, not DOFs."),
                    "note": (
                        "SPARTA is a Direct Simulation Monte Carlo (DSMC) "
                        "rarefied-gas / particle code, NOT a FEM solver — "
                        "reachable through OASiS coupling (e.g. a continuum "
                        "FEM thermal wall coupled to DSMC gas)."),
                },
            }
            if solver:
                key_map = {"fourc": "4C Multiphysics", "4c": "4C Multiphysics",
                           "fenics": "FEniCSx (dolfinx)", "fenicsx": "FEniCSx (dolfinx)",
                           "dealii": "deal.II", "deal.ii": "deal.II",
                           "ngsolve": "NGSolve", "skfem": "scikit-fem", "scikit-fem": "scikit-fem",
                           "kratos": "Kratos Multiphysics", "dune": "DUNE-fem", "dune-fem": "DUNE-fem",
                           "febio": "FEBio", "sparta": "SPARTA (DSMC)"}
                name = key_map.get(solver.lower(), solver)
                if name in hw:
                    return json.dumps({name: hw[name]}, indent=2)
                return f"No hardware info for {solver}"
            return json.dumps(hw, indent=2)

        elif topic == "cross_backend":
            # Cross-backend collation pitfalls — failures that live
            # in the delta between two backends both claiming to
            # solve the same problem. See src/backends/_cross.py
            # for content + rationale. The `physics` arg here is
            # repurposed as a topic filter (e.g. 'units', 'mesh',
            # 'bc', 'restart', 'mpi') to narrow the response.
            if _ABLATE_PITFALLS:
                return "No cross-backend collation entries available."
            from backends._cross import get_cross_backend_pitfalls
            result = get_cross_backend_pitfalls(physics or signal or None)
            return json.dumps(result, indent=2)

        else:
            # Topics list must match the docstring + dispatch
            # branches. Audit 2026-06-01: 'postmortems' was
            # documented in the docstring and implemented at
            # line 326 but missing from this usage hint, so
            # LLMs hitting an invalid topic never learned that
            # postmortems exists. (Same drift class as
            # session_insights' missing 'ingest'.)
            return (
                "Usage: knowledge(topic, solver, physics, signal='')\n"
                "Topics: physics, pitfalls, postmortems, materials, "
                "overview, coupling, tsi, precice, input_guide, "
                "solver_guidance, hardware, cross_backend"
            )

    # ═══════════════════════════════════════════════════════════
    # 2. DISCOVER (replaces 6 discovery tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def discover(query: str = "list", solver: str = "") -> str:
        """Discover available solvers and their capabilities.

        Args:
            query: What to discover. Options:
                - "list" — list all solvers with status
                - "physics" — list all physics types per solver
                - "capabilities" — full capabilities matrix
                - "recommend" — recommend solver for a physics (set solver= to physics name)
            solver: Filter by solver name, or physics name for "recommend"
        """
        if query == "list":
            # Show ALL registered backends, not only the
            # installed ones. The MCP server instructions
            # advertise 8 backends; if discover('list') hides
            # the ones the user has not installed yet, an LLM
            # asking for (say) DUNE-fem or FEBio gets no entry,
            # no status, and no install hint — total dead end.
            # Surface every backend with its actual availability
            # status and the install hint that
            # check_availability() returns. (Audit 2026-06-02.)
            lines = []
            for b in all_backends():
                status, msg = b.check_availability()
                core = (f"- **{b.display_name()}** ({b.name()}): "
                        f"{status.value} — "
                        f"{b.input_format().value} input")
                if status.value != "available" and msg:
                    # Inline a ONE-LINE reason/hint (not a raw traceback) so the
                    # LLM does not have to call a second tool and the list stays
                    # readable.
                    core += f"\n  *{_short_reason(msg)}*"
                lines.append(core)
            return "\n".join(lines) if lines else "No backends registered."

        elif query == "physics":
            # Show physics for ALL registered backends (same
            # rationale as discover('list')) so an LLM can
            # learn what dune-fem or febio offer even before
            # installing them. Tag unavailable backends with
            # their status so the LLM does not call
            # run_simulation against a backend that will
            # error out on availability. (Audit 2026-06-02.)
            lines = []
            backends = [get_backend(solver)] if solver else all_backends()
            backends = [b for b in backends if b]
            for b in backends:
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                lines.append(f"## {b.display_name()}{tag}")
                for p in b.supported_physics():
                    lines.append(f"- **{p.name}**: {p.description} (variants: {', '.join(p.template_variants)})")
                lines.append("")
            return "\n".join(lines)

        elif query == "capabilities":
            # Show ALL registered backends (see discover('list')
            # rationale above) so an LLM sees the full
            # capabilities matrix including not-yet-installed
            # backends. (Audit 2026-06-02.)
            lines = ["| Solver | Physics Count | Input | Status |",
                     "|--------|--------------|-------|--------|"]
            for b in all_backends():
                status, _ = b.check_availability()
                lines.append(f"| {b.display_name()} | {len(b.supported_physics())} | {b.input_format().value} | {status.value} |")
            return "\n".join(lines)

        elif query == "recommend":
            physics = solver  # in this case solver param holds the physics name
            # Empty / whitespace-only physics matches every
            # backend's first physics (substring-of-everything).
            # Same class of bug as the empty-physics
            # prepare_simulation match — reject it explicitly so
            # the LLM gets a clear usage hint instead of a fake
            # "all backends recommend this" result. Audit
            # 2026-06-01.
            if not physics or not physics.strip():
                return ("Empty physics name. Pass the physics "
                        "as the 'solver' parameter, e.g. "
                        "discover(query='recommend', "
                        "solver='poisson').")
            # Route the physics query through the canonical
            # fuzzy resolver per backend so short shorthands
            # ('ns', 'em', 'cfd', ...) hit the synonym map
            # before a loose substring scan. The raw
            # substring-on-name-or-description recommendation
            # silently matched 'ns' to heat in fenics; 'em' to
            # eigenvalue in fenics; 'pd' to nonlinear_pde in
            # fenics. (Audit 2026-06-02; same drift class as
            # the prepare_simulation fix.)
            #
            # Iterate ALL registered backends, not just the
            # installed ones, so the recommendation includes
            # backends the user has not installed yet. Tag
            # unavailable backends inline so the LLM knows
            # they need an install step. (Audit 2026-06-02;
            # same hide-unavailable bug as discover('list').)
            results = []
            for b in all_backends():
                matched = _fuzzy_match_physics(b, physics)
                if not matched:
                    continue
                for p in b.supported_physics():
                    if p.name == matched:
                        status, _ = b.check_availability()
                        tag = "" if status.value == "available" else f" *[{status.value}]*"
                        results.append(
                            f"- **{b.display_name()}**{tag}: {p.description}")
                        break
            return "\n".join(results) if results else f"No solver found for '{physics}'"

        return "Usage: discover(query='list'|'physics'|'capabilities'|'recommend', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 3. EXAMPLES (replaces 7 example/search tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def examples(keyword: str, solver: str = "fourc", action: str = "search",
                 max_results: int = 3, variant: str = "") -> str:
        """Find and retrieve example input files from solver test suites.

        IMPORTANT: Always call this before writing new input files to study
        real, validated configurations.

        Args:
            keyword: Search term (e.g. 'peridynamic', 'fsi', 'poisson', 'heat')
            solver: Backend name (default: 'fourc')
            action: What to do. Options:
                - "search" — find matching test files with content preview
                - "template" — get a generated template for this physics
                - "tutorials" — list available tutorials
            max_results: Maximum results (default 3)
        """
        if action == "search":
            # Empty / whitespace-only keyword matches every
            # filename (substring-of-everything) and silently
            # returns the first few random files in the test
            # tree. Surface a usage hint instead. Audit
            # 2026-06-01 (mirror of the empty-physics fix in
            # prepare_simulation).
            if not keyword or not keyword.strip():
                return ("Empty keyword. Provide a substring "
                        "to match, e.g. 'poisson', 'fluid', "
                        "'contact'.")
            from tools.examples_search import register_example_tools
            # Shared discovery with prepare_simulation —
            # discover_test_dirs returns local demo paths for all
            # backends; resolve_search_keywords applies the same
            # alias map (ngsolve hyperelasticity -> nonlin,
            # fenics navier_stokes -> navier-stokes, ...) so the
            # two LLM-facing tools surface the same content for
            # the same (solver, keyword) pair. Audit 2026-06-01.
            from tools.knowledge import (discover_test_dirs,
                                         resolve_search_keywords)
            results = []
            test_dirs = discover_test_dirs()
            solver_key = solver.lower()
            test_dir = test_dirs.get(solver_key)
            ext = "*.4C.yaml" if solver_key in ("fourc", "4c") else "*.cc" if solver_key == "dealii" else "*.py"

            if test_dir and test_dir.is_dir():
                # Apply solver-aware aliases on top of the raw
                # keyword (the LLM may have used the catalog
                # physics name verbatim, e.g. 'hyperelasticity'
                # — which doesn't match NGSolve's nonlin.py
                # demo without aliasing).
                kw_candidates = list(dict.fromkeys(
                    [keyword] + resolve_search_keywords(solver, keyword)))
                seen: set = set()
                for kw in kw_candidates:
                    for f in sorted(test_dir.rglob(ext)):
                        if kw.lower() not in f.name.lower():
                            continue
                        if f in seen:
                            continue
                        seen.add(f)
                        try:
                            content = f.read_text()[:5000]
                            rel = f.relative_to(test_dir)
                            results.append(f"### `{rel}`\n```\n{content}\n```\n")
                        except Exception:
                            pass
                        if len(results) >= max_results:
                            break
                    if len(results) >= max_results:
                        break

            # Also search templates. Route the keyword through the
            # canonical fuzzy resolver so short shorthands ('ns',
            # 'em', 'cfd', ...) resolve to the right physics via
            # the synonym map BEFORE a loose substring scan. The
            # old code did a raw substring match on name OR
            # description; keyword='ns' matched heat /
            # thermal_structural / reaction_diffusion /
            # multiphase / time_dependent_heat (all contain "ns"
            # somewhere) — five wrong templates and never a
            # navier_stokes one. (Audit 2026-06-02; same drift
            # class as the prepare_simulation fix.)
            #
            # Same 12000-char limit as prepare_simulation — the
            # harder Layer F templates (ngsolve hdivdiv /
            # nonlinear_elasticity, fenics navier_stokes) exceed
            # 3000 chars and lose their solver/output blocks if
            # truncated lower. (Audit 2026-06-01.)
            backend = get_backend(solver)
            if backend:
                EX_TEMPLATE_LIMIT = 12000
                matched = _fuzzy_match_physics(backend, keyword)
                for p in backend.supported_physics():
                    if p.name == matched:
                        _sel, _note = _select_template_variant(
                            keyword, list(p.template_variants))
                        for v in ([_sel] if _sel else []):
                            try:
                                content = backend.generate_input(p.name, v, {})
                                truncated = len(content) > EX_TEMPLATE_LIMIT
                                body = content[:EX_TEMPLATE_LIMIT]
                                suffix = (f"\n... [truncated {len(content) - EX_TEMPLATE_LIMIT} chars]"
                                          if truncated else "")
                                _n = f"{_note}\n\n" if _note else ""
                                results.append(f"### Template: `{p.name}/{v}`\n{_n}```\n{body}{suffix}\n```\n")
                            except Exception as exc:
                                # Same rationale as the
                                # prepare_simulation generator-
                                # failure surfacing: a silent
                                # except: pass made
                                # examples('search') return a
                                # "no template, no error" reply
                                # for any catalog regression.
                                # Now the failure is visible.
                                # (Audit 2026-06-02.)
                                results.append(
                                    f"### Template: `{p.name}/{v}`\n"
                                    f"⚠ Template generation FAILED: "
                                    f"`{type(exc).__name__}: {exc}`\n")
                        break

            if not results:
                return f"No examples found for '{keyword}' in {solver}"
            return f"## {len(results)} example(s) for '{keyword}' from {solver}\n\n" + "\n---\n".join(results)

        elif action == "template":
            if not keyword or not keyword.strip():
                return ("Empty keyword. Provide a physics name "
                        "(or substring), e.g. 'poisson', 'fluid', "
                        "'contact'.")
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            # Route the keyword through the canonical fuzzy
            # resolver so short shorthands ('ns' -> navier_stokes,
            # 'em' -> maxwell, ...) route via the synonym map
            # first. The raw substring path here matched 'ns'
            # against 'transient_heat' and never against
            # 'navier_stokes' (no adjacent 'ns' substring in
            # 'navier_stokes' itself). (Audit 2026-06-02.)
            matched = _fuzzy_match_physics(backend, keyword)
            for p in backend.supported_physics():
                if p.name == matched:
                    avail = list(p.template_variants)
                    if variant and variant not in avail:
                        return (f"No variant `{variant}` for "
                                f"`{matched}` in {solver}. Available: "
                                + (", ".join(avail) if avail else "none"))
                    if variant:
                        chosen, note = variant, ""
                    else:
                        chosen, note = _select_template_variant(keyword, avail)
                        chosen = chosen or "2d"
                    try:
                        content = backend.generate_input(p.name, chosen, {})
                        fmt = detect_template_language(content, backend.input_format().value)
                        head = f"Template `{matched}/{chosen}`. {note}\n\n" if note else ""
                        return f"{head}```{fmt}\n{content}\n```"
                    except Exception as e:
                        return f"Error generating template: {e}"
            return f"No template for '{keyword}' in {solver}"

        elif action == "tutorials":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            lines = [f"## {backend.display_name()} Templates\n"]
            for p in backend.supported_physics():
                lines.append(f"- **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            return "\n".join(lines)

        return "Usage: examples(keyword, solver, action='search'|'template'|'tutorials')"

    # ═══════════════════════════════════════════════════════════
    # 4. SIMULATE (replaces run_simulation + run_with_generator)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def submit_critic_review(solver: str, findings: str,
                                   setup: str = "", coupling_args: str = "",
                                   ttl_s: float = 3600.0) -> str:
        """Put an independent critic's review of a setup ON RECORD, so a run of
        that setup can be verified.

        OASiS's critic requirement is enforced, not requested. The run and
        coupling tools do not take your word for it: they look up whether THIS
        server holds a review of the EXACT setup being executed. Passing
        critic_approved=True without a matching review here leaves the result
        NOT VERIFIED, whatever else the run does.

        The workflow is: spawn a sub-agent as an independent critic; have it
        challenge the parameters, units, discretisation, problem statement and
        boundary conditions, and cross-check against literature and benchmarks;
        then submit what it actually found here; then run.

        The review is bound to the setup by digest, so a setup edited after
        review no longer matches and must be reviewed again. That is deliberate:
        reviewing a clean deck and running a different one is the obvious way to
        defeat a critic requirement, and it is the route this closes.

        This server cannot judge whether a critique was any GOOD — it is not an
        oracle for review quality. It enforces that a substantive review of this
        setup exists and is auditable, and refuses a review too short to have
        said anything.

        Args:
            solver: the backend the run will use. For `couple` pass "couple",
                for `couple_precice` pass "couple_precice", and for the legacy
                `coupled_solve` pass "<solver_a>-><solver_b>".
            findings: what the critic actually checked and concluded. Substance
                is required; an empty approval is indistinguishable from no
                review and is refused.
            setup: for run_simulation / run_with_generator /
                verify_mesh_independence — the EXACT deck text you will run
                (input_content, generator_script, or input_template).
            coupling_args: for the coupling tools instead of `setup` — a JSON
                object of the arguments you will pass. Keys per tool:
                coupled_solve: problem, solver_a, solver_b, nx, ny, max_iter,
                tol, relaxation, params; couple: participants, max_iter, tol,
                accelerator, theta, monolithic, probe; couple_precice:
                participants, data, exchanges, scheme, dimensions, max_time,
                time_window, max_iterations, convergence_tol, relaxation,
                mapping. Pass EVERY key for the tool you will call, with the
                values you will call it with — a missing or different key is a
                different setup and the run will come back NOT VERIFIED. For
                `couple` the CONTENTS of each participant's script are part of
                the setup too, so the scripts must already be written when the
                review is submitted, and editing one afterwards invalidates it.
            ttl_s: how long the review stays valid (default 1 hour).

        Returns: JSON with a `critic_token`. Passing it to run_simulation or
            run_with_generator makes the review single-use and binds it to that
            job; omitting it still works, since the deck is matched by digest.
        """
        if bool(setup) == bool(coupling_args):
            return json.dumps({
                "accepted": False,
                "error": "pass exactly one of `setup` (a deck) or "
                         "`coupling_args` (a JSON object of coupling "
                         "arguments), so the review binds to one setup."},
                indent=2)
        if coupling_args:
            try:
                parsed = json.loads(coupling_args)
            except json.JSONDecodeError as exc:
                return json.dumps({"accepted": False,
                                   "error": f"coupling_args is not JSON: {exc}"},
                                  indent=2)
            if not isinstance(parsed, dict):
                return json.dumps({
                    "accepted": False,
                    "error": "coupling_args must be a JSON OBJECT of the "
                             "arguments you will pass."}, indent=2)
            # Through the same single definition the run tools use, so the
            # participant-script fingerprints are part of both digests.
            setup_text = _coupling_setup_text(**parsed)
        else:
            setup_text = setup
        try:
            rec = _CRITIC_REGISTRY.submit_review(
                solver=solver, findings=findings,
                digest=setup_digest(solver, setup_text), ttl_s=ttl_s)
        except CriticGateError as exc:
            return json.dumps({"accepted": False, "error": str(exc)}, indent=2)
        _get_journal().record("critic_review", "submit_critic_review",
                              solver=solver,
                              input_snapshot=_make_input_snapshot(
                                  setup_text, solver, {"type": "critic_review"}))
        return json.dumps({
            "accepted": True,
            "critic_token": rec.token,
            "solver": solver,
            "valid_for_s": ttl_s,
            "note": ("This review is on record for this exact setup. Editing "
                     "the setup invalidates it. Pass critic_token to the run "
                     "tool to make the review single-use and bound to that "
                     "job."),
        }, indent=2)

    @mcp.tool()
    async def run_with_generator(solver: str, generator_script: str,
                                  job_name: str = "", np: int = 1,
                                  critic_approved: bool = False,
                                  critic_token: str = "",
                                  verify_pde: str = "",
                                  ctx: Context = None) -> str:
        """Run a generator script that creates an input file, then execute the solver.

        Use this for solvers that need a COMPILED binary or separate input files:
        - 4C: generator creates .4C.yaml + mesh, then 4C binary runs on them
        - deal.II: generator creates main.cpp, then cmake + make + ./fem_solve
        - Kratos (with real binary): generator creates ProjectParameters.json +
          .mdpa + MainKratos.py, then Kratos Python runs MainKratos.py

        DO NOT use this for:
        - FEniCS, NGSolve, scikit-fem, DUNE-fem: use run_simulation() instead
        - Kratos manual-assembly scripts (numpy/scipy): use run_simulation()
          since those are standalone Python scripts, not input-file generators

        The generator script runs in the server's Python. It must produce an
        input file matching one of: *.4C.yaml, *.yaml, input.*, solve.py,
        MainKratos.py

        Args:
            solver: Backend name (fourc, dealii, kratos)
            generator_script: Python script that creates the input file
            job_name: Optional job directory name
            np: MPI processes (default 1)
            critic_approved: recorded, not trusted. The result is verified only
                if a critic review of THIS generator_script is on record — call
                submit_critic_review first.
            critic_token: optional token from submit_critic_review; makes the
                review single-use and binds it to this job.
            verify_pde: optional JSON declaring the problem being solved, so
                OASiS can check the result actually SATISFIES it rather than
                merely looking well-formed. Example:
                {"operator": "diffusion",
                 "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                 "coefficient": "1.0", "dim": 2, "domain_measure": 1.0}
                `source` and `coefficient` are numeric expressions in x, y, z.
                A field that does not satisfy the declared equations is NOT
                VERIFIED, whatever else the run did. Currently covers scalar
                diffusion on simplex meshes; anything else is reported as not
                checked, never as passed.
        """
        import subprocess
        import sys

        _journal = _get_journal()
        _snap = _make_input_snapshot(generator_script, solver, {"type": "generator"})
        _journal.record("tool_call", "run_with_generator", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {_short_reason(msg)}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_gen_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        gen_path = work_dir / "generate_input.py"
        gen_path.write_text(generator_script)

        python = sys.executable
        gen_result = subprocess.run(
            [python, str(gen_path)],
            capture_output=True, text=True,
            cwd=str(work_dir),
        )

        if gen_result.returncode != 0:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=f"Generator failed: {gen_result.stderr[-200:]}",
                            input_snapshot=_snap)
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": gen_result.stderr[-500:],
                "work_dir": str(work_dir),
            }, indent=2)

        from core.backend import find_generated_input
        input_file = find_generated_input(work_dir, backend)

        if not input_file:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message="Generator did not produce an input file",
                            input_snapshot=_snap)
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": "Generator did not produce an input file",
                "work_dir": str(work_dir),
                "files": sorted(f.name for f in work_dir.iterdir())[:50],
            }, indent=2)

        input_content = input_file.read_text()
        # Update snapshot with the generated input's shape
        _snap_run = _make_input_snapshot(input_content, solver,
                                         {"type": "generated_input", "input_file": input_file.name})
        from core.backend import JobHandle
        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_with_generator", solver=solver,
                            error_message=job.error[:300],
                            input_snapshot=_snap_run)
        else:
            _journal.record("tool_success", "run_with_generator", solver=solver,
                            input_snapshot=_snap_run)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "input_file": input_file.name,
        }
        out_files = []
        if job.error:
            result["error"] = job.error[:500]
        nonfinite = []
        _stdout_text = ""
        if job.status == "completed":
            out_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in out_files]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                _stdout_text = stdout_log.read_text()
                result["stdout_tail"] = (_stdout_text[-2000:]
                                         if len(_stdout_text) > 2000 else _stdout_text)
            # Attestation: a process that exits 0 but produces NO output files is
            # NOT a verified solve — the canonical silent failure. Flag it loudly.
            if not out_files:
                result["status"] = "completed_unverified"
            else:
                # ... and a file full of NaN/Inf is a fabricated-looking result.
                nonfinite = check_result_files_finite(out_files)
            # Also scan the headline numbers (results_summary.json + stdout).
            nonfinite += check_summary_finite(work_dir, _stdout_text)
            # Structural defects in the solver's own data output. The scans
            # above look at headline numbers and mesh files; a FIELD that is
            # mostly NaN, or a wholly degenerate mesh, passed both and was
            # stamped verified. (Anti-fabrication gate.)
            nonfinite += inspect_result_artefacts(out_files)
            if nonfinite:
                result.setdefault("validation", []).extend(nonfinite)
            # The 'finiteness not asserted' honesty note is a coverage gap,
            # not evidence of a bad number — it must not flip the verdict to
            # 'non-finite values' (FEBio .xplt / .bp-without-adios2 runs).
            nonfinite = [x for x in nonfinite
                         if not x.startswith("finiteness not asserted")]
            # OASiS computes the run's headline numbers from the run's own
            # data, so the agent never has to assert one of its own.
            if out_files:
                result["oasis_computed"] = _attest_run_quantities(
                    work_dir, job.job_id)
                # …and, if the run declared what it is solving, whether that
                # data satisfies those equations at all.
                if verify_pde:
                    result["residual_check"] = _check_declared_pde(
                        verify_pde, out_files)
        # Verification gate: bind the verdict to run evidence (attestation).
        if job.error:
            reason = "the solver run errored, so no number is backed by a valid run"
        elif job.status != "completed":
            reason = f"the run did not complete (status={job.status})"
        elif not out_files:
            reason = ("the process exited cleanly but produced NO output files, "
                      "so no reported number is backed by run evidence")
        elif nonfinite:
            reason = ("the result contains non-finite (NaN/Inf) values, so it is "
                      "numerically invalid"
                      if any("non-finite" in x for x in nonfinite)
                      else "a result file is unreadable/corrupt, so the gate "
                           "could not assert the output's integrity")
        elif _residual_blocks_verification(result):
            reason = ("the field this run produced does NOT satisfy the "
                      "equations it declared: "
                      + str(result["residual_check"].get("detail", "")))
        else:
            reason = ""
        _stamp_verification(result,
                            evidence_ok=(bool(out_files) and not job.error
                                         and not nonfinite
                                         and not _residual_blocks_verification(result)),
                            reason=reason, critic_approved=critic_approved,
                            solver=solver, setup_text=generator_script,
                            critic_token=critic_token,
                            job_id=str(result.get("job_id", job_name or "")))
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def run_simulation(solver: str, input_content: str,
                             job_name: str = "", np: int = 1,
                             critic_approved: bool = False,
                             critic_token: str = "",
                             verify_pde: str = "",
                             ctx: Context = None) -> str:
        """Run a simulation directly with input content.

        Use this for Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE-fem)
        where the input IS a Python script. The tool routes through the correct
        Python environment automatically (e.g., conda env for FEniCS).

        For 4C/deal.II/Kratos where a separate input file must be generated
        first, use run_with_generator() instead.

        Args:
            solver: Backend name (best for: fenics, ngsolve, skfem, dune)
            input_content: The input content (Python script / YAML / C++ / XML)
            job_name: Optional job name
            np: MPI processes
            critic_approved: recorded, not trusted. The result is verified only
                if a critic review of THIS input_content is on record — call
                submit_critic_review first.
            critic_token: optional token from submit_critic_review; makes the
                review single-use and binds it to this job.
            verify_pde: optional JSON declaring the problem being solved, so
                OASiS can check the result actually SATISFIES it rather than
                merely looking well-formed. Example:
                {"operator": "diffusion",
                 "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                 "coefficient": "1.0", "dim": 2, "domain_measure": 1.0}
                `source` and `coefficient` are numeric expressions in x, y, z.
                A field that does not satisfy the declared equations is NOT
                VERIFIED, whatever else the run did. Currently covers scalar
                diffusion on simplex meshes; anything else is reported as not
                checked, never as passed.
        """
        _journal = _get_journal()
        _snap = _make_input_snapshot(input_content, solver)
        _journal.record("tool_call", "run_simulation", solver=solver,
                        input_snapshot=_snap)

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=f"Not available: {msg}",
                            input_snapshot=_snap)
            return f"Solver {solver} not available: {_short_reason(msg)}"

        # CP-4: validate the input BEFORE running (was skipped on the live path)
        _input_warnings = []
        try:
            _input_warnings = backend.validate_input(input_content) or []
        except Exception:
            pass

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        if job.error:
            _journal.record("tool_error", "run_simulation", solver=solver,
                            error_message=job.error[:300],
                            input_snapshot=_snap)
        else:
            _journal.record("tool_success", "run_simulation", solver=solver,
                            input_snapshot=_snap)

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
        }
        out_files = []
        if job.error:
            result["error"] = job.error[:500]
        if _input_warnings:
            result["input_validation_warnings"] = _input_warnings
        nonfinite = []
        _stdout_text = ""
        if job.status == "completed":
            out_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in out_files]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                _stdout_text = stdout_log.read_text()
                result["stdout_tail"] = (_stdout_text[-2000:]
                                         if len(_stdout_text) > 2000 else _stdout_text)
            # CP-4: a process that exits 0 but produces NO output is NOT a verified
            # success — the canonical silent failure. Downgrade the status loudly.
            if not out_files:
                result["status"] = "completed_unverified"
                result["warning"] = ("Process exited cleanly but produced NO output files "
                                     "— this is NOT a verified solve. Do not treat as a result.")
            else:
                # ... and output full of NaN/Inf is a fabricated-looking result.
                nonfinite = check_result_files_finite(out_files)
            # Also scan the HEADLINE numbers (results_summary.json + stdout): a
            # summary can report max_value: Infinity while the mesh stays finite.
            nonfinite += check_summary_finite(work_dir, _stdout_text)
            # Structural defects in the solver's own data output. The scans
            # above look at headline numbers and mesh files; a FIELD that is
            # mostly NaN, or a wholly degenerate mesh, passed both and was
            # stamped verified. (Anti-fabrication gate.)
            nonfinite += inspect_result_artefacts(out_files)
            if nonfinite:
                result.setdefault("validation", []).extend(nonfinite)
            # The 'finiteness not asserted' honesty note is a coverage gap,
            # not evidence of a bad number — it must not flip the verdict to
            # 'non-finite values' (FEBio .xplt / .bp-without-adios2 runs).
            nonfinite = [x for x in nonfinite
                         if not x.startswith("finiteness not asserted")]
            # OASiS computes the run's headline numbers from the run's own
            # data, so the agent never has to assert one of its own.
            if out_files:
                result["oasis_computed"] = _attest_run_quantities(
                    work_dir, job.job_id)
                # …and, if the run declared what it is solving, whether that
                # data satisfies those equations at all.
                if verify_pde:
                    result["residual_check"] = _check_declared_pde(
                        verify_pde, out_files)
        # Verification gate: attestation binds the verdict to run evidence.
        if job.error:
            reason = "the solver run errored, so no number is backed by a valid run"
        elif job.status != "completed":
            reason = f"the run did not complete (status={job.status})"
        elif not out_files:
            reason = ("the process exited cleanly but produced NO output files, "
                      "so no reported number is backed by run evidence")
        elif nonfinite:
            reason = ("the result contains non-finite (NaN/Inf) values, so it is "
                      "numerically invalid"
                      if any("non-finite" in x for x in nonfinite)
                      else "a result file is unreadable/corrupt, so the gate "
                           "could not assert the output's integrity")
        elif _residual_blocks_verification(result):
            reason = ("the field this run produced does NOT satisfy the "
                      "equations it declared: "
                      + str(result["residual_check"].get("detail", "")))
        else:
            reason = ""
        _stamp_verification(result,
                            evidence_ok=(bool(out_files) and not job.error
                                         and not nonfinite
                                         and not _residual_blocks_verification(result)),
                            reason=reason, critic_approved=critic_approved,
                            solver=solver, setup_text=input_content,
                            critic_token=critic_token,
                            job_id=str(result.get("job_id", job_name or "")))
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def verify_mesh_independence(
            solver: str, input_template: str, resolution: float,
            refinement_factor: float = 2.0, levels: int = 1,
            parameter_kind: str = "divisions", field: str = "",
            probe_points: str = "", rel_tol: float = 0.01,
            job_name: str = "", np: int = 1,
            critic_approved: bool = False, ctx: Context = None) -> str:
        """Heuristic mesh-independence study for problems WITHOUT an exact
        solution: re-run the SAME problem at successively refined
        resolutions and accept it as converged only if ALL monitored
        quantities stop changing materially.

        MMS convergence tests need a manufactured exact solution; real
        application problems have none. This tool automates the
        established recourse: halve the discretisation length (once by
        default, more via `levels`), then compare (a) a volume-weighted
        global L2 norm and the global max of the primary field, (b) the
        field value at probe points (auto-chosen from the mesh — field
        hotspot, domain centre, off-centre interior points — or supplied
        explicitly), and (c) any scalar QoIs the script writes to
        results_summary.json. Verdict: CONVERGED only if every monitored
        quantity changes by less than `rel_tol` on the finest refinement
        step; otherwise NOT CONVERGED, with all numbers in the report.

        The input template must contain the placeholder __RESOLUTION__
        where the characteristic discretisation parameter goes, e.g.
        `nx = __RESOLUTION__`. For Python-scripted solvers (fenics,
        ngsolve, skfem, dune) the template is the solve script itself; for
        compiled/file-input solvers (fourc, dealii, kratos, febio) it is a
        generator script that writes the input file, exactly as in
        run_with_generator. The solve must write the primary field as
        nodal data in a VTU/VTK/VTP result file.

        IMPORTANT — this tool checks discretisation convergence only. It
        does not validate the model physics; have the MANDATORY critic
        review the setup and pass critic_approved=True as with the run
        tools.

        Args:
            solver: Backend name (any registered backend).
            input_template: Solve/generator script containing __RESOLUTION__.
            resolution: Coarsest value of the discretisation parameter.
            refinement_factor: Refinement per level (default 2 = halving h).
            levels: Number of refinements (default 1; runs levels+1 cases).
            parameter_kind: 'divisions' (parameter counts elements; refining
                multiplies) or 'size' (parameter is h; refining divides).
            field: Field name to monitor (default: auto-select from result).
            probe_points: Optional JSON list of probe coordinates, e.g.
                "[[0.5, 0.5], [0.25, 0.75]]" (default: auto from the mesh).
            rel_tol: Acceptance threshold on relative change (default 0.01).
            job_name: Optional study directory name.
            np: MPI processes per run.
            critic_approved: True only after the critic approved the setup.
        """
        import subprocess
        import sys
        from core import mesh_independence as mi
        from core.backend import InputFormat, find_generated_input, sorted_by_step

        _journal = _get_journal()
        _snap = _make_input_snapshot(input_template, solver,
                                     {"type": "mesh_independence_template"})
        _journal.record("tool_call", "verify_mesh_independence", solver=solver,
                        input_snapshot=_snap)

        def _fail(msg: str) -> str:
            _journal.record("tool_error", "verify_mesh_independence",
                            solver=solver, error_message=msg[:300],
                            input_snapshot=_snap)
            res = {"tool": "verify_mesh_independence", "solver": solver,
                   "status": "failed", "error": msg}
            _stamp_verification(res, evidence_ok=False, reason=msg[:200],
                                critic_approved=critic_approved,
                                solver=solver, setup_text=input_template)
            return json.dumps(res, indent=2)

        # Structured failures for these early exits too (Copilot review,
        # PR #49): every failure path of THIS tool returns the same JSON
        # shape with the verification stamp and a tool_error journal
        # record — a client must never have to branch on plain strings.
        backend = get_backend(solver)
        if not backend:
            return _fail(f"Unknown solver: {solver}")
        status, msg = backend.check_availability()
        if status.value != "available":
            return _fail(
                f"Solver {solver} not available: {_short_reason(msg)}")

        try:
            resolutions = mi.refinement_resolutions(
                resolution, refinement_factor, levels, parameter_kind)
            mi.substitute_resolution(input_template, resolutions[0])
        except ValueError as e:
            return _fail(str(e))
        if not (0 < rel_tol < 1):
            return _fail(f"rel_tol must be in (0, 1), got {rel_tol}")

        user_probes = None
        if probe_points.strip():
            try:
                user_probes = json.loads(probe_points)
                if (not isinstance(user_probes, list) or not user_probes
                        or not all(isinstance(p, (list, tuple)) for p in user_probes)):
                    raise ValueError("expected a JSON list of coordinate lists")
            except (json.JSONDecodeError, ValueError) as e:
                return _fail(f"probe_points is not a JSON list of coordinates: {e}")

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        study_dir = _OUTPUT_DIR / (job_name or f"{solver}_meshcheck_{ts}")
        is_python = backend.input_format() == InputFormat.PYTHON

        level_reports = []      # user-facing per-level info
        level_metrics = []      # input to mi.compare_levels
        shared_probes = user_probes
        pinned_field = field
        for lvl, res_val in enumerate(resolutions):
            content = mi.substitute_resolution(input_template, res_val)
            work_dir = study_dir / f"level{lvl}_res{mi.format_resolution(res_val)}"
            work_dir.mkdir(parents=True, exist_ok=True)

            if not is_python:
                # generator path (fourc / dealii / kratos / febio / sparta):
                # the template writes the solver input file first.
                gen_path = work_dir / "generate_input.py"
                gen_path.write_text(content)
                gen = subprocess.run([sys.executable, str(gen_path)],
                                     capture_output=True, text=True,
                                     cwd=str(work_dir))
                if gen.returncode != 0:
                    return _fail(f"level {lvl} (resolution "
                                 f"{mi.format_resolution(res_val)}): generator "
                                 f"failed: {gen.stderr[-400:]}")
                input_file = find_generated_input(work_dir, backend)
                if not input_file:
                    return _fail(f"level {lvl}: generator produced no input file")
                content = input_file.read_text()

            run_coro = backend.run(content, work_dir, np=np, timeout=None)
            if ctx is not None:
                job = await _run_with_progress(
                    ctx, run_coro,
                    f"Mesh study level {lvl}/{levels} on {solver}")
            else:
                job = await run_coro
            _jobs[job.job_id] = job

            if job.status != "completed" or job.error:
                return _fail(
                    f"level {lvl} (resolution {mi.format_resolution(res_val)}) "
                    f"did not complete: {(job.error or job.status)[:400]}")

            # plain FILES only: dolfinx VTXWriter can emit a DIRECTORY named
            # *.vtu, which no mesh reader can open (agent-validation S1 hit
            # exactly this and burned an iteration on it)
            out_files = [f for f in backend.get_result_files(job)
                         if f.suffix.lower() in (".vtu", ".vtk", ".vtp")
                         and not f.name.endswith(".pvtu") and f.is_file()]
            if not out_files:
                return _fail(
                    f"level {lvl} exited cleanly but produced no readable "
                    f"result file (.vtu/.vtk/.vtp) — no number is backed by "
                    f"run evidence")
            result_file = sorted_by_step(out_files)[-1]

            try:
                metrics = mi.extract_level_metrics(
                    result_file, field=pinned_field, probe_points=shared_probes)
            except Exception as e:
                # unreadable/fieldless output is a verdict, never a crash
                return _fail(f"level {lvl}: {e}")
            if lvl == 0:
                # Pin the auto-chosen field and probe locations so every
                # level monitors the SAME quantities at the SAME points.
                pinned_field = metrics["field"]
                shared_probes = metrics["probe_points"]
            metrics["resolution"] = res_val
            metrics["qoi"] = mi.collect_qoi_scalars(work_dir)
            level_metrics.append(metrics)
            level_reports.append({
                "level": lvl, "resolution": res_val,
                "job_id": job.job_id, "work_dir": str(work_dir),
                "result_file": result_file.name,
                "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
                "n_points": metrics["n_points"], "n_cells": metrics["n_cells"],
                "global_l2": metrics["global_l2"],
                "global_max": metrics["global_max"],
                "probe_values": metrics["probe_values"],
                **({"qoi": metrics["qoi"]} if metrics["qoi"] else {}),
            })

        comparison = mi.compare_levels(level_metrics, rel_tol=rel_tol)
        result = {
            "tool": "verify_mesh_independence", "solver": solver,
            "status": "completed",
            "field": pinned_field,
            "norm_type": level_metrics[0]["norm_type"],
            "parameter_kind": parameter_kind,
            "refinement_factor": refinement_factor,
            "probe_points": shared_probes,
            "levels": level_reports,
            "refinement_steps": comparison["steps"],
            "rel_tol": rel_tol,
            "converged": comparison["converged"],
            "verdict": comparison["verdict"],
            "study_dir": str(study_dir),
        }
        if not comparison["converged"]:
            result["failures"] = comparison["failures"]

        _journal.record("tool_success", "verify_mesh_independence",
                        solver=solver, input_snapshot=_snap)
        # Verification gate: the runs are the evidence; the verdict is the
        # check. A study whose quantities still drift is NOT a verified
        # solution — stamp it so the coarse answer cannot be reported.
        _stamp_verification(
            result,
            evidence_ok=comparison["converged"],
            reason=("the mesh-independence study did NOT converge: "
                    + "; ".join(comparison["failures"])
                    + f" (threshold {rel_tol:.2%}). The solution still "
                    "depends on the mesh — refine further"),
            critic_approved=critic_approved,
            solver=solver, setup_text=input_template)
        return json.dumps(result, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 5. COUPLING (general couple() + legacy coupled_solve)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def coupled_solve(
        problem: str = "heat_dd", solver_a: str = "fenics",
        solver_b: str = "fourc", nx: int = 32, ny: int = 32,
        max_iter: int = 20, tol: float = 1e-6,
        relaxation: float = 1.0, params: str = "{}",
        critic_approved: bool = False,
    ) -> str:
        """LEGACY cross-solver coupling — FIXED toy geometries only. PREFER `couple`.

        DEPRECATED: this tool only handles a fixed enum of benchmark problems on a
        hardcoded unit-square split at x=0.5 (heat_dd/poisson_dd/one_way/tsi_dd/...).
        For ANY real or non-benchmark coupling use the general `couple` tool, which is
        physics-agnostic and validates the result for silent-wrong (flux balance,
        convergence, finiteness). Only use coupled_solve to reproduce the legacy
        benchmarks.

        Domain A (Dirichlet at interface) supports: fenics, ngsolve, skfem, dune.
        Domain B (Neumann at interface) supports: fenics, fourc, ngsolve, skfem, dune.
        Any combination of these works for heat_dd and poisson_dd problems.

        Args:
            problem: 'heat_dd', 'poisson_dd', 'one_way', 'tsi_dd',
                     'poisson_dd_study', 'l_bracket_tsi', 'heat_dd_precice'
            solver_a, solver_b: Backend names
            nx, ny: Elements per direction
            max_iter: Max iterations
            tol: Convergence tolerance
            relaxation: Under-relaxation parameter
            params: JSON with additional parameters
            critic_approved: Set True after critic review
        """
        _get_journal().record("tool_call", "coupled_solve",
                              solver=f"{solver_a}->{solver_b}",
                              physics=problem)
        # Import and delegate to the full coupling implementation
        from tools.coupling import register_coupling_tools
        # The coupling tools are complex — delegate to the original implementation
        from tools.coupling import (
            _heat_domain_decomposition, _poisson_domain_decomposition,
            _oneway_thermal_structural, _twoway_tsi_coupling,
            _relaxation_parameter_study, _l_bracket_tsi,
            _heat_dd_precice_comparison,
        )

        param_dict = json.loads(params)
        backend_a = get_backend(solver_a)
        backend_b = get_backend(solver_b)
        if not backend_a or not backend_b:
            return f"Backend not found: {solver_a} or {solver_b}"

        dispatch = {
            "heat_dd": lambda: _heat_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd": lambda: _poisson_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "one_way": lambda: _oneway_thermal_structural(backend_a, backend_b, nx, ny, param_dict),
            "tsi_dd": lambda: _twoway_tsi_coupling(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd_study": lambda: _relaxation_parameter_study(backend_a, backend_b, nx, ny, max_iter, tol, param_dict),
            "l_bracket_tsi": lambda: _l_bracket_tsi(backend_a, backend_b, nx, ny, param_dict),
            "heat_dd_precice": lambda: _heat_dd_precice_comparison(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
        }

        if problem not in dispatch:
            return f"Unknown problem: {problem}. Available: {list(dispatch.keys())}"

        out = await dispatch[problem]()
        # LEGACY path returns human-readable text, not a gated JSON verdict.
        # An audit found this tool accepted `critic_approved` and then never
        # read it — a dead parameter on a tool the server instructions list as
        # critic-gated, so an unreviewed run was indistinguishable from a
        # reviewed one. The critic state now governs the verdict shown here,
        # and it is resolved from the server's review record rather than from
        # the flag: every parameter that changes what is solved goes into the
        # digest, so reviewing one configuration does not approve another.
        critic_ok, critic_note = _critic_state(
            f"{solver_a}->{solver_b}",
            _coupling_setup_text(problem=problem, solver_a=solver_a,
                                 solver_b=solver_b, nx=nx, ny=ny,
                                 max_iter=max_iter, tol=tol,
                                 relaxation=relaxation, params=params))
        if critic_ok:
            note = ("\n\n[OASiS verification: LEGACY coupled_solve — critic-reviewed. "
                    "Trust is governed by the convergence report above; a "
                    "non-converged run is reported as failure, never a result. "
                    "For a machine-readable verification verdict use `couple`.]")
        else:
            note = ("\n\n[OASiS verification: NOT VERIFIED — OASiS's independent "
                    "critic has not reviewed this setup, and OASiS treats no "
                    "result as trustworthy until it has. Do NOT report the values "
                    "above as a result: have a critic challenge the parameters, "
                    "units, discretisation and boundary conditions, then call "
                    "submit_critic_review with what it found and re-run. "
                    f"({critic_note}.) Asserting critic_approved=True does not "
                    "work: OASiS looks the review up rather than taking your "
                    "word for it. For a machine-readable verdict use `couple`.]")
        return (out + note) if isinstance(out, str) else out

    @mcp.tool()
    async def couple(participants: str, max_iter: int = 50, tol: float = 1e-6,
                     accelerator: str = "aitken", theta: float = 0.5,
                     monolithic: str = "", probe: bool = True,
                     critic_approved: bool = False) -> str:
        """GENERAL partitioned multi-code coupling — works for ANY physics/coupling.

        Have an independent critic review the setup before coupling; pass
        critic_approved=True only after that review (every simulation must be
        critic-reviewed first).

        Unlike coupled_solve (legacy, fixed toy geometries), this is physics-agnostic:
        you write one self-contained solver script per subdomain/participant and OASiS
        runs the fixed-point iteration, relaxation, convergence-or-fail, AND the
        silent-wrong validation a partitioned coupling needs, because a partitioned
        coupling's characteristic failure is not a crash — it is a clean, converged,
        confidently wrong number. OASiS checks, and reports in the verdict:
          * convergence, and per-block convergence (a large settled block, e.g. force,
            cannot hide a small moving one, e.g. displacement, inside one global norm);
          * finiteness of every exchanged array, including coordinates and fluxes;
          * interface flux balance, naming SIGN-CONVENTION and UNIT-MISMATCH signatures;
          * that every participant exited 0 — a diverged solver often writes its last
            iterate and then aborts;
          * that every participant's output actually MOVED when its imports moved —
            the test for a participant that exits 0 having done nothing, or re-serves
            a cached answer; such a run "converges" at iteration 2 with residual 0;
          * that the coupling graph is wired as declared — an `imports_from` name that
            matches no participant is REFUSED, not silently dropped into a one-way run;
          * whether the two interface discretisations match;
          * and, when you pass `monolithic`, that the coupled answer equals an
            independent un-split solve of the same problem. That last one is the only
            check that can catch a consistent unit error or a wrongly applied interface
            sign, so when it is not supplied the verdict SAYS it was not run.

        PARTICIPANT CONTRACT — each iteration the driver, per participant:
          1. writes <work_dir>/imports.json = {partner_name: InterfaceData} (boundary
             data this participant consumes; empty on iteration 1).
          2. runs your `command` in <work_dir>.
          3. reads <work_dir>/exports.json = the InterfaceData your script produced on
             the shared interface.
        Your script decides HOW to apply imports (Dirichlet/Neumann/Robin/traction/
        flux/...) and WHAT to export — opaque to the driver, so it generalizes.

        InterfaceData JSON shape (read imports, write exports):
          {"field_name": str, "n_points": N, "coordinates": [[x,y(,z)],...],
           "values": [...], "normal_fluxes": [...]  # optional, for conservation check}

        FLUX SIGN: export `normal_fluxes` with respect to YOUR OWN outward normal.
        The two normals are anti-parallel, so the two participants' fluxes carry
        OPPOSITE signs and their sums cancel. The Dirichlet value you APPLY is the
        same number on both sides — the opposite rule. Getting this wrong is the
        single most common cause of the flux-balance finding.

        Args:
            participants: JSON list of {"name", "command":[argv...], "work_dir",
              "imports_from":[partner names], "timeout": seconds}. Every name in
              `imports_from` must be another participant's name.
            max_iter, tol: iteration controls.
            accelerator: "aitken" (theta recomputed each iteration from the residual
              history, starting at `theta`) or "constant" (theta held at `theta` for
              the whole run). There is no per-field or per-participant theta.
            theta: the relaxation factor. Under-relaxation (theta < 1) is what makes
              a Dirichlet-Neumann or FSI coupling converge at all when the physical
              stiffness/density ratio makes the un-relaxed iteration diverge; 0.5 is
              a neutral default, not a recommendation for your problem.
            probe: after the iteration settles, spend ONE extra solve per
              participant perturbing its final imports and measuring how far its
              answer moves. This is the only check here that can tell a solver
              which reads its boundary data from one that merely looks as if it
              does; turn it off only if that solve is genuinely unaffordable, and
              the verdict will then record that the question was not asked.
            monolithic: OPTIONAL JSON {"command":[argv...], "work_dir": str,
              "timeout": int} — a solve of the SAME problem un-split, in ONE code,
              which writes <work_dir>/monolithic.json in InterfaceData shape on the
              same interface. Supplying it is the strongest verification available
              here and needs no external benchmark.

        Returns: JSON with converged, iterations, residual, per-block residuals,
            exports, the coupling graph, per-participant responsiveness and exit
            codes, a `validation` block, a `checks_not_run` block, and the verdict.
            A coupling that failed any check is reported as NOT VERIFIED, never as a
            trustworthy result — and one that could not be fully checked says so.
        """
        from core.coupling_driver import Participant, run_coupling
        from core.quality_checks import (
            check_interface_balance, check_finite, check_convergence,
            check_coupling_directionality, check_participant_responsiveness,
            check_interface_meshes, check_residual_blocks, check_returncodes,
            check_interface_flux_profile, check_interfaces_are_the_same_surface,
            check_interface_sensitivity,
        )
        _get_journal().record("tool_call", "couple", solver="general", physics="coupling")
        try:
            specs = json.loads(participants)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid participants JSON: {e}"})
        if not isinstance(specs, list) or len(specs) < 2:
            return json.dumps({"error": "need a JSON list of >=2 participants"})
        parts = []
        for s in specs:
            try:
                wd = Path(s["work_dir"]); wd.mkdir(parents=True, exist_ok=True)
                parts.append(Participant(name=s["name"], command=list(s["command"]),
                                         work_dir=wd, imports_from=s.get("imports_from", []),
                                         timeout=int(s.get("timeout", 3600)),
                                         data_files=list(s.get("data_files", []))))
            except (KeyError, TypeError) as e:
                return json.dumps({"error": f"bad participant spec {s!r}: {e}"})
        r = run_coupling(parts, max_iter=max_iter, tol=tol, accelerator=accelerator,
                         theta0=theta, probe=probe)

        # ── silent-wrong validators ───────────────────────────────────────────
        # `val` holds findings (they flip the verdict); `not_run` holds checks
        # that could not look at anything (they never flip the verdict, and they
        # are printed in it, so "not checked" is never read as "checked and fine").
        val = list(r.warnings)
        not_run: list[str] = [_DIGEST_SCOPE_LIMIT]
        val += check_convergence(r.converged, r.residual, tol)
        for nm, ex in r.exports.items():
            val += check_finite(ex.get("values", []), label=f"{nm}.values")
            if ex.get("normal_fluxes") is not None:
                val += check_finite(ex["normal_fluxes"], label=f"{nm}.normal_fluxes")
            val += check_finite(ex.get("coordinates", []), label=f"{nm}.coordinates")
        f, n = check_returncodes(r.returncodes); val += f; not_run += n
        f, n = check_coupling_directionality(r.graph, max_iter); val += f; not_run += n
        f, n = check_participant_responsiveness(r.responsiveness); val += f; not_run += n
        if probe:
            f, n = check_interface_sensitivity(r.sensitivity); val += f; not_run += n
        else:
            not_run.append(
                "interface sensitivity: NOT probed (probe=False). The one solve "
                "that would have established whether each participant's answer "
                "depends on its imports was skipped, so a solver that ignores "
                "imports.json is indistinguishable here from one that reads it.")
        # Only interesting when the GLOBAL norm claims convergence: that is the
        # case where a still-moving small block is invisible. When the global
        # residual already says NOT CONVERGED it says everything, and repeating
        # it per block just buries the findings that are specific.
        if r.converged:
            f, n = check_residual_blocks(r.block_residuals, tol); val += f; not_run += n
        names = list(r.exports)
        if len(names) == 2:
            a, b = r.exports[names[0]], r.exports[names[1]]
            f, n = check_interfaces_are_the_same_surface(a, b, names[0], names[1])
            val += f; not_run += n
            f, n = check_interface_meshes(a, b, names[0], names[1]); val += f; not_run += n
            f, n = check_interface_flux_profile(a, b, names[0], names[1])
            val += f; not_run += n
            if a.get("normal_fluxes") is None or b.get("normal_fluxes") is None:
                not_run.append(
                    "interface flux balance: at least one participant exported no "
                    "`normal_fluxes`, so conservation across the interface was NOT "
                    "checked. Export the normal flux from both sides (each w.r.t. "
                    "its own outward normal) to enable the only conservation "
                    "evidence available here.")
            else:
                val += check_interface_balance(a, b, names[0], names[1])
        elif len(names) > 2:
            not_run.append(
                f"interface flux balance: {len(names)} participants — the pairwise "
                "conservation check only applies to a 2-participant interface, so "
                "conservation was NOT checked.")

        # ── monolithic consistency: the decisive silent-wrong detector ────────
        mono_block, f, n = _run_monolithic_check(monolithic, r.exports)
        val += f; not_run += n

        # `val` is the FINDINGS channel and `not_run` the COVERAGE channel, and
        # every check above puts its output in exactly one of them. So any
        # finding at all means the coupling cannot be trusted — no substring
        # matching on message text, which would silently stop working the moment
        # a message was reworded, and could equally raise a false alarm on a
        # benign sentence that happened to contain one of the keywords.
        # A non-matching interface is deliberately a coverage note rather than a
        # finding: it is a legitimate configuration whose consequence (unchecked
        # conservation) belongs in the coverage list, and whose failure mode is
        # caught by the flux balance.
        checks_ok = r.converged and not val
        result = {"converged": r.converged, "iterations": r.iterations,
                  "residual": r.residual, "history": r.history,
                  "block_residuals": r.block_residuals,
                  "returncodes": r.returncodes,
                  "responsiveness": r.responsiveness,
                  "graph": r.graph, "relaxation": r.theta,
                  "interface_sensitivity": r.sensitivity,
                  "monolithic_check": mono_block,
                  "exports": r.exports, "error": r.error,
                  "validation": val, "checks_not_run": not_run}
        reason = ("" if checks_ok else
                  "the coupling did not converge, or failed one of OASiS's "
                  "silent-wrong checks (see `validation`)")
        _stamp_verification(result, evidence_ok=checks_ok, reason=reason,
                            critic_approved=critic_approved,
                            solver="couple",
                            setup_text=_coupling_setup_text(
                                participants=participants, max_iter=max_iter,
                                tol=tol, accelerator=accelerator, theta=theta,
                                monolithic=monolithic, probe=probe))
        if not_run:
            result["verification"] += (
                " COVERAGE — these checks could NOT run on this coupling, so the "
                "verdict above does not cover what they would have caught: "
                + " | ".join(not_run))
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def couple_precice(participants: str, data: str, exchanges: str,
                             work_dir: str, scheme: str = "serial-explicit",
                             dimensions: int = 2, max_time: float = 10.0,
                             time_window: float = 1.0, timeout: int = 1800,
                             max_iterations: int = 20,
                             convergence_tol: float = 1e-6,
                             relaxation: float = 0.5,
                             mapping: str = "nearest-neighbor",
                             extra_env: str = "",
                             critic_approved: bool = False) -> str:
        """GENERAL preCICE coupling of ARBITRARY codes/paradigms, end-to-end.

        Have an independent critic review the setup before coupling; pass
        critic_approved=True only after that review.

        The standard-library (preCICE) path for cross-code coupling — works for any
        number of participants, any data fields, any exchange pattern. OASiS generates
        the preCICE config and launches every participant's solver command. Use this
        when each side is a separate executable/script that talks preCICE (e.g. a DSMC
        particle code <-> a FEM solid; FSI; TSI). Each backend's preCICE participant
        pattern is available via knowledge(topic='precice', solver=...).

        Args (all JSON strings except scheme/numbers):
            participants: list of {"name","mesh","writes":[data],"reads":[data],
                          "command":[argv...]} — one per coupled code.
            data:      list of {"name","type":"scalar"|"vector"}.
            exchanges: list of {"data","from","to"} — one per coupled field.
            work_dir:  directory to run in (config + participant cwd).
            scheme:    serial-explicit|serial-implicit|parallel-explicit|parallel-implicit.
                       An EXPLICIT scheme takes one pass per time window and measures
                       no convergence at all — it cannot establish a coupled fixed
                       point, and the verdict here says so rather than implying one.
            dimensions, max_time, time_window, timeout: coupling controls.
            max_iterations, convergence_tol, relaxation: implicit-scheme controls
                       (ignored for explicit). These were previously not forwarded
                       at all, so every implicit coupling ran on the defaults
                       whatever the caller asked for.
            mapping:   nearest-neighbor|nearest-projection. Mapped with
                       constraint="consistent", which preserves nodal values and
                       NOT integrals — a flux/force field on a non-matching
                       interface is therefore not conserved, and the tool says so.
            extra_env: optional JSON dict of extra env (e.g. {"LD_LIBRARY_PATH":...,
                       "PYTHONPATH":...}) for the participant processes.

        Returns: JSON {exit_codes_ok, exchanged, coupling_converged, returncodes,
            config, logs, evidence, validation, checks_not_run}. Every participant
            exiting 0 is NOT by itself a coupling: an implicit scheme that exhausts
            max-iterations without meeting its convergence measure logs that and
            exits 0, and two scripts that never call preCICE at all exit 0 too. The
            verdict is built from preCICE's own per-window record, and an explicit
            scheme — which measures no convergence — is reported as unmeasured
            rather than as converged.
        """
        from core.precice_config import run_precice_coupling, check_precice_available
        _get_journal().record("tool_call", "couple_precice", solver="general", physics="coupling")
        ok, msg = check_precice_available()
        if not ok:
            return json.dumps({"error": f"preCICE not usable: {msg}"})
        try:
            parts = json.loads(participants); ds = json.loads(data); exs = json.loads(exchanges)
            env = json.loads(extra_env) if extra_env else None
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid JSON argument: {e}"})
        if not isinstance(parts, list) or len(parts) < 2:
            return json.dumps({"error": "need a JSON list of >=2 participants"})
        try:
            r = run_precice_coupling(parts, ds, exs, Path(work_dir), scheme=scheme,
                                     dimensions=dimensions, max_time=max_time,
                                     time_window=time_window, timeout=timeout,
                                     extra_env=env, max_iterations=max_iterations,
                                     convergence_tol=convergence_tol,
                                     mapping=mapping,
                                     initial_relaxation=relaxation)
        except ValueError as e:
            return json.dumps({"error": f"preCICE configuration refused: {e}"})
        except Exception as e:
            return json.dumps({"error": f"coupling failed: {e}"})
        # The orchestrator returns exit codes, preCICE's own per-window record and
        # log tails — not the exchanged field values, so check_finite cannot run on
        # the data. Best-effort: a whole-word NaN/Inf in any participant log is a
        # broken exchange. This only ever DOWNGRADES the verdict (fails safe toward
        # "not verified"), never upgrades it — the anti-fabrication direction.
        import re as _re
        _logs = " ".join(str(v) for v in (r.get("logs") or {}).values())
        nonfinite = bool(_re.search(r"\b(nan|-?inf|-?infinity)\b", _logs, _re.I))
        val: list[str] = []
        not_run: list[str] = []
        if nonfinite:
            val.append("participant logs report non-finite (NaN/Inf) "
                       "values — the exchanged fields are invalid.")
        if not r.get("exit_codes_ok"):
            val.append(f"participant exit codes: {r.get('returncodes')} — a "
                       "non-zero (or unknown) exit is a failed participant.")
        if not r.get("exchanged"):
            val.append("NO EXCHANGE: preCICE's own record shows no completed "
                       "coupling time window for every participant. Exiting 0 is "
                       "not evidence of a coupling — a script that never calls "
                       "preCICE exits 0 too.")
        conv = r.get("coupling_converged")
        if conv is False:
            val.append("NOT CONVERGED: preCICE recorded a time window in which the "
                       "implicit scheme hit max-iterations without meeting its "
                       "convergence measure. preCICE logs that and exits 0.")
        elif conv is None:
            not_run.append(
                "coupling convergence: preCICE recorded no convergence measure"
                + (" — an EXPLICIT scheme takes one pass per time window by "
                   "construction, so nothing here established that the coupled "
                   "state settled. Use serial-implicit / parallel-implicit if you "
                   "need that." if "explicit" in scheme else
                   " for this implicit scheme, so whether it converged is unknown."))
        for d in (r.get("evidence") or []):
            (not_run if "NOT established" in d else val).append(
                f"preCICE record: {d}" if "NOT established" in d else d)
        for note in (r.get("config_notes") or []):
            not_run.append(f"config: {note}")
        if val:
            r["validation"] = val
        r["checks_not_run"] = not_run
        evidence_ok = (bool(r.get("exit_codes_ok")) and bool(r.get("exchanged"))
                       and conv is not False and not nonfinite and not r.get("error"))
        _stamp_verification(
            r, evidence_ok=evidence_ok, critic_approved=critic_approved,
            solver="couple_precice",
            # Every argument that changes what preCICE computes. max_iterations,
            # convergence_tol, relaxation and mapping were absent, so a review of
            # one coupling silently approved the same coupling run to a
            # convergence tolerance five orders of magnitude looser.
            setup_text=_coupling_setup_text(
                participants=participants, data=data, exchanges=exchanges,
                scheme=scheme, dimensions=dimensions, max_time=max_time,
                time_window=time_window, max_iterations=max_iterations,
                convergence_tol=convergence_tol, relaxation=relaxation,
                mapping=mapping),
            reason="" if evidence_ok else
                   ("participant logs report non-finite (NaN/Inf) values"
                    if nonfinite else
                    "preCICE's own record does not show a completed, converged "
                    "exchange between all participants (see `validation`)"))
        if not_run:
            r["verification"] += (
                " COVERAGE — these checks could NOT run, so the verdict above does "
                "not cover what they would have caught: " + " | ".join(not_run))
        return json.dumps(r, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 6. VISUALIZE (replaces 4 visualization tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def visualize(job_id: str = "", work_dir: str = "",
                        action: str = "summary", field: str = "",
                        ctx: Context = None) -> str:
        """Post-process and visualize simulation results.

        Args:
            job_id: Job ID from run_simulation (or leave empty and set work_dir)
            work_dir: Direct path to results directory
            action: What to do. Options:
                - "summary" — field statistics + results_summary.json content
                  (default; the fastest pulse-check on a finished run)
                - "list" — list every result file under the work dir
                - "plot" — generate a PNG of the named field (needs field=)
                - "validate" — automated sanity checks across the first
                  3 result files: NaN/Inf detection, constant-field
                  detection, suspiciously-large-magnitude detection
                  (>1e15). Use after summary when a field looks wrong.
            field: Specific field name to plot (e.g. 'temperature', 'displacement')
        """
        from core.backend import JobHandle

        # Find work directory
        if job_id and job_id in _jobs:
            wd = _jobs[job_id].work_dir
        elif work_dir:
            wd = Path(work_dir)
        else:
            return "Provide job_id or work_dir"

        if not wd.is_dir():
            return f"Directory not found: {wd}"

        # Collect result files — skip .pvtu (parallel wrappers that can hang PyVista).
        # Deliberately NOT globbing *.xdmf: pyvista's VTK XDMF reader SIGSEGVs on
        # some files, which is uncatchable and takes the whole MCP server down
        # (it wiped the in-memory job registry for the session). Every backend
        # that writes .xdmf also writes a readable .vtu companion, so nothing is
        # lost. (Mirrors quality_checks._FINITE_SCANNABLE excluding xdmf.)
        vtu_files = [f for f in sorted(wd.rglob("*.vtu")) if not f.name.endswith(".pvtu")]
        vtu_files += sorted(wd.rglob("*.vtk"))
        vtu_files += sorted(wd.rglob("*.vtp"))
        vtu_files += sorted(wd.rglob("*.bp"))  # ADIOS2/VTX output from dolfinx 0.10+

        if action == "list":
            return "\n".join(f"- {f.relative_to(wd)}" for f in vtu_files) or "No VTU/VTP files found"

        elif action == "summary":
            try:
                from core.post_processing import read_mesh
                import numpy as np
                import re

                # Layer F catalog templates (fenics / ngsolve /
                # skfem / kratos) write a per-run summary at
                # results_summary.json: max field values, dof
                # counts, convergence metrics. Without surfacing
                # this, visualize('summary') returns '[]' when
                # only the JSON summary exists — even though the
                # template printed exactly the info the LLM wants.
                # Audit 2026-06-01.
                summary_artifacts = []
                for js in sorted(wd.rglob("results_summary.json")):
                    try:
                        with open(js) as _f:
                            summary_artifacts.append({
                                "file": str(js.relative_to(wd)),
                                "summary": json.load(_f),
                            })
                    except Exception as e:
                        summary_artifacts.append({
                            "file": str(js.relative_to(wd)),
                            "error": f"unreadable: {e}",
                        })

                # Group VTU files by field type (structure, fluid, ale, etc.)
                # 4C multi-physics outputs separate files per field
                field_groups: dict[str, list] = {}
                for vtu in vtu_files:
                    name = vtu.stem
                    # Detect field type from filename patterns like
                    # structure-00-0, fluid-05-0, ale-03-0
                    match = re.match(r'^(.*?)(?:-\d+)?(?:-\d+)?$', name)
                    group_name = match.group(1) if match else name
                    # Also strip trailing -0 (processor rank)
                    group_name = re.sub(r'-\d+$', '', group_name)
                    field_groups.setdefault(group_name, []).append(vtu)

                def _safe_float(v):
                    """Convert to float, replacing NaN/Inf with string markers."""
                    f = float(v)
                    if np.isnan(f):
                        return "NaN"
                    if np.isinf(f):
                        return "Inf" if f > 0 else "-Inf"
                    return f

                results = []
                # Show summary per field group, using the last timestep
                # Limit to 10 groups to avoid extremely long responses
                group_idx = 0
                for group, files in sorted(field_groups.items())[:10]:
                    group_idx += 1
                    if ctx is not None:
                        try:
                            await ctx.report_progress(
                                group_idx, len(field_groups),
                                f"Reading {group} ({len(files)} timesteps)")
                        except Exception:
                            pass
                    # Use the last file in each group (latest timestep)
                    last_vtu = sorted(files)[-1]
                    try:
                        mesh = read_mesh(last_vtu)
                        fields = {}
                        for fname in mesh.point_data:
                            arr = np.asarray(mesh.point_data[fname])
                            n_nan = int(np.isnan(arr).sum())
                            n_inf = int(np.isinf(arr).sum())
                            finite = arr[np.isfinite(arr)]
                            stats = {
                                "shape": list(arr.shape),
                            }
                            if len(finite) > 0:
                                stats["min"] = _safe_float(finite.min())
                                stats["max"] = _safe_float(finite.max())
                                stats["mean"] = _safe_float(finite.mean())
                            if n_nan > 0:
                                stats["WARNING_NaN"] = f"{n_nan} values"
                            if n_inf > 0:
                                stats["WARNING_Inf"] = f"{n_inf} values"
                            fields[fname] = stats
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "latest_file": last_vtu.name,
                            "points": mesh.n_points,
                            "fields": fields,
                        })
                    except Exception as e:
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "error": str(e),
                        })
                # Prepend the JSON-summary artifacts (if any)
                # so the LLM sees them first.
                output = {
                    "results_summary_json": summary_artifacts,
                    "vtu_field_groups": results,
                }
                # If neither populated, drop the wrapper to keep
                # the legacy '[]' empty signal for "nothing here".
                if not summary_artifacts and not results:
                    return "[]"
                return json.dumps(output, indent=2)
            except Exception as e:
                return f"Error reading results: {e}"

        elif action == "plot" and field:
            try:
                from core.post_processing import read_mesh, plot_field
                vtu = vtu_files[-1] if vtu_files else None
                if not vtu:
                    return "No VTU files to plot"
                mesh = read_mesh(vtu)
                plot_path = wd / f"plot_{field}.png"
                plot_field(mesh, field, plot_path, title=field, spatial_dim=2)
                return f"Plot saved: {plot_path}"
            except Exception as e:
                return f"Error plotting: {e}"

        elif action == "validate":
            # Automated sanity checks on results
            try:
                from core.post_processing import read_mesh
                import numpy as np
                checks = []
                for vtu in vtu_files[:3]:
                    mesh = read_mesh(vtu)
                    for name in mesh.point_data:
                        arr = np.asarray(mesh.point_data[name])
                        issues = []
                        if np.any(np.isnan(arr)):
                            issues.append(f"CONTAINS NaN ({np.isnan(arr).sum()} values)")
                        if np.any(np.isinf(arr)):
                            issues.append(f"CONTAINS Inf ({np.isinf(arr).sum()} values)")
                        if arr.max() == arr.min() and len(arr) > 1:
                            issues.append(f"CONSTANT FIELD (all values = {arr.max():.6e})")
                        if arr.max() > 1e15:
                            issues.append(f"SUSPICIOUSLY LARGE max = {arr.max():.2e}")
                        status = "PASS" if not issues else "ISSUES FOUND"
                        checks.append(f"- {name} in {vtu.name}: {status}" +
                                     (f"\n  " + "\n  ".join(issues) if issues else ""))
                return "## Results Validation\n\n" + "\n".join(checks)
            except Exception as e:
                return f"Validation error: {e}"

        return "Usage: visualize(job_id, action='summary'|'plot'|'list'|'validate', field='')"

    # ═══════════════════════════════════════════════════════════
    # 7. DEVELOPER (replaces 3 developer tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def developer(action: str, solver: str = "", keyword: str = "") -> str:
        """Developer tools: architecture, source files, capabilities matrix.

        Args:
            action: What to surface. Options:
                - "architecture" — extension points + source-tree
                  layout for the requested solver
                - "files" — source-file listing filtered by keyword
                - "capabilities" — full backend × physics × variant
                  matrix dump
            solver: Backend name
            keyword: File pattern for "files" action
        """
        if action == "files":
            _get_journal().record("source_read", "developer",
                                  solver=solver, notes=f"keyword={keyword}")
        if action == "architecture" and solver:
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            if not info:
                return f"Unknown solver: {solver}"
            return json.dumps(info, indent=2)

        elif action == "capabilities":
            # All registered backends so the developer-side
            # capabilities listing matches discover('capabilities'):
            # consistent visibility across both surfaces.
            # (Audit 2026-06-02.)
            lines = []
            for b in all_backends():
                status, _ = b.check_availability()
                tag = "" if status.value == "available" else f" *[{status.value}]*"
                physics = [p.name for p in b.supported_physics()]
                lines.append(f"**{b.display_name()}**{tag}: {', '.join(physics)}")
            return "\n".join(lines)

        elif action == "files" and solver:
            # Check if solver has a source root set via env var
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            source_root = info.get("root", "")
            source_env = info.get("source_env_var", "")

            # If keyword starts with "src/" or similar, search the solver source tree
            if keyword and source_root and Path(source_root).is_dir():
                base = Path(source_root)
                pattern = keyword
                files = sorted(base.rglob(pattern))[:30]
                if files:
                    return "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files)

            # Default: search the MCP backend files
            base = Path(__file__).resolve().parents[1] / "backends" / solver
            if not base.exists():
                hint = f"\nTo browse {solver} source code, set {source_env} in .claude/settings.json" if source_env else ""
                return f"No source directory for {solver}{hint}"
            pattern = keyword or "*.py"
            files = sorted(base.rglob(pattern))
            result = "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files[:20])
            if source_env and not (source_root and Path(source_root).is_dir()):
                result += f"\n\nNote: Set {source_env} env var to browse the full {solver} source tree"
            return result

        return "Usage: developer(action='architecture'|'capabilities'|'files', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 8. PREPARE (meta-tool: knowledge + examples + template in one call)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def prepare_simulation(solver: str, physics: str) -> str:
        """Prepare everything needed to set up a simulation — in ONE call.

        Returns: knowledge + real test file examples + generated template.
        This eliminates 3 separate tool calls before every simulation.

        Supports fuzzy matching: e.g. 'magnetostatics' finds 'maxwell',
        'thermal' finds 'heat', 'elasticity' finds 'linear_elasticity'.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'particle_pd', 'navier_stokes',
                     'magnetostatics', 'thermal', 'elasticity')
        """
        _get_journal().record("knowledge_lookup", "prepare_simulation",
                              solver=solver, physics=physics)
        parts = []

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # Warn up front if the REQUESTED backend is not usable on this install.
        # Otherwise a user follows the returned template and only discovers at
        # run time that the solver can't run (user-session finding). The guidance
        # is still returned — it is valid — just clearly flagged.
        _avail_status, _avail_msg = backend.check_availability()
        if _avail_status.value != "available":
            parts.append(
                f"> ⚠ **{backend.display_name()} ({backend.name()}) is NOT available "
                f"on this install** — {_short_reason(_avail_msg)}\n>\n> The setup below "
                f"is still accurate, but install/enable {backend.name()} (or choose an "
                f"available backend) before running.\n")

        # Fuzzy match: find the best matching physics name
        matched_physics = _fuzzy_match_physics(backend, physics)
        if not matched_physics:
            # Empty / whitespace-only query — surface the
            # available-physics list so the LLM can pick a real
            # name. Without this guard prepare_simulation silently
            # builds a half-empty response for a physics it never
            # had. Audit 2026-06-01.
            available = ", ".join(
                p.name for p in backend.supported_physics())
            return (f"Empty physics query. Available physics in "
                    f"{backend.display_name()}: {available}")
        if matched_physics != physics:
            parts.append(f"*Note: '{physics}' matched to '{matched_physics}'*\n")

        # 0. Also available on — show which other backends support this physics (informational)
        alternatives = _list_alternative_solvers(solver, matched_physics)
        if alternatives:
            parts.append("## Also available on\n" + alternatives + "\n")

        # 1. Knowledge
        # Render pitfalls OUTSIDE the JSON dump so the 3000-char
        # truncation does not silently hide them. Audited
        # 2026-06-01: large KNOWLEDGE blocks like ngsolve::
        # hyperelasticity (7 pitfalls / ~4.4 KB) and skfem::
        # poisson (6 / ~4 KB) showed 0/N pitfalls fully visible
        # to the LLM client; every Layer F fix landed but never
        # reached the prepare_simulation surface that's meant to
        # teach the agent.
        k = _strip_pitfalls(backend.get_knowledge(matched_physics))
        if k:
            pitfalls_separate = None
            json_payload = k
            if isinstance(k, dict) and isinstance(k.get("pitfalls"), list):
                pitfalls_separate = k["pitfalls"]
                json_payload = {kk: vv for kk, vv in k.items() if kk != "pitfalls"}
            # After the pitfalls carve-out the remaining JSON
            # is description / spaces / solver / elements /
            # materials / time_integration / typical_experiments.
            # Most backends sit < 1.5 KB but fourc::solid_mechanics
            # is ~12 KB (rich plasticity_models + materials dict).
            # The old 3000-char cap silently dropped most of that.
            # Match the TEMPLATE_LIMIT of 12000 set above so the
            # LLM gets the full materials table. Audit 2026-06-01.
            KNOWLEDGE_LIMIT = 16000
            payload_text = json.dumps(json_payload, indent=2, default=str)
            payload_truncated = len(payload_text) > KNOWLEDGE_LIMIT
            payload_body = payload_text[:KNOWLEDGE_LIMIT]
            payload_suffix = (f"\n... [truncated {len(payload_text) - KNOWLEDGE_LIMIT} chars]"
                              if payload_truncated else "")
            parts.append("## Knowledge\n```json\n"
                         + payload_body + payload_suffix
                         + "\n```\n")
            if pitfalls_separate:
                bullets = "\n".join(f"- {p}" for p in pitfalls_separate)
                parts.append(
                    f"### Pitfalls ({len(pitfalls_separate)})\n{bullets}\n")

        # 1a. Installed-version API reference (VERIFIED by actually running here).
        # The dominant failure mode is writing API calls for the WRONG version of the
        # installed code (NGSolve Integrate signature, deal.II DEAL_II_DIR build-tree,
        # the 4C YAML schema, ...). Surface a verified smoke test + version gotchas so
        # the agent adapts a known-good call instead of guessing from memory.
        try:
            from backends._installed_api import render as _render_installed_api
            api_ref = _render_installed_api(backend.name())
            if api_ref:
                parts.append(api_ref + "\n")
        except Exception:
            pass

        # 1b. General input-format pitfalls (ExodusII IDs, FUNCT syntax, etc.)
        # These apply to ALL physics in this solver, not just the current one
        general_k = _strip_pitfalls(backend.get_knowledge("input_format"))
        if isinstance(general_k, dict):
            gp = general_k.get("general_pitfalls")
            if gp:
                pitfall_text = "\n".join(f"- {p}" for p in gp)
                parts.append(f"## General Input Pitfalls\n{pitfall_text}\n")

        # 2. Real test file examples
        from tools.knowledge import _find_reference_test_files
        ref = _find_reference_test_files(solver, matched_physics)
        if ref:
            parts.append(ref)

        # 3. Generated template
        # Templates can exceed 3000 chars on the harder physics
        # (ngsolve hdivdiv 3.2KB, nonlinear_elasticity 3.4KB,
        # fenics navier_stokes 3.8KB ...) — truncating at 3000
        # cuts off the trailing solver / output / summary
        # blocks the LLM needs to actually run the template.
        # Raise to 12000 chars so the standard Layer F-class
        # templates (typically 2-5KB) render in full. Audit
        # 2026-06-01.
        TEMPLATE_LIMIT = 12000
        for p in backend.supported_physics():
            if p.name == matched_physics and p.template_variants:
                # Honour the qualifiers in the request. Reading variants[0] and
                # nothing else served a 2D plane-stress deck for "3d linear
                # elasticity" while a 3d variant sat unreachable in the catalog.
                variant, variant_note = _select_template_variant(
                    physics, list(p.template_variants))
                try:
                    content = backend.generate_input(matched_physics, variant, {})
                    fmt = backend.input_format().value
                    truncated = len(content) > TEMPLATE_LIMIT
                    body = content[:TEMPLATE_LIMIT]
                    suffix = (f"\n... [truncated {len(content) - TEMPLATE_LIMIT} chars]"
                              if truncated else "")
                    stub_tag = _stub_template_tag(content, fmt)
                    note = f"\n{variant_note}\n" if variant_note else ""
                    parts.append(f"## Template ({variant}){stub_tag}\n{note}```{fmt}\n{body}{suffix}\n```\n")
                except Exception as exc:
                    # Surface the failure: the catalog claims a
                    # template exists (p.template_variants is
                    # non-empty) but the generator raised. The
                    # old `except Exception: pass` silently
                    # produced an LLM-visible "successful" reply
                    # with no template and no hint that the
                    # generator was broken — masking Layer-F
                    # class regressions both from the LLM and
                    # the developer running it. (Audit 2026-06-02.)
                    parts.append(
                        f"## Template ({variant})\n"
                        f"⚠ Template generation FAILED for "
                        f"`{matched_physics}/{variant}`: "
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        f"This is a catalog generator bug — the "
                        f"physics is advertised in "
                        f"`{backend.display_name()}.supported_physics()` "
                        f"but `generate_input` raised. The other "
                        f"sections of this response (knowledge, "
                        f"pitfalls, real-test references) are still "
                        f"valid; only the auto-generated template "
                        f"is missing.\n")
                break

        if not parts:
            # List available physics as hint
            avail = [p.name for p in backend.supported_physics()]
            return f"No information found for '{physics}' in {solver}. Available physics: {', '.join(avail)}"

        return f"# Preparation for {matched_physics} on {solver}\n\n" + "\n---\n".join(parts)

    # ═══════════════════════════════════════════════════════════
    # 9. TRANSFER FIELD (keep — needed for coupling)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def transfer_field(
        source_vtu: str, field_name: str,
        interface_coord: float, interface_axis: int = 0,
        target_format: str = "json", output_path: str = "",
    ) -> str:
        """Extract field values at an interface from a VTU file and format for transfer.

        Universal data connector for cross-solver coupling. Reads VTU output
        from any solver, extracts values at the interface plane, and formats
        them for the target solver's expected input shape.

        Args:
            source_vtu: Path to VTU result file from the source solver.
            field_name: Field to extract (e.g. 'temperature', 'displacement').
            interface_coord: Coordinate value defining the interface plane.
            interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z).
            target_format: Output format. Options:
                - "json"        — interface coordinates + values (default)
                - "fenics"      — Python BoundaryCondition snippet (Dirichlet
                                  at this interface), saved as .py
                - "4c_neumann"  — 4C-format YAML snippet for a Neumann
                                  boundary condition, saved as .yaml
            output_path: Where to save the formatted output. If empty,
                auto-generated next to the source VTU as
                'interface_<field_name>.<ext>'.

        Returns:
            A summary string with the interface min/max/mean and the path
            of the saved file.
        """
        from core.field_transfer import extract_interface_from_vtu

        vtu_path = Path(source_vtu)
        if not vtu_path.exists():
            return f"VTU file not found: {source_vtu}"

        try:
            iface = extract_interface_from_vtu(
                vtu_path, field_name, interface_coord, interface_axis)
        except Exception as e:
            return f"Error extracting interface: {e}"

        if not output_path:
            output_path = str(
                vtu_path.parent / f"interface_{field_name}.json")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if target_format == "json":
            iface.to_json(out)
        elif target_format == "fenics":
            from core.field_transfer import format_for_fenics
            code = format_for_fenics(
                iface, "dirichlet", interface_axis, interface_coord)
            out = out.with_suffix(".py")
            out.write_text(code)
        elif target_format == "4c_neumann":
            from core.field_transfer import format_for_4c_neumann
            yaml_snippet = format_for_4c_neumann(iface)
            out = out.with_suffix(".yaml")
            out.write_text(yaml_snippet)
        else:
            return (f"Unknown format: {target_format}. Use 'json', "
                    "'fenics', or '4c_neumann'.")

        vals = iface.values
        summary = (
            f"## Field Transfer: {field_name}\n\n"
            f"- Source: {vtu_path.name}\n"
            f"- Interface: {'xyz'[interface_axis]}={interface_coord}\n"
            f"- Nodes: {len(iface.coordinates)}\n"
            f"- Values: [{vals.min():.6e}, {vals.max():.6e}], "
            f"mean={vals.mean():.6e}\n"
            f"- Output: {out}\n"
        )
        if iface.normal_fluxes is not None:
            fl = iface.normal_fluxes
            summary += f"- Fluxes: [{fl.min():.6e}, {fl.max():.6e}]\n"
        return summary

    # ═══════════════════════════════════════════════════════════
    # 10. MESH (keep — needed for Gmsh)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.1,
                      output_dir: str = "") -> str:
        """Generate a mesh using Gmsh for non-trivial geometries.

        Args:
            geometry: One of the built-in geometries:
                - "l_domain"          — 2D L-shaped domain
                - "plate_with_hole"   — 2D plate with circular hole
                - "channel_cylinder"  — 2D channel with cylindrical obstacle
                (No "custom" passthrough yet — passing any other name
                returns a 'Unknown geometry' message with this list.)
            mesh_size: Target element size
            output_dir: Where to save (auto if empty)
        """
        from tools.mesh_generation import register_mesh_tools
        # Delegate to original. Importer names must match the
        # functions in tools.mesh_generation EXACTLY — the prior
        # _generate_channel_cylinder_2d (without 'with') did not
        # exist there (actual name is _generate_channel_with_
        # cylinder_2d) and the ImportError short-circuited the
        # dispatch dict for ALL three geometries, including
        # l_domain and plate_with_hole. (Audit 2026-06-01.)
        try:
            from tools.mesh_generation import (
                _generate_l_domain_2d,
                _generate_plate_with_hole_2d,
                _generate_channel_with_cylinder_2d,
            )
            generators = {
                "l_domain": _generate_l_domain_2d,
                "plate_with_hole": _generate_plate_with_hole_2d,
                "channel_cylinder": _generate_channel_with_cylinder_2d,
            }
            gen = generators.get(geometry)
            if gen:
                # The three generators have DIFFERENT positional
                # signatures (l_domain: (mesh_size, output_path);
                # plate_with_hole: (mesh_size, radius, width,
                # height, output_path); channel_with_cylinder:
                # (mesh_size, cyl_radius, center, length, height,
                # output_path)). Always pass output_path via
                # keyword. The functions expect a FULL FILE path
                # (gmsh.write needs an extension to pick the
                # output format) — append "<geom>.msh" to the
                # directory the user passed in. (Audit 2026-06-01.)
                out_dir = Path(output_dir or str(_OUTPUT_DIR / "meshes"))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{geometry}.msh"
                result = gen(mesh_size, output_path=out_file)
                # generators return either Path or (Path, n_nodes,
                # n_elements). Surface a friendly summary.
                if isinstance(result, tuple):
                    path, *meta = result
                    return f"mesh: {path} (nodes={meta[0]}, elements={meta[1]})"
                return str(result) if result is not None else "ok"
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"
        except Exception as e:
            return f"Error: {e}"

    # ═══════════════════════════════════════════════════════════
    # 12. BACKEND DISCOVERY
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def reload_catalog() -> str:
        """Hot-reload the per-backend KNOWLEDGE dicts from disk.

        Closes the gap identified by the
        mcp-catalog-staleness-runtime-isolation post-mortem
        (2026-06-01): the MCP server normally imports
        src/backends/<be>/generators/<physics>.py modules ONCE at
        startup and never refreshes them, so catalog edits made
        during a long-running session are invisible. Postmortems
        in data/postmortems/ are scanned on every request (already
        hot), but pitfall dicts are not.

        This tool walks every imported `backends.<be>.generators.*`
        and `backends.<be>.backend` module, runs importlib.reload
        on each, and re-runs load_all_backends() so the registry
        re-binds the backend objects to the refreshed module
        attributes. After the call, the very next
        mcp__oasis__knowledge call returns the on-disk
        catalog without having to restart Claude Code.

        Returns a one-line summary of which modules were
        successfully reloaded vs which raised, so the caller can
        tell when a syntax error in a newly-edited generator
        prevented its module from re-importing (in that case the
        OLD dict is still served from the previous import).
        """
        import importlib
        import sys
        reload_ok: list[str] = []
        reload_fail: list[tuple[str, str]] = []

        # Reload data/*_knowledge.py first (sourced by some
        # backends).
        for mod_name in list(sys.modules.keys()):
            if (mod_name.endswith("_knowledge")
                    and not mod_name.startswith("backends.")):
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Reload every imported backends.* submodule.
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("backends.") and "." in mod_name:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_ok.append(mod_name)
                except Exception as exc:  # noqa: BLE001
                    reload_fail.append((mod_name, str(exc)))

        # Re-bind backend objects to refreshed modules.
        try:
            from core.registry import load_all_backends
            load_all_backends()
            re_register = "ok"
        except Exception as exc:  # noqa: BLE001
            re_register = f"FAILED: {exc}"

        msg = (f"reload_catalog: {len(reload_ok)} modules reloaded, "
               f"{len(reload_fail)} failed; "
               f"re-register backends: {re_register}.")
        if reload_fail:
            msg += "\n\nFailures:\n" + "\n".join(
                f"  - {n}: {e[:200]}" for n, e in reload_fail[:10])
        return msg

    @mcp.tool()
    def rediscover_backends(confirm: bool = False) -> str:
        """Probe the system for solver backends and report findings.

        Searches pip packages, conda environments, common build directories,
        and source roots. In developer mode, reports git branch and status.

        Args:
            confirm: If True, save the discovered config for future sessions.
                     If False (default), just report what was found.
        """
        from core.autodiscovery import (
            discover_backends as _discover,
            format_discovery,
            save_discovered_config,
        )

        # A negative dune interpreter result is cached for the server's
        # lifetime, so a dune installed after startup would otherwise stay
        # invisible. Reset the cache so re-discovery genuinely re-probes.
        try:
            from backends.dune.backend import _reset_dune_python_cache
            _reset_dune_python_cache()
        except Exception:
            pass

        results = _discover()
        report = format_discovery(results)

        if confirm:
            path = save_discovered_config(results)
            report += f"\n\nConfig saved to `{path}`. Will be used on next restart."
        else:
            found_count = sum(1 for r in results if r.found)
            if found_count > 0:
                report += (
                    f"\n\nCall `rediscover_backends(confirm=True)` to save this "
                    f"config for future sessions."
                )

        return report

    @mcp.tool()
    def setup_backend(action: str = "status", solver: str = "",
                      route: str = "") -> str:
        """Guided backend setup: detect, plan, install, verify, persist.

        Helps a new user get any of the 8 FEM backends working on THIS
        machine — picking the fastest install route for the current OS
        (pip > conda > binary download > source build), executing it,
        running the backend's smoke test, and persisting the resolved
        paths into ~/.config/oasis/sources.json so every future MCP
        session finds the install without re-discovery.

        Actions:
          status  — one-row-per-backend table: available? source tree?
                    build dir? (no args needed)
          plan    — recommended install route for `solver` on this OS,
                    incl. system deps (apt/brew), human notes, and
                    whether the route is verified on this OS. Nothing
                    executes. Optional `route` (pip|conda|binary|source)
                    forces an alternative.
          install — execute the planned route. pip/conda run inline
                    (minutes); source builds start in the BACKGROUND
                    (30-120 min) — re-run with action='verify' when
                    done. binary routes return manual instructions.
          verify  — run the smoke test for `solver` and, on success,
                    persist its paths.

        macOS note: darwin routes are structured but mostly UNVERIFIED
        (flagged in the plan output). They are extension points — when
        a route is validated on a Mac, its os_support['darwin'] entry
        in src/core/backend_setup.py should be updated with the working
        steps (e.g. the 4C brew/CMake settings).

        Args:
            action: status | plan | install | verify
            solver: backend name (required for plan/install/verify)
            route:  optional route kind override (pip|conda|binary|source)
        """
        import json as _json
        from core.backend_setup import (
            render_status_markdown, plan_setup, execute_setup,
            _verify_and_persist,
        )
        if action == "status":
            return render_status_markdown()
        if not solver:
            return ("setup_backend: actions plan/install/verify need "
                    "solver=<backend>. Usage: action=status|plan|install"
                    "|verify, solver=fourc|fenics|dealii|ngsolve|skfem"
                    "|kratos|dune|febio, route=pip|conda|binary|source "
                    "(optional).")
        if action == "plan":
            return _json.dumps(plan_setup(solver, prefer=route or None),
                               indent=2, default=str)
        if action == "install":
            return _json.dumps(
                execute_setup(solver, route_kind=route or None),
                indent=2, default=str)
        if action == "verify":
            return _json.dumps(_verify_and_persist(solver), indent=2,
                               default=str)
        return (f"setup_backend: unknown action {action!r}. "
                "Use status | plan | install | verify.")

    # ═══════════════════════════════════════════════════════════
    # 13. SESSION INSIGHTS (knowledge capture)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def session_insights(action: str = "review", path: str = "") -> str:
        """Review knowledge discovered during this session or from saved
        journals on disk.

        Two flows are supported:

        * In-session flow: call ``review`` -> ``approve_all`` /
          ``reject_all`` during the live MCP session to surface
          candidates from the current journal and save approved ones
          to ``data/community_knowledge/pending/``.
        * Ingest flow: call ``ingest`` with ``path`` pointing at a
          previously-saved session journal (``data/sessions/session_*.json``,
          which the server writes on shutdown) or at a directory of
          such files.  Candidates are surfaced just like ``review``
          and can be approved with ``approve_all``.

        Args:
            action:
                - "review" — show candidate knowledge from the current
                  session for approval
                - "ingest" — load saved journal(s) from ``path`` and
                  analyse them; requires ``path``
                - "approve_all" — approve all pending candidates and
                  save to community_knowledge/pending/
                - "reject_all" — dismiss all pending candidates
                - "stats" — current session statistics
            path: file or directory used by the ``ingest`` action;
                ignored otherwise.  Directories are scanned for
                ``session_*.json``.
        """
        from pathlib import Path as _Path

        from core.session_journal import get_journal
        from core.session_analyzer import (
            CandidateKnowledge,
            analyze_journal,
            analyze_journal_file,
            filter_against_existing,
            format_candidates,
        )

        journal = get_journal()

        if action == "stats":
            return json.dumps({
                "session_id": journal.session_id,
                "events": len(journal.events),
                "errors": journal.error_count,
                "solvers_used": sorted(journal.solvers_used),
                "physics_used": sorted(journal.physics_used),
                "duration_seconds": round(journal.duration_seconds, 1),
            }, indent=2)

        if action == "review":
            if len(journal.events) < 3:
                return "Session too short for knowledge extraction (< 3 tool calls)."
            candidates = analyze_journal(journal)
            # Filter against existing knowledge
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            if not candidates:
                return "No new knowledge candidates discovered in this session."
            # Store candidates for potential approval
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            return format_candidates(candidates)

        if action == "ingest":
            if not path:
                return (
                    "Usage: session_insights('ingest', path='<file_or_dir>')\n"
                    "Point at a session journal saved by the MCP server "
                    "(data/sessions/session_*.json) or a directory of "
                    "such files."
                )
            p = _Path(path)
            if not p.exists():
                return f"Path not found: {p}"
            sources: list[_Path] = (
                sorted(p.glob("session_*.json")) if p.is_dir() else [p]
            )
            if not sources:
                return f"No session_*.json files found in {p}"
            all_candidates: list = []
            errors: list[str] = []
            for s in sources:
                try:
                    all_candidates.extend(analyze_journal_file(s))
                except Exception as e:
                    # repr(e) keeps the exception type so a contributor
                    # can tell `KeyError('events')` from a `FileNotFoundError`.
                    errors.append(f"{s.name}: {e!r}")
            # Cross-source de-duplication on a normalised key (the in-file
            # analyzer runs fuzzy dedup already; cross-file dedup needs to
            # match that contract or near-identical entries from N journals
            # all survive as separate candidates).
            import re as _re
            _retry_re = _re.compile(r"\s*\(retry \d+\)\s*$", _re.IGNORECASE)
            def _norm_title(t: str) -> str:
                return " ".join(_retry_re.sub("", t).lower().split())
            best: dict[tuple[str, str, str], CandidateKnowledge] = {}
            for c in all_candidates:
                key = (
                    c.category.strip().lower(),
                    (c.solver or "").strip().lower(),
                    _norm_title(c.title),
                )
                if key not in best or c.confidence > best[key].confidence:
                    best[key] = c
            candidates = list(best.values())
            existing = _collect_existing_pitfalls()
            candidates = filter_against_existing(candidates, existing)
            _pending_candidates.clear()
            _pending_candidates.extend(candidates)
            header = (
                f"Ingested {len(sources)} journal file(s); "
                f"{len(all_candidates)} raw candidates -> "
                f"{len(candidates)} novel after dedup + filter.\n"
            )
            if errors:
                header += "Errors:\n  " + "\n  ".join(errors) + "\n"
            if not candidates:
                return header + "No new candidates."
            return header + format_candidates(candidates)

        if action == "approve_all":
            if not _pending_candidates:
                return "No pending candidates. Call session_insights('review') first."
            saved = _save_candidates(_pending_candidates, journal.session_id)
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Approved {count} candidate(s). Saved to: {saved}"

        if action == "reject_all":
            count = len(_pending_candidates)
            _pending_candidates.clear()
            return f"Rejected {count} candidate(s)."

        # The Actions list must match the docstring + the
        # actual dispatch branches in this function. Audit
        # 2026-06-01: 'ingest' was documented but missing
        # from this usage hint, so LLMs that hit an invalid
        # action never learned that ingest exists.
        return (
            "Usage: session_insights(action, path='')\n"
            "Actions: review, ingest, approve_all, reject_all, stats\n"
            "Use ingest with path=<session.json|dir> to "
            "analyse saved session journals."
        )

    # Storage for pending candidates between review and approve
    _pending_candidates: list = []


def _collect_existing_pitfalls() -> list[str]:
    """Gather all existing pitfall strings for novelty checking.

    Includes both built-in knowledge AND community contributions.

    Uses all_backends() (not available_backends): the pitfall
    library is a static catalog and the novelty check should
    compare against EVERY known pitfall, including those of
    backends the user has not installed locally. Filtering by
    availability would let a candidate that duplicates a
    dune-fem pitfall slip through as "novel" on any host
    without dune-fem. (Audit 2026-06-02.)
    """
    pitfalls = []
    try:
        for b in all_backends():
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if k and isinstance(k, dict) and "pitfalls" in k:
                    for pit in k["pitfalls"]:
                        if isinstance(pit, str):
                            pitfalls.append(pit)
                        elif isinstance(pit, dict) and "text" in pit:
                            pitfalls.append(pit["text"])
    except Exception:
        pass
    # Also include community contributions
    for c in _load_community_knowledge():
        pitfalls.append(c.get("title", ""))
    return pitfalls


def _load_community_knowledge(solver: str = "") -> list[dict]:
    """Load approved community knowledge from pending/ directory.

    Returns list of candidate dicts. Optionally filter by solver.
    """
    from pathlib import Path
    pending_dir = Path(__file__).parent.parent.parent / "data" / "community_knowledge" / "pending"
    if not pending_dir.exists():
        return []
    entries = []
    for f in sorted(pending_dir.glob("session_*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for entry in data:
                    if solver and entry.get("solver", "") != solver:
                        continue
                    entries.append(entry)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def _save_candidates(candidates: list, session_id: str) -> str:
    """Save approved candidates to community_knowledge/pending/."""
    from pathlib import Path
    pending_dir = Path(__file__).parent.parent.parent / "data" / "community_knowledge" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for c in candidates:
        entries.append(c.to_dict())

    path = pending_dir / f"session_{session_id}.json"
    path.write_text(json.dumps(entries, indent=2, default=str))
    return str(path)


# ═══════════════════════════════════════════════════════════════
# Helper functions for knowledge (copied from original tools)
# ═══════════════════════════════════════════════════════════════

def _capture_knowledge_fn(fn_name: str) -> str:
    """Reach into tools.knowledge.register_knowledge_tools to pull
    out one of the inline get_*_knowledge closure bodies.

    The three knowledge providers (coupling / TSI / preCICE) live
    inside register_knowledge_tools as nested @mcp.tool() closures,
    not as module-level functions. The consolidated tool surface
    needs to call them outside FastMCP's tool-dispatch path, so
    this helper builds a throwaway FastMCP instance, monkey-
    patches its `tool` decorator to capture every registered
    function by name, runs register_knowledge_tools, then calls
    the requested one.

    Failures here used to be wrapped in `except Exception: pass`
    and produced a bare "...knowledge not available" string to
    the LLM — silent degradation that hid genuine breakage of the
    capture trick (FastMCP API change, register_knowledge_tools
    refactor, missing tools.knowledge module, ...). The wrapper
    now surfaces the exception so the LLM and the developer can
    diagnose. (Audit 2026-06-02.)
    """
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        return (f"⚠ Cannot load knowledge subsystem: "
                f"`{type(exc).__name__}: {exc}`")
    mcp = FastMCP("tmp")
    captured: dict = {}
    orig = mcp.tool

    def cap(*a, **kw):
        d = orig(*a, **kw)

        def w(fn):
            r = d(fn)
            captured[fn.__name__] = fn
            return r
        return w

    mcp.tool = cap
    try:
        register_knowledge_tools(mcp)
    except Exception as exc:
        return (f"⚠ register_knowledge_tools failed while capturing "
                f"`{fn_name}`: `{type(exc).__name__}: {exc}`")
    if fn_name not in captured:
        return (f"⚠ `{fn_name}` was not registered by "
                f"register_knowledge_tools. Captured: "
                f"{sorted(captured.keys())}")
    try:
        return captured[fn_name]()
    except Exception as exc:
        return (f"⚠ `{fn_name}()` raised: "
                f"`{type(exc).__name__}: {exc}`")


def _get_coupling_knowledge():
    """Return coupling knowledge string (or a visible error block)."""
    return _capture_knowledge_fn("get_coupling_knowledge")


def _get_tsi_knowledge():
    return _capture_knowledge_fn("get_tsi_knowledge")


def _get_precice_knowledge():
    return _capture_knowledge_fn("get_precice_knowledge")


