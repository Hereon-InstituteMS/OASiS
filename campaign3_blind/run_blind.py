#!/usr/bin/env python
"""Campaign-3 blind runner.

Executes one agent per (problem, model, condition, seed) on a BLIND task and
records everything needed for cost and energy accounting. It deliberately does
NOT grade: grading happens afterwards, offline, with grade_blind.py, which is
the only component that opens a sealed key.

Safety properties:
  * the answer keys are sealed (unreadable) while this runs — the runner
    refuses to start otherwise;
  * every run gets a fresh directory and an existing ledger is never
    overwritten (a relaunch resumes and fills gaps);
  * tokens in/out, wall time and tool calls are recorded per run;
  * the live trajectory is written as it happens, so a crash or timeout cannot
    destroy the evidence the leak audit needs.

Usage:
  ../../open-fem-agent/.venv-lg/bin/python run_blind.py \
      --model 27b --conditions BARE MCP --problems B1 B2 D1 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_envf = ROOT / ".env"
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

REPO = Path(os.environ.get("OASIS_REPO", "/home/alexander/Schreibtisch/open-fem-agent"))
sys.path.insert(0, str(REPO / "langgraph_eval"))
import agent as _agent                                            # noqa: E402
from agent import build_bare_agent, build_mcp_agent               # noqa: E402
from langchain_openai import ChatOpenAI                           # noqa: E402
from langchain_core.callbacks import (UsageMetadataCallbackHandler,  # noqa: E402
                                      BaseCallbackHandler)

OR_MODELS = {
    "27b": "qwen/qwen3.5-27b",
    "122b": "qwen/qwen3.5-122b-a10b",
    "397b": "qwen/qwen3.5-397b-a17b",
}

ENVIRON = (
    "\nENVIRONMENT: NGSolve & scikit-fem -> "
    f"{REPO}/.venv/bin/python ; "
    "FEniCSx/dolfinx -> /home/alexander/miniconda3/envs/fenics/bin/python ; "
    "DUNE-fem -> /home/alexander/miniconda3/envs/dune-fem-env/bin/python ; "
    "Kratos Multiphysics (and gmsh) -> /usr/bin/python3 ; "
    "deal.II -> build C++ with cmake using DEAL_II_DIR=/home/alexander/dealii/build "
    "(run with LD_LIBRARY_PATH=/opt/4C-dependencies/lib) ; "
    "4C binary -> /home/alexander/4C/build/4C ; "
    "FEBio binary -> /home/alexander/FEBio/bin/febio4 .\n"
)

_USAGE_CB = UsageMetadataCallbackHandler()


def _usage_totals():
    ti = sum(int(u.get("input_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    to = sum(int(u.get("output_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    return ti, to


def _or_llm(size, *, temperature, seed):
    return ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"],
                      model=OR_MODELS[size], temperature=temperature, seed=seed,
                      timeout=600, max_retries=30, callbacks=[_USAGE_CB])


_agent._llm = _or_llm

_INFRA_ERRS = ("APIConnectionError", "Connection error", "UnicodeDecodeError",
               "InternalServerError", "empty response", "Error code: 5",
               "RateLimit", "ReadTimeout", "ServiceUnavailable")


class TrajLiveLog(BaseCallbackHandler):
    """Append tool activity to disk as it happens, so a crash or timeout
    cannot destroy the trajectory the leak audit depends on."""

    def __init__(self, path):
        self._f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def on_tool_start(self, serialized, input_str, **kw):
        try:
            self._f.write(f"TOOL_CALL {(serialized or {}).get('name')} | "
                          f"{str(input_str)[:600]}\n")
        except Exception:
            pass

    def on_tool_end(self, output, **kw):
        try:
            self._f.write(f"TOOL_RESULT {str(output)[:600]}\n")
        except Exception:
            pass


sys.path.insert(0, str(REPO / "src"))
from blind_eval import keyvault as _kv                             # noqa: E402


def keys_are_sealed() -> bool:
    """The runner must never execute while the answer keys are readable.

    The previous implementation returned True when ``keys/`` was MISSING or
    EMPTY, so a deleted keys tree read as sealed and the campaign would have
    started with nothing to grade against.  Absence is not a seal.  The real
    check lives in the OASiS repo, under version control and under test, and is
    imported rather than restated here -- Amendment 1 declared this fixed while
    the runner still carried the broken copy and never imported the fix.
    """
    return _kv.is_sealed(HERE / "keys")


def preflight_or_die(problems: list) -> None:
    """Every custody control, executed, before a single paid run starts.

    One cell run with the keys readable makes every cell's blindness arguable
    afterwards, and there is no way to prove afterwards that they were sealed.
    So this is a hard gate, it runs the checks rather than asserting them, and
    it refuses on anything it cannot verify.
    """
    import subprocess
    keys = HERE / "keys"
    failures = []

    if not keys_are_sealed():
        failures.append(f"answer keys at {keys} are not sealed "
                        f"(exists={keys.exists()})")
    else:
        probe = _kv.verify_unreadable(keys)
        if not probe.get("sealed"):
            failures.append(f"the seal did not survive an executed read: {probe}")

    plain = [p for p in keys.rglob("*.json")
             if not p.name.endswith(".enc")] if keys.is_dir() else []
    if plain:
        failures.append(f"{len(plain)} plaintext key file(s) still on disk")

    for src in (HERE / "build_problems.py", HERE / "build_extra.py",
                HERE / "build_coupled.py"):
        if src.exists() and os.access(src, os.R_OK):
            failures.append(f"{src.name} is readable and holds hidden fields "
                            f"as literals two directories above the agent's "
                            f"workdir")

    man = REPO / "data" / "blind_key_commitment.json"
    if not man.is_file():
        failures.append(f"no key commitment at {man}: without it, nobody can "
                        f"check the solutions were fixed before the runs")

    # The exposure sweep, executed, over the whole agent-reachable tree.
    sweep = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(HERE)],
        capture_output=True, text=True, timeout=600)
    if sweep.returncode != 0:
        failures.append("exposure sweep found readable solution-bearing files:\n"
                        + sweep.stdout[-1500:])

    if failures:
        sys.exit("REFUSING TO RUN — custody preflight failed:\n  - "
                 + "\n  - ".join(failures))
    print(f"[preflight] custody controls verified for {len(problems)} "
          f"problem(s); keys sealed and unreadable by execution")


def per_cell_exposure_check(run_dir: Path) -> None:
    """After every cell, sweep the tree the tooling just wrote.

    Amendment 1 found seven plaintext copies of live solutions in scratch space
    AFTER the design declared custody solved -- sub-agent transcripts that had
    quoted key files, a keys snapshot, its __pycache__.  Custody has to cover
    the space tooling writes, not only the directory the design names, and a
    sweep that runs once at the start cannot see what a run creates.
    """
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(run_dir)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        (run_dir / "EXPOSURE_ALERT.txt").write_text(r.stdout)
        print(f"  !! EXPOSURE in {run_dir.name}: a solution is readable from "
              f"the run tree; see EXPOSURE_ALERT.txt", flush=True)


def run_one(pid: str, model: str, cond: str, seed: int, timeout_s: int) -> dict:
    run_dir = HERE / "runs" / f"{pid}_{model}_{cond}_seed{seed}"
    ledger = run_dir / "ledger.json"
    if ledger.exists():
        print(f"[{pid} {model} {cond} s{seed}] SKIP: ledger exists", flush=True)
        return json.loads(ledger.read_text())

    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    task = (HERE / "problems" / pid / "task.txt").read_text(encoding="utf-8")
    prompt = task + ENVIRON

    _USAGE_CB.usage_metadata.clear()
    ag = (build_bare_agent if cond == "BARE" else build_mcp_agent)(
        size=model, seed=seed, workdir=work)

    live = TrajLiveLog(work / "trajectory_live.txt")
    t0, err, final, n_calls = time.time(), None, None, 0

    async def _invoke():
        return await asyncio.wait_for(
            ag.ainvoke({"messages": [("user", prompt)]},
                       config={"recursion_limit": 250, "callbacks": [live]}),
            timeout=timeout_s)

    try:
        final = asyncio.run(_invoke())
    except asyncio.TimeoutError:
        err = f"TimeoutError: exceeded {timeout_s}s"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    try:
        msgs = (final or {}).get("messages", []) if isinstance(final, dict) else []
        lines = []
        for m in msgs:
            for tc in (getattr(m, "tool_calls", None) or []):
                lines.append(f"TOOL_CALL {tc.get('name')}")
                n_calls += 1
            c = getattr(m, "content", "") or ""
            if c:
                lines.append(str(c)[:1500])
        (work / "trajectory.txt").write_text("\n".join(lines), errors="replace")
    except Exception:
        pass
    if n_calls == 0 and (work / "trajectory_live.txt").exists():
        n_calls = sum(1 for ln in (work / "trajectory_live.txt")
                      .read_text(errors="replace").splitlines()
                      if ln.startswith("TOOL_CALL "))

    tin, tout = _usage_totals()
    rec = dict(problem=pid, model=model, condition=cond, seed=seed,
               wall_s=round(time.time() - t0, 1), tool_calls=n_calls,
               tokens_in=tin, tokens_out=tout, error=err,
               graded=False, note="grading is offline: run grade_blind.py")
    if err and any(s in err for s in _INFRA_ERRS):
        rec["outcome"] = "INVALID_INFRA"
    ledger.write_text(json.dumps(rec, indent=2))
    print(f"[{pid} {model} {cond} s{seed}] done  calls={n_calls} "
          f"tok={tin}/{tout} {rec['wall_s']}s"
          + (f"  ERR {err[:60]}" if err else ""), flush=True)
    # Custody is checked per cell, not once at launch: a run creates scratch
    # that did not exist when the campaign started.
    per_cell_exposure_check(run_dir)
    if not keys_are_sealed():
        sys.exit(f"HALTING AFTER {run_dir.name}: the answer keys became "
                 f"readable during the run. Every later cell would be "
                 f"unarguably non-blind.")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(OR_MODELS))
    ap.add_argument("--conditions", nargs="+", required=True,
                    choices=["BARE", "MCP"])
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phase", required=True,
                    choices=["development", "evaluation"],
                    help="development: refine knowledge, run post-mortems, "
                         "check convergence. evaluation: the paper's numbers — "
                         "requires a freeze marker and held-out instances.")
    ap.add_argument("--timeout", type=int, default=2700,
                    help="per-run wall-clock cap in seconds")
    a = ap.parse_args()

    preflight_or_die(a.problems)

    # Which PHASE is this? Development runs against spent instances are fine and
    # expected; an EVALUATION run must not grade a problem the knowledge was
    # tuned on. Nothing in this runner used to distinguish them, so the whole
    # protection rested on somebody remembering — and a rule that depends on
    # memory is not a rule. `--phase evaluation` demands a freeze marker and
    # refuses any instance that matches a development one by id or by the
    # fingerprint of its task text, so a rename cannot launder it.
    if a.phase == "evaluation":
        import phase as _phase
        try:
            _phase.assert_evaluation_is_clean(HERE / "problems")
        except _phase.EvaluationNotCleanError as exc:
            sys.exit(f"REFUSING TO RUN AN EVALUATION: {exc}")
        overlap = sorted(set(a.problems) & set(_phase.DEVELOPMENT))
        if overlap:
            sys.exit("REFUSING TO RUN AN EVALUATION on development instances "
                     f"{overlap}: the knowledge was shaped against these, so "
                     f"grading them measures the tuning and not the tool.")
    else:
        print("[phase: development] results are for post-mortems and "
              "convergence checking, NOT for the paper's evaluation table.")

    for pid in a.problems:
        for cond in a.conditions:
            run_one(pid, a.model, cond, a.seed, a.timeout)


if __name__ == "__main__":
    main()
