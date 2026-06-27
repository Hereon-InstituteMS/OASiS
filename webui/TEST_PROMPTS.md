# OASiS WebUI — manual test prompts

Paste each prompt into the chat box (Ctrl+↵ to send). For every test
this file tells you what to look at in the UI, what the expected
observable is, and how to know it failed. After you've worked through
them, hand me back the screen/event-log notes and we'll verify.

## Setup

```bash
cd ~/Schreibtisch/open-fem-agent
# kill any old instance
fuser -k 8080/tcp 2>/dev/null || true
# start the UI (foreground; Ctrl+C to stop)
.venv-lg/bin/uvicorn webui.app:app --port 8080
```

Open http://localhost:8080. The first cell of each test tells you
which model to pick in the left sidebar and whether the OASiS MCP
toggle should be on or off.

For the Qwen tests, you also need vLLM running on the chosen port:
```bash
# in a separate terminal, pick the size:
./langgraph_eval/start_vllm.sh 7b      # → http://localhost:8000
./langgraph_eval/start_vllm.sh 14b     # → http://localhost:8001
./langgraph_eval/start_vllm.sh 32b     # → http://localhost:8002
```
Tests that can run on the **mock** model are explicitly marked — those
work without any GPU and are useful for verifying wiring quickly.

---

## Test 1 — Smoke test on the mock model (no GPU)

**Settings:** Model = `Mock LLM (no GPU)`, Mode = `accept`, MCP = **off**.

```
Plan a Poisson MMS demo on a 32x32 grid and use the critic to
validate the setup.
```

**What to look for in the UI:**
- A green “You” bubble appears with the prompt.
- An amber “Tool call” bubble shows `spawn_subagent` with the args.
- A violet “Sub-agent” bubble appears with role=`critic` and the task
  text.
- Another violet “Sub-agent return” bubble shows `APPROVED: setup
  and units look consistent; proceed.`
- A grey “Tool result” bubble carries the same APPROVED message.
- An accent-coloured “Done” bubble closes the turn with text like
  *“Parent: critic approved; recording result.”*
- The token counter in the left sidebar shows `in=3 out=3`.
- The connection dot in the header stays green (pulsing).

**Pass criteria:** all six bubble types above appear in that order;
token counter increments; no `Error` bubble.

**Why this matters:** confirms the agent loop, sub-agent spawning,
mode gating, token accounting and the chat rendering all work
end-to-end without any GPU.

---

## Test 2 — Real Qwen, single-step Poisson (mock vs 7B vs 14B vs 32B)

**Repeat once per model.** Settings: Model = the Qwen of that
iteration, Mode = `accept`, MCP = **off**.

```
Solve the 2D Poisson equation -Δu = f on the unit square [0,1]² with
homogeneous Dirichlet BC and the manufactured solution
u_exact(x,y) = sin(π x) sin(π y). Use scikit-fem with P1 elements on
a uniform 32×32 mesh. Write a Python script in the sandbox, run it,
report the L2 error of (u_h - u_exact) as a single line:
    RESULT l2_error = <value>
Then write that line to /tmp/oasis_webui_t2.txt.
```

**What to look for:**
- The agent should call `run_bash` (and maybe `write_file`) — you see
  amber Tool-call bubbles for each.
- Watch the right pane: refresh the sandbox root, the new file
  `webui_<sid>/work/<something>.py` should appear under
  `eval_interactive/`.
- Click the `.py` file → the parameter sliders should populate from
  any top-level numeric assignments (N, mesh size, etc.).
- The L2 error reported in the chat should be around `1e-2` for a 32²
  mesh (analytic O(h²) ≈ 0.01).
- Token counter increases monotonically.

**Pass criteria:**
- Final chat message contains `RESULT l2_error =` with a number in
  [0.003, 0.05].
- The file `/tmp/oasis_webui_t2.txt` exists and contains the same
  number.
- The script the agent wrote is visible in the file browser and Edit
  → Save → reopen round-trips.

**Compare across models:** the 7B run will usually take 2–5× longer
than 14B, which is slower than 32B. Failure modes should be reported:
e.g. 7B forgetting to write the file, 7B mis-importing skfem.

---

## Test 3 — Plan mode with approve / reject

**Settings:** Model = `Mock LLM (no GPU)`, Mode = `plan`, MCP = **off**.

```
Begin the first major step of solving a 1D heat equation on [0,1].
```

**What to look for:**
1. An amber `Tool call (pending)` bubble appears with two buttons:
   **Approve** and **Reject**.
2. Click **Approve** → an amber `Tool call` bubble (without buttons)
   shows up, followed by a grey `Tool result`, then a `Done`.
3. Start a second turn with the same prompt — this time hit **Reject**
   and type any reason → the tool call should NOT execute. You should
   see a `Tool result` containing `[rejected by user: <reason>]` and
   then a `Done`.

**Pass criteria:** Approve actually runs the tool; Reject genuinely
short-circuits it. The chat should not be locked up after Reject (you
should be able to send another prompt).

---

## Test 4 — Real Qwen + plan mode + a multi-step problem (battle test)

**Settings:** Model = `qwen2.5-14b` (or 7B if 14B isn't loaded),
Mode = `plan`, MCP = **off**.

```
Set up a 2D Stokes flow demo in scikit-fem on the unit square. Use
Taylor-Hood (P2 velocity / P1 pressure), inflow on the left edge
u=(1,0), outflow on the right, no-slip top and bottom. Mesh 24x24.
After you compute the velocity field, write a single line
"RESULT u_max = <value>" to the sandbox.
For every major step (mesh, BCs, weak form, solve, post-process) you
MUST first spawn a critic sub-agent to review your plan, wait for
APPROVED, then continue.
```

**What to look for:**
- Multiple sub-agent bubbles (one per major step), each with role=
  `critic`.
- Between each critic round you should get a `Tool call (pending)` for
  the next action — approve them yourself.
- The final `Done` should arrive with a value in the chat.

**Pass criteria:**
- ≥ 3 distinct `Sub-agent` events fire across the turn.
- The pending tool calls actually wait for your approval (the chat
  pauses; the “Send” button is greyed out — wait, it isn't, but the
  agent doesn't progress).
- The final `RESULT u_max` is in [0.6, 1.1] (for unit inflow this
  cavity-like setup).

**Failure modes worth reporting:** the agent forgets to spawn the
critic between steps (the prompt explicitly mandates it); reject one
of the critic verdicts and see whether the agent re-plans.

---

## Test 5 — File browser + in-browser editor (round-trip to disk)

No prompt needed first. Settings irrelevant.

1. In the right pane, navigate into `E5_BARE_seed0_v2/work`.
2. Click `cavity.py`. The text body should appear in the lower pane
   along with a row of detected parameters: `N`, `nx`, `ny`.
3. Click **✎ Edit** at the top of the file view → a textarea opens.
4. Change one comment line (e.g. add `# touched by manual test`).
5. Click **Save** → you should see `saved (<N> B)` in green.
6. Click another file and come back to `cavity.py` → your edit should
   be there.

**Pass criteria:** edits actually persist to disk; cancelling without
Save discards changes; trying to save with the path manipulated to
escape the sandbox (you'd have to do this via the API; not via the UI)
is blocked.

---

## Test 6 — Interactive plot with editable axes and PNG/SVG export

1. In the file browser, open
   `B3_BARE_seed0_v2/work/forces.csv`. A Plotly chart appears with
   three traces (`t`, `CL`, `CD`).
2. Click the chart **title** — it becomes editable. Type a new title.
3. Click an **axis label** — editable. Rename it.
4. Drag a legend entry into the chart → it sticks.
5. Click the **PNG** button (above the chart) → a PNG download starts
   with filename `forces.png`.
6. Repeat with **SVG**.
7. Use Plotly's native toolbar (top-right of the plot) — there's a
   camera icon for high-res PNG export too.

**Pass criteria:** every edit is in-place; both downloads produce a
real image (open and confirm the title/axis edits are baked in).

---

## Test 7 — VTK render (real mesh)

1. In the file browser, navigate into a workdir that produced VTK
   output, e.g.
   `B4_MCP_NO_CRITIC_seed1_v2/work` (mortar contact run).
2. Click any `.vtp` file (vtk.js full pipeline) — a 3D viewport
   should appear with the mesh.
   - Drag with the mouse to rotate.
   - **Reset camera** button restores the default view.
   - The scalar-range inputs let you remap the colour scale; click
     **Apply** and the rendering should update.
   - **Save PNG** captures the canvas to disk.

> Caveat: the production solvers ship `.vtu` / `.pvd` files for
> unstructured grids. The current pipeline uses
> `vtkXMLPolyDataReader` (best fit for `.vtp`). For full `.vtu`
> rendering, swap in `vtkXMLUnstructuredGridReader` in
> `static/app.js::renderVtk`. The file URL serving (`/sandbox-file/`)
> already works for all formats.

**Pass criteria:** mesh appears, rotates with mouse drag, range
sliders re-tint the surface, PNG save downloads a file. If no `.vtp`
exists in your workdirs, this test is skipped; the URL fetch still
shows the right `vtk.js` was loaded by checking
`http://localhost:8080/sandbox-file/<rel>.vtu` returns a binary blob.

---

## Test 8 — Multi-prompt session + reconnect

**Settings:** any model; **stay on the same session id** the whole
test. Mode = `accept`.

Prompt #1:
```
Plan a Poisson MMS demo and use the critic.
```

Wait for `Done`. Then **close the browser tab**. Reopen
http://localhost:8080. In the sidebar, **click the previous session
id under "Recent sessions"**. The chat should rehydrate the full
event history.

Prompt #2 in the same session:
```
Extend the previous plan to 3D and re-validate with the critic.
```

Wait for `Done`. Then click **Restart** in the sidebar. The chat
should clear; token counters go back to 0; the session id stays the
same.

Prompt #3 after restart:
```
Summarise what we did so far.
```

**Pass criteria:**
- Reconnect rehydrates events (you see the prompt-1 history in the
  chat).
- Prompt-2 result lands in the same session JSON
  (`data/webui_sessions/<sid>.json` grows).
- Restart empties the chat and tokens.
- Prompt-3 starts from an empty history.

---

## Test 9 — OASiS MCP toggle on / off

**Setup:** kill vLLM (or use mock) so you can run with and without
MCP without GPU pressure.

A. **Mock model, MCP off:** ask a problem prompt. Expect: agent
   only has bash/web/spawn_subagent. The mock answer ignores the MCP
   so this is a sanity check that the agent builds.

B. **Mock model, MCP ON:** tick the OASiS checkbox in the sidebar
   → a status bubble announces *"MCP servers updated; agent will
   rebuild on next prompt"*. Send another prompt. The OASiS
   server boots as a subprocess (you'll briefly see backend-
   registration log lines if you watch the uvicorn terminal). The
   mock LLM still ignores them, but the OASiS tools should appear in
   the wrapped tool list (visible in plan mode as available tool
   names).

C. **Real Qwen 14B, MCP ON:** Mode = `plan`, prompt:
   ```
   Use the OASiS knowledge tool to retrieve the FEniCSx Poisson
   pitfalls and summarise them.
   ```
   Expect: a `tool_call_pending` for `knowledge` (an OASiS tool, not
   bash). Approve. The result should be a JSON blob from OASiS.

**Pass criteria:** toggling MCP triggers a rebuild status; with MCP
on, OASiS tool names (knowledge, prepare_simulation, examples,
discover, …) are reachable by the real model in plan mode.

---

## Reporting back

For each test, jot down:

- The model + mode + MCP setting you used
- Whether each bullet of "What to look for" actually happened
- Anything that surprised you (slow steps, missing events, an Approve
  button that didn't reappear after a Reject, a plot that wouldn't
  export, etc.)
- For real Qwen runs: the final answer, the wall-clock time, whether
  the script the agent wrote in the sandbox actually runs when you
  re-execute it manually
- Screenshots of any rendering quirks

When you hand it back I'll pair the observed behaviour with the event
stream in `data/webui_sessions/<sid>.json` and we'll either tighten
the checks or fix the regressions.
