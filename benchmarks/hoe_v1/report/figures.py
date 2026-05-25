"""Paper-ready figures for HOE-v1.

After redesign per reviewer critique:
  Fig R1: per-task × condition pass-rate heatmap with discrete colormap,
          bold tier dividers, marginal column means, k/N annotations on every cell.
  Fig R2: per-task observed convergence rate (Tier C only) — bars per
          condition with a horizontal line at the formal order. (Replaces
          the degenerate scatter from v1.)
  Fig R3: behavioural ablation — fraction of cells where the agent
          actually spawned a critic sub-agent, per condition. (Replaces
          the cost-Pareto plot, which was misleading because the MCP
          conditions did not have metered cost.)
  Fig R4: per-tier pass rate × condition with 95% Wilson CIs, k/N
          numerical labels, and an inset showing the paired pooled deltas
          with task-bootstrap 95% intervals.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORT_OUT = ROOT / "report" / "out"
CSV = REPORT_OUT / "all_rows.csv"
STATS_JSON = REPORT_OUT / "stats.json"
SESSION_DIR = Path("/home/alexander/.claude/projects/-home-alexander-Schreibtisch-open-fem-agent")

TIER_OF = {
    **{f"A{i}": "A" for i in range(1, 6)},
    **{f"B{i}": "B" for i in range(1, 6)},
    **{f"C{i}": "C" for i in range(1, 6)},
    **{f"D{i}": "D" for i in range(1, 3)},
}
TASK_ORDER = sorted(TIER_OF, key=lambda t: (t[0], int(t[1:])))
COND_ORDER = ["BARE", "MCP_FULL", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC"]
COND_LABEL_TWO_LINE = {
    "BARE":              "BARE\n(no MCP)",
    "MCP_FULL":          "MCP\n(full)",
    "MCP_NO_PITFALL_DB": "MCP\n(no pitfall DB)",
    "MCP_NO_CRITIC":     "MCP\n(no critic)",
}
COND_LABEL = {
    "BARE":              "BARE (no MCP)",
    "MCP_FULL":          "MCP (full)",
    "MCP_NO_PITFALL_DB": "MCP (no pitfall DB)",
    "MCP_NO_CRITIC":     "MCP (no critic)",
}
COND_COLOR = {
    "BARE":              "#9E9E9E",
    "MCP_FULL":          "#2E7D32",
    "MCP_NO_PITFALL_DB": "#FB8C00",
    "MCP_NO_CRITIC":     "#1565C0",
}
TIER_DESC = {
    "A": "Textbook (5 tasks)",
    "B": "Compositional engineering (5 tasks)",
    "C": "MMS verification (5 tasks)",
    "D": "Adversarial / developer (2 tasks)",
}


def _load() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CSV.open() as fh:
        for r in csv.DictReader(fh):
            r["passed"] = r["passed"] in ("True", "1", "true")
            try: r["seed"] = int(r["seed"])
            except: r["seed"] = -1
            r["tier"] = TIER_OF.get(r["task_id"], "?")
            rows.append(r)
    return rows


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0: return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = (z/(1 + z*z/n)) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0.0, centre-half), min(1.0, centre+half)


# --------------------------------------------------------------------------- #
# Fig R1 — heatmap (redesigned)
# --------------------------------------------------------------------------- #
def fig_r1_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    # k/3 per (task, condition)
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        buckets[(r["task_id"], r["condition"])].append(bool(r["passed"]))
    M = np.full((len(TASK_ORDER), len(COND_ORDER)), np.nan)
    for i, t in enumerate(TASK_ORDER):
        for j, c in enumerate(COND_ORDER):
            cs = buckets.get((t, c), [])
            if cs:
                M[i, j] = sum(cs) / 3.0  # always 3 seeds expected

    # Discrete 4-level colormap
    cmap = ListedColormap(["#C62828", "#EF6C00", "#F9A825", "#2E7D32"])  # 0, 1/3, 2/3, 3/3
    norm = BoundaryNorm([0.0, 0.17, 0.5, 0.83, 1.001], ncolors=4)

    fig = plt.figure(figsize=(6.8, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[5.2, 0.6],
                          height_ratios=[14.0, 0.7],
                          hspace=0.05, wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_cb = fig.add_subplot(gs[0, 1])

    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_LABEL_TWO_LINE[c] for c in COND_ORDER], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER, fontsize=8.5)

    # k/3 annotations on every cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{int(round(3*v))}/3"
            color = "white" if (np.isnan(v) or v <= 0.5) else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")

    # Bold tier dividers
    for cum, name in zip([5, 10, 15], "BCD"):
        ax.axhline(cum - 0.5, color="black", lw=1.7)
    # Tier brackets on left
    for tier, (top, bot) in zip("ABCD", [(0, 4), (5, 9), (10, 14), (15, 16)]):
        ax.annotate(
            "", xy=(-0.85, top - 0.4), xytext=(-0.85, bot + 0.4),
            xycoords="data",
            arrowprops=dict(arrowstyle="-", lw=1.5, color="black"),
            annotation_clip=False,
        )
        ax.text(-1.05, (top + bot) / 2, f"Tier {tier}",
                rotation=90, va="center", ha="center",
                fontsize=9, fontweight="bold")

    # Highlight failing cells with a black outline
    for i, t in enumerate(TASK_ORDER):
        for j, c in enumerate(COND_ORDER):
            v = M[i, j]
            if not np.isnan(v) and v < 0.5:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="black",
                                       lw=1.6, zorder=10))

    ax.set_xlim(-0.55, len(COND_ORDER) - 0.45)
    ax.set_title("Seeds passed per task and condition",
                 fontsize=11, pad=6)

    # Bottom marginal: column means with k/51 labels AND condition labels on x
    col_mean = np.array([np.nanmean(M[:, j]) for j in range(M.shape[1])])
    ax_m.imshow(col_mean[None, :], cmap=cmap, norm=norm,
                aspect="auto", interpolation="nearest")
    ax_m.set_yticks([0]); ax_m.set_yticklabels(["pooled"], fontsize=8.5)
    # k/51 pooled annotation per column
    for j, cond in enumerate(COND_ORDER):
        cs = [r for r in rows if r["condition"] == cond]
        k = sum(1 for r in cs if r["passed"])
        n = len(cs)
        ax_m.text(j, 0, f"{k}/{n}",
                  ha="center", va="center",
                  color="white", fontsize=9, fontweight="bold")
    ax_m.set_xticks(range(len(COND_ORDER)))
    ax_m.set_xticklabels([COND_LABEL_TWO_LINE[c] for c in COND_ORDER],
                         fontsize=8.5)
    ax_m.tick_params(axis="x", length=0, pad=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0)

    # Discrete 4-level colour bar with one tick at the centre of each band
    # (red 0/3, orange 1/3, yellow 2/3, green 3/3). BoundaryNorm boundaries
    # are [0.0, 0.17, 0.5, 0.83, 1.001], so band centres are:
    band_centres = [0.085, 0.335, 0.665, 0.917]
    cb = fig.colorbar(im, cax=ax_cb, ticks=band_centres)
    cb.ax.set_yticklabels(["0/3", "1/3", "2/3", "3/3"], fontsize=8.5)
    cb.ax.tick_params(length=0, pad=4)
    cb.outline.set_visible(False)
    cb.ax.set_title("seeds\npassed", fontsize=8, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig R2 — per-task Tier C observed convergence orders
# --------------------------------------------------------------------------- #
def fig_r2_tier_c_orders(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-task bar chart of observed mean convergence rate (averaged over seeds),
    with a horizontal dashed line at the formal order. Replaces the previous
    scatter, which collapsed to two vertical strips."""
    # We need to recompute by reading grader verdicts (interactive cells store
    # them under <cell>/grader_verdict.json? — actually they don't; we re-grade
    # on the fly to get observed rates).
    sys.path.insert(0, str(ROOT.parent.parent))
    from benchmarks.hoe_v1.grader.grader import grade, load_spec

    obs_by_task_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
    formal_by_task: dict[str, float] = {}
    for r in rows:
        if not r["task_id"].startswith("C"):
            continue
        cell_dir = Path(r["cell_dir"])
        result_path = cell_dir / ("work" / Path("result.txt")) \
            if "eval_interactive" in str(cell_dir) else cell_dir / "work" / "result.txt"
        if not result_path.exists():
            continue
        spec = load_spec(ROOT / "tasks" / f"{r['task_id']}.yaml")
        try:
            v = grade(spec, result_path, seed=r["seed"], run_dir=result_path.parent)
        except Exception:
            continue
        details = v.details or {}
        # single-axis MMS
        if "mean_rate" in details:
            obs = details["mean_rate"]
            formal = details.get("formal_order", 2.0)
        else:
            axes = [d for d in details.values()
                    if isinstance(d, dict) and "mean_rate" in d]
            if not axes:
                continue
            obs = sum(a["mean_rate"] for a in axes) / len(axes)
            formal = sum(a.get("formal_order", 2.0) for a in axes) / len(axes)
        obs_by_task_cond[(r["task_id"], r["condition"])].append(float(obs))
        formal_by_task[r["task_id"]] = float(formal)

    c_tasks = sorted([t for t in TASK_ORDER if t.startswith("C")])
    NARR_ORDER = ["BARE", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC", "MCP_FULL"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(c_tasks))
    bar_w = 0.19
    pad_in_group = 0.012
    for j, cond in enumerate(NARR_ORDER):
        means, errs = [], []
        for t in c_tasks:
            vals = obs_by_task_cond.get((t, cond), [])
            if vals:
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0); errs.append(0.0)
        offset = (j - 1.5) * (bar_w + pad_in_group)
        ax.bar(x + offset, means, bar_w,
               yerr=errs, capsize=3,
               color=COND_COLOR[cond], edgecolor="white",
               linewidth=0.8, label=COND_LABEL[cond],
               error_kw={"elinewidth": 0.9, "ecolor": "#222222"},
               zorder=2)
    # Reference lines at formal order per task
    for i, t in enumerate(c_tasks):
        p = formal_by_task.get(t, 2.0)
        ax.hlines(p, i - 0.45, i + 0.45, colors="#222222",
                  linestyles="--", linewidths=1.1,
                  label="formal order $p$" if i == 0 else None,
                  zorder=3)
        ax.text(i, p + 0.07, f"$p={p:g}$", ha="center", va="bottom",
                fontsize=8, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(c_tasks, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlabel("Tier C task", fontsize=10)
    ax.set_ylabel("observed convergence rate", fontsize=10)
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=5, fontsize=8.5, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig R3 — behavioural ablation: critic spawn rate per condition
# --------------------------------------------------------------------------- #
def _agent_spawns_per_cell() -> dict[str, dict[str, Any]]:
    """Return {cell_id: {agent_spawns, mcp_calls}} from interactive session
    logs. BARE cells get filled as agent=0, mcp=0 from sandboxed transcripts."""
    out: dict[str, dict[str, Any]] = {}
    if SESSION_DIR.exists():
        for f in SESSION_DIR.glob("*.jsonl"):
            cell = None
            n_agent = n_mcp = 0
            try:
                for line in f.open():
                    try: d = json.loads(line)
                    except: continue
                    if cell is None and d.get("type") == "user":
                        c = d.get("message", {}).get("content", "")
                        if isinstance(c, str):
                            m = re.search(
                                r"eval_interactive/([A-Z]\d+_(?:MCP_[A-Z_]+|BARE)_seed\d+)(?:_rerun)?/", c
                            )
                            if m:
                                cell = m.group(1)
                    msg = d.get("message") or {}
                    for blk in msg.get("content", []) or []:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            name = blk.get("name", "")
                            if name == "Agent": n_agent += 1
                            if "mcp__" in name: n_mcp += 1
            except Exception:
                continue
            if cell:
                # Multiple sessions can map to the same cell (rerun); keep max
                prev = out.get(cell, {"agent_spawns": 0, "mcp_calls": 0})
                out[cell] = {
                    "agent_spawns": max(prev["agent_spawns"], n_agent),
                    "mcp_calls": max(prev["mcp_calls"], n_mcp),
                }
    return out


def fig_r3_critic_spawn(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Critic-sub-agent spawn rate per condition (share of cells with at
    least one spawn). Same visual idiom as Figures 5 and 6: vertical
    bars, condition colours, Wilson 95% CIs as whiskers, value labels
    above the CI caps. BARE has no critic instruction and no MCP → 0
    by construction; its CI is therefore degenerate at zero.
    """
    spawns = _agent_spawns_per_cell()
    by_cond_cells: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        cell_id = f"{r['task_id']}_{r['condition']}_seed{r['seed']}"
        if r["condition"] == "BARE":
            by_cond_cells[r["condition"]].append(False)
        else:
            spawn = spawns.get(cell_id)
            by_cond_cells[r["condition"]].append(
                bool(spawn and spawn["agent_spawns"] > 0)
            )

    # Same narrative order as Figures 5 and 6
    NARR_ORDER = ["BARE", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC", "MCP_FULL"]
    counts = []
    for c in NARR_ORDER:
        cs = by_cond_cells.get(c, [])
        n = len(cs)
        k = sum(cs)
        p = k / max(n, 1)
        lo, hi = _wilson(k, n)
        counts.append((c, k, n, p, lo, hi))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(NARR_ORDER))
    rates = np.array([c[3] for c in counts])
    lo_arr = rates - np.array([c[4] for c in counts])
    hi_arr = np.array([c[5] for c in counts]) - rates

    ax.bar(x, rates, width=0.62,
           color=[COND_COLOR[c[0]] for c in counts],
           edgecolor="white", linewidth=1.0, zorder=2)
    ax.errorbar(x, rates, yerr=[lo_arr, hi_arr],
                fmt="none", ecolor="#222222",
                capsize=5, capthick=1.2, lw=1.2, zorder=3)

    for i, c in enumerate(counts):
        ax.text(i, c[5] + 0.025,
                f"{c[3]*100:.1f}%\n({c[1]}/{c[2]})",
                ha="center", va="bottom",
                fontsize=9, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL_TWO_LINE[c[0]] for c in counts], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("critic-sub-agent spawn rate", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig SI tier-bars — four panels (one per tier), Figure 6 style
# --------------------------------------------------------------------------- #
def fig_r4_tier_bars(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-tier pooled pass rate per condition with Wilson 95% CIs.

    Four side-by-side panels (Tier A--D), each in the same idiom as
    Figures 5--7 (vertical bars in condition colours, Wilson 95% CI
    whiskers, value labels above the cap). The redundant bootstrap-
    delta inset has been removed (Figure 5 already shows those).
    """
    tiers = ["A", "B", "C", "D"]
    NARR_ORDER = ["BARE", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC", "MCP_FULL"]
    SHORT_LABEL = {
        "BARE":              "BARE",
        "MCP_NO_PITFALL_DB": "MCP\n−PitDB",
        "MCP_NO_CRITIC":     "MCP\n−Critic",
        "MCP_FULL":          "MCP\nFull",
    }
    tier_titles = [f"Tier {t}\n{TIER_DESC[t].split('(')[1].rstrip(')')}"
                   for t in tiers]

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.8),
                             sharey=True, gridspec_kw={"wspace": 0.18})

    for ti, ax in enumerate(axes):
        tier = tiers[ti]
        vals, lo_arr, hi_arr, kns = [], [], [], []
        for c in NARR_ORDER:
            cs = [r for r in rows if r["condition"] == c and r["tier"] == tier]
            k = sum(1 for r in cs if r["passed"])
            n = len(cs)
            m = k / n if n else 0.0
            lo, hi = _wilson(k, n)
            vals.append(m); lo_arr.append(m - lo); hi_arr.append(hi - m); kns.append((k, n))
        vals = np.array(vals)
        x = np.arange(len(NARR_ORDER))
        ax.bar(x, vals, width=0.62,
               color=[COND_COLOR[c] for c in NARR_ORDER],
               edgecolor="white", linewidth=1.0, zorder=2)
        ax.errorbar(x, vals, yerr=[lo_arr, hi_arr],
                    fmt="none", ecolor="#222222",
                    capsize=4, capthick=1.0, lw=1.0, zorder=3)
        for xi, v, hi, (k, n) in zip(x, vals, hi_arr, kns):
            ax.text(xi, v + hi + 0.025,
                    f"{v*100:.0f}%\n({k}/{n})",
                    ha="center", va="bottom",
                    fontsize=8.5, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABEL[c] for c in NARR_ORDER], fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_title(tier_titles[ti], fontsize=10.5, pad=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.7, len(NARR_ORDER) - 0.3)

    axes[0].set_ylabel("pooled pass rate", fontsize=10)
    axes[0].set_ylim(0.0, 1.18)
    axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_r4_deltas_only(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Pooled pass rate per condition with Wilson 95% confidence intervals.

    Bar chart in the standard AI-benchmark idiom (cf. SWE-bench, AgentBench,
    tau-bench). Bars are coloured by condition consistent with the rest of
    the paper; whiskers are exact two-sided Wilson 95% CIs on a binomial
    rate of k passes out of n attempts. Pairwise bootstrap deltas between
    conditions are reported in the figure caption rather than on the figure.
    """
    NARR_ORDER = ["BARE", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC", "MCP_FULL"]
    counts = []
    for c in NARR_ORDER:
        cs = [r for r in rows if r["condition"] == c]
        k = sum(1 for r in cs if r["passed"])
        n = len(cs)
        p = k / max(n, 1)
        lo, hi = _wilson(k, n)
        counts.append((c, k, n, p, lo, hi))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(NARR_ORDER))
    rates = np.array([c[3] for c in counts])
    lo_arr = np.array([c[3] - c[4] for c in counts])
    hi_arr = np.array([c[5] - c[3] for c in counts])

    ax.bar(x, rates, width=0.62,
           color=[COND_COLOR[c[0]] for c in counts],
           edgecolor="white", linewidth=1.0, zorder=2)
    ax.errorbar(x, rates, yerr=[lo_arr, hi_arr],
                fmt="none", ecolor="#222222",
                capsize=5, capthick=1.2, lw=1.2, zorder=3)

    # Value labels above the upper CI cap
    for i, c in enumerate(counts):
        ax.text(i, c[5] + 0.025,
                f"{c[3]*100:.1f}%\n({c[1]}/{c[2]})",
                ha="center", va="bottom",
                fontsize=9, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL_TWO_LINE[c[0]] for c in counts], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("Pooled pass rate", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def fig_r5_pass_at_k(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Reliability metric: fraction of tasks passing >= k of 3 seeds, per condition.

    Three side-by-side panels (k=1, k=2, k=3), each a vertical bar
    chart in the same style as Figure 5 (four conditions, condition
    colours, value label above each bar). Shared y-axis so panels are
    directly comparable; the k=3 panel carries the headline reliability
    gap between MCP-Full and the rest.
    """
    by_ct: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        by_ct[(r["condition"], r["task_id"])].append(bool(r["passed"]))
    n_tasks = len({t for (_, t) in by_ct})

    ks = [1, 2, 3]
    rho_by_cond: dict[str, list[float]] = {}
    for cond in COND_ORDER:
        per_task_k = [sum(by_ct[(cond, t)])
                      for (cc, t) in by_ct if cc == cond]
        if not per_task_k:
            continue
        rho_by_cond[cond] = [
            sum(1 for k in per_task_k if k >= kk) / len(per_task_k)
            for kk in ks
        ]

    # Match COND_ORDER for left-to-right placement; use the same NARR_ORDER
    # as Figure 5 (BARE -> NoPitDB -> NoCritic -> Full) so the reader sees
    # the same progression in both figures.
    NARR_ORDER = ["BARE", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC", "MCP_FULL"]
    NARR_ORDER = [c for c in NARR_ORDER if c in rho_by_cond]

    # Wider, more compact labels: short condition tag on top, the
    # qualifier in parentheses on a second line at small size.
    SHORT_LABEL = {
        "BARE":              "BARE",
        "MCP_NO_PITFALL_DB": "MCP\n−PitDB",
        "MCP_NO_CRITIC":     "MCP\n−Critic",
        "MCP_FULL":          "MCP\nFull",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8),
                             sharey=True, gridspec_kw={"wspace": 0.18})
    k_titles = [r"$k = 1$ (any seed)",
                r"$k = 2$ (majority)",
                r"$k = 3$ (all seeds)"]

    # Wilson 95% CIs on k_pass / n_tasks per (condition, k)
    def _ci(cond: str, ki: int) -> tuple[float, float]:
        rho = rho_by_cond[cond][ki]
        k = int(round(rho * n_tasks))
        return _wilson(k, n_tasks)

    for ki, ax in enumerate(axes):
        vals = np.array([rho_by_cond[c][ki] for c in NARR_ORDER])
        cis  = [_ci(c, ki) for c in NARR_ORDER]
        lo_arr = vals - np.array([c[0] for c in cis])
        hi_arr = np.array([c[1] for c in cis]) - vals
        x = np.arange(len(NARR_ORDER))
        ax.bar(x, vals, width=0.58,
               color=[COND_COLOR[c] for c in NARR_ORDER],
               edgecolor="white", linewidth=1.0, zorder=2)
        ax.errorbar(x, vals, yerr=[lo_arr, hi_arr],
                    fmt="none", ecolor="#222222",
                    capsize=4, capthick=1.0, lw=1.0, zorder=3)
        for xi, v, (lo, hi) in zip(x, vals, cis):
            ax.text(xi, hi + 0.018, f"{v:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABEL[c] for c in NARR_ORDER],
                           fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_title(k_titles[ki], fontsize=10.5, pad=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.7, len(NARR_ORDER) - 0.3)

    axes[0].set_ylabel(f"share of {n_tasks} tasks", fontsize=10)
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = _load()
    if not rows:
        print("Run aggregate.py first.")
        return 1
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    # Main text:
    fig_r1_heatmap(rows, REPORT_OUT / "fig_R1_heatmap.png")
    fig_r3_critic_spawn(rows, REPORT_OUT / "fig_R3_critic_spawn.png")
    fig_r4_deltas_only(rows, REPORT_OUT / "fig_R4_deltas.png")
    fig_r5_pass_at_k(rows, REPORT_OUT / "fig_R5_pass_at_k.png")
    # Appendix (supplementary):
    fig_r2_tier_c_orders(rows, REPORT_OUT / "fig_SI_tier_c_orders.png")
    fig_r4_tier_bars(rows, REPORT_OUT / "fig_SI_tier_bars.png")
    print(f"Wrote 4 main-text + 2 SI figures to {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
