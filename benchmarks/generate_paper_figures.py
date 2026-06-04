#!/usr/bin/env python3
"""
Generate publication-quality figures for the OASiS paper.

Figures:
1. Architecture diagram (text-based, convert to TikZ)
2. h-Convergence plot
3. Cross-solver benchmark summary table
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_convergence():
    """Generate h-convergence plot from convergence study results."""
    data_file = RESULTS_DIR / "convergence_poisson.json"
    if not data_file.exists():
        print("Run convergence_study.py first!")
        return

    with open(data_file) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"fenics": "#2196F3", "dealii": "#4CAF50", "fourc": "#FF5722"}
    markers = {"fenics": "o", "dealii": "s", "fourc": "^"}
    labels = {"fenics": "FEniCSx (P1 tri)", "dealii": "deal.II (Q1 quad)", "fourc": "4C (Q1 quad)"}

    # Plot 1: max(u) vs h
    for solver, entries in data.items():
        if not entries:
            continue
        h_vals = [e["h"] for e in entries]
        u_vals = [e["max_u"] for e in entries]
        ax1.plot(h_vals, u_vals, f"-{markers[solver]}", color=colors[solver],
                 label=labels[solver], markersize=8, linewidth=2)

    ax1.set_xlabel("Mesh size $h$", fontsize=13)
    ax1.set_ylabel("max($u$)", fontsize=13)
    ax1.set_title("Poisson $-\\Delta u = 1$ on $[0,1]^2$, $u=0$ on $\\partial\\Omega$", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Plot 2: Error convergence (difference from finest)
    finest_val = max(e["max_u"] for entries in data.values() for e in entries if e["h"] == min(ee["h"] for ee in entries))

    for solver, entries in data.items():
        if len(entries) < 2:
            continue
        h_vals = [e["h"] for e in entries[:-1]]  # skip finest
        errors = [abs(e["max_u"] - entries[-1]["max_u"]) for e in entries[:-1]]
        if all(err > 0 for err in errors):
            ax2.loglog(h_vals, errors, f"-{markers[solver]}", color=colors[solver],
                       label=labels[solver], markersize=8, linewidth=2)

    # Reference O(h²) slope
    h_ref = np.array([0.125, 0.0625, 0.03125])
    e_ref = 0.001 * (h_ref / 0.03125) ** 2
    ax2.loglog(h_ref, e_ref, "k--", alpha=0.5, linewidth=1.5, label="$O(h^2)$")

    ax2.set_xlabel("Mesh size $h$", fontsize=13)
    ax2.set_ylabel("$|u_h - u_{h/64}|$", fontsize=13)
    ax2.set_title("Convergence rate", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = RESULTS_DIR / "convergence_poisson.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    outpath_png = RESULTS_DIR / "convergence_poisson.png"
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved to {outpath}")


def generate_architecture_tikz():
    """Generate TikZ code for architecture diagram."""
    tikz = r"""
% OASiS Architecture Diagram
% Compile with: pdflatex -shell-escape architecture.tex
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, fit, calc}

\definecolor{userblue}{HTML}{1565C0}
\definecolor{llmgreen}{HTML}{2E7D32}
\definecolor{mcpgold}{HTML}{F57F17}
\definecolor{fenicsblue}{HTML}{2196F3}
\definecolor{dealigreen}{HTML}{4CAF50}
\definecolor{fourcred}{HTML}{FF5722}
\definecolor{febiogray}{HTML}{607D8B}

\begin{document}
\begin{tikzpicture}[
    >=Stealth,
    node distance=1.5cm,
    every node/.style={font=\sffamily},
    box/.style={draw, rounded corners=3pt, minimum height=1cm, minimum width=2.5cm,
                text centered, font=\sffamily\small, line width=0.8pt},
    tool/.style={draw, rounded corners=2pt, minimum height=0.6cm, minimum width=2cm,
                 text centered, font=\sffamily\scriptsize, fill=mcpgold!15},
]

% User
\node[box, fill=userblue!15, draw=userblue, minimum width=3cm] (user)
    {\textbf{User / Scientist}};

% LLM
\node[box, fill=llmgreen!15, draw=llmgreen, minimum width=3cm, below=1cm of user] (llm)
    {\textbf{LLM} (Claude, GPT, ...)};

% MCP Server
\node[box, fill=mcpgold!15, draw=mcpgold, minimum width=9cm, minimum height=2.5cm,
      below=1cm of llm] (mcp) {};
\node[above=0.1cm] at (mcp.north) {\textbf{OASiS (MCP Server)}};
\node[font=\sffamily\scriptsize, text=gray] at (mcp.center)
    {35 tools $\cdot$ 133 pitfalls $\cdot$ 34 materials $\cdot$ cross-solver validation};

% Backends
\node[box, fill=fenicsblue!15, draw=fenicsblue, below left=1.5cm and 3cm of mcp] (fenics)
    {\textbf{FEniCSx}\\[-2pt]\scriptsize dolfinx, Python};
\node[box, fill=dealigreen!15, draw=dealigreen, below left=1.5cm and 0cm of mcp] (dealii)
    {\textbf{deal.II}\\[-2pt]\scriptsize C++, CMake};
\node[box, fill=fourcred!15, draw=fourcred, below right=1.5cm and 0cm of mcp] (fourc)
    {\textbf{4C}\\[-2pt]\scriptsize YAML, MPI};
\node[box, fill=febiogray!15, draw=febiogray, below right=1.5cm and 3cm of mcp] (febio)
    {\textbf{FEBio}\\[-2pt]\scriptsize XML};

% Arrows
\draw[->, line width=1pt, userblue] (user) -- node[right, font=\sffamily\scriptsize] {natural language} (llm);
\draw[->, line width=1pt, llmgreen] (llm) -- node[right, font=\sffamily\scriptsize] {MCP protocol} (mcp);
\draw[->, line width=0.8pt, fenicsblue] (mcp.south west) ++(0.5,0) -- (fenics);
\draw[->, line width=0.8pt, dealigreen] (mcp.south) ++(-1,0) -- (dealii);
\draw[->, line width=0.8pt, fourcred] (mcp.south) ++(1,0) -- (fourc);
\draw[->, line width=0.8pt, febiogray] (mcp.south east) ++(-0.5,0) -- (febio);

\end{tikzpicture}
\end{document}
"""
    outpath = RESULTS_DIR / "architecture.tex"
    outpath.write_text(tikz)
    print(f"Architecture TikZ saved to {outpath}")


def generate_benchmark_table():
    """Generate LaTeX benchmark table."""
    benchmarks = [
        ("Poisson $[0,1]^2$", "3", "0.07361", "0.07373", "0.07373", "99.8\\%"),
        ("Elasticity 10$\\times$1", "3", "12.961", "13.228", "13.228", "98.0\\%"),
        ("Heat $T$=100$\\to$0", "3", "100.0", "100.0", "100.0", "100\\%"),
        ("Poisson L-domain", "3", "0.1493", "0.1492", "0.1492", "99.9\\%"),
        ("Poisson $[0,2]{\\times}[0,1]$", "3", "0.1138", "0.1139", "0.1139", "99.95\\%"),
        ("Heat $[0,2]{\\times}[0,1]$", "3", "100.0", "100.0", "100.0", "100\\%"),
        ("Thick beam 5$\\times$2", "3", "0.2437", "0.2469", "0.2469", "98.7\\%"),
        ("Heat $T$=50$\\to$200", "3", "200.0", "200.0", "200.0", "100\\%"),
        ("Elasticity steel", "3", "0.06711", "0.06859", "0.06859", "97.8\\%"),
        ("\\textbf{Poisson 3D} $[0,1]^3$", "3", "0.05588", "0.05760", "0.05760", "97\\%"),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cross-solver benchmark results. All problems solved with the same boundary conditions and comparable mesh resolution.}",
        r"\label{tab:benchmarks}",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"Problem & Solvers & FEniCSx & deal.II & 4C & Agreement \\",
        r"\midrule",
    ]
    for row in benchmarks:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    outpath = RESULTS_DIR / "benchmark_table.tex"
    outpath.write_text("\n".join(lines))
    print(f"Benchmark table saved to {outpath}")


if __name__ == "__main__":
    plot_convergence()
    generate_architecture_tikz()
    generate_benchmark_table()
    print("\nAll paper figures generated!")
