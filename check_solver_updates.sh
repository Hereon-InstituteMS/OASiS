#!/bin/bash
# Check for solver source and pip updates that might affect the MCP knowledge base.
# Run periodically (e.g., before a test campaign) to catch API changes.

set -u

echo "=== Solver Freshness Check ==="
echo "Date: $(date)"
echo

# Resolve a pip executable that works inside or outside a venv.
PIP="python3 -m pip"

# check_source <display_name> <env_var_value> <env_var_name>
# Runs git inspection inside a subshell so cwd never leaks between checks.
check_source() {
    local name=$1 root=$2 var=$3
    echo "--- $name (source) ---"
    if [ -n "$root" ] && [ -d "$root" ]; then
        (
            cd "$root" || exit 1
            echo "  Path: $root"
            echo "  Branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
            echo "  Last commit: $(git log -1 --oneline 2>/dev/null || echo 'N/A')"
            local last_ct
            last_ct=$(git log -1 --format=%ct 2>/dev/null)
            if [ -n "$last_ct" ]; then
                local days=$(( ($(date +%s) - last_ct) / 86400 ))
                echo "  Age: $days days"
                if [ "$days" -gt 14 ]; then
                    echo "  WARNING: $name source is $days days old. Consider: (cd $root && git pull)"
                fi
            else
                echo "  Age: N/A (not a git repo or no commits)"
            fi
        )
    else
        echo "  Not configured (set $var)"
    fi
}

# check_pip <pkg> <label>
check_pip() {
    local pkg=$1 label=$2
    local ver
    ver=$($PIP show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')
    if [ -n "$ver" ]; then
        echo "  pip: $label ($pkg) $ver"
    else
        echo "  pip: $label ($pkg) not installed"
    fi
}

# 4C
check_source "4C" "${FOURC_ROOT:-}" "FOURC_ROOT"
if [ -n "${FOURC_BINARY:-}" ] && [ -f "$FOURC_BINARY" ]; then
    echo "  Binary: $FOURC_BINARY ($(stat -c '%y' "$FOURC_BINARY" | cut -d' ' -f1))"
fi
echo

# Kratos
check_source "Kratos" "${KRATOS_ROOT:-}" "KRATOS_ROOT"
check_pip "KratosMultiphysics" "Kratos core"
check_pip "KratosDEMApplication" "Kratos DEM"
check_pip "KratosStructuralMechanicsApplication" "Kratos SMA"
echo

# deal.II
check_source "deal.II" "${DEALII_ROOT:-}" "DEALII_ROOT"
DEALII_VER=$(dpkg -l 2>/dev/null | awk '/^ii +libdeal\.ii-dev / {print $3; exit}')
if [ -n "$DEALII_VER" ]; then
    echo "  System package: deal.II $DEALII_VER"
fi
echo

# FEniCSx
check_source "FEniCSx" "${FENICS_ROOT:-}" "FENICS_ROOT"
check_pip "fenics-dolfinx" "FEniCSx"
echo

# NGSolve
check_source "NGSolve" "${NGSOLVE_ROOT:-}" "NGSOLVE_ROOT"
check_pip "ngsolve" "NGSolve"
echo

# scikit-fem
check_source "scikit-fem" "${SKFEM_ROOT:-}" "SKFEM_ROOT"
check_pip "scikit-fem" "scikit-fem"
echo

# DUNE-fem
check_source "DUNE-fem" "${DUNE_ROOT:-}" "DUNE_ROOT"
check_pip "dune-fem" "DUNE-fem"
echo

# FEBio (binary; not a Python package)
check_source "FEBio" "${FEBIO_ROOT:-}" "FEBIO_ROOT"
if [ -n "${FEBIO_BINARY:-}" ] && [ -f "$FEBIO_BINARY" ]; then
    echo "  Binary: $FEBIO_BINARY ($(stat -c '%y' "$FEBIO_BINARY" | cut -d' ' -f1))"
fi
echo

echo "=== Summary ==="
echo "If any solver has been updated, review the changelog and update"
echo "the MCP knowledge base pitfalls if needed."
echo "Run: pytest tests/test_physics_coverage.py::TestSolverFreshness -v"
