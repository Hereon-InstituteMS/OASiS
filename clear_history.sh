#!/bin/bash
# Clear conversation history before launching a fresh agent session.
# This prevents anchoring bias from previous conversations.
#
# Preserves the per-project memory/ directory (auto-memory persisted across
# conversations) and any other non-conversation subdirectory.  Only the
# .jsonl conversation transcripts and the todos/ subdir are removed.
#
# Usage: ./clear_history.sh [-y|--yes]   (-y skips the confirmation prompt)

set -u

ASSUME_YES=0
case "${1:-}" in
    -y|--yes) ASSUME_YES=1 ;;
    "") ;;
    *) echo "Usage: $0 [-y|--yes]" >&2; exit 2 ;;
esac

# Derive the project-specific Claude history directory from the current path.
# Claude Code encodes the project path as: leading slash dropped, remaining
# slashes replaced with dashes, prefixed with a single dash.
PROJECT_DIR=$(pwd)
ENCODED_PATH=$(echo "$PROJECT_DIR" | sed 's|^/||;s|/|-|g')
HISTORY_DIR="$HOME/.claude/projects/-$ENCODED_PATH"

if [ ! -d "$HISTORY_DIR" ]; then
    echo "No history directory found at: $HISTORY_DIR"
    echo "Run this script from the oasis project root."
    exit 0
fi

# Enumerate what would be removed.  Use nullglob so empty matches collapse.
shopt -s nullglob
jsonl_files=("$HISTORY_DIR"/*.jsonl)
todos_dir="$HISTORY_DIR/todos"
shopt -u nullglob

if [ ${#jsonl_files[@]} -eq 0 ] && [ ! -d "$todos_dir" ]; then
    echo "Nothing to clear in: $HISTORY_DIR"
    [ -d "$HISTORY_DIR/memory" ] && echo "(memory/ preserved as expected)"
    exit 0
fi

echo "Will remove from $HISTORY_DIR:"
[ ${#jsonl_files[@]} -gt 0 ] && printf '  %s\n' "${jsonl_files[@]}"
[ -d "$todos_dir" ] && echo "  $todos_dir/"
echo "Will PRESERVE: $HISTORY_DIR/memory/ (auto-memory) and any other subdirs"

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

[ ${#jsonl_files[@]} -gt 0 ] && rm -f -- "${jsonl_files[@]}"
[ -d "$todos_dir" ] && rm -rf -- "$todos_dir"

echo "History cleared: $HISTORY_DIR"
