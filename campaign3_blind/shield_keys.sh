#!/usr/bin/env bash
# Seal / unseal the answer keys. Keys are unreadable while agents run.
D="$(cd "$(dirname "$0")" && pwd)"
case "$1" in
  seal)   chmod -R a-rwx "$D/keys" 2>/dev/null; chmod 000 "$D/keys"
          echo "SEALED  $(stat -c %A "$D/keys")  keys are unreadable" ;;
  unseal) chmod 755 "$D/keys"; chmod -R u+rwX "$D/keys"
          echo "UNSEALED $(stat -c %A "$D/keys")  grader may read" ;;
  status) stat -c '%A %n' "$D/keys" ;;
  *) echo "usage: shield_keys.sh {seal|unseal|status}"; exit 2 ;;
esac
