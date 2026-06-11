#!/usr/bin/env bash
# repopulate.sh — fill the scrubbed placeholders in this snapshot with real values
# before installing it on a host. The public repo ships with placeholders
# (__LAN_IP__, __HOSTNAME__) so it leaks no live network coordinates; run this
# once after cloning to populate them for your machine.
#
#   ./repopulate.sh                       # auto-detect LAN IP + short hostname
#   ./repopulate.sh --ip 10.0.0.5 --host mybox
#   LAN_IP=10.0.0.5 HOST_NAME=mybox ./repopulate.sh
#   ./repopulate.sh --dry-run             # show what would change, edit nothing
#
# NOTE: edits files in place. Do NOT commit the populated files back to the
# public repo — `git checkout -- .` (or re-clone) to restore placeholders.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAN_IP="${LAN_IP:-}"
HOST_NAME="${HOST_NAME:-}"
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --ip)      LAN_IP="${2:?--ip needs a value}"; shift 2 ;;
    --host)    HOST_NAME="${2:?--host needs a value}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 1 ;;
  esac
done

# Auto-detect the primary private IPv4 (the source IP used for outbound routing,
# falling back to the first RFC-1918 address on any interface).
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1 || true)"
  if [ -z "$LAN_IP" ]; then
    LAN_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' \
      | grep -E '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' | head -1 || true)"
  fi
fi
[ -z "$HOST_NAME" ] && HOST_NAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || true)"

[ -z "$LAN_IP" ]    && { echo "ERROR: could not detect a LAN IP; pass --ip <addr>"  >&2; exit 1; }
[ -z "$HOST_NAME" ] && { echo "ERROR: could not detect hostname; pass --host <name>" >&2; exit 1; }

echo "LAN_IP   = $LAN_IP"
echo "HOSTNAME = $HOST_NAME"

# Files still containing placeholders (exclude this script and .git)
mapfile -t FILES < <(grep -rlE '__LAN_IP__|__HOSTNAME__' "$DIR" 2>/dev/null \
  | grep -v -e '/\.git/' -e 'repopulate.sh' || true)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "No placeholders found — already populated (or run from the wrong directory)."
  exit 0
fi

for f in "${FILES[@]}"; do
  if [ "$DRY" -eq 1 ]; then
    echo "--- would edit: ${f#$DIR/}"
    grep -nE '__LAN_IP__|__HOSTNAME__' "$f" | sed 's/^/    /'
  else
    sed -i -e "s/__LAN_IP__/${LAN_IP}/g" -e "s/__HOSTNAME__/${HOST_NAME}/g" "$f"
    echo "populated: ${f#$DIR/}"
  fi
done

[ "$DRY" -eq 1 ] && echo "(dry run — no files changed)" \
  || echo "Done. Reminder: do not commit populated files back to the public repo."
