#!/usr/bin/env bash
# Set-attention adaptive host access (user directive 2026-07-23).
#
# Single entry point for ALL ssh/scp to the experiment hosts. Prefers the
# current LAN; falls back to Tailscale (blue direct, lizmark via blue jump)
# when the LAN is unreachable. The detected mode is cached for 10 minutes;
# any ssh connection failure forces one re-probe + retry, so moving between
# networks is seamless.
#
# Usage:
#   scripts/sa_ssh.sh blue    [remote command...]
#   scripts/sa_ssh.sh lizmark [remote command...]
#   scripts/sa_ssh.sh blue    scp <normal scp args...>
#   scripts/sa_ssh.sh lizmark scp <normal scp args...>
#   scripts/sa_ssh.sh mode              # print current detected mode
#
# Auth: sshpass -f ~/.ssh/.sshpass (override with SA_PASS). Never print secrets.
set -u
PASS="${SA_PASS:-$HOME/.ssh/.sshpass}"
USER_="${SA_USER:-iarroyof}"
BLUE_LAN="192.168.241.149"
BLUE_TS="100.64.104.123"
LIZ_LAN="192.168.241.205"
CACHE="${SA_MODE_CACHE:-/tmp/sa_ssh_mode}"
TTL=600

lan_up () {
  sshpass -f "$PASS" ssh -o ConnectTimeout=3 -o BatchMode=no \
    -o StrictHostKeyChecking=accept-new -o NumberOfPasswordPrompts=1 \
    "$USER_@$BLUE_LAN" true >/dev/null 2>&1
}

detect_mode () {
  if lan_up; then echo lan; else echo tailscale; fi
}

current_mode () {
  if [ -f "$CACHE" ] && [ $(( $(date +%s) - $(stat -c %Y "$CACHE") )) -lt $TTL ]; then
    cat "$CACHE"
  else
    detect_mode | tee "$CACHE"
  fi
}

resolve () { # host mode -> sets TARGET and JUMPFLAGS
  local host="$1" mode="$2"
  JUMPFLAGS=()
  case "$host" in
    blue)
      [ "$mode" = lan ] && TARGET="$BLUE_LAN" || TARGET="$BLUE_TS" ;;
    lizmark)
      TARGET="$LIZ_LAN"
      [ "$mode" = tailscale ] && JUMPFLAGS=(-J "$USER_@$BLUE_TS") ;;
    *) echo "unknown host: $host" >&2; return 2 ;;
  esac
}

run_once () { # mode, then remaining args already in "$@"
  :
}

case "${1:-}" in
  mode)
    echo "$(current_mode)"
    exit 0 ;;
  blue|lizmark) HOST_="$1"; shift ;;
  *) echo "usage: sa_ssh.sh blue|lizmark [cmd...] | sa_ssh.sh blue|lizmark scp <args> | sa_ssh.sh mode" >&2; exit 2 ;;
esac

MODE="$(current_mode)"
resolve "$HOST_" "$MODE"

if [ "${1:-}" = "scp" ]; then
  shift
  SCPJ=()
  if [ "${#JUMPFLAGS[@]}" -gt 0 ]; then SCPJ=(-o "ProxyJump=${JUMPFLAGS[1]}"); fi
  sshpass -f "$PASS" scp -o ConnectTimeout=8 "${SCPJ[@]}" "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    # access error: re-probe once and retry in the other mode
    MODE="$(detect_mode | tee "$CACHE")"
    resolve "$HOST_" "$MODE"
    SCPJ=()
    if [ "${#JUMPFLAGS[@]}" -gt 0 ]; then SCPJ=(-o "ProxyJump=${JUMPFLAGS[1]}"); fi
    sshpass -f "$PASS" scp -o ConnectTimeout=8 "${SCPJ[@]}" "$@"
    rc=$?
  fi
  exit $rc
fi

sshpass -f "$PASS" ssh -o ConnectTimeout=8 "${JUMPFLAGS[@]}" "$USER_@$TARGET" "$@"
rc=$?
if [ $rc -eq 255 ]; then
  # access error: re-probe once and retry in the other mode
  MODE="$(detect_mode | tee "$CACHE")"
  resolve "$HOST_" "$MODE"
  sshpass -f "$PASS" ssh -o ConnectTimeout=8 "${JUMPFLAGS[@]}" "$USER_@$TARGET" "$@"
  rc=$?
fi
exit $rc
