---
name: blue-demon-access
description: How to reach the blue-demon experiment server (credentials location, ssh pattern, gotchas)
metadata:
  type: reference
---

Blue-demon is the short/medium experiment host assigned by the current matrix. Host
`192.168.241.149`, user `iarroyof`, repo `~/set-attention`, 2× RTX 4090. Lizmark is
`192.168.241.205`, the same user/repo, 2× RTX 6000 Ada. Each host is authoritative only for its
assigned cells; the local workspace holds canonical documentation and merged analysis.

HOST PREFERENCE (user directive 2026-07-20): if a test/probe/launch needs
≤24 GB VRAM, use **blue-demon** — its RTX 4090s are faster. Reserve Lizmark
for work that needs >24 GB (or massive parallelism). Applies to probes,
tests, and any launch that fits a 4090.

OFF-LAN ACCESS (verified 2026-07-23): Tailscale network `ai.deploys@gmail.com`.

- blue-demon Tailscale IP: `100.64.104.123` (tailscaled on the host). From
  anywhere: `sshpass -f ~/.ssh/.sshpass ssh iarroyof@100.64.104.123`.
- Lizmark has NO Tailscale; reach it through blue-demon as jump host:
  `sshpass -f ~/.ssh/.sshpass ssh -J iarroyof@100.64.104.123 iarroyof@192.168.241.205`
  (sshpass answers both password prompts; same password file). `scp`/`ssh -J`
  works the same way (`scp -J ...` or `-o ProxyJump=`).
- Local machine: Tailscale runs on the Windows host (this laptop,
  100.67.206.68); WSL2 routes through it — no tailscale client needed inside
  WSL. The Windows Tailscale app must be running and logged in.
- Optional hardening: install Tailscale on Lizmark (needs sudo + `tailscale
  up` login) to drop the jump hop; not required.

CREDENTIALS (do not print or commit the secret): use `sshpass -f ~/.ssh/.sshpass` for both hosts.
Do not parse structured repo-parent credential files.

ACCESS PATTERN: from WSL/Linux use
`sshpass -f ~/.ssh/.sshpass ssh iarroyof@<host>`. Run project/PyTorch commands in the
`set-attention:latest` container. For multi-line remote commands, prefer a checked-in script or
`ssh ... 'bash -s'`.

Follow `docs/sd_dense_paper5_matrix.md` for host ownership and `audit/phase_sd_status.md` for live
state. Verify completion from processes, metadata, epoch rows, and logs before summarizing.
