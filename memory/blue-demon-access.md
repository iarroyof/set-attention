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

CREDENTIALS (do not print or commit the secret): use `sshpass -f ~/.ssh/.sshpass` for both hosts.
Do not parse structured repo-parent credential files.

ACCESS PATTERN: from WSL/Linux use
`sshpass -f ~/.ssh/.sshpass ssh iarroyof@<host>`. Run project/PyTorch commands in the
`set-attention:latest` container. For multi-line remote commands, prefer a checked-in script or
`ssh ... 'bash -s'`.

Follow `docs/sd_dense_paper5_matrix.md` for host ownership and `audit/phase_sd_status.md` for live
state. Verify completion from processes, metadata, epoch rows, and logs before summarizing.
