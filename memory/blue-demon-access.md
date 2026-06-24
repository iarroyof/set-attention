---
name: blue-demon-access
description: How to reach the blue-demon experiment server (credentials location, ssh pattern, gotchas)
metadata:
  type: reference
---

Blue-demon = remote experiment server, the authoritative source for completed runs/artifacts. Host `192.168.241.149`, user `iarroyof`, repo `~/set-attention`. 2× RTX 4090, runs in Docker service `set-attention`.

CREDENTIALS (do NOT print or commit the secret): file `../blue-demon.txt` relative to the repo root, i.e. `D:\UserFolders\Documents\GitHub\blue-demon.txt` (WSL: `/mnt/d/UserFolders/Documents/GitHub/blue-demon.txt`). Format = 4 lines `key: value`: line1 label, line2 `local IP address: 192.168.241.149`, line3 `user: iarroyof`, line4 `pass: <password>`. Extract pass with `grep -i '^pass' "$F" | sed 's/^[^:]*:[[:space:]]*//'`. A copy of the same password also exists at WSL `~/.ssh/.sshpass` (8 chars). The onboarding/context docs only mention `~/.ssh/.sshpass`, NOT `../blue-demon.txt` — this filled that gap.

ACCESS PATTERN that works from this Windows box: run through `wsl -d Ubuntu-24.04 -u iarroyof` (NOT the default Bash tool, which is Git-Bash as user `nachi` and has no creds). Put the password in `$SSHPASS` and use `sshpass -e ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149`. Key-based ssh is NOT configured (BatchMode → publickey/password denied), so a plain `ssh` or `sshpass -f` (which reads line1 of a structured file) FAILS — must use the parsed password. ALWAYS run a pre-written `.sh` file via `wsl ... bash /mnt/d/UserTemp/foo.sh` to dodge PowerShell→wsl→bash→ssh quoting hell; inline nested quoting breaks. For remote multi-line commands use `ssh ... 'bash -s' <<'EOF' ... EOF`.

LESSON: do not declare "blocked on credentials" before checking `../blue-demon.txt` and using `sshpass -e` with the parsed `pass:` value. Verify a remote job is actually finished (CSV count + row count + running procs) before running a summarizer/verdict — "should be done" is not "is done". See [[set-attention-research-direction]].
