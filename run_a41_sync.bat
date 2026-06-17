@echo off
title A4.1 Long-Context Smoke Sync
echo Running A4.1 post-run summarizer sync via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a41_steps.sh"
echo.
echo === Log written to run_a41_sync.log if redirected ===
pause
