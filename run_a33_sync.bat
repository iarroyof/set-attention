@echo off
title A3.3 Stride Sweep Sync
echo Running A3.3 post-run summarizer sync via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a33_steps.sh"
echo.
echo === Log written to run_a33_sync.log if redirected ===
pause
