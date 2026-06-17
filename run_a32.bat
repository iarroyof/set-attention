@echo off
title A3.2 Summarizer Sync
echo Running A3.2 steps via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a32_steps.sh"
echo.
echo === Log written to run_a32.log if redirected ===
pause
