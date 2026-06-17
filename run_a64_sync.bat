@echo off
title A6.4 Set-Stack Depth Sweep Sync
echo Running A6.4 post-run summarizer and syncing artifacts via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a64_steps.sh"
echo.
pause
