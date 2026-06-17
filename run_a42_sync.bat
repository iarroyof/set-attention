@echo off
title A4.2 Long-Context Slice Sync
echo Running A4.2 post-run summarizer sync via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a42_steps.sh"
echo.
pause
