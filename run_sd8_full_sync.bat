@echo off
title SD-8 All-Past Dense-Router Full Sync
echo Running SD-8 full summarizer sync via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_sd8_full_steps.sh"
echo.
echo === SD-8 full sync complete ===
pause
