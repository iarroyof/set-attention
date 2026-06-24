@echo off
title SD-8 All-Past Dense-Router Smoke Sync
echo Running SD-8 smoke summarizer sync via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_sd8_smoke_steps.sh"
echo.
echo === SD-8 smoke sync complete ===
pause
