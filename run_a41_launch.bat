@echo off
title A4.1 Long-Context Smoke Launch
echo Launching A4.1 long-context smoke on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a41_launch.sh"
echo.
echo === A4.1 smoke launched. Monitor blue-demon logs/a41_smoke/nohup_a41.log ===
echo === When runs finish, double-click run_a41_sync.bat to collect artifacts ===
pause
