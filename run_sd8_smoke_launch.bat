@echo off
title SD-8 All-Past Dense-Router Smoke Launch
echo Launching SD-8 all_past dense-router smoke on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_sd8_smoke_launch.sh"
echo.
echo === SD-8 smoke launched. Monitor blue-demon logs/sd8_all_past_dense_routerdense_smoke/nohup_sd8_smoke.log ===
echo === When it finishes, double-click run_sd8_smoke_sync.bat to collect artifacts ===
pause
