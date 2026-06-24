@echo off
title SD-8 All-Past Dense-Router Full Launch
echo Launching SD-8 all_past dense-router full ladder on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_sd8_full_launch.sh"
echo.
echo === SD-8 full ladder launched. Monitor blue-demon logs/sd8_all_past_dense_routerdense/nohup_sd8_full.log ===
echo === When it finishes, double-click run_sd8_full_sync.bat to collect artifacts ===
pause
