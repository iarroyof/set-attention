@echo off
title A3.3 Stride Sweep Launch
echo Launching A3.3 stride sweep on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a33_launch.sh"
echo.
echo === A3.3 sweep launched. Monitor blue-demon logs/a3_stride_sweep/nohup_a33.log ===
echo === When runs finish, double-click run_a33_sync.bat to collect artifacts ===
pause
