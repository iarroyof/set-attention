@echo off
title A6.4 Set-Stack Depth Sweep Launch
echo Launching A6.4 set-stack depth sweep on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a64_launch.sh"
echo.
echo === A6.4 launched. Workers running detached inside the container on blue-demon ===
echo === Monitor with run_a64_status_to_file.bat, then sync with run_a64_sync.bat ===
pause
