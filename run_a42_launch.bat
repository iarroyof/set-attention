@echo off
title A4.2 Long-Context Slice Launch
echo Launching A4.2 long-context family slice on blue-demon via WSL...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_run_a42_launch.sh"
echo.
echo === A4.2 slice launched. Monitor blue-demon logs/a42_slice/nohup_a42.log ===
echo === When runs finish, double-click run_a42_sync.bat to collect artifacts ===
pause
