@echo off
title A6.4 GPU Status to File
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "sshpass -f $HOME/.ssh/.sshpass ssh iarroyof@192.168.241.149 'nvidia-smi && echo --- && tail -10 ~/set-attention/logs/a64_depth_sweep/container_worker_gpu0.log && echo --- && tail -10 ~/set-attention/logs/a64_depth_sweep/container_worker_gpu1.log' | tee /mnt/d/UserFolders/Documents/GitHub/set-attention/a64_status.txt"
pause
