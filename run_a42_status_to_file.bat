@echo off
title A4.2 GPU Status to File
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "sshpass -f $HOME/.ssh/.sshpass ssh iarroyof@192.168.241.149 'nvidia-smi && echo --- && tail -5 ~/set-attention/logs/a42_slice/worker_gpu0.log && echo --- && tail -5 ~/set-attention/logs/a42_slice/worker_gpu1.log' | tee /mnt/d/UserFolders/Documents/GitHub/set-attention/a42_status.txt"
pause
