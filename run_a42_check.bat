@echo off
title A4.2 GPU Status Check
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "sshpass -f $HOME/.ssh/.sshpass ssh iarroyof@192.168.241.149 'nvidia-smi && echo --- && ps aux | grep run_a42 | grep -v grep'"
pause
