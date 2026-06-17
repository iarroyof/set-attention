@echo off
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc ^
  "SSHPASS=\"sshpass -f $HOME/.ssh/.sshpass\"; REMOTE=\"iarroyof@192.168.241.149\"; $SSHPASS ssh \"$REMOTE\" 'echo \"=== nvidia-smi ===\"; nvidia-smi; echo; echo \"=== A3.3 running processes ===\"; ps aux | grep run_a3_stride | grep -v grep || echo \"(none)\"; echo; echo \"=== last 30 lines of nohup log ===\"; tail -30 ~/set-attention/logs/a3_stride_sweep/nohup_a33.log 2>/dev/null || echo \"log not found\"'"
pause
