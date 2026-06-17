@echo off
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc ^
  "SSHPASS=\"sshpass -f $HOME/.ssh/.sshpass\"; REMOTE=\"iarroyof@192.168.241.149\"; $SSHPASS ssh \"$REMOTE\" 'echo \"=== nvidia-smi ===\"; nvidia-smi; echo; echo \"=== A4.1 running processes ===\"; ps aux | grep run_a41_smoke | grep -v grep || echo \"(none)\"; echo; echo \"=== last 30 lines of nohup log ===\"; tail -30 ~/set-attention/logs/a41_smoke/nohup_a41.log 2>/dev/null || echo \"log not found\"'"
pause
