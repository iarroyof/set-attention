@echo off
title Paper Build
echo Building example_paper_working_agent.tex via WSL pdflatex...
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_build_paper.sh"
echo.
echo Build log: out\paper_build\build.log
echo Issues:    out\paper_build\build_issues.txt
pause
