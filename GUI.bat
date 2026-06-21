@echo off
REM Sovereign Domain AI - offline desktop GUI launcher (double-click to run)
pushd "%~dp0sovereign_ai"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM Fully offline: load the cached embedding model with no network calls
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
start "Sovereign AI" pythonw -m sovereign.cli gui
popd
