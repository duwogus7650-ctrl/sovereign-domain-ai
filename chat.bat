@echo off
REM YJH AI - offline chat launcher
pushd "%~dp0sovereign_ai"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ============================================
echo  YJH AI  (offline RAG chat)
echo  - Ollama must be running with qwen3:8b
echo  - This PC has no GPU: each answer takes minutes. Please wait.
echo  - Exit: type  exit  or press Enter on empty line
echo ============================================
echo.
python -m sovereign.cli chat
popd
echo.
echo (chat ended)
pause
