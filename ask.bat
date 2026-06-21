@echo off
REM YJH AI - single question launcher.  Usage: ask.bat "your question"
pushd "%~dp0sovereign_ai"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "Q=%~1"
if "%Q%"=="" set /p Q="Question> "
python -m sovereign.cli ask "%Q%"
popd
echo.
pause
