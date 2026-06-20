@echo off
chcp 65001 >nul
REM 소버린 도메인 AI — 단일 질문 실행기. 사용: ask.bat "질문 내용"
cd /d "%~dp0sovereign_ai"
if "%~1"=="" (
  set /p Q="질문을 입력하세요> "
) else (
  set "Q=%~1"
)
python -m sovereign.cli ask "%Q%"
echo.
pause
