@echo off
chcp 65001 >nul
REM 소버린 도메인 AI — 오프라인 대화 실행기 (더블클릭 또는 cmd에서 실행)
cd /d "%~dp0sovereign_ai"
echo [소버린 AI] sovereign_ai 폴더에서 chat 실행...
echo  - Ollama가 실행 중이고 qwen3:8b가 설치되어 있어야 합니다.
echo.
python -m sovereign.cli chat
echo.
pause
