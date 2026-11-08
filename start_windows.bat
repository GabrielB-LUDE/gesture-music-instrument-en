@echo off
REM Double-click to start the instrument.
cd /d "%~dp0"

if not exist venv (
  echo == Creating virtual environment ==
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

python tools\check_setup.py
python main.py %*
pause
