@echo off
rem
rem Run NDLOCR‑Lite web server inside a Python virtual environment.
rem
rem This batch script creates a virtual environment in the repository root (.venv)
rem if it does not already exist, installs required Python packages, and then
rem launches the FastAPI server defined in local_server/main.py.
rem Any additional command line arguments are passed through to the Python
rem interpreter.

set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV%
  python -m venv %VENV%
  call "%VENV%\Scripts\activate"
  python -m pip install --upgrade pip
  if exist requirements.txt (
    pip install -r requirements.txt
  ) else if exist ndlocr-lite-gui\requirements.txt (
    pip install -r ndlocr-lite-gui\requirements.txt
  )
  pip install fastapi uvicorn pypdfium2 pillow
) else (
  call "%VENV%\Scripts\activate"
)

rem Launch the server; forward any arguments
python server/main.py %*
