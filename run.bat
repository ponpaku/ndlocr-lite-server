@echo off
setlocal enabledelayedexpansion
rem
rem Run NDLOCR‑Lite web server inside a Python virtual environment.
rem
rem This batch script creates a virtual environment in the repository root (.venv)
rem if it does not already exist, installs required Python packages, and then
rem launches the FastAPI server defined in server/main.py.
rem Any additional command line arguments are passed through to the Python
rem interpreter.

set "VENV=.venv"

rem --- Copy config.toml.example → config.toml if not present ---
if not exist "config.toml" (
  if exist "config.toml.example" (
    echo Copying config.toml.example to config.toml
    copy /y "config.toml.example" "config.toml" >nul
  )
)

rem --- Always use onnxruntime-gpu on Windows (includes CPU fallback) ---
set "REQ_FILE=requirements-gpu.txt"

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV%
  python -m venv %VENV%
  call "%VENV%\Scripts\activate"
  python -m pip install --upgrade pip
  echo Installing dependencies from %REQ_FILE% ...
  pip install -r "%REQ_FILE%"
) else (
  call "%VENV%\Scripts\activate"
)

rem Launch the server; forward any arguments
python server/main.py %*
