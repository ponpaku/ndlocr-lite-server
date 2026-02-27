@echo off
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

rem --- Read runtime.package from config.toml (default: gpu) ---
rem     Supported values: gpu / directml / cpu
rem     !! runtime.package を変更した場合は .venv を削除して再実行してください !!
set "NDLOCR_RUNTIME=gpu"
if exist "config.toml" (
    for /f "usebackq tokens=*" %%a in (`findstr /B "package" config.toml 2^>nul`) do (
        for /f "tokens=2 delims=""" %%b in ("%%a") do set "NDLOCR_RUNTIME=%%b"
    )
)

if "%NDLOCR_RUNTIME%"=="gpu"       set "REQ_FILE=requirements-gpu.txt"
if "%NDLOCR_RUNTIME%"=="directml"  set "REQ_FILE=requirements-directml.txt"
if "%NDLOCR_RUNTIME%"=="cpu"       set "REQ_FILE=requirements-cpu.txt"
if not defined REQ_FILE            set "REQ_FILE=requirements-gpu.txt"

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
