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

rem --- Python version check (3.14+ は非対応) ---
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if %PY_MAJOR% GTR 3 goto :warn_ver
if %PY_MAJOR% EQU 3 if %PY_MINOR% GEQ 14 goto :warn_ver
goto :ver_ok
:warn_ver
echo.
echo [警告] Python %PY_VER% は非対応です（推奨: 3.11〜3.13）
echo        Python 3.13 がインストール済みであれば run-py313.bat を使用してください。
echo        インストールされていない場合は以下のコマンドでインストールできます:
echo          py -3.13 （py ランチャー経由）または https://python.org からダウンロード
echo.
pause
:ver_ok

rem --- Copy config.toml.example → config.toml if not present ---
if not exist "config.toml" (
  if exist "config.toml.example" (
    echo Copying config.toml.example to config.toml
    copy /y "config.toml.example" "config.toml" >nul
  )
)

rem --- Select requirements based on CUDA availability ---
nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "REQ_FILE=requirements-cpu.txt"
    echo CUDA not detected - using CPU (onnxruntime^)
) else (
    set "REQ_FILE=requirements-gpu.txt"
    echo CUDA detected - using GPU (onnxruntime-gpu^)
)

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV%
  python -m venv %VENV%
  call "%VENV%\Scripts\activate"
  python -m pip install --upgrade pip
  echo Installing dependencies from %REQ_FILE% ...
  pip install --prefer-binary -r "%REQ_FILE%"
) else (
  call "%VENV%\Scripts\activate"
)

rem Launch the server; forward any arguments
python server/main.py %*
