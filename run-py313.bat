@echo off
setlocal enabledelayedexpansion
rem
rem Run NDLOCR-Lite web server using Python 3.13 (py launcher).
rem
rem Python 3.14+ is not supported due to onnxruntime-gpu requiring numpy<2.3.
rem Use this script when your default python is 3.14+ but Python 3.13 is also
rem installed.  Install Python 3.13 from https://python.org if needed.
rem

set "VENV=.venv313"

rem --- Check that py launcher is available and Python 3.13 is installed ---
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [エラー] Python 3.13 が見つかりません。
    echo         https://python.org から Python 3.13 をインストールしてください。
    echo.
    pause
    exit /b 1
)

rem --- Copy config.toml.example -> config.toml if not present ---
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
  echo Creating virtual environment in %VENV% using Python 3.13
  py -3.13 -m venv %VENV%
  call "%VENV%\Scripts\activate"
  python -m pip install --upgrade pip
  echo Installing dependencies from %REQ_FILE% ...
  pip install --prefer-binary -r "%REQ_FILE%"
) else (
  call "%VENV%\Scripts\activate"
)

rem Launch the server; forward any arguments
python server/main.py %*
