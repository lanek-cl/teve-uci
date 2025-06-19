@echo off
cd /d "%~dp0\.."
REM Set the path to the Python script and the environment folder
set ENV_PATH=env

REM Check if the environment folder exists
if not exist "%ENV_PATH%" (
    echo Virtual environment not found. Creating a new one...
    python -m venv "%ENV_PATH%"
    echo Activating virtual environment...
    call "%ENV_PATH%\Scripts\activate.bat"
    echo Inatalling requirements...
    pip install uv
    uv pip install -r requirements.txt
    echo Deactivating virtual environment...
    deactivate
)

