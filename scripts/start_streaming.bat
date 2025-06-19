@echo off
cd /d "%~dp0\.."
REM Set the path to the Python script and the environment folder
set SCRIPT_PATH=cms50.py
set ENV_PATH=env

REM Create venv and install dependencies
echo Checking virtual environment...
call scripts/env.bat

REM Activate the virtual environment
echo Activating virtual environment...
call "%ENV_PATH%\Scripts\activate"

REM Run the Python script
echo Running script...
python "%SCRIPT_PATH%"

REM Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate
