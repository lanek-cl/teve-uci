@echo off
setlocal

REM Define variables
set REPO_URL=https://github.com/lanek-cl/usach.git
set REPO_DIR=%USERPROFILE%\usach-repo

REM Clone or pull the repository
if exist "%REPO_DIR%" (
    echo Repository directory exists. Pulling latest changes...
    cd /d "%REPO_DIR%"
    git pull
) else (
    echo Cloning repository...
    git clone %REPO_URL% "%REPO_DIR%"
)

echo Deployment complete.

endlocal
