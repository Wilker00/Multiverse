@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "CLI_PY=%ROOT_DIR%tools\multiverse_cli.py"

if not exist "%CLI_PY%" (
  echo Error: CLI entrypoint not found at "%CLI_PY%"
  exit /b 1
)

python "%CLI_PY%" %*
if not %ERRORLEVEL%==9009 (
  exit /b %ERRORLEVEL%
)

py -3 "%CLI_PY%" %*
exit /b %ERRORLEVEL%
