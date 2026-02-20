@echo off
setlocal EnableExtensions

set "REMOTE_URL=https://github.com/Wilker00/Multiverse"
set "COMMIT_MSG=%~1"
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=chore: repository cleanup"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Not a git repository.
  exit /b 1
)

echo [1/4] Staging changes...
git add -A
if errorlevel 1 (
  echo [ERROR] Failed to stage changes.
  exit /b 1
)

echo [2/4] Creating commit...
git diff --cached --quiet
if not errorlevel 1 (
  git commit -m "%COMMIT_MSG%"
  if errorlevel 1 (
    echo [ERROR] Commit failed.
    exit /b 1
  )
) else (
  echo [INFO] Nothing to commit.
)

echo [3/4] Ensuring origin remote...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin "%REMOTE_URL%"
  if errorlevel 1 (
    echo [ERROR] Failed to add origin remote.
    exit /b 1
  )
)

for /f "delims=" %%B in ('git branch --show-current') do set "CURRENT_BRANCH=%%B"
if "%CURRENT_BRANCH%"=="" set "CURRENT_BRANCH=main"

echo [4/4] Pushing %CURRENT_BRANCH%...
git push -u origin "%CURRENT_BRANCH%"
if errorlevel 1 (
  echo [ERROR] Push failed.
  exit /b 1
)

echo [OK] Push complete.
exit /b 0
