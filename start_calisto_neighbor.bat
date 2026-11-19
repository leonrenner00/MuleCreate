@echo off
rem Calisto Studio Create neighbor launcher (personal Workstation pin).
rem Product chrome is Calisto Studio — this only starts the GPL sketch_machine process.
rem Studio dashboard stays on :8770; sketch_machine MCP uses :8772.
setlocal
cd /d "%~dp0"

set "SKETCH_MACHINE_MCP_PORT=8772"
set "VPY=%~dp0.venv\Scripts\python.exe"
set "VPW=%~dp0.venv\Scripts\pythonw.exe"
set "EXE=%~dp0Mule\Mule.exe"
if exist "%~dp0..\Mule\Mule.exe" set "EXE=%~dp0..\Mule\Mule.exe"

if exist "%VPY%" goto venv_ok
echo Neighbor .venv missing. Creating with Python 3.13 (or 3.12+) ...
where py >nul 2>nul || goto need_python
py -3.13 -c "import sys" >nul 2>nul && goto make313
py -3.12 -c "import sys" >nul 2>nul && goto make312
goto need_python

:make313
py -3.13 -m venv .venv || goto failed
goto install

:make312
py -3.12 -m venv .venv || goto failed
goto install

:install
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt || goto failed

:venv_ok
"%VPY%" -c "import PySide6, numpy, moderngl, scipy, pxr" >nul 2>nul || (
  echo Installing packages into .venv ...
  "%VPY%" -m pip install --disable-pip-version-check -r requirements.txt || goto failed
)

if /I "%~1"=="--setup-only" (
  echo Neighbor venv ready. MCP port %SKETCH_MACHINE_MCP_PORT%.
  exit /b 0
)

rem Prefer console python when CALISTO_NEIGHBOR_CONSOLE=1 (logs visible); else pythonw for a normal GUI start.
if /I "%CALISTO_NEIGHBOR_CONSOLE%"=="1" (
  start "Calisto Create neighbor" "%VPY%" -m sketch_machine %*
) else if exist "%VPW%" (
  start "" "%VPW%" -m sketch_machine %*
) else (
  start "Calisto Create neighbor" "%VPY%" -m sketch_machine %*
)
echo Started sketch_machine with SKETCH_MACHINE_MCP_PORT=%SKETCH_MACHINE_MCP_PORT%
echo Fallback packaged exe (no Calisto port patch): %EXE%
exit /b 0

:need_python
echo.
echo Need Python 3.12+ on PATH ^(py launcher^), or unzip Mule_Windows.zip beside this folder.
echo Upstream also ships start.bat ^(wants 3.14^). Calisto uses this launcher + .venv instead.
pause
exit /b 1

:failed
echo.
echo Neighbor setup failed. Delete .venv and re-run, with internet once.
pause
exit /b 1
