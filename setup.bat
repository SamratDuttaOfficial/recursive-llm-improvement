@echo off
rem ---------------------------------------------------------------------------
rem  Prepares everything this project needs, and touches nothing outside its own
rem  folder. No admin rights, no PATH changes, no effect on any Python you
rem  already use for other work.
rem
rem    .python\   a private CPython, downloaded ONLY when this machine has none
rem               that is new enough. It lives in this folder and nowhere else.
rem    .venv\     the environment every script runs in
rem
rem  Safe to run as often as you like: it checks before it does anything, and
rem  does nothing at all once the environment exists.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
set "ROOT=%~dp0"
set "PYDIR=%ROOT%.python"
set "VENV=%ROOT%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "PYVER=3.12.7"
set "PBSTAG=20241016"

if exist "%VPY%" (
    if not "%~1"=="-v" exit /b 0
    echo [setup] environment already prepared: %VENV%
    exit /b 0
)

rem ---- 1. is there already a Python we may use? -----------------------------
set "PY="
call :try "%PYDIR%\python.exe"
if not defined PY call :try python
if not defined PY call :try py
if defined PY (
    for /f "delims=" %%v in ('"%PY%" -c "import sys;print(sys.version.split()[0])"') do set "FOUND=%%v"
    echo [setup] using the Python already on this machine: %PY%  ^(!FOUND!^)
    goto :makevenv
)

rem ---- 2. none found: fetch a private one just for this project -------------
echo [setup] no Python 3.9+ found on this machine.
echo [setup] downloading a private CPython %PYVER% into %PYDIR%
echo [setup] (this copy is used only by this project - nothing else changes)
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (set "TRIPLE=aarch64-pc-windows-msvc") else (set "TRIPLE=x86_64-pc-windows-msvc")
set "RLI_TRIPLE=%TRIPLE%"
set "RLI_TGZ=%TEMP%\rli-cpython-%TRIPLE%.tar.gz"
set "RLI_TAG=%PBSTAG%"
set "RLI_VER=%PYVER%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $t=$env:RLI_TAG; $v=$env:RLI_VER; $tr=$env:RLI_TRIPLE; $u='https://github.com/astral-sh/python-build-standalone/releases/download/'+$t+'/cpython-'+$v+'+'+$t+'-'+$tr+'-install_only.tar.gz'; try { $h=@{'User-Agent'='rli-setup'}; $r=Invoke-RestMethod 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest' -Headers $h; $a=$r.assets ^| Where-Object { $_.name -like ('cpython-3.12.*-'+$tr+'-install_only.tar.gz') } ^| Select-Object -First 1; if ($a) { $u=$a.browser_download_url } } catch { }; Write-Host ('[setup] ' + $u); Invoke-WebRequest $u -OutFile $env:RLI_TGZ -UseBasicParsing"
if errorlevel 1 goto :nopython
if not exist "%RLI_TGZ%" goto :nopython

if exist "%PYDIR%_tmp" rmdir /s /q "%PYDIR%_tmp"
mkdir "%PYDIR%_tmp"
tar -xzf "%RLI_TGZ%" -C "%PYDIR%_tmp"
if errorlevel 1 goto :nopython
if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
move "%PYDIR%_tmp\python" "%PYDIR%" >nul
rmdir /s /q "%PYDIR%_tmp" 2>nul
del "%RLI_TGZ%" 2>nul
set "PY=%PYDIR%\python.exe"
if not exist "%PY%" goto :nopython
echo [setup] private Python ready: %PY%

rem ---- 3. the virtual environment every script runs in ----------------------
:makevenv
echo [setup] creating the virtual environment in %VENV%
"%PY%" -m venv "%VENV%"
if errorlevel 1 (
    echo [setup] could not create the virtual environment.
    exit /b 1
)
"%VPY%" -m pip install -q --upgrade pip wheel
echo [setup] ready. Heavier pieces ^(PyTorch, ruff, ...^) install themselves when first needed.
exit /b 0

:try
rem  %1 is a command or a path; sets PY when it is a usable Python 3.9+
if defined PY goto :eof
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%~1"
goto :eof

:nopython
echo.
echo [setup] Could not download a private Python automatically.
echo [setup] Install Python 3.9 or newer from https://www.python.org/downloads/
echo [setup] and run this again - it will use it and change nothing on your system.
exit /b 1
