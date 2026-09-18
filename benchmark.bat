@echo off
rem Phase 3: measure every model that has no results yet.
rem Prepares Python and the virtual environment on first use, then runs benchmark.py.
rem Any arguments are passed straight through, e.g.  benchmark.bat --models base,v1
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0benchmark.py" %*
