@echo off
rem Phase 1: write exercises and build the corpus of good answers.
rem Prepares Python and the virtual environment on first use, then runs run.py.
rem Any arguments are passed straight through, e.g.  run.bat --generator latest --new
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
