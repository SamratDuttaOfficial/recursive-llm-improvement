@echo off
rem All three phases in order (optional convenience).
rem Prepares Python and the virtual environment on first use, then runs pipeline.py.
rem Any arguments are passed straight through, e.g.  pipeline.bat --cycles 2
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0pipeline.py" %*
