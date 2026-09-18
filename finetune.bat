@echo off
rem Phase 2: fine-tune on the answers the corpus holds.
rem Prepares Python and the virtual environment on first use, then runs finetune.py.
rem Any arguments are passed straight through, e.g.  finetune.bat --from-model latest
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0finetune.py" %*
