@echo off
REM Double-click launcher for the Flashdrive Folder Generator on Windows.
REM Runs the GUI using whatever Python is installed. If Python is missing,
REM use the standalone .exe built by build_folder_generator.sh instead.
cd /d "%~dp0"
python folder_generator.py
if errorlevel 1 (
  py folder_generator.py
)
if errorlevel 1 (
  echo.
  echo Could not find Python. Either install Python from python.org
  echo or use the standalone FolderGenerator.exe.
  pause
)
