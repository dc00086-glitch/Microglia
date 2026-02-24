@echo off
REM ============================================================
REM Build MMPS as a standalone Windows app
REM
REM Run:     build_app.bat
REM Output:  dist\MMPS\MMPS.exe
REM ============================================================

echo ========================================
echo   MMPS App Builder (Windows)
echo ========================================

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo [2/3] Building app with PyInstaller...
pyinstaller MMPS.spec --noconfirm

echo.
echo [3/3] Build complete!
echo.
echo   App location: dist\MMPS\MMPS.exe
echo.
echo   Double-click dist\MMPS\MMPS.exe to run
pause
