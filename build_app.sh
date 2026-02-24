#!/bin/bash
# ============================================================
# Build MMPS as a standalone app
#
# macOS:   ./build_app.sh        -> dist/MMPS.app
# Windows: build_app.bat         -> dist\MMPS\MMPS.exe
# ============================================================

set -e

echo "========================================"
echo "  MMPS App Builder"
echo "========================================"

# 1. Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller

# 2. Build
echo ""
echo "[2/3] Building app with PyInstaller..."
pyinstaller MMPS.spec --noconfirm

# 3. Done
echo ""
echo "[3/3] Build complete!"
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  App location: dist/MMPS.app"
    echo ""
    echo "  To run:   open dist/MMPS.app"
    echo "  To move:  drag dist/MMPS.app to /Applications"
else
    echo "  App location: dist/MMPS/MMPS"
    echo ""
    echo "  To run:   ./dist/MMPS/MMPS"
fi
