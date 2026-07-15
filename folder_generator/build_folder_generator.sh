#!/bin/bash
# ============================================================================
# Build the Flashdrive Folder Generator as a fully standalone executable
# using PyInstaller -- so it runs on lab computers that have NO Python.
#
# Run this once on each OS you want a native build for:
#   * On macOS  -> produces dist/FolderGenerator.app  (and a unix binary)
#   * On Windows-> produces dist\FolderGenerator.exe
#
# Then copy the result onto your USB stick next to your presets.
#
# Prerequisites:
#   pip install pyinstaller
# (No other dependencies -- the tool is standard-library only.)
#
# Usage:
#   cd folder_generator
#   bash build_folder_generator.sh
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

rm -rf build/FolderGenerator dist/FolderGenerator dist/FolderGenerator.app dist/FolderGenerator.exe

echo "Building FolderGenerator..."
pyinstaller \
    --name FolderGenerator \
    --onefile \
    --windowed \
    --noconfirm \
    folder_generator.py

echo ""
echo "======================================"
echo "  BUILD COMPLETE"
echo ""
echo "  Result is in the dist/ folder:"
ls -1 dist/ 2>/dev/null | sed 's/^/    dist\//'
echo ""
echo "  Copy it onto your USB stick. It runs with no Python installed."
echo "======================================"
