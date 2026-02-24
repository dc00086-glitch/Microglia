#!/bin/bash
# ============================================================
# Build MMPS.app — run from the Microglia folder
# ============================================================
set -e

# Copy MMPS.icns from Downloads if not already here
if [[ ! -f MMPS.icns ]]; then
    cp ~/Downloads/MMPS.icns .
    echo "Copied MMPS.icns from Downloads"
fi

# Build
python3 -m PyInstaller --onefile --windowed --name "MMPS" --icon MMPS.icns MMPSv2.py

echo ""
echo "Done: dist/MMPS.app"
