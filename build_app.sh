#!/bin/bash
# ============================================================
# Build MMPS.app — run from the Microglia folder
# ============================================================
set -e

# Copy MMPS.icns from Downloads if not already here
if [[ ! -f MMPS.icns ]]; then
    if [[ -f ~/Downloads/MMPS.icns ]]; then
        cp ~/Downloads/MMPS.icns .
        echo "Copied MMPS.icns from Downloads"
    else
        echo "Warning: MMPS.icns not found in ~/Downloads or current directory"
        echo "         The app will build without a custom icon."
    fi
fi

# Build using the spec file (which now references the icon)
python3 -m PyInstaller MMPS.spec --noconfirm

echo ""
echo "Done: dist/MMPS"
