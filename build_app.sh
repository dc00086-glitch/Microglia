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

# Build using the spec file (produces .app bundle on macOS)
python3 -m PyInstaller MMPS.spec --noconfirm

echo ""
if [[ -d dist/MMPS.app ]]; then
    echo "Done: dist/MMPS.app"
    echo ""
    echo "First launch: right-click > Open to bypass Gatekeeper."
    echo "After that it will open normally and fast."
else
    echo "Done: dist/MMPS/"
fi
