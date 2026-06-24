#!/bin/bash
# ============================================================================
# Build MMPS62426 as a macOS .app bundle using PyInstaller
#
# Prerequisites:
#   pip install pyinstaller PyQt5 numpy scipy scikit-image opencv-python-headless \
#               tifffile Pillow matplotlib
#
# Usage:
#   cd /path/to/Microglia
#   bash build_MMPS62426.sh
#
# Output:
#   dist/MMPS62426.app    — double-click to run
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  Building MMPS62426.app"
echo "======================================"

# Check for required files
if [ ! -f "MMPS62426.py" ]; then
    echo "ERROR: MMPS62426.py not found in current directory"
    echo "  Run this script from the directory containing it"
    exit 1
fi

# Look for icon
ICON=""
for candidate in \
    "$HOME/Desktop/MMPS.icns" \
    "$HOME/MMPS.icns" \
    "$HOME/Downloads/MMPS.icns" \
    "./MMPS.icns"; do
    if [ -f "$candidate" ]; then
        ICON="$candidate"
        echo "Found icon: $ICON"
        break
    fi
done

if [ -z "$ICON" ]; then
    echo "WARNING: MMPS.icns not found. Building without custom icon."
    echo "  Place MMPS.icns in this directory or on your Desktop to include it."
fi

# Check PyInstaller is installed (invoked as a module so PATH doesn't matter)
if ! python3 -c "import PyInstaller" &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    python3 -m pip install pyinstaller
fi

# Check key dependencies are installed
echo "Checking dependencies..."
python3 -c "
import PyQt5, numpy, scipy, skimage, cv2, tifffile, PIL, matplotlib
print('All dependencies found')
" || {
    echo ""
    echo "Missing dependencies. Install with:"
    echo "  python3 -m pip install PyQt5 numpy scipy scikit-image opencv-python-headless tifffile Pillow matplotlib"
    exit 1
}

# Copy icon to build directory if found elsewhere
if [ -n "$ICON" ] && [ ! -f "MMPS.icns" ]; then
    cp "$ICON" ./MMPS.icns
    echo "Copied icon to build directory"
fi

# Clean previous builds
rm -rf build/MMPS62426 dist/MMPS62426 dist/MMPS62426.app

echo ""
echo "Running PyInstaller..."
echo ""

# Build using the spec file (invoked as a module so PATH doesn't matter)
python3 -m PyInstaller MMPS62426.spec --noconfirm

echo ""
echo "======================================"

if [ -d "dist/MMPS62426.app" ]; then
    APP_SIZE=$(du -sh "dist/MMPS62426.app" | cut -f1)

    # Zip the .app for upload as a GitHub release asset
    ZIP_NAME="MMPS62426-mac.zip"
    rm -f "dist/$ZIP_NAME"
    ( cd dist && zip -r -q -y "$ZIP_NAME" MMPS62426.app )
    ZIP_SIZE=$(du -sh "dist/$ZIP_NAME" | cut -f1)

    echo "  BUILD SUCCESSFUL"
    echo ""
    echo "  App:  dist/MMPS62426.app   ($APP_SIZE)"
    echo "  Zip:  dist/$ZIP_NAME  ($ZIP_SIZE)  <- upload this to the GitHub release"
    echo ""
    echo "  To run:  open dist/MMPS62426.app"
    echo "  To install: drag dist/MMPS62426.app to /Applications"
    echo ""
    echo "  NOTE: On first launch macOS may block it."
    echo "  Fix: System Settings > Privacy & Security > Open Anyway"
    echo "  Or:  Right-click the app > Open"
    echo "======================================"
else
    echo "  BUILD FAILED"
    echo "  Check the output above for errors"
    echo "======================================"
    exit 1
fi
